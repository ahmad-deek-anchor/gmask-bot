"""Dynamic token universe on top of Coin Metrics reference data and asset metrics.

The curated list in ``tools.metrics.FULL_TOKEN_UNIVERSE`` (~28 tokens) is what the daily
jobs and the z-score reports run on, because every token there costs API calls and model
time. The chat tools should not be limited to it: any asset Coin Metrics knows can be
priced, charted and z-scored on demand. This module provides that:

    TokenUniverse.resolve("tao")        -> "tao_bittensor"   (Coin Metrics asset id)
    TokenUniverse.spot_markets(asset)   -> ["coinbase-tao_bittensor-usd-spot", ...]
    TokenUniverse.top_assets(100)       -> ranked by estimated market cap (or spot volume)

Data (checked 2026-09-16 on our key): ``reference_data_assets`` lists 6,252 assets;
``CapMrktEstUSD`` and ``volume_trusted_spot_usd_1d`` are served daily for ~840 / ~720
assets; ``catalog_market_candles_v2(asset=..., market_type="spot")`` lists every spot
market with candles. A full ranking pull is 6 calls / ~20 s, so it is cached for a day
(in memory and in ``data/universe_cache.json``) and can be warmed in the background.

Stablecoins, gold tokens and wrapped / staked duplicates (usdt, wbtc, steth, bnb_bsc ...)
are excluded from ``top_assets`` by default: a trading-desk "top 100" means tradeable
underlyings, not the same BTC or ETH counted four times.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CAP_METRIC = "CapMrktEstUSD"
VOL_METRIC = "volume_trusted_spot_usd_1d"
DEFAULT_TTL_S = 24 * 3600
DEFAULT_CACHE_PATH = Path(os.getenv("UNIVERSE_CACHE_PATH", "data/universe_cache.json"))
CHUNK = 150                # assets per asset-metrics call
MARKET_MAX_AGE_DAYS = 7    # a spot market counts as live if its last daily candle is this recent

# Preferred spot markets for an asset we have no curated mapping for, in order.
PREFERRED_MARKETS: tuple[tuple[str, str], ...] = (
    ("coinbase", "usd"), ("coinbase", "usdc"), ("kraken", "usd"), ("binance", "usdt"),
    ("bybit", "usdt"), ("okex", "usdt"), ("binance", "usdc"), ("kraken", "usdt"),
    ("bitstamp", "usd"), ("gate.io", "usdt"), ("crypto.com", "usd"), ("kucoin", "usdt"),
    ("mexc", "usdt"), ("hyperliquid", "usdc"),
)
MAX_MARKETS = 4

# Not "tokens" for a desk ranking: fiat / gold stables and wrapped or staked duplicates.
STABLES_AND_WRAPPED: frozenset[str] = frozenset({
    # fiat-referenced stablecoins
    "usdt", "usdc", "usds", "usde", "dai", "usd1", "usdg", "pyusd", "rlusd", "usdy", "usdd",
    "usdf", "bfusd", "u_unitedstables", "susde", "usd0", "gho", "stable", "fdusd", "tusd",
    "usdp", "frax", "lusd", "crvusd", "usdb", "usda", "eurc", "eurt", "usdtb", "usdx",
    "usdl", "deusd", "syrupusdc", "usdai", "rusd", "usdh", "usr", "buidl", "usyc",
    # gold
    "xaut", "paxg",
    # wrapped / staked / bridged duplicates of an underlying that is already in the list
    "steth", "wsteth", "wbeth", "weeth", "weth", "reth", "rseth", "lseth", "meth", "cbeth",
    "oseth", "frxeth", "sfrxeth", "ezeth", "ethx", "wbtc", "cbbtc", "lbtc", "fbtc", "tbtc",
    "solvbtc", "wsol", "bnsol", "jitosol", "msol", "bsol", "jupsol", "bnb_bsc", "wmatic",
    "wpol", "wavax", "wbnb", "rndr",
})


def display_symbol(asset_id: str) -> str:
    """'tao_bittensor' -> 'tao', 'sky_sky' -> 'sky', 'btc' -> 'btc'."""
    return asset_id.split("_", 1)[0] if "_" in asset_id else asset_id


class TokenUniverse:
    """Symbol resolution, spot-market discovery and market-cap ranking over Coin Metrics."""

    def __init__(self, client, cache_path: Optional[Path] = None, ttl_s: int = DEFAULT_TTL_S):
        self._client = client
        self._cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._ranking: Optional[pd.DataFrame] = None   # columns: asset, market_cap, spot_volume, as_of
        self._ranking_built_at: float = 0.0
        self._assets: Optional[set[str]] = None          # every Coin Metrics asset id
        self._resolved: dict[str, Optional[str]] = {}
        self._markets: dict[str, list[str]] = {}
        self._load_cache()

    # ------------------------------------------------------------------ cache

    def _load_cache(self) -> None:
        try:
            if not self._cache_path.exists():
                return
            raw = json.loads(self._cache_path.read_text())
            built = float(raw.get("built_at", 0))
            if time.time() - built > self._ttl_s:
                return
            self._ranking = pd.DataFrame(raw["ranking"])
            self._ranking_built_at = built
            self._markets = dict(raw.get("markets", {}))
            self._resolved = dict(raw.get("resolved", {}))
            logger.info("Universe cache loaded: %d ranked assets, built %s", len(self._ranking),
                        datetime.fromtimestamp(built, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        except Exception as e:  # noqa: BLE001 - a bad cache file is not fatal
            logger.warning("Universe cache unreadable (%s); rebuilding on demand", e)
            self._ranking = None

    def _save_cache(self) -> None:
        if self._ranking is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"built_at": self._ranking_built_at,
                       "ranking": self._ranking.assign(as_of=self._ranking["as_of"].astype(str)).to_dict("records"),
                       "markets": self._markets, "resolved": self._resolved}
            self._cache_path.write_text(json.dumps(payload))
        except Exception as e:  # noqa: BLE001
            logger.debug("Universe cache not written: %s", e)

    # ------------------------------------------------------------------ ranking

    def _fresh(self) -> bool:
        return self._ranking is not None and (time.time() - self._ranking_built_at) < self._ttl_s

    def ranking(self, force: bool = False) -> pd.DataFrame:
        """Latest daily market cap and trusted spot volume for every asset that has them."""
        with self._lock:
            if self._fresh() and not force:
                return self._ranking
            t0 = time.time()
            cat = self._client.catalog_asset_metrics_v2(metrics=[CAP_METRIC], page_size=10000).to_dataframe()
            if "frequency" in cat.columns:
                cat = cat[cat["frequency"] == "1d"]
            assets = sorted(cat["asset"].astype(str).unique().tolist())
            since = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
            frames = []
            for i in range(0, len(assets), CHUNK):
                try:
                    df = self._client.get_asset_metrics(
                        assets=assets[i:i + CHUNK], metrics=[CAP_METRIC, VOL_METRIC], frequency="1d",
                        start_time=since, page_size=10000,
                        ignore_forbidden_errors=True, ignore_unsupported_errors=True,
                    ).to_dataframe()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Universe ranking chunk %d failed: %s", i // CHUNK, e)
                    continue
                if df is not None and not df.empty:
                    frames.append(df)
            if not frames:
                raise RuntimeError("Coin Metrics returned no market-cap data for the universe ranking")
            df = pd.concat(frames, ignore_index=True)
            for c in (CAP_METRIC, VOL_METRIC):
                df[c] = pd.to_numeric(df.get(c), errors="coerce")
            df["time"] = pd.to_datetime(df["time"], utc=True)
            latest = df.sort_values("time").groupby("asset", as_index=False).last()
            out = pd.DataFrame({
                "asset": latest["asset"].astype(str),
                "market_cap": latest[CAP_METRIC],
                "spot_volume": latest[VOL_METRIC],
                "as_of": latest["time"].dt.strftime("%Y-%m-%d"),
            }).dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False).reset_index(drop=True)
            self._ranking, self._ranking_built_at = out, time.time()
            logger.info("Universe ranking built: %d assets in %.1fs (%d calls)", len(out), time.time() - t0, len(frames))
            self._save_cache()
            return out

    def top_assets(self, n: int = 100, by: str = "market_cap", exclude_stables_and_wrapped: bool = True,
                   classifier=None) -> pd.DataFrame:
        """Top-``n`` assets by 'market_cap' or 'spot_volume': rank, asset, symbol, market_cap, spot_volume, as_of.

        ``classifier(symbols, cm_ids)`` (optional, e.g. ``MessariProvider.classify``) returns a frame
        with ``symbol`` plus ``sector`` / ``sub_sector`` list columns that is left-joined on; the
        Coin Metrics asset id is passed per symbol so ticker collisions resolve correctly. A failing
        classifier only logs - the ranking is returned without sectors."""
        if by not in ("market_cap", "spot_volume"):
            raise ValueError("by must be 'market_cap' or 'spot_volume'")
        df = self.ranking()
        if exclude_stables_and_wrapped:
            df = df[~df["asset"].isin(STABLES_AND_WRAPPED)]
        df = df.dropna(subset=[by]).sort_values(by, ascending=False)
        df = df.assign(symbol=df["asset"].map(display_symbol)).drop_duplicates("symbol", keep="first")
        df = df.head(max(1, int(n))).reset_index(drop=True)
        df = df.assign(rank=range(1, len(df) + 1))
        out = df[["rank", "asset", "symbol", "market_cap", "spot_volume", "as_of"]]
        if classifier is not None and len(out):
            try:
                cls = classifier(out["symbol"].tolist(), dict(zip(out["symbol"], out["asset"])))
                if cls is not None and len(cls) and "symbol" in cls.columns:
                    keep = [c for c in ("sector", "sub_sector") if c in cls.columns]
                    cls = cls[["symbol"] + keep].drop_duplicates("symbol")
                    cls["symbol"] = cls["symbol"].astype(str).str.lower()
                    out = out.merge(cls, on="symbol", how="left")
            except Exception as e:  # noqa: BLE001
                logger.warning("top_assets classifier failed: %s", e)
        return out

    def warm(self) -> None:
        """Build the ranking if stale; safe to call from a background thread."""
        try:
            self.ranking()
        except Exception as e:  # noqa: BLE001
            logger.warning("Universe warm-up failed: %s", e)

    # ------------------------------------------------------------------ resolution

    def _all_assets(self) -> set[str]:
        if self._assets is None:
            try:
                df = self._client.reference_data_assets(page_size=10000).to_dataframe()
                self._assets = set(df["asset"].astype(str).tolist())
            except Exception as e:  # noqa: BLE001
                logger.warning("reference_data_assets failed: %s", e)
                self._assets = set()
        return self._assets

    def resolve(self, symbol: str) -> Optional[str]:
        """Coin Metrics asset id for a user symbol, or None when Coin Metrics has never heard of it.

        Exact id first ('btc' -> 'btc'); otherwise ids of the form '<symbol>_<qualifier>'
        ('tao' -> 'tao_bittensor', 'sky' -> 'sky_sky'), preferring the one with the largest
        market cap when several exist. Curated overrides live in providers.coinmetrics.ASSET_MAP
        and are applied by the provider before this is consulted.
        """
        s = (symbol or "").strip().lower()
        if not s:
            return None
        if s in self._resolved:
            return self._resolved[s]
        assets = self._all_assets()
        candidates = [a for a in assets if a == s or a.startswith(s + "_")]
        result: Optional[str] = None
        if len(candidates) == 1:
            result = candidates[0]
        elif candidates:
            try:
                caps = self.ranking().set_index("asset")["market_cap"]
                ranked = sorted(candidates, key=lambda a: float(caps.get(a, float("nan")) or 0.0) if a in caps.index else -1.0, reverse=True)
                # a qualified id ('sky_sky') with a real market cap beats a bare id with none
                result = ranked[0] if ranked else candidates[0]
            except Exception:  # noqa: BLE001
                result = s if s in candidates else candidates[0]
        self._resolved[s] = result
        return result

    def is_known(self, symbol: str) -> bool:
        return self.resolve(symbol) is not None

    # ------------------------------------------------------------------ markets

    def spot_markets(self, asset_id: str) -> list[str]:
        """Live spot markets for an asset id, preferred exchanges first (max MAX_MARKETS)."""
        if asset_id in self._markets:
            return list(self._markets[asset_id])
        try:
            cat = self._client.catalog_market_candles_v2(asset=asset_id, market_type="spot", page_size=5000).to_dataframe()
        except Exception as e:  # noqa: BLE001
            logger.debug("catalog_market_candles_v2 failed for %s: %s", asset_id, e)
            cat = None
        found: list[str] = []
        if cat is not None and not cat.empty:
            if "frequency" in cat.columns:
                cat = cat[cat["frequency"] == "1d"]
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=MARKET_MAX_AGE_DAYS)
            live = set(cat[pd.to_datetime(cat["max_time"], utc=True) >= cutoff]["market"].astype(str))
            for ex, quote in PREFERRED_MARKETS:
                m = f"{ex}-{asset_id}-{quote}-spot"
                if m in live:
                    found.append(m)
                if len(found) >= MAX_MARKETS:
                    break
            if not found:   # any live market whose base is the asset, most recently updated first
                rest = cat[cat["market"].astype(str).str.split("-").str[1] == asset_id]
                found = rest.sort_values("max_time", ascending=False)["market"].astype(str).head(MAX_MARKETS).tolist()
        self._markets[asset_id] = found
        return list(found)


__all__ = ["TokenUniverse", "display_symbol", "STABLES_AND_WRAPPED", "PREFERRED_MARKETS", "CAP_METRIC", "VOL_METRIC"]
