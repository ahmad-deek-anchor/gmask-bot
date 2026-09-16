"""Coin Metrics implementation of MarketDataProvider (spot data).

Only the spot methods are used in production (derivatives come from Amberdata
via CompositeProvider), but the derivative methods are kept functional so the
provider is complete on its own.

Spot price resolution order
---------------------------
1. asset metric PriceUSD              (not offered for several small-cap assets)
2. daily candle price_close from the token's spot markets (works for all)
3. asset metric ReferenceRateUSD      (returns 403 on our key at every frequency; treated as no data)

Intraday / live data (added 2026-09-16)
---------------------------------------
The key has full access to the market-level endpoints, and they are not delayed: 1-minute
candles land ~1 minute after the bar closes, market trades and top-of-book quotes are
live. Asset-level reference rates (ReferenceRateUSD) are forbidden at every frequency, so
"live price" means the latest trade / quote on the token's primary spot market (Coinbase
USD first, then the fallbacks in EXCHANGE_AVAILABILITY / DEFAULT_EXCHANGES), not an index.

    get_intraday_candles(token, frequency, lookback_minutes)  OHLCV bars, 1m .. 4h
    get_latest_trade(token)                                   last print: time, price, amount, side
    get_latest_quote(token)                                   top of book: bid / ask, sizes, time
    get_recent_trades(token, minutes)                         the tape for the last N minutes
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

# Asset IDs that differ between our token symbols and Coin Metrics IDs.
# Polygon is listed as "pol" since the rebrand (the old "matic" markets are dead);
# Sky is "sky_sky" ("sky" is Skycoin, "mkr" the legacy Maker token).
ASSET_MAP = {
    "sky": "sky_sky",
}

# Per-token spot markets to try, in order, as (exchange, quote)
EXCHANGE_AVAILABILITY = {
    "hype":    [("coinbase", "usd"), ("kraken", "usd"), ("bybit", "usdt"), ("okex", "usdt")],
    "syrup":   [("coinbase", "usd"), ("binance", "usdt"), ("kraken", "usd")],
    "fluid":   [("coinbase", "usd"), ("bybit", "usdt")],
    "aero":    [("coinbase", "usd"), ("kraken", "usd"), ("bybit", "usdt")],
    "spx":     [("coinbase", "usd"), ("binance", "usdt"), ("kraken", "usd"), ("okex", "usdt")],
    "ray":     [("coinbase", "usd"), ("kraken", "usd"), ("bybit", "usdt")],
    "pendle":  [("coinbase", "usd"), ("binance", "usdt"), ("kraken", "usd"), ("bybit", "usdt")],
    "morpho":  [("coinbase", "usd"), ("kraken", "usd"), ("bybit", "usdt")],
}
DEFAULT_EXCHANGES = [
    ("coinbase", "usd"),
    ("binance", "usdt"),
    ("kraken", "usd"),
    ("bybit", "usdt"),
]

TOKENS_WITHOUT_LIQUIDATIONS = {"fluid", "morpho", "pendle", "pump"}

# Candle frequencies the market-candles endpoint serves below daily (catalog_market_candles_v2,
# checked 2026-09-16 for coinbase-btc-usd-spot). "1d" is handled by the daily methods above.
INTRADAY_FREQUENCIES = ("1m", "5m", "10m", "15m", "30m", "1h", "4h")
MAX_INTRADAY_LOOKBACK_MIN = 7 * 24 * 60   # one week of bars per call is plenty for a chat answer
MAX_TAPE_MINUTES = 60                     # market trades: cap the tape window at an hour


def asset_id_for(token: str) -> str:
    """Map a token symbol to its Coin Metrics asset id (curated mapping only; the provider
    consults the dynamic universe for symbols outside it)."""
    return ASSET_MAP.get(token.lower(), token.lower())


def spot_markets_for(token: str) -> list[str]:
    """Curated candidate spot market ids for a token, in preference order (no network)."""
    asset_id = asset_id_for(token)
    return [f"{ex}-{asset_id}-{quote}-spot" for ex, quote in EXCHANGE_AVAILABILITY.get(token.lower(), DEFAULT_EXCHANGES)]


def is_curated(token: str) -> bool:
    """True when the token is in the curated universe (tools.metrics.FULL_TOKEN_UNIVERSE)."""
    try:
        from tools.metrics import FULL_TOKEN_UNIVERSE
    except Exception:  # noqa: BLE001
        return True
    return token.lower() in FULL_TOKEN_UNIVERSE


def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce `time` to tz-naive UTC daily datetime64[ns], sorted ascending."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None).dt.normalize()
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def _normalize_intraday_time(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce `time` to tz-aware UTC datetime64 (no day normalisation), sorted ascending."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def _utc_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class CoinMetricsProvider(MarketDataProvider):

    def __init__(self, client, universe=None):
        self._client = client
        self._universe = universe   # providers.universe.TokenUniverse, built lazily

    @property
    def universe(self):
        """Dynamic token universe (symbol resolution, market discovery, top-N ranking)."""
        if self._universe is None:
            from providers.universe import TokenUniverse
            self._universe = TokenUniverse(self._client)
        return self._universe

    def _asset_id(self, token: str) -> str:
        """Curated mapping for curated tokens; otherwise the dynamic universe (falls back to the symbol)."""
        t = token.lower()
        if t in ASSET_MAP or is_curated(t):
            return asset_id_for(t)
        try:
            return self.universe.resolve(t) or t
        except Exception as e:  # noqa: BLE001
            logger.debug("universe.resolve(%s) failed: %s", t, e)
            return t

    def _spot_markets(self, token: str) -> list[str]:
        """Curated market list for curated tokens; discovered live markets for everything else."""
        t = token.lower()
        if t in EXCHANGE_AVAILABILITY or is_curated(t):
            return spot_markets_for(t)
        try:
            asset_id = self.universe.resolve(t)
            if asset_id:
                found = self.universe.spot_markets(asset_id)
                if found:
                    return found
        except Exception as e:  # noqa: BLE001
            logger.debug("universe.spot_markets(%s) failed: %s", t, e)
        return spot_markets_for(t)

    # ------------------------------------------------------------------
    # Internal fetch helpers
    # ------------------------------------------------------------------

    def _asset_metric(self, token: str, metric: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """One daily asset metric as (time, <metric>) or None on any failure / empty."""
        try:
            df = self._client.get_asset_metrics(
                assets=[self._asset_id(token)], metrics=[metric],
                start_time=start_date.strftime("%Y-%m-%d"), end_time=end_date.strftime("%Y-%m-%d"),
                frequency="1d",
            ).to_dataframe()
        except Exception as e:
            logger.debug(f"CoinMetrics {metric} failed for {token}: {type(e).__name__}: {e}")
            return None
        if df is None or df.empty or metric not in df.columns:
            return None
        df = df[["time", metric]].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        return _normalize_time(df)

    def _market_candles(self, market: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Daily candles for one market, time-normalised; None on error / empty."""
        sd, ed = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        try:
            df = self._client.get_market_candles(
                markets=[market], start_time=sd, end_time=ed, frequency="1d", page_size=1000
            ).to_dataframe()
        except Exception as e:
            logger.debug(f"CoinMetrics candles failed for {market}: {type(e).__name__}: {e}")
            return None
        if df is None or df.empty:
            return None
        return _normalize_time(df)

    def _candles(self, token: str, start_date: datetime, end_date: datetime) -> list[pd.DataFrame]:
        """Daily candles from every spot market of the token that returns data (preference order)."""
        frames = []
        for market in self._spot_markets(token):
            df = self._market_candles(market, start_date, end_date)
            if df is not None:
                frames.append(df)
        return frames

    # ------------------------------------------------------------------
    # Spot price / OHLCV
    # ------------------------------------------------------------------

    def get_spot_ohlcv(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """OHLC from the token's primary spot market; spot_volume (USD) summed over
        every market in the token's exchange list that returned candles."""
        frames = self._candles(token, start_date, end_date)
        if not frames:
            return None
        rename = {
            "price_open": "open", "price_high": "high", "price_low": "low", "price_close": "close",
            "candle_usd_volume": "spot_volume",
        }
        frames = [f.rename(columns=rename) for f in frames]
        cols = ["time", "open", "high", "low", "close", "spot_volume"]
        primary = frames[0]
        if not all(c in primary.columns for c in cols):
            return None
        df = primary[cols].copy()
        for c in cols[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

        # Sum USD volume across all markets that returned data (NaN if none did on a day).
        vols = [
            pd.to_numeric(f.set_index("time")["spot_volume"], errors="coerce").astype("float64")
            for f in frames if "spot_volume" in f.columns
        ]
        if vols:
            total = pd.concat(vols, axis=1).sum(axis=1, min_count=1)
            df = df.merge(total.rename("spot_volume").reset_index(), on="time", how="outer", suffixes=("_primary", ""))
            df = df.drop(columns=["spot_volume_primary"]).sort_values("time").reset_index(drop=True)
        logger.info("CoinMetrics: %s spot volume summed over %d market(s)", token, len(vols))
        return df[cols]

    def get_spot_price(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        # 1. PriceUSD asset metric
        df = self._asset_metric(token, "PriceUSD", start_date, end_date)
        if df is not None:
            return df.rename(columns={"PriceUSD": "price"})

        # 2. candle close from the token's spot markets
        candles = self.get_spot_ohlcv(token, start_date, end_date)
        if candles is not None and candles["close"].notna().any():
            return candles[["time", "close"]].rename(columns={"close": "price"}).reset_index(drop=True)

        # 3. ReferenceRateUSD (403 on the trial key -> _asset_metric returns None; no retry)
        df = self._asset_metric(token, "ReferenceRateUSD", start_date, end_date)
        if df is not None:
            return df.rename(columns={"ReferenceRateUSD": "price"})

        logger.debug(f"CoinMetrics: no spot price for {token}")
        return None

    # ------------------------------------------------------------------
    # Intraday / live (market-level endpoints; see module docstring)
    # ------------------------------------------------------------------

    def _markets(self, token: str, market: Optional[str]) -> list[str]:
        return [market] if market else self._spot_markets(token)

    def get_intraday_candles(
        self, token: str, frequency: str = "5m", lookback_minutes: int = 180, market: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """OHLCV bars for the token's primary spot market (first market that returns data).

        Columns: time (UTC, tz-aware), open, high, low, close, vwap, volume (base units),
        usd_volume, trades. ``df.attrs["market"]`` names the market the bars came from.
        Returns None when the frequency is unsupported or no market returns data.
        """
        if frequency not in INTRADAY_FREQUENCIES:
            logger.debug("CoinMetrics: unsupported intraday frequency %r", frequency)
            return None
        lookback_minutes = max(1, min(int(lookback_minutes), MAX_INTRADAY_LOOKBACK_MIN))
        start = _utc_iso(datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes))
        rename = {"price_open": "open", "price_high": "high", "price_low": "low", "price_close": "close",
                  "candle_usd_volume": "usd_volume", "candle_trades_count": "trades"}
        cols = ["time", "open", "high", "low", "close", "vwap", "volume", "usd_volume", "trades"]
        for m in self._markets(token, market):
            try:
                df = self._client.get_market_candles(
                    markets=[m], frequency=frequency, start_time=start, page_size=1000
                ).to_dataframe()
            except Exception as e:
                logger.debug("CoinMetrics intraday candles failed for %s %s: %s: %s", m, frequency, type(e).__name__, e)
                continue
            if df is None or df.empty:
                continue
            df = _normalize_intraday_time(df.rename(columns=rename))
            for c in cols[1:]:
                if c not in df.columns:
                    df[c] = pd.NA
                df[c] = pd.to_numeric(df[c], errors="coerce")
            out = df[cols].reset_index(drop=True)
            out.attrs["market"] = m
            out.attrs["frequency"] = frequency
            return out
        return None

    def _latest_row(self, fetch, token: str, market: Optional[str], what: str) -> Optional[pd.DataFrame]:
        """Newest row of a market endpoint (paging from the end), trying markets in order."""
        for m in self._markets(token, market):
            try:
                df = fetch(m).to_dataframe()
            except Exception as e:
                logger.debug("CoinMetrics %s failed for %s: %s: %s", what, m, type(e).__name__, e)
                continue
            if df is None or df.empty:
                continue
            df = _normalize_intraday_time(df)
            row = df.iloc[[-1]].reset_index(drop=True)
            row.attrs["market"] = m
            return row
        return None

    def get_latest_trade(self, token: str, market: Optional[str] = None) -> Optional[dict]:
        """Last print on the token's primary spot market: {market, time, price, amount, side, usd}."""
        row = self._latest_row(
            lambda m: self._client.get_market_trades(markets=[m], paging_from="end", page_size=1, limit_per_market=1),
            token, market, "latest trade",
        )
        if row is None:
            return None
        r = row.iloc[0]
        price, amount = float(r["price"]), float(r["amount"])
        return {"market": row.attrs["market"], "time": r["time"].to_pydatetime(), "price": price,
                "amount": amount, "side": str(r.get("side", "")), "usd": price * amount}

    def get_latest_quote(self, token: str, market: Optional[str] = None) -> Optional[dict]:
        """Top of book on the token's primary spot market: {market, time, bid, ask, bid_size, ask_size, mid, spread_bp}."""
        row = self._latest_row(
            lambda m: self._client.get_market_quotes(markets=[m], paging_from="end", page_size=1, limit_per_market=1),
            token, market, "latest quote",
        )
        if row is None:
            return None
        r = row.iloc[0]
        bid, ask = float(r["bid_price"]), float(r["ask_price"])
        mid = (bid + ask) / 2 if bid and ask else None
        return {"market": row.attrs["market"], "time": r["time"].to_pydatetime(), "bid": bid, "ask": ask,
                "bid_size": float(r.get("bid_size", float("nan"))), "ask_size": float(r.get("ask_size", float("nan"))),
                "mid": mid, "spread_bp": ((ask - bid) / mid * 1e4) if mid else None}

    def get_recent_trades(
        self, token: str, minutes: int = 5, market: Optional[str] = None, max_rows: int = 20000
    ) -> Optional[pd.DataFrame]:
        """The tape for the last ``minutes`` on the token's primary spot market.

        Columns: time (UTC), price, amount (base), side ('buy'/'sell' = taker side), usd.
        ``df.attrs["market"]`` names the market. None when no market returns trades.
        """
        minutes = max(1, min(int(minutes), MAX_TAPE_MINUTES))
        start = _utc_iso(datetime.now(timezone.utc) - timedelta(minutes=minutes))
        for m in self._markets(token, market):
            try:
                df = self._client.get_market_trades(
                    markets=[m], start_time=start, page_size=10000, limit_per_market=max_rows
                ).to_dataframe()
            except Exception as e:
                logger.debug("CoinMetrics trades failed for %s: %s: %s", m, type(e).__name__, e)
                continue
            if df is None or df.empty:
                continue
            df = _normalize_intraday_time(df)
            out = pd.DataFrame({
                "time": df["time"],
                "price": pd.to_numeric(df["price"], errors="coerce"),
                "amount": pd.to_numeric(df["amount"], errors="coerce"),
                "side": df["side"].astype(str) if "side" in df.columns else "",
            })
            out["usd"] = out["price"] * out["amount"]
            out = out.dropna(subset=["price", "amount"]).reset_index(drop=True)
            out.attrs["market"] = m
            return out
        return None

    # ------------------------------------------------------------------
    # Derivatives (unused when composed with Amberdata; kept for completeness)
    # ------------------------------------------------------------------

    def get_funding_rate(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        metric = "futures_aggregate_funding_rate_usd_margin_1y_period"
        df = self._asset_metric(token, metric, start_date, end_date)
        return None if df is None else df.rename(columns={metric: "funding_rate"})

    def get_perp_oi(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        metric = "open_interest_reported_future_perpetual_usd"
        df = self._asset_metric(token, metric, start_date, end_date)
        return None if df is None else df.rename(columns={metric: "perp_oi"})

    def get_perp_volume(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        metric = "volume_reported_future_perpetual_usd_1d"
        df = self._asset_metric(token, metric, start_date, end_date)
        return None if df is None else df.rename(columns={metric: "perp_volume"})

    def get_liquidations(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        if token.lower() in TOKENS_WITHOUT_LIQUIDATIONS:
            return None
        pair = f"{self._asset_id(token)}-usd"
        buy, sell = "liquidations_reported_future_buy_usd_1d", "liquidations_reported_future_sell_usd_1d"
        try:
            df = self._client.get_pair_metrics(
                pairs=[pair], metrics=[buy, sell],
                start_time=start_date.strftime("%Y-%m-%d"), end_time=end_date.strftime("%Y-%m-%d"),
                frequency="1d",
            ).to_dataframe()
        except Exception as e:
            logger.debug(f"CoinMetrics liquidations failed for {token}: {e}")
            return None
        if df is None or df.empty or buy not in df.columns or sell not in df.columns:
            return None
        df = df.rename(columns={buy: "long_liquidations", sell: "short_liquidations"})
        for c in ("long_liquidations", "short_liquidations"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["total_liquidations"] = df["long_liquidations"].fillna(0) + df["short_liquidations"].fillna(0)
        return _normalize_time(df[["time", "long_liquidations", "short_liquidations", "total_liquidations"]])
