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
import re
import threading
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
MAX_INTRADAY_LOOKBACK_MIN = 10 * 24 * 60  # ten days of hourly bars: enough for a T-7d comparison with slack
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


# ---------------------------------------------------------------------------
# CME crypto futures and BTC ETF on-chain flows (added 2026-09-16)
# ---------------------------------------------------------------------------
#
# Our key has CME reference data (reference_data_markets exchange="cme" type="future"),
# daily / hourly candles with USD volume, ticks, and per-contract open interest that CME
# publishes once a day (timestamp 21:00 UTC). No mark or index price. Contracts that are
# listed but have never traded raise "market not supported" on the candle endpoint.
# Outright symbols are <product><month code><year>: BTCV6 = Oct 2026 (5 BTC), MBTV6 =
# micro (0.1 BTC), ETHV6 (50 ETH) / METV6 (0.1 ETH), SOLV6 (500 SOL) / MSLV6 (25 SOL,
# base 'msl'), XRPV6 (50,000 XRP) / MXPV6 (2,500 XRP, base 'mxp'); BFF<mmd> are the weekly
# Bitcoin Friday futures (0.02 BTC). Calendar spreads carry a hyphen (BTCU6-BTCV6).

CME_BASES = ("btc", "eth", "sol", "xrp")
# Coin Metrics 'base' codes that belong to an underlying (micro contracts of SOL / XRP and the
# weekly BTC product are booked under their own base codes).
CME_BASE_ALIASES = {"btc": ("btc", "bff"), "eth": ("eth",), "sol": ("sol", "msl"), "xrp": ("xrp", "mxp")}
CME_OUTRIGHT_RE = re.compile(r"^([A-Z]+?)([FGHJKMNQUVXZ])(\d{1,3})$")
CME_MONTH_CODES = "FGHJKMNQUVXZ"
CME_REFDATA_TTL_S = 3600
ETF_FLOW_METRICS = ("FlowInEtfUSD", "FlowOutEtfUSD")
ETF_SUPPLY_METRICS = ("SplyEtfNtv", "SplyEtfUSD")

def cme_product(symbol: str) -> Optional[tuple[str, str, str]]:
    """'BTCV6' -> ('BTC', 'V', '6'); 'BFFU618' -> ('BFF', 'U', '618'); spreads -> None."""
    m = CME_OUTRIGHT_RE.match(symbol or "")
    return (m.group(1), m.group(2), m.group(3)) if m else None


def cme_contract_label(symbol: str) -> str:
    """'BTCV6' -> 'Oct-26'; 'BFFU618' -> 'wk Sep-18'."""
    p = cme_product(symbol)
    if not p:
        return symbol
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mon = months[CME_MONTH_CODES.index(p[1])]
    if len(p[2]) == 3:
        return f"wk {mon}-{p[2][1:]}"
    return f"{mon}-2{p[2]}"


class CoinMetricsProvider(MarketDataProvider):

    def __init__(self, client, universe=None):
        self._client = client
        self._universe = universe   # providers.universe.TokenUniverse, built lazily
        self._cme_ref: Optional[pd.DataFrame] = None   # CME reference data, refreshed hourly
        self._cme_ref_at: float = 0.0
        self._cme_lock = threading.Lock()

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

    # ------------------------------------------------------------------
    # CME crypto futures + BTC ETF on-chain flows (see module notes above the class)
    # ------------------------------------------------------------------

    def cme_contracts(self, base: str, include_micro: bool = True, include_weekly: bool = False,
                      now: Optional[datetime] = None) -> pd.DataFrame:
        """Active CME outright futures on ``base`` (btc, eth, sol, xrp ...).

        Columns: market, symbol, product, label, base, listing, expiration, days_to_expiry,
        contract_size, size_asset, is_standard (product code == underlying), is_weekly.
        Sorted by expiration. Reference data is cached for an hour on the provider."""
        b = (base or "").lower()
        now = now or datetime.now(timezone.utc)
        with self._cme_lock:
            ref = self._cme_ref
            if ref is None or (datetime.now(timezone.utc).timestamp() - self._cme_ref_at) > CME_REFDATA_TTL_S:
                ref = self._client.reference_data_markets(exchange="cme", type="future", page_size=5000).to_dataframe()
                self._cme_ref, self._cme_ref_at = ref, datetime.now(timezone.utc).timestamp()
        cols = ["market", "symbol", "base", "listing", "expiration", "contract_size", "size_asset"]
        df = ref[[c for c in cols if c in ref.columns]].copy()
        df = df[df["base"].astype(str).str.lower().isin(CME_BASE_ALIASES.get(b, (b,)))]
        df["listing"] = pd.to_datetime(df["listing"], utc=True, errors="coerce")
        df["expiration"] = pd.to_datetime(df["expiration"], utc=True, errors="coerce")
        df = df[(df["listing"] <= pd.Timestamp(now)) & (df["expiration"] > pd.Timestamp(now))]
        out_cols = ["market", "symbol", "product", "label", "base", "listing", "expiration", "days_to_expiry",
                    "contract_size", "size_asset", "is_standard", "is_weekly"]
        prods = df["symbol"].astype(str).map(cme_product)
        df = df[prods.notna()].copy()
        if df.empty:
            return pd.DataFrame(columns=out_cols)
        df["product"] = [p[0] for p in prods[prods.notna()]]
        df["is_weekly"] = [len(p[2]) == 3 for p in prods[prods.notna()]]
        df["is_standard"] = df["product"].astype(str).str.lower() == b
        df["label"] = df["symbol"].map(cme_contract_label)
        df["contract_size"] = pd.to_numeric(df["contract_size"], errors="coerce")
        df["days_to_expiry"] = ((df["expiration"] - pd.Timestamp(now)).dt.total_seconds() / 86400).round(1)
        if not include_micro:
            df = df[df["is_standard"]]
        if not include_weekly:
            df = df[~df["is_weekly"]]
        out = df.sort_values(["expiration", "is_standard"], ascending=[True, False]).reset_index(drop=True)
        return out[out_cols]

    def _cme_candles(self, markets: list[str], start: datetime, frequency: str = "1d") -> pd.DataFrame:
        """Daily candles for several CME markets; one failing market ("market not supported") is skipped."""
        frames = []
        sd = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            df = self._client.get_market_candles(markets=list(markets), frequency=frequency, start_time=sd,
                                                 page_size=10000).to_dataframe()
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:  # noqa: BLE001 - fall back to one call per market
            logger.debug("CME batch candles failed (%s); retrying per market", e)
            for m in markets:
                try:
                    df = self._client.get_market_candles(markets=[m], frequency=frequency, start_time=sd,
                                                         page_size=10000).to_dataframe()
                except Exception as e2:  # noqa: BLE001
                    logger.debug("CME candles failed for %s: %s", m, e2)
                    continue
                if df is not None and not df.empty:
                    frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["market", "time", "close", "usd_volume", "volume", "trades"])
        df = pd.concat(frames, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        out = pd.DataFrame({"market": df["market"].astype(str), "time": df["time"],
                            "close": pd.to_numeric(df.get("price_close"), errors="coerce"),
                            "usd_volume": pd.to_numeric(df.get("candle_usd_volume"), errors="coerce"),
                            "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
                            "trades": pd.to_numeric(df.get("candle_trades_count"), errors="coerce")})
        return out.sort_values(["market", "time"]).reset_index(drop=True)

    def _cme_open_interest(self, markets: list[str], start: datetime) -> pd.DataFrame:
        """Daily open interest rows (market, time, oi_contracts, oi_usd) for several CME markets."""
        sd = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        frames = []
        try:
            df = self._client.get_market_open_interest(markets=list(markets), start_time=sd, page_size=10000).to_dataframe()
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:  # noqa: BLE001
            logger.debug("CME batch OI failed (%s); retrying per market", e)
            for m in markets:
                try:
                    df = self._client.get_market_open_interest(markets=[m], start_time=sd, page_size=10000).to_dataframe()
                except Exception as e2:  # noqa: BLE001
                    logger.debug("CME OI failed for %s: %s", m, e2)
                    continue
                if df is not None and not df.empty:
                    frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["market", "time", "oi_contracts", "oi_usd"])
        df = pd.concat(frames, ignore_index=True)
        out = pd.DataFrame({"market": df["market"].astype(str), "time": pd.to_datetime(df["time"], utc=True),
                            "oi_contracts": pd.to_numeric(df.get("contract_count"), errors="coerce"),
                            "oi_usd": pd.to_numeric(df.get("value_usd"), errors="coerce")})
        return out.sort_values(["market", "time"]).reset_index(drop=True)

    def cme_curve(self, base: str, include_micro: bool = False, now: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """The CME futures curve for ``base``: one row per active outright contract.

        Columns: symbol, label, product, contract_size, expiration, days_to_expiry, close (last
        complete daily settlement-area close, USD), close_time, usd_volume (that day), oi_contracts,
        oi_usd, oi_base (contracts x size), oi_time, basis_pct (close / spot - 1, %),
        basis_ann_pct (simple ACT/365 annualisation). ``attrs``: spot, spot_market, spot_time,
        as_of (latest candle day), base. Spot is the last trade on the token's primary spot market
        (Coinbase USD for the majors) - not a synchronous mark, and the tools say so.
        None when no contracts are listed or nothing returned data."""
        now = now or datetime.now(timezone.utc)
        contracts = self.cme_contracts(base, include_micro=include_micro, now=now)
        if contracts.empty:
            return None
        markets = contracts["market"].tolist()
        start = now - timedelta(days=8)
        candles = self._cme_candles(markets, start)
        oi = self._cme_open_interest(markets, start)
        if candles.empty and oi.empty:
            return None
        last_c = candles.dropna(subset=["close"]).sort_values("time").groupby("market").last() if len(candles) else pd.DataFrame()
        last_oi = oi.dropna(subset=["oi_contracts"]).sort_values("time").groupby("market").last() if len(oi) else pd.DataFrame()
        trade = None
        try:
            trade = self.get_latest_trade(base)
        except Exception as e:  # noqa: BLE001
            logger.debug("spot for CME basis unavailable: %s", e)
        spot = float(trade["price"]) if trade else None
        rows = []
        for _, c in contracts.iterrows():
            m = c["market"]
            cl = last_c.loc[m] if m in getattr(last_c, "index", []) else None
            o = last_oi.loc[m] if m in getattr(last_oi, "index", []) else None
            close = float(cl["close"]) if cl is not None else None
            dte = float(c["days_to_expiry"])
            basis = (close / spot - 1) * 100 if (close and spot) else None
            rows.append({
                "symbol": c["symbol"], "label": c["label"], "product": c["product"], "contract_size": c["contract_size"],
                "is_standard": bool(c["is_standard"]), "expiration": c["expiration"], "days_to_expiry": dte,
                "close": close, "close_time": cl["time"] if cl is not None else None,
                "usd_volume": float(cl["usd_volume"]) if cl is not None and pd.notna(cl["usd_volume"]) else None,
                "oi_contracts": float(o["oi_contracts"]) if o is not None else None,
                "oi_usd": float(o["oi_usd"]) if o is not None and pd.notna(o["oi_usd"]) else None,
                "oi_base": float(o["oi_contracts"]) * float(c["contract_size"]) if o is not None and pd.notna(c["contract_size"]) else None,
                "oi_time": o["time"] if o is not None else None,
                "basis_pct": basis,
                "basis_ann_pct": (basis * 365.0 / dte) if (basis is not None and dte > 0.5) else None,
            })
        out = pd.DataFrame(rows)
        out = out[out["close"].notna() | out["oi_contracts"].notna()].reset_index(drop=True)
        if out.empty:
            return None
        out.attrs.update({"base": base.lower(), "spot": spot, "spot_market": trade["market"] if trade else None,
                          "spot_time": trade["time"] if trade else None,
                          "as_of": out["close_time"].max() if out["close_time"].notna().any() else None,
                          "oi_as_of": out["oi_time"].max() if out["oi_time"].notna().any() else None})
        return out

    def cme_history(self, base: str, contract: Optional[str] = None, days: int = 30,
                    now: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Daily CME open interest and volume for one contract or for all active outrights of
        ``base`` summed (standard + micro + weekly), with the all-venue futures OI / volume from
        Coin Metrics asset metrics so the CME share can be quoted.

        Columns: time, close (single contract only), usd_volume, oi_contracts, oi_usd,
        all_venue_oi_usd, all_venue_volume_usd, cme_share_oi_pct. ``attrs``: contracts (list),
        base, contract. None when no data."""
        now = now or datetime.now(timezone.utc)
        days = max(2, min(int(days), 365))
        start = now - timedelta(days=days + 1)
        if contract:
            sym = contract.upper().replace("CME-", "").replace("-FUTURE", "")
            markets = [f"cme-{sym}-future"]
            single = True
        else:
            contracts = self.cme_contracts(base, include_micro=True, include_weekly=True, now=now)
            markets = contracts["market"].tolist()
            single = False
        if not markets:
            return None
        candles = self._cme_candles(markets, start)
        oi = self._cme_open_interest(markets, start)
        if candles.empty and oi.empty:
            return None
        c_day = candles.assign(day=candles["time"].dt.floor("D")) if len(candles) else candles
        o_day = oi.assign(day=oi["time"].dt.floor("D")) if len(oi) else oi
        vol = c_day.groupby("day")["usd_volume"].sum(min_count=1) if len(c_day) else pd.Series(dtype=float)
        oic = o_day.groupby("day")["oi_contracts"].sum(min_count=1) if len(o_day) else pd.Series(dtype=float)
        oiu = o_day.groupby("day")["oi_usd"].sum(min_count=1) if len(o_day) else pd.Series(dtype=float)
        out = pd.DataFrame({"usd_volume": vol, "oi_contracts": oic, "oi_usd": oiu})
        if single and len(c_day):
            out["close"] = c_day.groupby("day")["close"].last()
        out.index.name = "time"
        out = out.reset_index().sort_values("time")
        out = out[out["time"] >= pd.Timestamp(now - timedelta(days=days)).floor("D")]
        # all-venue context
        try:
            am = self._client.get_asset_metrics(
                assets=[self._asset_id(base)], metrics=["open_interest_reported_future_usd", "volume_reported_future_usd_1d"],
                frequency="1d", start_time=start.strftime("%Y-%m-%d"), page_size=1000,
            ).to_dataframe()
            if am is not None and not am.empty:
                am["time"] = pd.to_datetime(am["time"], utc=True)
                ctx = pd.DataFrame({"time": am["time"],
                                    "all_venue_oi_usd": pd.to_numeric(am.get("open_interest_reported_future_usd"), errors="coerce"),
                                    "all_venue_volume_usd": pd.to_numeric(am.get("volume_reported_future_usd_1d"), errors="coerce")})
                out = out.merge(ctx, on="time", how="left")
        except Exception as e:  # noqa: BLE001
            logger.debug("all-venue futures context unavailable for %s: %s", base, e)
        for c in ("all_venue_oi_usd", "all_venue_volume_usd"):
            if c not in out.columns:
                out[c] = pd.NA
        out["cme_share_oi_pct"] = (out["oi_usd"] / pd.to_numeric(out["all_venue_oi_usd"], errors="coerce") * 100)
        out = out.reset_index(drop=True)
        out.attrs.update({"base": base.lower(), "contract": markets[0] if single else None, "contracts": markets})
        return out if len(out) else None

    def etf_onchain_flows(self, base: str = "btc", days: int = 30, frequency: str = "1d",
                          now: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Coin Metrics on-chain ETF metrics (BTC only): FlowInEtfUSD / FlowOutEtfUSD (1d or 1h) and
        ETF supply SplyEtfNtv / SplyEtfUSD (1d).

        Columns: time, flow_in_usd, flow_out_usd, net_flow_usd, supply_btc, supply_usd (supply
        NaN on hourly rows). Values are inferred from ETF-labelled on-chain addresses, so they
        lag the issuers' own reports by about a day and differ from Blockworks / Messari figures.
        None for other assets or when nothing is returned."""
        if (base or "").lower() != "btc":
            return None
        if frequency not in ("1d", "1h"):
            return None
        now = now or datetime.now(timezone.utc)
        days = max(1, min(int(days), 730))
        start = (now - timedelta(days=days + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            flows = self._client.get_asset_metrics(assets=["btc"], metrics=list(ETF_FLOW_METRICS), frequency=frequency,
                                                   start_time=start, page_size=10000).to_dataframe()
        except Exception as e:  # noqa: BLE001
            logger.debug("ETF flow metrics failed: %s", e)
            return None
        if flows is None or flows.empty:
            return None
        flows["time"] = pd.to_datetime(flows["time"], utc=True)
        out = pd.DataFrame({"time": flows["time"],
                            "flow_in_usd": pd.to_numeric(flows.get("FlowInEtfUSD"), errors="coerce"),
                            "flow_out_usd": pd.to_numeric(flows.get("FlowOutEtfUSD"), errors="coerce")})
        out["net_flow_usd"] = out["flow_in_usd"] - out["flow_out_usd"]
        out["supply_btc"] = pd.NA
        out["supply_usd"] = pd.NA
        try:
            sup = self._client.get_asset_metrics(assets=["btc"], metrics=list(ETF_SUPPLY_METRICS), frequency="1d",
                                                 start_time=start, page_size=10000).to_dataframe()
            if sup is not None and not sup.empty:
                sup["time"] = pd.to_datetime(sup["time"], utc=True)
                s = pd.DataFrame({"time": sup["time"], "supply_btc": pd.to_numeric(sup.get("SplyEtfNtv"), errors="coerce"),
                                  "supply_usd": pd.to_numeric(sup.get("SplyEtfUSD"), errors="coerce")})
                out = out.drop(columns=["supply_btc", "supply_usd"]).merge(s, on="time", how="left")
        except Exception as e:  # noqa: BLE001
            logger.debug("ETF supply metrics failed: %s", e)
        out = out.dropna(subset=["flow_in_usd", "flow_out_usd"], how="all").sort_values("time").reset_index(drop=True)
        out.attrs.update({"frequency": frequency, "base": "btc"})
        return out if len(out) else None
