"""Amberdata implementation of MarketDataProvider (derivatives only).

Covers perpetual-futures metrics at daily granularity, aggregated across the
major exchanges: funding rate, open interest, volume and liquidations. Spot
methods raise NotImplementedError — Coin Metrics owns spot data.

Endpoints (all under ``https://api.amberdata.com/markets/derivatives``,
header ``x-api-key``; verified live against the trial key in Sept 2026):

    /analytics/futures-perpetuals/open-interest-total   ?asset=BTC
        daily rows per (exchange, type in {swaps, futures}); ``usd`` is OI in
        millions of USD, ``volumeMilUSD`` is that bucket's daily volume in
        millions of USD. No range cap observed (400 days OK).
    /analytics/futures-perpetuals/volumes               ?asset=BTC
        daily rows per exchange (futures + perps combined),
        ``totalDailyVolumeMilUSD``. Max range 31 days per request.
    /analytics/futures-perpetuals/liquidations-total    ?asset=BTC
        rows per exchange, ``buyLiquidationsUSD`` / ``sellLiquidationsUSD``
        (plain USD). Max range 31 days; bucket size is dynamic (4h for <=8d,
        12h for ~15d, daily for ~30d) so we always re-sum to UTC days.
    /analytics/futures-perpetuals/funding-rates         ?underlying=BTC
        one row per instrument per funding event (1h or 8h) with
        ``fundingRateNormalized8h`` (decimal, e.g. 0.0001 = 0.01% / 8h),
        ``marginType`` (linear | inverse | 'null'). Max range 366 days.

``startDate`` is inclusive, ``endDate`` exclusive. Unknown assets return
HTTP 200 with an empty ``payload.data`` (not 404). ``payload.metadata.next``
carries a cursor URL when a response is paginated.

Aggregation formulas
--------------------
perp_oi            = sum over SUPPORTED_EXCHANGES of OI (type == "swaps") * 1e6
perp_volume        = sum over SUPPORTED_EXCHANGES of perp volume * 1e6
                     (``volumeMilUSD`` of the swaps rows; falls back to the
                     /volumes endpoint's ``totalDailyVolumeMilUSD`` if absent)
long_liquidations  = sum of ``sellLiquidationsUSD``  (longs are sold to close)
short_liquidations = sum of ``buyLiquidationsUSD``   (shorts are bought to close)
                     excluding LIQUIDATION_EXCLUDED_EXCHANGES (bitget's feed is
                     mis-scaled by 20x-30000x vs its OI as of Sept 2026)
funding_rate       = mean over exchanges of the per-exchange daily mean of
                     ``fundingRateNormalized8h`` on the exchange's USD-margined
                     (linear) perp, annualised: rate_8h * 3 * 365 * 100  (%)
"""

from __future__ import annotations

import logging
import time  # noqa: F401  (kept: tests patch ``amberdata.time.sleep`` to silence backoff)
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
import requests

from providers._http import AmberdataHTTP
from providers._http import date_chunks as _date_chunks
from providers._http import day_str as _day_str
from providers._http import to_utc_day as _to_utc_day
from providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

BASE_URL = "https://api.amberdata.com/markets/derivatives"

# Amberdata exchange identifiers (note: OKX is "okex"). Order is cosmetic.
SUPPORTED_EXCHANGES: tuple[str, ...] = (
    "binance",
    "bybit",
    "okex",
    "deribit",
    "bitget",
    "hyperliquid",
)

# Exchanges dropped from the liquidation aggregate only. Verified Sept 2026:
# bitget's liquidations-total rows are implausible (BTC median $262B/day vs
# ~$3B OI; ETH ~1000x, SOL ~20x too high) while its OI/volume/funding are fine.
LIQUIDATION_EXCLUDED_EXCHANGES: frozenset[str] = frozenset({"bitget"})

# Common names -> Amberdata identifiers
EXCHANGE_ALIASES = {
    "okx": "okex",
    "huobi": "huobi",
    "htx": "huobi",
}

# Our token symbols -> Amberdata ``asset`` / ``underlying`` identifiers, where
# they differ from a plain upper-case of the symbol.
TOKEN_OVERRIDES = {
    "matic": "POL",  # Polygon rebrand
}

# Perp instrument naming per exchange for the USD-margined (linear) contract.
_INSTRUMENT_PATTERNS = {
    "binance": "{sym}USDT",
    "bybit": "{sym}USDT",
    "bitget": "{sym}USDT",
    "okex": "{sym}-USDT-SWAP",
    "deribit": "{sym}_USDC-PERPETUAL",
    "hyperliquid": "{sym}_USDT-PERP",
    "huobi": "{sym}-USDT",
    "kraken": "PF_{sym}USD",
    "bitmex": "{sym}USDT",
}

# Exchange-specific asset symbols (kraken/bitmex still call bitcoin XBT).
_INSTRUMENT_SYMBOL_OVERRIDES = {
    ("kraken", "BTC"): "XBT",
    ("bitmex", "BTC"): "XBT",
}

# Quote assets accepted when falling back from the canonical instrument.
_LINEAR_QUOTES = {"USDT", "USDC", "USD"}

_FUNDING_PERIODS_PER_YEAR = 3 * 365  # 8h periods
_MILLION = 1_000_000.0

_DEFAULT_CHUNK_DAYS = 360     # funding-rates (cap 366d); OI has no observed cap
_SHORT_CHUNK_DAYS = 30        # volumes / liquidations-total (cap 31d)
_MAX_PAGES = 50


def asset_symbol(token: str) -> str:
    """Amberdata asset identifier for one of our token symbols ("btc" -> "BTC")."""
    t = token.strip().lower()
    return TOKEN_OVERRIDES.get(t, t.upper())


def normalize_exchange(exchange: str) -> str:
    e = exchange.strip().lower()
    return EXCHANGE_ALIASES.get(e, e)


def to_instrument(token: str, exchange: str) -> Optional[str]:
    """Canonical USD-margined perpetual instrument name for ``token`` on ``exchange``.

    >>> to_instrument("btc", "binance")
    'BTCUSDT'
    >>> to_instrument("eth", "okx")
    'ETH-USDT-SWAP'
    >>> to_instrument("sol", "deribit")
    'SOL_USDC-PERPETUAL'

    Returns None for exchanges without a known naming pattern.
    """
    ex = normalize_exchange(exchange)
    pattern = _INSTRUMENT_PATTERNS.get(ex)
    if pattern is None:
        return None
    sym = asset_symbol(token)
    sym = _INSTRUMENT_SYMBOL_OVERRIDES.get((ex, sym), sym)
    return pattern.format(sym=sym)


def annualize_funding(rate_8h: float) -> float:
    """8-hour funding rate (decimal) -> annualised percentage.

    0.0001 per 8h  ->  0.0001 * 3 * 365 * 100 = 10.95 % p.a.
    """
    return rate_8h * _FUNDING_PERIODS_PER_YEAR * 100.0


class AmberdataProvider(MarketDataProvider):
    """Derivatives data from Amberdata, aggregated across major perp venues."""

    def __init__(
        self,
        api_key: str,
        session: Optional[requests.Session] = None,
        timeout: int = 30,
        exchanges: Optional[Iterable[str]] = None,
        max_retries: int = 3,
        backoff: float = 1.0,
    ):
        self._http = AmberdataHTTP(api_key, session=session, timeout=timeout,
                                   max_retries=max_retries, backoff=backoff)
        self._session = self._http.session
        exs = exchanges if exchanges is not None else SUPPORTED_EXCHANGES
        self.exchanges: tuple[str, ...] = tuple(normalize_exchange(e) for e in exs)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _request(self, url: str, params: Optional[dict]) -> Optional[dict]:
        """GET with retry/backoff on 429/5xx (see providers._http). JSON or None."""
        return self._http.request(url, params)

    def _get(self, path: str, params: dict) -> Optional[list]:
        """Fetch ``payload.data`` for an analytics endpoint, following cursors.

        Returns a list of row dicts (possibly empty) or None on error.
        """
        return self._http.get_rows(f"{BASE_URL}{path}", params, _MAX_PAGES)

    def _fetch_chunked(self, path: str, base_params: dict,
                       start: datetime, end: datetime, max_days: int) -> Optional[pd.DataFrame]:
        frames = []
        any_ok = False
        for s, e in _date_chunks(start, end, max_days):
            params = dict(base_params, startDate=_day_str(s), endDate=_day_str(e),
                          timeFormat="milliseconds")
            rows = self._get(path, params)
            if rows is None:
                continue
            any_ok = True
            if rows:
                frames.append(pd.DataFrame(rows))
        if not any_ok:
            return None
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _finalize(self, df: pd.DataFrame, token: str, metric: str,
                  start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Restrict to [start, end] days, sort, tz-naive daily ``time``."""
        if df is None or df.empty:
            logger.info("Amberdata: no %s data for %s", metric, token)
            return None
        lo = pd.Timestamp(start.date())
        hi = pd.Timestamp(end.date())
        df = df[(df["time"] >= lo) & (df["time"] <= hi)]
        if df.empty:
            logger.info("Amberdata: no %s data for %s in range", metric, token)
            return None
        df = df.sort_values("time").reset_index(drop=True)
        df["time"] = df["time"].astype("datetime64[ns]")
        return df

    def _filter_exchanges(self, df: pd.DataFrame) -> pd.DataFrame:
        if "exchange" not in df.columns:
            return df
        ex = df["exchange"].astype(str).str.lower().map(normalize_exchange)
        return df[ex.isin(self.exchanges)]

    @staticmethod
    def _num(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(float("nan"), index=df.index, dtype="float64")
        return pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Spot — not supported here
    # ------------------------------------------------------------------

    def get_spot_ohlcv(self, token, start_date, end_date):
        raise NotImplementedError("AmberdataProvider is derivatives-only; use Coin Metrics for spot")

    def get_spot_price(self, token, start_date, end_date):
        raise NotImplementedError("AmberdataProvider is derivatives-only; use Coin Metrics for spot")

    # ------------------------------------------------------------------
    # Open interest (+ shared loader for volume)
    # ------------------------------------------------------------------

    def _load_oi_rows(self, token: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        df = self._fetch_chunked(
            "/analytics/futures-perpetuals/open-interest-total",
            {"asset": asset_symbol(token)}, start, end, _DEFAULT_CHUNK_DAYS,
        )
        if df is None or df.empty:
            return df
        df = self._filter_exchanges(df)
        if "type" in df.columns:
            df = df[df["type"].astype(str).str.lower() == "swaps"]
        if df.empty:
            return df
        df = df.assign(time=_to_utc_day(df["timestamp"]))
        return df

    def get_perp_oi(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Perp open interest in USD: sum of swaps OI over the supported exchanges."""
        df = self._load_oi_rows(token, start_date, end_date)
        if df is None or df.empty:
            return self._finalize(df, token, "perp_oi", start_date, end_date)
        df = df.assign(usd=self._num(df, "usd") * _MILLION)
        # one row per (exchange, day); keep the last observation if duplicated
        df = df.dropna(subset=["usd"]).drop_duplicates(subset=["exchange", "time"], keep="last")
        out = df.groupby("time", as_index=False)["usd"].sum().rename(columns={"usd": "perp_oi"})
        out = out[out["perp_oi"] > 0]
        return self._finalize(out, token, "perp_oi", start_date, end_date)

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    def get_perp_volume(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Perp volume in USD: sum over supported exchanges.

        Primary source is the swaps ``volumeMilUSD`` of the open-interest
        endpoint (perp-only, uncapped range). Falls back to the /volumes
        endpoint (futures + perps combined, 31-day chunks) if that field is
        missing.
        """
        df = self._load_oi_rows(token, start_date, end_date)
        out = None
        if df is not None and not df.empty and "volumeMilUSD" in df.columns:
            vol = self._num(df, "volumeMilUSD") * _MILLION
            tmp = df.assign(vol=vol).dropna(subset=["vol"])
            tmp = tmp.drop_duplicates(subset=["exchange", "time"], keep="last")
            out = tmp.groupby("time", as_index=False)["vol"].sum().rename(columns={"vol": "perp_volume"})
            out = out[out["perp_volume"] > 0]
        if out is None or out.empty:
            out = self._volume_from_volumes_endpoint(token, start_date, end_date)
        return self._finalize(out, token, "perp_volume", start_date, end_date)

    def _volume_from_volumes_endpoint(self, token, start, end) -> Optional[pd.DataFrame]:
        df = self._fetch_chunked(
            "/analytics/futures-perpetuals/volumes",
            {"asset": asset_symbol(token)}, start, end, _SHORT_CHUNK_DAYS,
        )
        if df is None or df.empty:
            return df
        df = self._filter_exchanges(df)
        if df.empty:
            return df
        df = df.assign(time=_to_utc_day(df["timestamp"]),
                       vol=self._num(df, "totalDailyVolumeMilUSD") * _MILLION)
        df = df.dropna(subset=["vol"]).drop_duplicates(subset=["exchange", "time"], keep="last")
        out = df.groupby("time", as_index=False)["vol"].sum().rename(columns={"vol": "perp_volume"})
        return out[out["perp_volume"] > 0]

    # ------------------------------------------------------------------
    # Liquidations
    # ------------------------------------------------------------------

    def get_liquidations(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Daily liquidations in USD summed over supported exchanges.

        ``sellLiquidationsUSD`` (sell-to-close) = long positions liquidated;
        ``buyLiquidationsUSD`` (buy-to-close) = short positions liquidated.
        Sub-daily buckets returned for short windows are summed into UTC days.
        Exchanges in LIQUIDATION_EXCLUDED_EXCHANGES are skipped (bad feed).
        """
        df = self._fetch_chunked(
            "/analytics/futures-perpetuals/liquidations-total",
            {"asset": asset_symbol(token)}, start_date, end_date, _SHORT_CHUNK_DAYS,
        )
        if df is None or df.empty:
            return self._finalize(df, token, "liquidations", start_date, end_date)
        df = self._filter_exchanges(df)
        if "exchange" in df.columns and LIQUIDATION_EXCLUDED_EXCHANGES:
            ex = df["exchange"].astype(str).str.lower().map(normalize_exchange)
            df = df[~ex.isin(LIQUIDATION_EXCLUDED_EXCHANGES)]
        if df.empty:
            return self._finalize(df, token, "liquidations", start_date, end_date)
        df = df.assign(
            time=_to_utc_day(df["timestamp"]),
            long_liquidations=self._num(df, "sellLiquidationsUSD").fillna(0.0),
            short_liquidations=self._num(df, "buyLiquidationsUSD").fillna(0.0),
        )
        df = df.drop_duplicates(subset=["exchange", "timestamp"], keep="last")
        out = df.groupby("time", as_index=False)[["long_liquidations", "short_liquidations"]].sum()
        out["total_liquidations"] = out["long_liquidations"] + out["short_liquidations"]
        return self._finalize(out, token, "liquidations", start_date, end_date)

    # ------------------------------------------------------------------
    # Funding rate
    # ------------------------------------------------------------------

    def _select_funding_instruments(self, df: pd.DataFrame, token: str) -> pd.DataFrame:
        """Keep, per exchange, the canonical USD-margined perp (or a linear fallback)."""
        keep = []
        for ex, grp in df.groupby("exchange"):
            canonical = to_instrument(token, ex)
            sel = grp[grp["instrument"] == canonical] if canonical else grp.iloc[0:0]
            if sel.empty:
                margin = grp.get("marginType", pd.Series("", index=grp.index)).astype(str).str.lower()
                quote = grp.get("quoteAsset", pd.Series("", index=grp.index)).astype(str).str.upper()
                sel = grp[(margin == "linear") & quote.isin(_LINEAR_QUOTES)]
                if not sel.empty:
                    logger.info("Amberdata funding: %s on %s not found, using %s",
                                canonical, ex, sorted(sel["instrument"].unique()))
            if not sel.empty:
                keep.append(sel)
        if not keep:
            return df.iloc[0:0]
        return pd.concat(keep, ignore_index=True)

    def get_funding_rate(
        self, token: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Annualised funding rate (%) on USD-margined perps.

        For each exchange in ``self.exchanges`` take the canonical linear perp
        (``to_instrument``), average its ``fundingRateNormalized8h`` over the
        funding events of each UTC day, then take the simple mean across
        exchanges and annualise: ``rate_8h * 3 * 365 * 100``.
        """
        df = self._fetch_chunked(
            "/analytics/futures-perpetuals/funding-rates",
            {"underlying": asset_symbol(token)}, start_date, end_date, _DEFAULT_CHUNK_DAYS,
        )
        if df is None or df.empty:
            return self._finalize(df, token, "funding_rate", start_date, end_date)
        df = self._filter_exchanges(df)
        if df.empty or "instrument" not in df.columns:
            return self._finalize(df.iloc[0:0], token, "funding_rate", start_date, end_date)
        df = df.assign(exchange=df["exchange"].astype(str).str.lower().map(normalize_exchange))
        df = self._select_funding_instruments(df, token)
        if df.empty:
            return self._finalize(df, token, "funding_rate", start_date, end_date)
        rate = self._num(df, "fundingRateNormalized8h")
        df = df.assign(time=_to_utc_day(df["timestamp"]), rate=rate).dropna(subset=["rate"])
        if df.empty:
            return self._finalize(df, token, "funding_rate", start_date, end_date)
        per_ex = df.groupby(["time", "exchange"], as_index=False)["rate"].mean()
        out = per_ex.groupby("time", as_index=False)["rate"].mean()
        out["funding_rate"] = out["rate"].map(annualize_funding)
        out = out[["time", "funding_rate"]]
        return self._finalize(out, token, "funding_rate", start_date, end_date)
