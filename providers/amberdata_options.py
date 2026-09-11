"""Amberdata options analytics (Deribit by default).

Exposes daily options metrics that the z-score pipeline merges alongside the
spot / perp metrics, plus snapshot views (term structure, delta surface,
gamma exposure, block trades) for the LLM analysis and the chat tools.

Endpoints (all GET under ``https://api.amberdata.com/markets/derivatives/analytics``,
header ``x-api-key``; **``Accept-Encoding: gzip`` is mandatory** — without it the
options endpoints answer HTTP 400 "Compression required"). Verified live on
2026-09-10; ``startDate`` inclusive, ``endDate`` exclusive; unknown currencies
return HTTP 200 with an empty ``payload.data``.

    volatility/index                                         ?exchange&currency&startDate&endDate&timeInterval=day
        DVOL index OHLC (vol points). ``exchangeTimestamp`` in ms. With
        ``timeInterval=day`` a 366-day range works in one call (without it the
        range is capped at 1 day). SOL_USDC has no DVOL index (0 rows).
    volatility/term-structures/richness                      ?exchange&currency&startDate&endDate
        daily ATM IV by constant tenor: atm7days/30/60/90/180days, ratio,
        richness. ``timestamp`` ISO-8601 (00:00Z). 366 days per call OK.
    volatility/term-structures/forward-volatility/constant   ?exchange&currency[&timestamp=YYYY-MM-DD]
        snapshot (10 tenors: 1,2,3,7,14,21,30,60,90,180 days): atm, fwdAtm.
    volatility/delta-surfaces/constant                       ?exchange&currency[&startDate&endDate&timeInterval=day]
        IV per delta (deltaCall05..45, deltaPut05..45, atm, delta50) for the
        same 10 tenors. ``timestamp`` is NOT accepted; without dates it is a
        live snapshot; with dates + ``timeInterval=day`` it returns one 00:00Z
        surface per day (366 days per call OK) -> used for skew history.
    trades-flow/put-call-ratio                               ?exchange&currency&startDate&endDate&timeInterval=day
        putCallRatioOpenInterest, putCallRatioVolume24hr; ``timestamp`` ms.
    trades-flow/volume-aggregates                            ?exchange&currency&startDate&endDate&timeInterval=day
        contract/notional/premium volume split OnScreen vs Blocked. With
        ``timeInterval=day`` one row per day (366 days per call OK); without
        it per-minute rows capped at 1 day. Notional / premium in USD.
    trades-flow/gamma-exposures-snapshots                    ?exchange&currency[&startDate&endDate]
        per (instrument) row: dealerNetInventory, dealerTotalInventory,
        gammaLevel, indexPrice, putCall, strike, snapshotTimestamp (ms).
        Hourly snapshots; without dates -> latest snapshot (~800 rows BTC);
        a date range returns up to 10000 rows with no cursor, so a specific
        day is fetched as the [23:00, 24:00) window (falling back to the day).
    trades-flow/block-volumes                                ?exchange&currency&startDate&endDate
        block trades aggregated over the range per (expirationTimestamp ms,
        strike, putCall): contractVolume, premiumVolume (USD).
    instruments/information                                  ?exchange[&currency]
        listed options; used once per instance to discover which currencies
        the exchange lists (Deribit Sept 2026: AVAX_USDC, BTC, BTC_USDC, ETH,
        ETH_USDC, HYPE_USDC, SOL_USDC, TRX_USDC, XRP_USDC).
    403 on the current tier: volatility/implied-vs-realized, trades-flow/top-trades.

Conventions match providers/base.py: daily frames have a tz-naive UTC
``time`` column (``datetime64[ns]``, one row per day, sorted ascending),
float64 metrics, the partial current UTC day is dropped, ``None`` when empty.
IV / DVOL / skew are in vol points (annualised %); skew = put IV - call IV,
so positive = puts richer than calls.

Every public method memoises its result within the instance per
(method, token, dates, args), so repeated calls in one run are free.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd
import requests

from providers._http import AmberdataHTTP, date_chunks, day_str, to_utc_day, to_utc_timestamp
from providers.amberdata import asset_symbol

logger = logging.getLogger(__name__)

BASE_URL = "https://api.amberdata.com/markets/derivatives/analytics"
DEFAULT_EXCHANGE = "deribit"

# Fallback token -> Amberdata options currency map, used only when
# instruments/information cannot be reached (verified for Deribit, Sept 2026).
STATIC_CURRENCIES: dict[str, str] = {
    "btc": "BTC",
    "eth": "ETH",
    "sol": "SOL_USDC",
    "xrp": "XRP_USDC",
    "avax": "AVAX_USDC",
    "hype": "HYPE_USDC",
    "trx": "TRX_USDC",
}

# Preference order for settlement suffixes when a token has several currencies
# (e.g. BTC and BTC_USDC): the un-suffixed (inverse) market carries the DVOL
# index and the deepest liquidity.
_SUFFIX_PREFERENCE = ("", "USDC", "USDT", "USD")

SURFACE_TENORS = (1, 2, 3, 7, 14, 21, 30, 60, 90, 180)

_DEFAULT_CHUNK_DAYS = 360   # every daily endpoint accepted 366 days with timeInterval=day
_MAX_PAGES = 50
_RANGE_CAP_MARKER = "maximum allowed"


def _utc_today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).date())


def _memoised(method: Callable) -> Callable:
    """Cache a method's result on the instance, keyed by name + arguments."""

    def wrapper(self, *args, **kwargs):
        key = (method.__name__,) + tuple(_key_part(a) for a in args) \
            + tuple(sorted((k, _key_part(v)) for k, v in kwargs.items()))
        if key in self._memo:
            hit = self._memo[key]
            return hit.copy() if isinstance(hit, pd.DataFrame) else hit
        result = method(self, *args, **kwargs)
        self._memo[key] = result
        return result.copy() if isinstance(result, pd.DataFrame) else result

    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


def _key_part(value):
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        return value.lower()
    return value


class AmberdataOptionsProvider:
    """Options analytics from Amberdata for one exchange (default Deribit)."""

    def __init__(
        self,
        api_key: str,
        exchange: str = DEFAULT_EXCHANGE,
        session: Optional[requests.Session] = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff: float = 1.0,
        chunk_days: int = _DEFAULT_CHUNK_DAYS,
        pause: float = 0.1,
    ):
        self._http = AmberdataHTTP(api_key, session=session, timeout=timeout,
                                   max_retries=max_retries, backoff=backoff)
        self.exchange = exchange.strip().lower()
        self._chunk_days = max(1, int(chunk_days))
        self._pause = pause
        self._currencies: Optional[dict[str, str]] = None
        self._memo: dict = {}

    # ------------------------------------------------------------------
    # Accounting / plumbing
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Number of HTTP requests issued so far (incl. retries and cursor pages)."""
        return self._http.call_count

    def _get(self, path: str, params: dict) -> Optional[list]:
        """Rows of ``payload.data`` for ``path`` (list, possibly empty) or None on error."""
        return self._http.get_rows(f"{BASE_URL}/{path.lstrip('/')}", params, _MAX_PAGES)

    def _base_params(self, currency: str) -> dict:
        return {"exchange": self.exchange, "currency": currency}

    # ------------------------------------------------------------------
    # Currency discovery
    # ------------------------------------------------------------------

    def _load_currencies(self) -> dict[str, str]:
        if self._currencies is not None:
            return self._currencies
        rows = self._get("instruments/information", {"exchange": self.exchange})
        if rows is None:
            logger.warning("Amberdata options: instrument discovery failed on %s; "
                           "using static currency map", self.exchange)
            mapping = dict(STATIC_CURRENCIES)
        else:
            mapping = currencies_from_instruments(rows)
            if not mapping:
                logger.info("Amberdata options: no listed options on %s", self.exchange)
        self._currencies = mapping
        return mapping

    def currency_for(self, token: str) -> Optional[str]:
        """Amberdata options currency for a token ("btc" -> "BTC", "sol" -> "SOL_USDC")."""
        mapping = self._load_currencies()
        t = token.strip().lower()
        return mapping.get(t) or mapping.get(asset_symbol(t).lower())

    def supported_tokens(self) -> list[str]:
        """Lower-case tokens with listed options on the exchange (cached per instance)."""
        return sorted(self._load_currencies())

    def _currency_or_log(self, token: str, metric: str) -> Optional[str]:
        cur = self.currency_for(token)
        if cur is None:
            logger.info("Amberdata options: %s has no options on %s (%s)", token, self.exchange, metric)
        return cur

    # ------------------------------------------------------------------
    # Range fetching
    # ------------------------------------------------------------------

    def _fetch_range(self, path: str, params: dict, start: datetime, end: datetime,
                     chunk_days: Optional[int] = None) -> Optional[list]:
        """Fetch rows for [start, end] (inclusive days), chunked.

        A chunk rejected with the 400 "range over the maximum allowed" reply
        is re-fetched day by day; other chunk failures are logged and
        skipped. Returns None only if every chunk failed.
        """
        rows: list = []
        any_ok = False
        for s, e in date_chunks(start, end, chunk_days or self._chunk_days):
            got = self._get(path, dict(params, startDate=day_str(s), endDate=day_str(e)))
            if got is None and (e - s).days > 1 and self._is_range_cap_error():
                logger.info("Amberdata options: %s capped at 1 day/request — looping %s..%s",
                            path, day_str(s), day_str(e))
                got = self._fetch_day_by_day(path, params, s, e)
            if got is None:
                logger.warning("Amberdata options: chunk %s..%s of %s failed — skipped",
                               day_str(s), day_str(e), path)
                continue
            any_ok = True
            rows.extend(got)
        return rows if any_ok else None

    def _is_range_cap_error(self) -> bool:
        return self._http.last_status == 400 and _RANGE_CAP_MARKER in (self._http.last_error or "")

    def _fetch_day_by_day(self, path: str, params: dict, start: datetime, end_excl: datetime) -> Optional[list]:
        rows: list = []
        any_ok = False
        cur = start
        while cur < end_excl:
            nxt = cur + timedelta(days=1)
            got = self._get(path, dict(params, startDate=day_str(cur), endDate=day_str(nxt)))
            if got is None:
                logger.warning("Amberdata options: %s failed for %s — day skipped", path, day_str(cur))
            else:
                any_ok = True
                rows.extend(got)
            cur = nxt
            if self._pause and cur < end_excl:
                time.sleep(self._pause)
        return rows if any_ok else None

    # ------------------------------------------------------------------
    # Daily normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _num(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(float("nan"), index=df.index, dtype="float64")
        return pd.to_numeric(df[col], errors="coerce").astype("float64")

    def _daily(self, rows: Optional[list], ts_col: str, columns: dict[str, str],
               token: str, metric: str, start: datetime, end: datetime,
               how: str = "last") -> Optional[pd.DataFrame]:
        """Rows -> daily frame ``time`` + renamed float64 columns.

        ``how="last"`` keeps the last observation per UTC day, ``how="sum"``
        adds sub-daily rows up (nulls as 0). Restricts to [start, end],
        drops the current (partial) UTC day, sorts, one row per day.
        """
        if rows is None:
            return None
        if not rows:
            logger.info("Amberdata options: no %s data for %s", metric, token)
            return None
        df = pd.DataFrame(rows)
        if ts_col not in df.columns:
            logger.warning("Amberdata options: %s rows lack %s for %s", metric, ts_col, token)
            return None
        out = pd.DataFrame({"time": to_utc_day(df[ts_col])})
        for src, dst in columns.items():
            out[dst] = self._num(df, src).values
        if how == "sum":
            out = out.fillna(0.0).groupby("time", as_index=False).sum()
        else:
            out = out.assign(_ts=to_utc_timestamp(df[ts_col]).values)
            out = out.sort_values("_ts").drop_duplicates("time", keep="last").drop(columns="_ts")
        return self._finalize(out, token, metric, start, end)

    def _finalize(self, df: pd.DataFrame, token: str, metric: str,
                  start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        lo, hi = pd.Timestamp(start.date()), pd.Timestamp(end.date())
        today = _utc_today()
        df = df[(df["time"] >= lo) & (df["time"] <= hi) & (df["time"] < today)]
        value_cols = [c for c in df.columns if c != "time"]
        df = df.dropna(subset=value_cols, how="all")
        if df.empty:
            logger.info("Amberdata options: no %s data for %s in range", metric, token)
            return None
        df = df.sort_values("time").reset_index(drop=True)
        df["time"] = df["time"].astype("datetime64[ns]")
        for c in value_cols:
            df[c] = df[c].astype("float64")
        return df

    # ------------------------------------------------------------------
    # Daily series
    # ------------------------------------------------------------------

    @_memoised
    def get_dvol(self, token: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Deribit DVOL index, daily OHLC in vol points.

        Columns: time, dvol_open, dvol_high, dvol_low, dvol_close.
        """
        cur = self._currency_or_log(token, "dvol")
        if cur is None:
            return None
        rows = self._fetch_range("volatility/index", dict(self._base_params(cur), timeInterval="day"),
                                 start_date, end_date)
        return self._daily(rows, "exchangeTimestamp",
                           {"open": "dvol_open", "high": "dvol_high", "low": "dvol_low", "close": "dvol_close"},
                           token, "dvol", start_date, end_date)

    @_memoised
    def get_term_structure_history(self, token: str, start_date: datetime,
                                   end_date: datetime) -> Optional[pd.DataFrame]:
        """Daily ATM IV by constant tenor (vol points) plus the term-structure richness.

        Columns: time, atm_iv_7d, atm_iv_30d, atm_iv_60d, atm_iv_90d, atm_iv_180d, ts_richness.
        ``ts_richness`` > 1 means the front end is rich vs the back (inverted curve).
        """
        cur = self._currency_or_log(token, "term_structure_history")
        if cur is None:
            return None
        rows = self._fetch_range("volatility/term-structures/richness", self._base_params(cur),
                                 start_date, end_date)
        return self._daily(rows, "timestamp",
                           {"atm7days": "atm_iv_7d", "atm30days": "atm_iv_30d", "atm60days": "atm_iv_60d",
                            "atm90days": "atm_iv_90d", "atm180days": "atm_iv_180d", "richness": "ts_richness"},
                           token, "term_structure_history", start_date, end_date)

    @_memoised
    def get_put_call_ratio(self, token: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Daily put/call ratios. Columns: time, pcr_oi, pcr_volume_24h."""
        cur = self._currency_or_log(token, "put_call_ratio")
        if cur is None:
            return None
        rows = self._fetch_range("trades-flow/put-call-ratio", dict(self._base_params(cur), timeInterval="day"),
                                 start_date, end_date)
        return self._daily(rows, "timestamp",
                           {"putCallRatioOpenInterest": "pcr_oi", "putCallRatioVolume24hr": "pcr_volume_24h"},
                           token, "put_call_ratio", start_date, end_date)

    @_memoised
    def get_options_volume(self, token: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Daily options volume summed over on-screen and block trades.

        Columns: time, options_contract_volume (contracts), options_notional_volume (USD),
        options_premium_volume (USD), options_block_notional_volume (USD, block trades only).
        """
        cur = self._currency_or_log(token, "options_volume")
        if cur is None:
            return None
        rows = self._fetch_range("trades-flow/volume-aggregates", dict(self._base_params(cur), timeInterval="day"),
                                 start_date, end_date)
        if rows is None:
            return None
        if not rows:
            logger.info("Amberdata options: no options_volume data for %s", token)
            return None
        df = pd.DataFrame(rows)
        parts = {}
        for name in ("contractVolume", "notionalVolume", "premiumVolume"):
            parts[name] = self._num(df, f"{name}OnScreen").fillna(0.0) + self._num(df, f"{name}Blocked").fillna(0.0)
        parts["blockNotional"] = self._num(df, "notionalVolumeBlocked").fillna(0.0)
        df = df.assign(**parts)
        return self._daily(df.to_dict("records"), "timestamp",
                           {"contractVolume": "options_contract_volume",
                            "notionalVolume": "options_notional_volume",
                            "premiumVolume": "options_premium_volume",
                            "blockNotional": "options_block_notional_volume"},
                           token, "options_volume", start_date, end_date, how="sum")

    # ------------------------------------------------------------------
    # Delta surface / skew
    # ------------------------------------------------------------------

    @_memoised
    def _surface_history_rows(self, token: str, start_date: datetime, end_date: datetime) -> Optional[list]:
        cur = self._currency_or_log(token, "delta_surface_history")
        if cur is None:
            return None
        return self._fetch_range("volatility/delta-surfaces/constant",
                                 dict(self._base_params(cur), timeInterval="day"), start_date, end_date)

    @staticmethod
    def _surface_frame(rows: list) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        num = lambda c: pd.to_numeric(df[c], errors="coerce").astype("float64") if c in df.columns \
            else pd.Series(float("nan"), index=df.index, dtype="float64")
        out = pd.DataFrame({
            "timestamp": to_utc_timestamp(df["timestamp"]) if "timestamp" in df.columns else pd.NaT,
            "days_to_expiration": num("daysToExpiration"),
            "atm_iv": num("atm"),
            "iv_call_10d": num("deltaCall10"),
            "iv_call_25d": num("deltaCall25"),
            "iv_put_10d": num("deltaPut10"),
            "iv_put_25d": num("deltaPut25"),
            "index_price": num("indexPrice"),
        })
        out["skew_25d"] = out["iv_put_25d"] - out["iv_call_25d"]
        out["skew_10d"] = out["iv_put_10d"] - out["iv_call_10d"]
        return out

    @_memoised
    def get_delta_surface(self, token: str, timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Constant-tenor delta surface (live snapshot, or the 00:00Z surface of ``timestamp``'s day).

        Columns: days_to_expiration, atm_iv, iv_call_10d, iv_call_25d, iv_put_10d, iv_put_25d,
        skew_25d, skew_10d (vol points; skew = put IV - call IV, positive = puts richer).
        ``df.attrs``: snapshot_time (tz-naive UTC), index_price.
        """
        cur = self._currency_or_log(token, "delta_surface")
        if cur is None:
            return None
        if timestamp is None:
            rows = self._get("volatility/delta-surfaces/constant", self._base_params(cur))
        else:
            day = datetime(timestamp.year, timestamp.month, timestamp.day)
            rows = self._fetch_range("volatility/delta-surfaces/constant",
                                     dict(self._base_params(cur), timeInterval="day"), day, day)
        if not rows:
            if rows is not None:
                logger.info("Amberdata options: no delta surface for %s", token)
            return None
        df = self._surface_frame(rows)
        latest = df["timestamp"].max()
        df = df[df["timestamp"] == latest].dropna(subset=["atm_iv"])
        if df.empty:
            logger.info("Amberdata options: empty delta surface for %s", token)
            return None
        index_price = float(df["index_price"].dropna().iloc[0]) if df["index_price"].notna().any() else None
        out = df.drop(columns=["timestamp", "index_price"]).sort_values("days_to_expiration").reset_index(drop=True)
        out.attrs = {"snapshot_time": latest, "index_price": index_price}
        return out

    @_memoised
    def get_skew_history(self, token: str, start_date: datetime, end_date: datetime,
                         tenor_days: int = 30) -> Optional[pd.DataFrame]:
        """Daily 25-delta and 10-delta skew (put IV - call IV, vol points) at a constant tenor.

        Columns: time, skew_25d_{tenor}d, skew_10d_{tenor}d  (e.g. skew_25d_30d, skew_10d_30d).
        Built from the daily delta-surface history (one call for the whole range).
        """
        if tenor_days not in SURFACE_TENORS:
            logger.warning("Amberdata options: tenor %sd not on the surface grid %s", tenor_days, SURFACE_TENORS)
            return None
        rows = self._surface_history_rows(token, start_date, end_date)
        if rows is None:
            return None
        if not rows:
            logger.info("Amberdata options: no skew history for %s", token)
            return None
        df = self._surface_frame(rows)
        df = df[df["days_to_expiration"] == tenor_days]
        recs = [{"timestamp": t, "skew25": a, "skew10": b}
                for t, a, b in zip(df["timestamp"], df["skew_25d"], df["skew_10d"])]
        return self._daily(recs, "timestamp",
                           {"skew25": f"skew_25d_{tenor_days}d", "skew10": f"skew_10d_{tenor_days}d"},
                           token, "skew_history", start_date, end_date)

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    @_memoised
    def get_term_structure(self, token: str, timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """ATM term structure snapshot. Columns: days_to_expiration, atm_iv, fwd_atm_iv.

        ``fwd_atm_iv`` is the forward vol between consecutive tenors (NaN on the first).
        ``df.attrs["snapshot_time"]`` is the tz-naive UTC snapshot time.
        """
        cur = self._currency_or_log(token, "term_structure")
        if cur is None:
            return None
        params = self._base_params(cur)
        if timestamp is not None:
            params["timestamp"] = day_str(timestamp) if timestamp.time() == datetime.min.time() \
                else timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        rows = self._get("volatility/term-structures/forward-volatility/constant", params)
        if not rows:
            if rows is not None:
                logger.info("Amberdata options: no term structure for %s", token)
            return None
        df = pd.DataFrame(rows)
        out = pd.DataFrame({
            "days_to_expiration": self._num(df, "daysToExpiration"),
            "atm_iv": self._num(df, "atm"),
            "fwd_atm_iv": self._num(df, "fwdAtm"),
        }).dropna(subset=["atm_iv"]).sort_values("days_to_expiration").reset_index(drop=True)
        if out.empty:
            logger.info("Amberdata options: empty term structure for %s", token)
            return None
        snap = to_utc_timestamp(df["timestamp"]).max() if "timestamp" in df.columns else None
        out.attrs = {"snapshot_time": snap}
        return out

    @_memoised
    def get_gamma_exposure(self, token: str, date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Dealer gamma positioning by strike from the latest snapshot (or the last one of ``date``).

        Columns: strike, net_dealer_gamma (sum of dealerNetInventory across expiries and
        puts/calls), total_dealer_gamma (sum of dealerTotalInventory), gamma_level (sum of
        gammaLevel), index_price. Sorted by strike. ``df.attrs``: snapshot_time, index_price.
        """
        cur = self._currency_or_log(token, "gamma_exposure")
        if cur is None:
            return None
        path = "trades-flow/gamma-exposures-snapshots"
        if date is None:
            rows = self._get(path, self._base_params(cur))
        else:
            day = datetime(date.year, date.month, date.day)
            # hourly snapshots; ask for the last hour of the day, then widen if empty
            rows = self._get(path, dict(self._base_params(cur),
                                        startDate=(day + timedelta(hours=23)).strftime("%Y-%m-%dT%H:%M:%S"),
                                        endDate=(day + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")))
            if not rows:
                rows = self._get(path, dict(self._base_params(cur), startDate=day_str(day),
                                            endDate=day_str(day + timedelta(days=1))))
        if not rows:
            if rows is not None:
                logger.info("Amberdata options: no gamma exposure for %s", token)
            return None
        df = pd.DataFrame(rows)
        if "snapshotTimestamp" in df.columns:
            snaps = to_utc_timestamp(df["snapshotTimestamp"])
            latest = snaps.max()
            df = df[snaps == latest]
        else:
            latest = None
        df = df.assign(
            strike=self._num(df, "strike"),
            net=self._num(df, "dealerNetInventory").fillna(0.0),
            total=self._num(df, "dealerTotalInventory").fillna(0.0),
            gamma=self._num(df, "gammaLevel").fillna(0.0),
        ).dropna(subset=["strike"])
        if df.empty:
            logger.info("Amberdata options: empty gamma exposure for %s", token)
            return None
        index_price = self._num(df, "indexPrice").dropna()
        index_price = float(index_price.iloc[-1]) if not index_price.empty else None
        out = df.groupby("strike", as_index=False)[["net", "total", "gamma"]].sum()
        out = out.rename(columns={"net": "net_dealer_gamma", "total": "total_dealer_gamma", "gamma": "gamma_level"})
        out["index_price"] = index_price if index_price is not None else float("nan")
        out = out.sort_values("strike").reset_index(drop=True)
        out.attrs = {"snapshot_time": latest, "index_price": index_price}
        return out

    @_memoised
    def get_block_trades(self, token: str, start_date: datetime, end_date: datetime,
                         top_n: int = 20) -> Optional[pd.DataFrame]:
        """Largest block trades over [start, end] by premium.

        Columns: expiry (tz-naive UTC), strike, put_call ("C"/"P"), contract_volume,
        premium_volume (USD). Aggregated per (expiry, strike, put_call), sorted by
        premium_volume descending, top ``top_n`` rows.
        """
        cur = self._currency_or_log(token, "block_trades")
        if cur is None:
            return None
        rows = self._fetch_range("trades-flow/block-volumes", self._base_params(cur), start_date, end_date)
        if rows is None:
            return None
        if not rows:
            logger.info("Amberdata options: no block trades for %s", token)
            return None
        df = pd.DataFrame(rows)
        if "expirationTimestamp" not in df.columns:
            return None
        df = pd.DataFrame({
            "expiry": to_utc_timestamp(df["expirationTimestamp"]),
            "strike": self._num(df, "strike"),
            "put_call": df["putCall"].astype(str).str.upper().str[:1] if "putCall" in df.columns else "",
            "contract_volume": self._num(df, "contractVolume").fillna(0.0),
            "premium_volume": self._num(df, "premiumVolume").fillna(0.0),
        }).dropna(subset=["strike"])
        out = df.groupby(["expiry", "strike", "put_call"], as_index=False)[["contract_volume", "premium_volume"]].sum()
        out = out.sort_values(["premium_volume", "contract_volume"], ascending=False).head(top_n).reset_index(drop=True)
        return out if not out.empty else None


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def currencies_from_instruments(rows: list) -> dict[str, str]:
    """token -> options currency from ``instruments/information`` rows.

    ``BTC`` and ``BTC_USDC`` both map to token ``btc``; the un-suffixed
    currency wins, then USDC, USDT, USD.
    """
    by_token: dict[str, set[str]] = {}
    for r in rows:
        cur = r.get("currency") if isinstance(r, dict) else None
        if not cur or not isinstance(cur, str):
            continue
        base = cur.split("_")[0].split("-")[0].strip().lower()
        if base:
            by_token.setdefault(base, set()).add(cur)

    def rank(cur: str) -> tuple:
        suffix = cur.split("_", 1)[1] if "_" in cur else ""
        pref = _SUFFIX_PREFERENCE.index(suffix) if suffix in _SUFFIX_PREFERENCE else len(_SUFFIX_PREFERENCE)
        return (pref, cur)

    return {tok: sorted(curs, key=rank)[0] for tok, curs in by_token.items()}
