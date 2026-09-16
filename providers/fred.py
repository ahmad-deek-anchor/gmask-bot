"""FRED (Federal Reserve Bank of St. Louis) macro series: rates, curve, dollar, oil, equity indices,
volatility, credit spreads, inflation, labour.

The traditional-finance vendors on our Coin Metrics key (FMP, Databento) are not entitled and
Stooq blocks API use, so FRED is the desk's macro source for now (decision 2026-09-16: "FRED only").
It is free: create a key at https://fred.stlouisfed.org/docs/api/api_key.html and store it as
Secret Manager ``fred_api_key`` in anchorage-trading-solutions (or env ``FRED_API_KEY``).

API: ``https://api.stlouisfed.org/fred`` with ``api_key`` and ``file_type=json`` as query
parameters (120 requests / minute). Series are daily (business days) or monthly and publish
with a one-day lag; missing days arrive as the string ".". Every value returned here carries
its observation date - nothing in FRED is live. The key is never logged.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.stlouisfed.org/fred"
RETRY_STATUSES = {429, 500, 502, 503, 504}
TTL_S = 15 * 60
SEARCH_LIMIT = 25

# series id -> (name, units, group). Groups drive the dashboard layout.
MACRO_SERIES: "OrderedDict[str, tuple[str, str, str]]" = OrderedDict([
    ("DGS2", ("US 2y Treasury yield", "%", "rates")),
    ("DGS10", ("US 10y Treasury yield", "%", "rates")),
    ("DGS30", ("US 30y Treasury yield", "%", "rates")),
    ("T10Y2Y", ("10y - 2y Treasury spread", "pp", "rates")),
    ("DFII10", ("US 10y real yield (TIPS)", "%", "rates")),
    ("SOFR", ("SOFR", "%", "rates")),
    ("FEDFUNDS", ("Effective fed funds rate (monthly avg)", "%", "rates")),
    ("DTWEXBGS", ("Broad US dollar index (trade-weighted)", "index", "fx")),
    ("DCOILWTICO", ("WTI crude", "$/bbl", "commodities")),
    ("DCOILBRENTEU", ("Brent crude", "$/bbl", "commodities")),
    ("SP500", ("S&P 500", "index", "equities")),
    ("NASDAQCOM", ("Nasdaq Composite", "index", "equities")),
    ("DJIA", ("Dow Jones Industrial Average", "index", "equities")),
    ("VIXCLS", ("VIX", "index", "volatility")),
    ("BAMLH0A0HYM2", ("US high-yield OAS (ICE BofA)", "%", "credit")),
    ("BAMLC0A0CM", ("US investment-grade OAS (ICE BofA)", "%", "credit")),
    ("T5YIE", ("5y breakeven inflation", "%", "inflation")),
    ("T10YIE", ("10y breakeven inflation", "%", "inflation")),
    ("CPIAUCSL", ("CPI, all items (index, monthly)", "index", "inflation")),
    ("UNRATE", ("Unemployment rate (monthly)", "%", "labour")),
])
GROUPS = ("rates", "fx", "commodities", "equities", "volatility", "credit", "inflation", "labour")


class FredError(RuntimeError):
    """FRED returned an error payload (bad series id, bad key, rate limit after retries)."""


def _redact(params: dict) -> dict:
    return {k: ("***" if k == "api_key" else v) for k, v in (params or {}).items()}


class FredProvider:
    """Thin FRED client with retries and a 15-minute memo. Never raises for vendor failures
    except FredError from ``observations`` / ``series_info`` (the tools turn it into a message)."""

    def __init__(self, api_key: str, session: Optional[requests.Session] = None, timeout: int = 15,
                 max_retries: int = 2, backoff: float = 1.0):
        self._key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.call_count = 0
        self._memo: dict = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ HTTP

    def _get(self, path: str, params: dict) -> dict:
        params = dict(params, api_key=self._key, file_type="json")
        url = f"{BASE_URL}/{path.lstrip('/')}"
        attempt = 0
        while True:
            self.call_count += 1
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt >= self.max_retries:
                    raise FredError(f"FRED request failed: {e}") from e
                attempt += 1
                time.sleep(self.backoff * (2 ** (attempt - 1)))
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as e:
                    raise FredError("FRED returned a non-JSON response") from e
            if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                attempt += 1
                time.sleep(self.backoff * (2 ** (attempt - 1)))
                continue
            msg = ""
            try:
                j = resp.json()
                msg = j.get("error_message") or ""
            except ValueError:
                msg = (resp.text or "")[:200]
            logger.warning("FRED HTTP %s on %s params=%s: %s", resp.status_code, path, _redact(params), msg)
            raise FredError(f"FRED HTTP {resp.status_code}: {msg or 'error'}")

    def _memoised(self, key: tuple, fn):
        now = time.time()
        with self._lock:
            hit = self._memo.get(key)
            if hit is not None and hit[0] > now:
                val = hit[1]
                return val.copy() if isinstance(val, pd.DataFrame) else val
        val = fn()
        with self._lock:
            self._memo[key] = (now + TTL_S, val)
        return val.copy() if isinstance(val, pd.DataFrame) else val

    # ------------------------------------------------------------------ series

    def series_info(self, series_id: str) -> dict:
        """title, units, frequency, seasonal_adjustment, last_updated, observation_end for one series."""
        sid = (series_id or "").strip().upper()

        def fetch():
            j = self._get("series", {"series_id": sid})
            rows = j.get("seriess") or []
            if not rows:
                raise FredError(f"FRED has no series {sid}")
            r = rows[0]
            return {"id": r.get("id", sid), "title": r.get("title", ""), "units": r.get("units", ""),
                    "units_short": r.get("units_short", ""), "frequency": r.get("frequency", ""),
                    "frequency_short": r.get("frequency_short", ""), "seasonal_adjustment": r.get("seasonal_adjustment", ""),
                    "last_updated": r.get("last_updated", ""), "observation_start": r.get("observation_start", ""),
                    "observation_end": r.get("observation_end", ""), "notes": (r.get("notes") or "")[:400]}
        return self._memoised(("info", sid), fetch)

    def observations(self, series_id: str, days: int = 90) -> pd.DataFrame:
        """Observations for the last ``days`` calendar days: columns date (UTC midnight), value (float).
        FRED's "." placeholders (holidays, not yet published) are dropped."""
        sid = (series_id or "").strip().upper()
        days = max(1, min(int(days), 365 * 30))

        def fetch():
            start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            j = self._get("series/observations", {"series_id": sid, "observation_start": start, "sort_order": "asc",
                                                  "limit": 100000})
            obs = j.get("observations") or []
            df = pd.DataFrame([{"date": o.get("date"), "value": o.get("value")} for o in obs if isinstance(o, dict)],
                              columns=["date", "value"])
            if df.empty:
                return df
            df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
        return self._memoised(("obs", sid, days), fetch)

    def latest(self, series_ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Latest value and short-run changes for each series (default: MACRO_SERIES).

        Columns: series_id, name, units, group, value, date, prev_value, prev_date, chg_1 (vs the
        previous observation), value_1w (last observation at least 7 days earlier), chg_1w,
        value_1m (at least 28 days earlier), chg_1m, error (text when a series failed)."""
        ids = [s.upper() for s in (series_ids or list(MACRO_SERIES))]
        rows = []
        for sid in ids:
            name, units, group = MACRO_SERIES.get(sid, (sid, "", "other"))
            rec = {"series_id": sid, "name": name, "units": units, "group": group, "value": None, "date": None,
                   "prev_value": None, "prev_date": None, "chg_1": None, "value_1w": None, "chg_1w": None,
                   "value_1m": None, "chg_1m": None, "error": None}
            try:
                df = self.observations(sid, days=60)
                if len(df) < 3:
                    df = self.observations(sid, days=400)   # monthly / quarterly series
                if sid not in MACRO_SERIES:
                    try:
                        info = self.series_info(sid)
                        rec["name"], rec["units"] = info["title"], info["units_short"] or info["units"]
                    except FredError:
                        pass
            except FredError as e:
                rec["error"] = str(e)
                rows.append(rec)
                continue
            if df.empty:
                rec["error"] = "no observations"
                rows.append(rec)
                continue
            last = df.iloc[-1]
            rec["value"], rec["date"] = float(last["value"]), last["date"]
            if len(df) > 1:
                prev = df.iloc[-2]
                rec["prev_value"], rec["prev_date"] = float(prev["value"]), prev["date"]
                rec["chg_1"] = rec["value"] - rec["prev_value"]
            monthly = len(df) > 1 and (last["date"] - df.iloc[-2]["date"]) > pd.Timedelta(days=20)
            for span, key in ((7, "1w"), (28, "1m")):
                if key == "1w" and monthly:
                    continue          # a monthly print has no one-week change; leave it n/a
                older = df[df["date"] <= last["date"] - pd.Timedelta(days=span)]
                if len(older):
                    v = float(older.iloc[-1]["value"])
                    rec[f"value_{key}"], rec[f"chg_{key}"] = v, rec["value"] - v
            rows.append(rec)
        return pd.DataFrame(rows)

    def search(self, text: str, limit: int = 10) -> pd.DataFrame:
        """FRED full-text series search: id, title, units, frequency, last_updated, popularity, observation_end."""
        q = (text or "").strip()
        n = max(1, min(int(limit), SEARCH_LIMIT))

        def fetch():
            j = self._get("series/search", {"search_text": q, "limit": n, "order_by": "popularity", "sort_order": "desc"})
            rows = [{"id": r.get("id"), "title": r.get("title"), "units": r.get("units_short") or r.get("units"),
                     "frequency": r.get("frequency_short") or r.get("frequency"), "last_updated": (r.get("last_updated") or "")[:10],
                     "popularity": r.get("popularity"), "observation_end": r.get("observation_end")}
                    for r in (j.get("seriess") or []) if isinstance(r, dict)]
            return pd.DataFrame(rows, columns=["id", "title", "units", "frequency", "last_updated", "popularity", "observation_end"])
        return self._memoised(("search", q.lower(), n), fetch)


__all__ = ["BASE_URL", "FredError", "FredProvider", "GROUPS", "MACRO_SERIES", "TTL_S"]
