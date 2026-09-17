"""Free official macro feeds that need no key: the US Treasury daily yield curve and CBOE's VIX history.

Both publish after each US close (Treasury by about 18:00 New York, CBOE overnight), so they are
fresher than FRED's mirror of the same series (FRED runs one to two days behind) and carry the
publisher's own dates. Nothing here is live.

    TreasuryCurve().curve(days=45)   -> date + tenor columns in percent (1m ... 30y)
    CboeVix().history(days=90)       -> date, open, high, low, close

Endpoints (checked 2026-09-17):
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml
        ?data=daily_treasury_yield_curve&field_tdr_date_value_month=YYYYMM      (Atom XML, one month per call)
    https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv    (full history CSV, 1990 ->)
"""

from __future__ import annotations

import io
import logging
import threading
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

TREASURY_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
TTL_S = 30 * 60
USER_AGENT = "gmask-bot/1.0 (+anchorage global markets desk)"

# Treasury XML field -> tenor label, in curve order
TREASURY_TENORS = (
    ("BC_1MONTH", "1m"), ("BC_2MONTH", "2m"), ("BC_3MONTH", "3m"), ("BC_4MONTH", "4m"), ("BC_6MONTH", "6m"),
    ("BC_1YEAR", "1y"), ("BC_2YEAR", "2y"), ("BC_3YEAR", "3y"), ("BC_5YEAR", "5y"), ("BC_7YEAR", "7y"),
    ("BC_10YEAR", "10y"), ("BC_20YEAR", "20y"), ("BC_30YEAR", "30y"),
)
TENOR_ORDER = [t for _, t in TREASURY_TENORS]
_NS = {"a": "http://www.w3.org/2005/Atom",
       "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
       "d": "http://schemas.microsoft.com/ado/2007/08/dataservices"}


class OfficialFeedError(RuntimeError):
    """The publisher's endpoint failed or returned something unparseable."""


class _Feed:
    def __init__(self, session: Optional[requests.Session] = None, timeout: int = 45, retries: int = 2,
                 ttl_s: int = TTL_S):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.timeout = timeout          # treasury.gov regularly takes 10-30 s to answer
        self.retries = retries
        self.ttl_s = ttl_s
        self._memo: dict = {}
        self._lock = threading.Lock()

    def _get_text(self, url: str, params: Optional[dict] = None) -> str:
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last = e
                logger.info("official feed %s attempt %d failed: %s", url, attempt + 1, e)
                continue
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise OfficialFeedError(f"HTTP {resp.status_code} from {url}")
        raise OfficialFeedError(f"request failed after {self.retries + 1} attempts: {last}")

    def _memoised(self, key, fn):
        now = time.time()
        with self._lock:
            hit = self._memo.get(key)
            if hit is not None and hit[0] > now:
                return hit[1].copy()
        val = fn()
        with self._lock:
            self._memo[key] = (now + self.ttl_s, val)
        return val.copy()


def _months_back(today: date, days: int) -> list[str]:
    """YYYYMM strings covering [today - days, today], newest first."""
    out = []
    cur = today.replace(day=1)
    start = today - timedelta(days=days)
    while True:
        out.append(cur.strftime("%Y%m"))
        if cur <= start.replace(day=1):
            break
        cur = (cur - timedelta(days=1)).replace(day=1)
    return out


class TreasuryCurve(_Feed):
    """US Treasury par yield curve, daily, from the Treasury's own XML feed."""

    def _parse_month(self, text: str) -> list[dict]:
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            raise OfficialFeedError(f"Treasury XML unparseable: {e}") from e
        rows = []
        for entry in root.findall("a:entry", _NS):
            props = entry.find(".//m:properties", _NS)
            if props is None:
                continue
            rec: dict = {}
            for child in props:
                tag = child.tag.split("}")[-1]
                rec[tag] = child.text
            d = rec.get("NEW_DATE")
            if not d:
                continue
            row = {"date": pd.Timestamp(d[:10])}
            for field, tenor in TREASURY_TENORS:
                row[tenor] = pd.to_numeric(rec.get(field), errors="coerce")
            rows.append(row)
        return rows

    def curve(self, days: int = 45, today: Optional[date] = None) -> pd.DataFrame:
        """Daily par yields (percent) for the last ``days`` calendar days: date + TENOR_ORDER columns."""
        days = max(1, min(int(days), 3660))
        today = today or datetime.now(timezone.utc).date()

        def fetch():
            rows: list[dict] = []
            for month in _months_back(today, days):
                text = self._get_text(TREASURY_URL, {"data": "daily_treasury_yield_curve", "field_tdr_date_value_month": month})
                rows.extend(self._parse_month(text))
            df = pd.DataFrame(rows, columns=["date"] + TENOR_ORDER)
            if df.empty:
                raise OfficialFeedError("Treasury feed returned no rows")
            df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
            return df[df["date"] >= pd.Timestamp(today - timedelta(days=days))].reset_index(drop=True)
        return self._memoised(("curve", days, today.isoformat()), fetch)


class CboeVix(_Feed):
    """CBOE VIX daily OHLC from CBOE's published history file."""

    def history(self, days: int = 90, today: Optional[date] = None) -> pd.DataFrame:
        days = max(1, min(int(days), 365 * 40))
        today = today or datetime.now(timezone.utc).date()

        def fetch_all():
            text = self._get_text(CBOE_VIX_URL)
            try:
                df = pd.read_csv(io.StringIO(text))
            except Exception as e:  # noqa: BLE001
                raise OfficialFeedError(f"CBOE CSV unparseable: {e}") from e
            df.columns = [c.strip().lower() for c in df.columns]
            need = ["date", "open", "high", "low", "close"]
            if any(c not in df.columns for c in need):
                raise OfficialFeedError(f"CBOE CSV has unexpected columns {list(df.columns)}")
            df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
            for c in need[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)[need]

        full = self._memoised(("vix", today.isoformat()), fetch_all)
        return full[full["date"] >= pd.Timestamp(today - timedelta(days=days))].reset_index(drop=True)


__all__ = ["CBOE_VIX_URL", "CboeVix", "OfficialFeedError", "TENOR_ORDER", "TREASURY_TENORS", "TREASURY_URL", "TreasuryCurve"]
