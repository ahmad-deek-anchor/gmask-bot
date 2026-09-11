"""Haruko end-of-day derivatives PnL, computed live with Carson Levy's EOW method.

``HarukoEodPnl`` runs ``sql/haruko_eod_pnl.sql`` (Carson Levy / Nayshil Dalal's
end-of-week report query, verbatim) through the guarded ``DeskBigQuery.query_script``
path - read-only, dataset allow-list, ``BQ_MAX_BYTES_BILLED`` cost cap - and derives
every period figure from **life-to-date (LTD) differences** of the chosen 15:00
America/Chicago snapshots:

* ``daily``     LTD(day) - LTD(previous EOD row)                      (query column ``daily_pnl``)
* ``monthly``   LTD(last EOD row of the month) - LTD(last EOD row before the month);
                the first month in the data is measured from the first available row
* ``period``    mtd / wtd / last_week / last_month / month:YYYY-MM / range:A..B by the
                same rule; ``ytd`` reports BOTH Haruko's summed ``year_to_date_pnl`` and the
                LTD change since the first available row, labelled

Haruko's own ``month_to_date_pnl`` / ``week_to_date_pnl`` columns are carried for
reference only: they reset mid-period (August 2026 shows -$18,923 while the LTD
difference is +$2,174,523, which is what the EOW report shows).

The query scans ~15 GB (~$0.08, ~45 s). BigQuery's query cache (24 h, free for an
identical script) is left on and the result frame is cached in-process for
``ttl_s`` (15 min) so follow-ups (MTD, then YTD) do not re-scan.
"""

from __future__ import annotations

import calendar
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

logger = logging.getLogger(__name__)

SQL_PATH = Path(__file__).resolve().parent.parent / "sql" / "haruko_eod_pnl.sql"
DEFAULT_TTL_S = 15 * 60
MAX_ROWS = 1000            # one row per EOD day; > 2.5 years of history
TIMEOUT_S = 180            # the script takes ~45 s cold
EOD_ZONE = "America/Chicago"

METHOD_LINE = ("Carson Levy EOW method: 3pm America/Chicago EOD cut, LTD differences, "
               "one full-book snapshot/day")

# DECLARE overrides: name -> (declared variable, validator -> literal)
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)$")
_ZONE_RE = re.compile(r"^[A-Za-z_]+(?:/[A-Za-z0-9_+\-]+){0,2}$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_RANGE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})$")


class HarukoEodError(ValueError):
    """Bad parameter / period request (never a BigQuery failure)."""


def _lit_time(v: Any) -> str:
    s = str(v).strip()
    if not _TIME_RE.match(s):
        raise HarukoEodError(f"eod_time must be HH:MM:SS, got {v!r}")
    return f"TIME '{s}'"


def _lit_zone(v: Any) -> str:
    s = str(v).strip()
    if not _ZONE_RE.match(s):
        raise HarukoEodError(f"eod_zone must be an IANA zone name like America/Chicago, got {v!r}")
    try:
        ZoneInfo(s)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise HarukoEodError(f"Unknown time zone {s!r}") from e
    return f"'{s}'"


def _lit_date(v: Any) -> str:
    if isinstance(v, datetime):
        v = v.date()
    if isinstance(v, date):
        return f"DATE '{v.isoformat()}'"
    s = str(v).strip()
    try:
        d = date.fromisoformat(s)
    except ValueError as e:
        raise HarukoEodError(f"base_date must be YYYY-MM-DD, got {v!r}") from e
    return f"DATE '{d.isoformat()}'"


def _lit_skew(v: Any) -> str:
    try:
        n = int(v)
    except (TypeError, ValueError) as e:
        raise HarukoEodError(f"skew must be an integer number of seconds, got {v!r}") from e
    if not 0 <= n <= 6 * 3600:
        raise HarukoEodError("skew must be between 0 and 21600 seconds")
    return str(n)


OVERRIDES = {
    "eod_time": ("eod_time", _lit_time),
    "eod_zone": ("eod_zone", _lit_zone),
    "base_date": ("ytd_base_date", _lit_date),
    "skew": ("max_skew_secs", _lit_skew),
}


def substitute_declares(sql: str, **overrides: Any) -> str:
    """Replace the DEFAULT literal of the named DECLAREs after validating each value.

    Only the four known parameters are accepted; the new literal is produced by a
    validator (never raw user text) so the guard's 'plain literal' rule keeps holding.
    """
    out = sql
    for name, value in overrides.items():
        if value is None:
            continue
        if name not in OVERRIDES:
            raise HarukoEodError(f"Unknown parameter {name!r}; allowed: {', '.join(OVERRIDES)}")
        var, to_literal = OVERRIDES[name]
        literal = to_literal(value)
        pattern = re.compile(r"(DECLARE\s+" + re.escape(var) + r"\s+\w+\s+DEFAULT\s+)([^;]+)(;)", re.IGNORECASE)
        out, n = pattern.subn(lambda m: m.group(1) + literal + m.group(3), out, count=1)
        if n != 1:
            raise HarukoEodError(f"DECLARE {var} not found in the script")
    return out


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------

@dataclass
class PeriodResult:
    kind: str                       # 'mtd', 'ytd', 'month', 'range', ...
    label: str                      # e.g. 'August 2026', 'MTD September 2026'
    start: Optional[date]           # first calendar day of the period (None = from first row)
    end: date                       # last EOD date used
    baseline_date: Optional[date]   # EOD row the LTD difference starts from (None when from first row)
    baseline_ltd: Optional[float]   # LTD PnL at the baseline row
    end_ltd: Optional[float]        # LTD PnL at ``end``
    pnl: Optional[float]            # LTD difference
    n_days: int                     # EOD rows inside the period
    from_first_row: bool            # baseline is the first available row (period truncated)
    daily: pd.DataFrame             # rows inside the period (ascending)
    haruko_mtd: Optional[float] = None   # Haruko month_to_date_pnl summed at ``end`` (reference only)
    haruko_wtd: Optional[float] = None
    ytd_sum: Optional[float] = None      # ytd only: SUM(year_to_date_pnl) at ``end``
    ytd_ltd_change: Optional[float] = None  # ytd only: LTD change since the first available row
    months: Optional[pd.DataFrame] = None   # monthly table when the period spans > 1 month
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------

class HarukoEodPnl:
    """Carson Levy's EOW PnL series, live from BigQuery, with LTD-difference period maths."""

    def __init__(self, bq, *, ttl_s: int = DEFAULT_TTL_S, max_rows: int = MAX_ROWS,
                 timeout_s: int = TIMEOUT_S, sql_path: str | Path = SQL_PATH, now=time.time,
                 today=None):
        self.bq = bq
        self.ttl_s = int(ttl_s)
        self.max_rows = int(max_rows)
        self.timeout_s = int(timeout_s)
        self.sql_path = Path(sql_path)
        self._now = now
        self._today = today  # callable -> date (tests); default = today in America/Chicago
        self._lock = threading.Lock()
        self._cache: Dict[Tuple, Tuple[float, pd.DataFrame]] = {}

    # -- SQL ----------------------------------------------------------------

    def base_sql(self) -> str:
        return self.sql_path.read_text()

    def sql(self, **overrides: Any) -> str:
        return substitute_declares(self.base_sql(), **overrides)

    # -- data ---------------------------------------------------------------

    def frame(self, force: bool = False, **overrides: Any) -> pd.DataFrame:
        """The EOD series (ascending by ``eod_date``), cached ``ttl_s`` per parameter set.

        ``df.attrs``: ``bytes_processed``, ``cache_hit`` (BigQuery cache), ``fetched_at``,
        ``from_cache`` (served from the in-process TTL cache), ``sql``.
        """
        key = tuple(sorted((k, str(v)) for k, v in overrides.items() if v is not None))
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and not force and self._now() - hit[0] <= self.ttl_s:
                df = hit[1].copy()
                df.attrs.update(hit[1].attrs)
                df.attrs["from_cache"] = True
                df.attrs["cache_age_s"] = self._now() - hit[0]
                return df
            sql = self.sql(**overrides)
            raw = self.bq.query_script(sql, max_rows=self.max_rows, timeout_s=self.timeout_s)
            df = _normalise(raw)
            df.attrs["fetched_at"] = self._now()
            df.attrs["from_cache"] = False
            df.attrs["cache_age_s"] = 0.0
            self._cache[key] = (self._now(), df)
            out = df.copy()
            out.attrs.update(df.attrs)
            return out

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # -- helpers ------------------------------------------------------------

    def today(self) -> date:
        if self._today is not None:
            return self._today()
        return datetime.now(ZoneInfo(EOD_ZONE)).date()

    def latest(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Latest EOD row as a dict plus ``stale`` / ``business_days_behind``."""
        df = self.frame() if df is None else df
        if df.empty:
            return {"eod_date": None, "stale": True, "business_days_behind": None}
        row = df.iloc[-1].to_dict()
        behind = business_days_between(row["eod_date"], self.today())
        row["business_days_behind"] = behind
        row["stale"] = behind > 1
        return row

    # -- period maths -------------------------------------------------------

    def monthly(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """One row per calendar month: LTD difference at month-end EOD.

        Columns: month (YYYY-MM), start (first EOD row in month), end (last EOD row),
        baseline (EOD row the difference starts from), pnl, n_days, from_first_row,
        partial (month not complete as of the last row), haruko_mtd (reference only),
        notional_quote, n_trades.
        """
        df = self.frame() if df is None else df
        if df.empty:
            return pd.DataFrame(columns=["month", "start", "end", "baseline", "pnl", "n_days", "from_first_row",
                                         "partial", "haruko_mtd", "notional_quote", "n_trades"])
        rows = []
        months = sorted({(d.year, d.month) for d in df["eod_date"]})
        last_date = df["eod_date"].iloc[-1]
        for y, m in months:
            first_day = date(y, m, 1)
            last_day = date(y, m, calendar.monthrange(y, m)[1])
            res = self._ltd_change(df, first_day, last_day)
            rows.append({
                "month": f"{y:04d}-{m:02d}",
                "start": res.daily["eod_date"].iloc[0] if len(res.daily) else None,
                "end": res.end,
                "baseline": res.baseline_date,
                "pnl": res.pnl,
                "n_days": res.n_days,
                "from_first_row": res.from_first_row,
                "partial": last_date < last_day,
                "haruko_mtd": res.haruko_mtd,
                "notional_quote": float(res.daily["notional_quote"].sum()) if "notional_quote" in res.daily else None,
                "n_trades": int(res.daily["n_trades"].sum()) if "n_trades" in res.daily else None,
            })
        return pd.DataFrame(rows)

    def period(self, kind: str = "mtd", df: Optional[pd.DataFrame] = None) -> PeriodResult:
        """PnL for ``kind``: mtd | wtd | ytd | last_week | last_month | month:YYYY-MM | range:A..B."""
        df = self.frame() if df is None else df
        if df.empty:
            raise HarukoEodError("The EOD query returned no rows.")
        k = (kind or "mtd").strip().lower().replace(" ", "")
        latest = df["eod_date"].iloc[-1]
        first = df["eod_date"].iloc[0]

        if k in ("mtd", "month_to_date", "thismonth", "this_month"):
            start = latest.replace(day=1)
            res = self._ltd_change(df, start, latest, kind="mtd")
            res.label = f"MTD {latest.strftime('%B %Y')} (to {latest.isoformat()} EOD)"
            return res
        if k in ("wtd", "week_to_date", "thisweek", "this_week"):
            start = latest - timedelta(days=latest.weekday())  # Monday
            res = self._ltd_change(df, start, latest, kind="wtd")
            res.label = f"WTD week of {start.isoformat()} (to {latest.isoformat()} EOD)"
            return res
        if k in ("last_week", "lastweek", "previous_week", "prior_week"):
            this_monday = latest - timedelta(days=latest.weekday())
            end = this_monday - timedelta(days=1)          # last Sunday
            start = end - timedelta(days=6)                 # previous Monday
            res = self._ltd_change(df, start, end, kind="last_week")
            res.label = f"Last week {start.isoformat()} to {end.isoformat()} (EOD {res.end.isoformat()})"
            return res
        if k in ("ytd", "year_to_date", "thisyear", "this_year"):
            start = date(latest.year, 1, 1)
            res = self._ltd_change(df, start, latest, kind="ytd")
            res.label = f"YTD {latest.year} (to {latest.isoformat()} EOD)"
            last_row = df.iloc[-1]
            res.ytd_sum = _f(last_row.get("ytd_pnl"))
            res.ytd_ltd_change = _f(last_row.get("ltd_pnl")) - _f(df.iloc[0].get("ltd_pnl"))
            res.pnl = res.ytd_sum
            res.notes.append(
                f"YTD is reported two ways: (a) Haruko SUM(year_to_date_pnl) at {latest.isoformat()} EOD; "
                f"(b) LTD change since the first available EOD row ({first.isoformat()})"
                + (" - (b) is a partial-year figure because snapshots do not start on 1 January."
                   if (first.month, first.day) != (1, 1) else ".")
            )
            res.months = self.monthly(df)
            return res
        if k in ("last_month", "lastmonth", "previous_month", "prior_month"):
            y, m = (latest.year, latest.month - 1) if latest.month > 1 else (latest.year - 1, 12)
            return self.period(f"month:{y:04d}-{m:02d}", df)
        mm = _MONTH_RE.match(k) or _MONTH_RE.match(k.removeprefix("month:"))
        if mm:
            y, m = int(mm.group(1)), int(mm.group(2))
            if not 1 <= m <= 12:
                raise HarukoEodError(f"Bad month {kind!r}")
            start, end = date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
            if end < first or start > latest:
                raise HarukoEodError(
                    f"No EOD snapshots for {start.strftime('%B %Y')}; data covers {first.isoformat()} to {latest.isoformat()}.")
            res = self._ltd_change(df, start, end, kind="month")
            res.label = start.strftime("%B %Y") + (" (month to date)" if latest < end else "")
            return res
        rm = _RANGE_RE.match(k.removeprefix("range:"))
        if rm:
            try:
                start, end = date.fromisoformat(rm.group(1)), date.fromisoformat(rm.group(2))
            except ValueError as e:
                raise HarukoEodError(f"Bad range {kind!r}: {e}") from e
            if start > end:
                raise HarukoEodError("range start must not be after its end")
            if end < first or start > latest:
                raise HarukoEodError(
                    f"No EOD snapshots in {start.isoformat()}..{end.isoformat()}; data covers "
                    f"{first.isoformat()} to {latest.isoformat()}.")
            res = self._ltd_change(df, start, end, kind="range")
            res.label = f"{start.isoformat()} to {end.isoformat()} (EOD {res.end.isoformat()})"
            lo = max(start, first)
            if (res.end.year, res.end.month) != (lo.year, lo.month):
                months = self.monthly(df)
                res.months = months[(months["month"] >= lo.strftime("%Y-%m"))
                                    & (months["month"] <= res.end.strftime("%Y-%m"))].reset_index(drop=True)
            return res
        raise HarukoEodError(
            f"Unknown period {kind!r}; use mtd, wtd, ytd, last_week, last_month, month:YYYY-MM or range:YYYY-MM-DD..YYYY-MM-DD.")

    def _ltd_change(self, df: pd.DataFrame, start: date, end: date, kind: str = "range") -> PeriodResult:
        """LTD(last row <= end) - LTD(last row < start); from the first row when none precedes."""
        dates = df["eod_date"]
        inside = df[(dates >= start) & (dates <= end)]
        before = df[dates < start]
        upto = df[dates <= end]
        if upto.empty:
            raise HarukoEodError(f"No EOD snapshots on or before {end.isoformat()}.")
        end_row = upto.iloc[-1]
        haruko_mtd = _f(end_row.get("mtd_pnl"))
        haruko_wtd = _f(end_row.get("wtd_pnl"))
        if not before.empty:
            base = before.iloc[-1]
            pnl = _f(end_row["ltd_pnl"]) - _f(base["ltd_pnl"])
            return PeriodResult(kind=kind, label="", start=start, end=end_row["eod_date"],
                                baseline_date=base["eod_date"], baseline_ltd=_f(base["ltd_pnl"]),
                                end_ltd=_f(end_row["ltd_pnl"]), pnl=pnl, n_days=len(inside),
                                from_first_row=False, daily=inside.reset_index(drop=True),
                                haruko_mtd=haruko_mtd, haruko_wtd=haruko_wtd)
        # no snapshot before the period: measure from the first available row
        first_row = df.iloc[0]
        pnl = _f(end_row["ltd_pnl"]) - _f(first_row["ltd_pnl"])
        res = PeriodResult(kind=kind, label="", start=start, end=end_row["eod_date"],
                           baseline_date=first_row["eod_date"], baseline_ltd=_f(first_row["ltd_pnl"]),
                           end_ltd=_f(end_row["ltd_pnl"]), pnl=pnl, n_days=len(inside),
                           from_first_row=True, daily=inside.reset_index(drop=True),
                           haruko_mtd=haruko_mtd, haruko_wtd=haruko_wtd)
        res.notes.append(
            f"No EOD snapshot before {start.isoformat()}: measured from the first available row "
            f"({first_row['eod_date'].isoformat()} EOD), so this excludes PnL earned on that first day.")
        return res


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _f(v) -> float:
    try:
        if v is None or pd.isna(v):
            return 0.0
    except (TypeError, ValueError):
        pass
    return float(v)


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Ascending by eod_date, eod_date as datetime.date, numeric columns as float."""
    df = raw.copy()
    df.attrs.update(raw.attrs)
    if df.empty or "eod_date" not in df.columns:
        return df
    df["eod_date"] = pd.to_datetime(df["eod_date"]).dt.date
    for col in ("ytd_pnl", "mtd_pnl", "wtd_pnl", "ltd_pnl", "ytd_pnl_check", "daily_pnl", "trailing_7d_pnl",
                "notional_quote", "skew_secs", "n_positions", "n_trades"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("eod_date").drop_duplicates("eod_date", keep="last").reset_index(drop=True)
    return df


def business_days_between(start: date, end: date) -> int:
    """Weekdays (Mon-Fri) in the half-open interval (start, end]; 0 when end <= start."""
    if end <= start:
        return 0
    n, d = 0, start + timedelta(days=1)
    while d <= end:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


__all__ = [
    "DEFAULT_TTL_S",
    "HarukoEodError",
    "HarukoEodPnl",
    "MAX_ROWS",
    "METHOD_LINE",
    "OVERRIDES",
    "PeriodResult",
    "SQL_PATH",
    "TIMEOUT_S",
    "business_days_between",
    "substitute_declares",
]
