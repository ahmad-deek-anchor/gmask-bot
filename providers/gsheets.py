"""Read-only access to the spot desk's "A1 Metrics Dashboard" Google Sheet.

The sheet (owner Joao Luis, maintained daily; each dashboard tab carries a
"Data as of" cell) is the booked PnL of the **spot** business, as opposed to the
Haruko mark-to-market PnL of the OTC / derivatives book in BigQuery
(``providers/bigquery.py``):

* ``HOLD`` - Anchorage's client spot trading business; the sheet's ``hold db`` tab is
  fed by ``trading_client_trades_*`` volume and ``trading_commissions_commission_in_usd``
  fields, i.e. HOLD PnL is trading commissions on client trades.
* ``A1`` - A1 Ltd, the principal spot trading desk. Weekly A1 PnL is split into
  "Realized PNL" and an "Unrealized PNL Approximation".
* ``TOTAL`` - A1 + HOLD combined.

Layouts observed live on 2026-09-11 (hand-built tabs: merged group headers,
blank spacer columns; every parser locates header rows by content, never by fixed
row numbers):

``Volume & PNL`` / ``2025 Volume & PNL`` (29 cols)
    row 1  ``Data as of | 2026-09-11``          (2025 tab: one cell "Data as of 12/31/25")
    row 2  ``HOLD`` (col B) ``A1`` (col G) ``TOTAL`` (col L)   - merged group labels
    row 3  ``Month | Volume | PNL | Take Rate`` per group; the TOTAL group adds
           ``Cumulative Volume | Cumulative PNL | Target PNL | % of Target``
    rows 4-15 months 1..12 (future months blank / 0); row 16 ``Total YTD``.
    Take Rate is stored in **bps** (PNL / Volume * 1e4).
``Weekly PNL``
    row 1 as-of; row 2 ``HOLD`` | ``A1`` | ``A1 + HOLD``; row 3
    ``Week | PNL`` (HOLD), ``Week | Realized PNL | Unrealized PNL Approximation | Total``
    (A1), ``Week | PNL | Start Date | End Date`` (combined; dates like
    ``Jan 1, 2026 (Thu)``); weeks 1..52+ then a ``Total`` row. Future weeks are 0.
``Financing Fees`` / ``2025 Financing Fees``
    ``Month | HOLD Financing Fees | HOLD Delta Sales | Total`` (A-D) and a weekly
    block ``Week | HOLD Financing Fees | HOLD Delta Sales | Total`` (G-J). No as-of cell.
``Nonclient PNL``
    only a title ("HOLD PNL") and a link to another spreadsheet - not populated here.
``db`` (counterparty trade blotter, one row per A1 client trade)
    ``Date (UTC) | Counterparty | Side | Symbol | Buy QTY | Buy Asset | Sell QTY |
    Sell Asset | Price | PNL | Currency | bps | Month``. Covers 2024 only (the fuller
    ``Trades`` tab stops in May 2025 and has no PnL column; ``Dealer Trades`` /
    ``Exchange Trades`` are broken IMPORTRANGEs). Accessors report the coverage window.

Auth: Application Default Credentials (user account) with the
``spreadsheets.readonly`` scope, via ``google.auth.default()``; requests go straight
to the Sheets REST API v4 with ``requests`` and carry ``x-goog-user-project`` so the
quota is charged to the ADC quota project. No Google API client library is needed.

Nothing here ever writes to the sheet.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

import pandas as pd

logger = logging.getLogger(__name__)

SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets.readonly"
ADC_LOGIN_CMD = (
    "gcloud auth application-default login --scopes="
    "https://www.googleapis.com/auth/cloud-platform,"
    "https://www.googleapis.com/auth/drive.file,"
    "https://www.googleapis.com/auth/presentations.readonly,"
    "https://www.googleapis.com/auth/spreadsheets.readonly,"
    "https://www.googleapis.com/auth/drive.readonly"
)

# tab names (overridable per instance for tests / sheet re-organisations)
TAB_VOLUME_PNL = "Volume & PNL"
TAB_WEEKLY_PNL = "Weekly PNL"
TAB_NONCLIENT_PNL = "Nonclient PNL"
TAB_FINANCING_FEES = "Financing Fees"
TAB_COUNTERPARTY_TRADES = "db"      # the trade-level blotter; 'Counterparty Trades' is a summary tab

# generous ranges: the tabs are ~20 wide and < 60 rows; the blotter is read whole
RANGE_VOLUME_PNL = "A1:T40"
RANGE_WEEKLY_PNL = "A1:N80"
RANGE_FINANCING_FEES = "A1:L70"
RANGE_NONCLIENT_PNL = "A1:Z40"
RANGE_TRADES = "A:M"

ENTITY_ALIASES = {
    "HOLD": "HOLD",
    "A1": "A1",
    "TOTAL": "TOTAL",
    "TOTAL/COMBINED": "TOTAL",
    "COMBINED": "TOTAL",
    "A1 + HOLD": "TOTAL",
    "A1+HOLD": "TOTAL",
    "HOLD + A1": "TOTAL",
    "HOLD+A1": "TOTAL",
}

STABLE_ASSETS = {
    "USD", "USDC", "USDT", "USDG", "PYUSD", "USDC.E", "USDC_SPL", "USDC_SOL", "USDC_ARB",
    "USDC_BASE", "USDC_NOBLE_DYDX", "USDT_SOL", "USDG_SOL", "USAT", "DAI", "TUSD", "FDUSD", "EURC",
}

_DATE_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
_A1_RE = re.compile(r"^\s*(?:'?(?P<tab>[^'!]+)'?!)?(?P<range>[A-Za-z]{0,3}\d*(?::[A-Za-z]{0,3}\d*)?)\s*$")


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class SheetsUnavailable(RuntimeError):
    """google-auth / requests missing or no Application Default Credentials."""


class SheetsAccessError(RuntimeError):
    """The Sheets API refused the request (403 / 404 / other HTTP error) - message is
    operator-readable and safe to relay to the model."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# cell helpers (module level so tests can exercise them directly)
# ---------------------------------------------------------------------------

def _norm(v: Any) -> str:
    """Normalised header text: str, stripped, collapsed whitespace, upper-case."""
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v)).strip().upper()


def _is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def to_float(v: Any) -> float:
    """Sheet cell -> float. Numbers pass through; blanks, '#N/A ...', '-' -> NaN;
    strings like '$1,234.5', '5.882 bps', '12%' are parsed leniently."""
    if v is None or isinstance(v, bool):
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s.startswith("#") or s in {"-", "n/a", "N/A"}:
        return float("nan")
    m = _NUMBER_RE.search(s.replace(" ", ""))
    if not m:
        return float("nan")
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return float("nan")


def to_int(v: Any) -> Optional[int]:
    f = to_float(v)
    if f != f:  # NaN
        return None
    if abs(f - round(f)) > 1e-9:
        return None
    return int(round(f))


def parse_sheet_date(v: Any) -> Optional[pd.Timestamp]:
    """'Jan 1, 2026 (Thu)', '2026-09-11', '12/31/25', '2024-02-12 20:05:00' -> Timestamp
    (tz-naive). Google serial numbers (days since 1899-12-30) are accepted too."""
    if _is_blank(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if 20000 < float(v) < 80000:
            return pd.Timestamp("1899-12-30") + timedelta(days=float(v))
        return None
    s = _DATE_PAREN_RE.sub("", str(v)).strip()
    if not s:
        return None
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    try:
        ts = pd.to_datetime(s, errors="coerce")
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT or ts is None:
        return None
    return pd.Timestamp(ts).tz_localize(None) if getattr(ts, "tzinfo", None) else pd.Timestamp(ts)


def _cell(rows: Sequence[Sequence[Any]], r: int, c: int) -> Any:
    if r < 0 or r >= len(rows):
        return None
    row = rows[r]
    return row[c] if 0 <= c < len(row) else None


def _find_header_row(rows: Sequence[Sequence[Any]], label: str, min_count: int = 1) -> Optional[int]:
    """Index of the first row with at least `min_count` cells equal to `label`."""
    want = _norm(label)
    for i, row in enumerate(rows):
        if sum(1 for v in row if _norm(v) == want) >= min_count:
            return i
    return None


def _group_label(rows: Sequence[Sequence[Any]], header_idx: int, col: int) -> str:
    """Label of the merged group header above `col`: the nearest non-blank cell at or
    left of `col` in the row above the header row (merged cells arrive as a single
    value in their first column)."""
    if header_idx == 0:
        return ""
    row = rows[header_idx - 1]
    for c in range(min(col, len(row) - 1), -1, -1):
        if not _is_blank(row[c]):
            return str(row[c]).strip()
    return ""


def _entity_from_label(label: str, ordinal: int) -> str:
    key = _norm(label).replace("  ", " ")
    if key in ENTITY_ALIASES:
        return ENTITY_ALIASES[key]
    if "HOLD" in key and "A1" in key:
        return "TOTAL"
    if key.startswith("TOTAL"):
        return "TOTAL"
    if key.startswith("HOLD"):
        return "HOLD"
    if key.startswith("A1"):
        return "A1"
    # unlabeled: fall back to the conventional order of the dashboard
    return ("HOLD", "A1", "TOTAL")[ordinal] if ordinal < 3 else (label or f"GROUP{ordinal + 1}")


def _blocks(rows: Sequence[Sequence[Any]], header_idx: int, key_label: str) -> List[Tuple[int, Dict[str, int]]]:
    """Split a header row into blocks that start at each `key_label` cell.

    Returns [(key_col, {normalised header -> col}), ...]; each block runs until the
    next key cell. Blank header cells inside a block are skipped, so a spacer column
    never becomes a field.
    """
    header = rows[header_idx]
    keys = [c for c, v in enumerate(header) if _norm(v) == _norm(key_label)]
    out = []
    for i, kc in enumerate(keys):
        end = keys[i + 1] if i + 1 < len(keys) else len(header)
        fields = {}
        for c in range(kc + 1, end):
            name = _norm(header[c])
            if name and name not in fields:
                fields[name] = c
        out.append((kc, fields))
    return out


def _pick(fields: Dict[str, int], *candidates: str) -> Optional[int]:
    """First matching column for any candidate header (exact, then prefix match)."""
    for cand in candidates:
        if cand in fields:
            return fields[cand]
    for cand in candidates:
        for name, col in fields.items():
            if name.startswith(cand):
                return col
    return None


def parse_a1_range(a1: str) -> Tuple[Optional[str], str]:
    """"'Weekly PNL'!A1:M40" -> ("Weekly PNL", "A1:M40"); "A1:B2" -> (None, "A1:B2")."""
    m = _A1_RE.match(a1 or "")
    if not m or not m.group("range"):
        raise ValueError(f"Invalid A1 range: {a1!r}")
    tab = m.group("tab")
    return (tab.strip() if tab else None), m.group("range").upper()


def col_to_index(col: str) -> int:
    """'A' -> 0, 'Z' -> 25, 'AA' -> 26."""
    n = 0
    for ch in col.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def index_to_col(idx: int) -> str:
    """0 -> 'A', 26 -> 'AA'."""
    idx += 1
    s = ""
    while idx:
        idx, r = divmod(idx - 1, 26)
        s = chr(65 + r) + s
    return s


def clamp_a1_range(rng: str, max_rows: int, max_cols: int) -> Tuple[str, bool]:
    """Shrink an A1 range (no tab) so it covers at most max_rows x max_cols cells.

    Unbounded parts ('A:M', 'A1:M', '3:8') are bounded first. Returns (range, clamped).
    """
    m = re.match(r"^([A-Z]*)(\d*)(?::([A-Z]*)(\d*))?$", rng.upper().strip())
    if not m or not (m.group(1) or m.group(2)):
        raise ValueError(f"Invalid A1 range: {rng!r}")
    c1, r1, c2, r2 = m.group(1), m.group(2), m.group(3), m.group(4)
    if m.group(3) is None and m.group(4) is None:      # single cell like 'B7' or bare column/row
        c2, r2 = c1, r1
    start_col = col_to_index(c1) if c1 else 0
    start_row = int(r1) if r1 else 1
    end_col = col_to_index(c2) if c2 else start_col + max_cols - 1
    end_row = int(r2) if r2 else start_row + max_rows - 1
    if end_col < start_col:
        start_col, end_col = end_col, start_col
    if end_row < start_row:
        start_row, end_row = end_row, start_row
    clamped = False
    if end_col - start_col + 1 > max_cols:
        end_col = start_col + max_cols - 1
        clamped = True
    if end_row - start_row + 1 > max_rows:
        end_row = start_row + max_rows - 1
        clamped = True
    if not (c2 and r2) or (c1 and not r1) or (r1 and not c1):
        clamped = clamped or not (c2 and r2)
    return f"{index_to_col(start_col)}{start_row}:{index_to_col(end_col)}{end_row}", clamped


# ---------------------------------------------------------------------------
# the sheet
# ---------------------------------------------------------------------------

class A1MetricsSheet:
    """Typed, cached, read-only view of the A1 Metrics Dashboard spreadsheet.

    Parameters
    ----------
    spreadsheet_id, quota_project : from utils.config.Config (see from_config()).
    cache_ttl_s : seconds a fetched range / tab list stays valid in-process.
    session : requests.Session-like object with .get(url, params=, headers=, timeout=);
              injectable for tests. Default: a real requests.Session.
    credentials : google.auth credentials-like object (.token, .expired/.valid,
              .refresh(request)); injectable for tests. Default: google.auth.default()
              with the spreadsheets.readonly scope, resolved lazily on first request
              (raises SheetsUnavailable when ADC is missing).
    clock : monotonic time source (tests inject a fake).
    """

    def __init__(self, spreadsheet_id: str, quota_project: Optional[str] = None,
                 cache_ttl_s: int = 300, session=None, credentials=None,
                 clock: Callable[[], float] = time.monotonic, timeout: int = 30):
        if not spreadsheet_id:
            raise ValueError("spreadsheet_id is required")
        self.spreadsheet_id = spreadsheet_id
        self.quota_project = quota_project or None
        self.cache_ttl_s = max(0, int(cache_ttl_s))
        self._session = session
        self._credentials = credentials
        self._clock = clock
        self.timeout = timeout
        self._cache: Dict[Tuple[str, ...], Tuple[float, Any]] = {}
        self._lock = threading.RLock()
        self.tabs = {
            "volume_pnl": TAB_VOLUME_PNL,
            "weekly_pnl": TAB_WEEKLY_PNL,
            "nonclient_pnl": TAB_NONCLIENT_PNL,
            "financing_fees": TAB_FINANCING_FEES,
            "trades": TAB_COUNTERPARTY_TRADES,
        }

    @classmethod
    def from_config(cls, cfg=None, **kwargs) -> "A1MetricsSheet":
        if cfg is None:
            from utils.config import Config
            cfg = Config()
        return cls(cfg.A1_METRICS_SHEET_ID, cfg.GSHEETS_QUOTA_PROJECT,
                   cache_ttl_s=cfg.GSHEETS_CACHE_TTL_S, **kwargs)

    @property
    def url(self) -> str:
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/edit"

    # ------------------------------------------------------------------ auth/http

    def _get_credentials(self):
        if self._credentials is None:
            try:
                import google.auth
                from google.auth.exceptions import DefaultCredentialsError
            except ImportError as e:  # pragma: no cover - google-auth is a hard dep of the project
                raise SheetsUnavailable(f"google-auth is not installed: {e}") from e
            try:
                creds, _ = google.auth.default(scopes=[SHEETS_SCOPE])
            except DefaultCredentialsError as e:
                raise SheetsUnavailable(
                    f"No Application Default Credentials for Google Sheets ({e}). Run: {ADC_LOGIN_CMD}"
                ) from e
            self._credentials = creds
        return self._credentials

    def _token(self, force_refresh: bool = False) -> str:
        creds = self._get_credentials()
        needs = force_refresh or not getattr(creds, "token", None) or getattr(creds, "expired", False)
        if needs:
            from google.auth.transport.requests import Request
            try:
                creds.refresh(Request())
            except Exception as e:  # noqa: BLE001 - RefreshError, network, invalid_grant ...
                raise SheetsUnavailable(
                    f"Could not refresh Google credentials for the Sheets API ({type(e).__name__}: {e}). "
                    f"Re-run: {ADC_LOGIN_CMD}"
                ) from e
        return creds.token

    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET {SHEETS_API}/{spreadsheet_id}{path}; JSON dict. Retries once on 401 after a
        token refresh; maps 403/404/429 to SheetsAccessError with operator-readable text."""
        url = f"{SHEETS_API}/{self.spreadsheet_id}{path}"
        resp = None
        for attempt in (0, 1):
            headers = {"Authorization": f"Bearer {self._token(force_refresh=attempt == 1)}"}
            if self.quota_project:
                headers["x-goog-user-project"] = self.quota_project
            resp = self._http().get(url, params=params or {}, headers=headers, timeout=self.timeout)
            if resp.status_code != 401 or attempt == 1:
                break
        status = resp.status_code
        if 200 <= status < 300:
            try:
                return resp.json() or {}
            except ValueError as e:
                raise SheetsAccessError(f"Sheets API returned non-JSON for {path}: {e}", status) from e
        detail = self._error_detail(resp)
        if status == 403:
            raise SheetsAccessError(
                "Google Sheets API returned 403 (permission denied) for the A1 Metrics Dashboard. "
                "Either the Application Default Credentials lack the spreadsheets.readonly scope - "
                f"re-run `{ADC_LOGIN_CMD}` - or the account does not have view access to the sheet"
                + (f" (quota project {self.quota_project})" if self.quota_project else "")
                + (f". API detail: {detail}" if detail else "."),
                status,
            )
        if status == 404:
            raise SheetsAccessError(
                f"Google Sheets API returned 404: spreadsheet {self.spreadsheet_id} was not found or is not "
                "shared with the current account. Ask the sheet owner to share the A1 Metrics Dashboard "
                "(view access is enough)" + (f". API detail: {detail}" if detail else "."),
                status,
            )
        if status == 429:
            raise SheetsAccessError(
                "Google Sheets API rate limit hit (429). Retry in a minute; cached ranges keep working"
                + (f". API detail: {detail}" if detail else "."),
                status,
            )
        if status == 400:
            raise SheetsAccessError(f"Google Sheets API rejected the request (400): {detail or 'bad range/tab?'}", status)
        raise SheetsAccessError(f"Google Sheets API error {status} for {path}: {detail}", status)

    @staticmethod
    def _error_detail(resp) -> str:
        try:
            j = resp.json()
            return str(j.get("error", {}).get("message") or "")[:300]
        except Exception:  # noqa: BLE001
            try:
                return (resp.text or "")[:300]
            except Exception:  # noqa: BLE001
                return ""

    # ------------------------------------------------------------------ cache

    def _cached(self, key: Tuple[str, ...], build: Callable[[], Any]) -> Any:
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and hit[0] > now:
                return hit[1]
        value = build()
        with self._lock:
            self._cache[key] = (now + self.cache_ttl_s, value)
        return value

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------ raw access

    def list_tabs(self) -> List[Dict[str, Any]]:
        """[{title, sheet_id, rows, cols}] in sheet order (cached)."""
        def build():
            j = self._request("", {"fields": "properties.title,sheets.properties"})
            self._title = j.get("properties", {}).get("title")
            out = []
            for s in j.get("sheets", []):
                p = s.get("properties", {})
                g = p.get("gridProperties", {}) or {}
                out.append({"title": p.get("title"), "sheet_id": p.get("sheetId"),
                            "rows": g.get("rowCount"), "cols": g.get("columnCount")})
            return out
        return self._cached(("tabs",), build)

    @property
    def title(self) -> Optional[str]:
        self.list_tabs()
        return getattr(self, "_title", None)

    def get_range(self, tab: str, a1: str) -> List[List[Any]]:
        """Raw values (UNFORMATTED_VALUE; dates as formatted strings) for `tab`!`a1`.

        Rows are ragged exactly as the API returns them (trailing blanks dropped, empty
        rows as []). Cached per (tab, a1) for cache_ttl_s.
        """
        if not tab:
            raise ValueError("tab is required")
        rng = f"'{tab.replace(chr(39), chr(39) * 2)}'!{a1}"

        def build():
            j = self._request(f"/values/{quote(rng, safe='')}",
                              {"valueRenderOption": "UNFORMATTED_VALUE",
                               "dateTimeRenderOption": "FORMATTED_STRING"})
            return j.get("values", []) or []
        return self._cached(("range", tab, a1.upper()), build)

    def data_as_of(self, tab: str, rows: Optional[Sequence[Sequence[Any]]] = None) -> Optional[str]:
        """The tab's "Data as of" date as 'YYYY-MM-DD', or None when the tab has no such
        cell. Looks at the first 5 rows; accepts 'Data as of | 2026-09-11' (two cells) and
        'Data as of 12/31/25' (one cell)."""
        if rows is None:
            rows = self.get_range(tab, "A1:F5")
        for row in list(rows)[:5]:
            for c, v in enumerate(row):
                if isinstance(v, str) and _norm(v).startswith("DATA AS OF"):
                    tail = re.sub(r"(?i)^\s*data\s+as\s+of\s*:?\s*", "", v).strip()
                    ts = parse_sheet_date(tail) if tail else None
                    if ts is None:
                        for nxt in row[c + 1:]:
                            ts = parse_sheet_date(nxt)
                            if ts is not None:
                                break
                    return ts.strftime("%Y-%m-%d") if ts is not None else None
        return None

    def dashboard_as_of(self) -> Optional[str]:
        """As-of date of the main 'Volume & PNL' tab (the dashboard's reference date)."""
        try:
            return self.data_as_of(self.tabs["volume_pnl"])
        except SheetsAccessError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.debug("dashboard_as_of failed: %s", e)
            return None

    # ------------------------------------------------------------------ typed accessors

    def _year_tab(self, base: str, year: Optional[int]) -> Tuple[str, int]:
        """('Volume & PNL', 2026) for the current year, ('2025 Volume & PNL', 2025) otherwise."""
        as_of = None
        try:
            as_of = self.data_as_of(base)
        except SheetsAccessError:
            raise
        except Exception:  # noqa: BLE001
            pass
        current_year = int(as_of[:4]) if as_of else date.today().year
        if year is None or int(year) == current_year:
            return base, current_year
        return f"{int(year)} {base}", int(year)

    def monthly_volume_pnl(self, year: Optional[int] = None) -> pd.DataFrame:
        """Long-format monthly volume / PnL / take rate.

        Columns: entity (HOLD | A1 | TOTAL), year, month (1-12; None on the YTD row),
        is_ytd, volume_usd, pnl_usd, take_rate_bps, cum_volume_usd, cum_pnl_usd,
        target_pnl_usd, pct_of_target (TOTAL group only). Months with no data yet are
        dropped (all of volume / pnl blank or zero). Take rate is taken from the sheet
        (bps) and back-filled as pnl / volume * 1e4 when the cell is blank.
        `df.attrs['tab']` / `df.attrs['data_as_of']` carry provenance.
        """
        tab, yr = self._year_tab(self.tabs["volume_pnl"], year)
        rows = self.get_range(tab, RANGE_VOLUME_PNL)
        df = parse_monthly_volume_pnl(rows, year=yr)
        df.attrs["tab"] = tab
        df.attrs["data_as_of"] = self.data_as_of(tab, rows)
        return df

    def weekly_pnl(self, include_future: bool = False) -> pd.DataFrame:
        """Weekly PnL: week, start_date, end_date, hold_pnl, a1_realized, a1_unrealized,
        a1_total, total, is_total (the sheet's 'Total' row, week=None). Weeks whose start
        date is after the tab's as-of date (or with no PnL at all) are dropped unless
        include_future=True."""
        tab = self.tabs["weekly_pnl"]
        rows = self.get_range(tab, RANGE_WEEKLY_PNL)
        as_of = self.data_as_of(tab, rows)
        df = parse_weekly_pnl(rows, as_of=as_of, include_future=include_future)
        df.attrs["tab"] = tab
        df.attrs["data_as_of"] = as_of
        return df

    def financing_fees(self, year: Optional[int] = None) -> pd.DataFrame:
        """Monthly HOLD financing fees: month, hold_financing_fees_usd, hold_delta_sales_usd,
        total_usd (months with no data dropped). attrs: tab, data_as_of (None - the tab has
        no as-of cell; callers fall back to dashboard_as_of())."""
        tab, yr = self._year_tab(self.tabs["financing_fees"], year) if year is not None else (self.tabs["financing_fees"], None)
        rows = self.get_range(tab, RANGE_FINANCING_FEES)
        df = parse_financing_fees(rows, block="Month")
        df.attrs["tab"] = tab
        df.attrs["data_as_of"] = self.data_as_of(tab, rows)
        df.attrs["year"] = yr
        return df

    def financing_fees_weekly(self) -> pd.DataFrame:
        """Weekly block of the Financing Fees tab: week, hold_financing_fees_usd,
        hold_delta_sales_usd, total_usd (weeks with no data dropped)."""
        tab = self.tabs["financing_fees"]
        rows = self.get_range(tab, RANGE_FINANCING_FEES)
        df = parse_financing_fees(rows, block="Week")
        df.attrs["tab"] = tab
        df.attrs["data_as_of"] = self.data_as_of(tab, rows)
        return df

    def nonclient_pnl(self) -> pd.DataFrame:
        """Whatever table the 'Nonclient PNL' tab holds (generic header + numeric rows), or
        an empty frame when the tab is not populated. attrs: tab, title, link (a URL found
        on the tab, e.g. the external sheet it points to), data_as_of."""
        tab = self.tabs["nonclient_pnl"]
        rows = self.get_range(tab, RANGE_NONCLIENT_PNL)
        df = parse_generic_table(rows)
        df.attrs["tab"] = tab
        df.attrs["data_as_of"] = self.data_as_of(tab, rows)
        df.attrs["title"] = next((str(v).strip() for row in rows[:3] for v in row
                                  if isinstance(v, str) and v.strip() and not v.startswith("http")), None)
        df.attrs["link"] = next((v for row in rows[:10] for v in row
                                 if isinstance(v, str) and v.startswith("http")), None)
        return df

    def counterparty_trades(self, days: Optional[int] = None) -> pd.DataFrame:
        """Typed trade blotter from the `db` tab.

        Columns: date (datetime64), counterparty, side (BUY/SELL, upper), symbol, buy_qty,
        buy_asset, sell_qty, sell_asset, price, pnl_usd, currency, bps, month,
        notional_usd (stable/USD leg quantity; else pnl / bps * 1e4). Rows without a date
        are dropped. `days` keeps trades in the last N days **relative to the latest trade
        in the blotter** (the blotter is not kept current - attrs['coverage_start'] /
        ['coverage_end'] report the window actually available).
        """
        tab = self.tabs["trades"]
        rows = self.get_range(tab, RANGE_TRADES)
        df = parse_trades(rows)
        start = df["date"].min() if len(df) else None
        end = df["date"].max() if len(df) else None
        if days is not None and len(df):
            cutoff = end.normalize() - timedelta(days=int(days) - 1)
            df = df[df["date"] >= cutoff].reset_index(drop=True)
        df.attrs["tab"] = tab
        df.attrs["coverage_start"] = start.strftime("%Y-%m-%d") if start is not None else None
        df.attrs["coverage_end"] = end.strftime("%Y-%m-%d") if end is not None else None
        df.attrs["data_as_of"] = df.attrs["coverage_end"]
        return df


# ---------------------------------------------------------------------------
# pure parsers (operate on the raw `values` payload; unit-tested offline)
# ---------------------------------------------------------------------------

_MONTHLY_FIELDS = (
    ("volume_usd", ("VOLUME",)),
    ("pnl_usd", ("PNL", "PNL (USD)", "P&L")),
    ("take_rate_bps", ("TAKE RATE", "TAKE RATE (BPS)")),
    ("cum_volume_usd", ("CUMULATIVE VOLUME",)),
    ("cum_pnl_usd", ("CUMULATIVE PNL", "CUMULATIVE P&L")),
    ("target_pnl_usd", ("TARGET PNL", "TARGET")),
    ("pct_of_target", ("% OF TARGET", "PCT OF TARGET")),
)


def parse_monthly_volume_pnl(rows: Sequence[Sequence[Any]], year: Optional[int] = None) -> pd.DataFrame:
    cols = ["entity", "year", "month", "is_ytd", "volume_usd", "pnl_usd", "take_rate_bps",
            "cum_volume_usd", "cum_pnl_usd", "target_pnl_usd", "pct_of_target"]
    h = _find_header_row(rows, "Month", min_count=1)
    if h is None:
        return pd.DataFrame(columns=cols)
    out = []
    for ordinal, (kc, fields) in enumerate(_blocks(rows, h, "Month")):
        entity = _entity_from_label(_group_label(rows, h, kc), ordinal)
        colmap = {name: _pick(fields, *cands) for name, cands in _MONTHLY_FIELDS}
        for r in range(h + 1, len(rows)):
            label = _cell(rows, r, kc)
            if _is_blank(label):
                continue
            month = to_int(label)
            is_ytd = False
            if month is None:
                text = _norm(label)
                if "YTD" in text or text.startswith("TOTAL"):
                    is_ytd = True
                else:
                    continue
            elif not 1 <= month <= 12:
                continue
            rec = {"entity": entity, "year": year, "month": None if is_ytd else month, "is_ytd": is_ytd}
            for name, col in colmap.items():
                rec[name] = to_float(_cell(rows, r, col)) if col is not None else float("nan")
            v, p = rec["volume_usd"], rec["pnl_usd"]
            if (v != v or v == 0) and (p != p or p == 0):
                continue  # month not populated yet
            if rec["take_rate_bps"] != rec["take_rate_bps"] and v == v and v and p == p:
                rec["take_rate_bps"] = p / v * 1e4
            out.append(rec)
    df = pd.DataFrame(out, columns=cols)
    if len(df):
        df["is_ytd"] = df["is_ytd"].astype(bool)
        df["month"] = df["month"].astype("Int64")
    return df


def parse_weekly_pnl(rows: Sequence[Sequence[Any]], as_of: Optional[str] = None,
                     include_future: bool = False) -> pd.DataFrame:
    cols = ["week", "start_date", "end_date", "hold_pnl", "a1_realized", "a1_unrealized",
            "a1_total", "total", "is_total"]
    h = _find_header_row(rows, "Week", min_count=1)
    if h is None:
        return pd.DataFrame(columns=cols)
    blocks = _blocks(rows, h, "Week")
    by_entity: Dict[str, Tuple[int, Dict[str, int]]] = {}
    for ordinal, (kc, fields) in enumerate(blocks):
        by_entity[_entity_from_label(_group_label(rows, h, kc), ordinal)] = (kc, fields)
    hold = by_entity.get("HOLD")
    a1 = by_entity.get("A1")
    tot = by_entity.get("TOTAL")
    key_block = tot or hold or a1
    if key_block is None:
        return pd.DataFrame(columns=cols)
    kc = key_block[0]
    as_of_ts = parse_sheet_date(as_of) if as_of else None

    def val(block, r, *names):
        if block is None:
            return float("nan")
        col = _pick(block[1], *names)
        return to_float(_cell(rows, r, col)) if col is not None else float("nan")

    out = []
    prev_week = 0
    for r in range(h + 1, len(rows)):
        label = _cell(rows, r, kc)
        if _is_blank(label):
            continue
        week = to_int(label)
        is_total = False
        if week is None:
            if _norm(label).startswith("TOTAL"):
                is_total = True
            else:
                continue
        rec = {
            "hold_pnl": val(hold, r, "PNL", "P&L"),
            "a1_realized": val(a1, r, "REALIZED PNL", "REALISED PNL", "REALIZED"),
            "a1_unrealized": val(a1, r, "UNREALIZED PNL APPROXIMATION", "UNREALIZED PNL", "UNREALISED PNL", "UNREALIZED"),
            "a1_total": val(a1, r, "TOTAL"),
            "total": val(tot, r, "PNL", "TOTAL", "P&L"),
            "is_total": is_total,
        }
        if rec["a1_total"] != rec["a1_total"]:
            re_, un = rec["a1_realized"], rec["a1_unrealized"]
            if re_ == re_ or un == un:
                rec["a1_total"] = (0 if re_ != re_ else re_) + (0 if un != un else un)
        if rec["total"] != rec["total"]:
            hp, at = rec["hold_pnl"], rec["a1_total"]
            if hp == hp or at == at:
                rec["total"] = (0 if hp != hp else hp) + (0 if at != at else at)
        if is_total:
            rec.update(week=None, start_date=None, end_date=None)
            out.append(rec)
            continue
        if week <= prev_week:          # the desk's sheet repeats week numbers at year end
            week = prev_week + 1
        prev_week = week
        start = parse_sheet_date(_cell(rows, r, _pick(tot[1], "START DATE", "START"))) if tot and _pick(tot[1], "START DATE", "START") is not None else None
        end = parse_sheet_date(_cell(rows, r, _pick(tot[1], "END DATE", "END"))) if tot and _pick(tot[1], "END DATE", "END") is not None else None
        rec.update(week=week, start_date=start, end_date=end)
        if not include_future:
            all_zero = all((rec[k] != rec[k]) or rec[k] == 0 for k in ("hold_pnl", "a1_realized", "a1_unrealized", "a1_total", "total"))
            if as_of_ts is not None and start is not None and start > as_of_ts:
                continue
            if all_zero and (as_of_ts is None or start is None):
                continue
            if all_zero and start is not None and as_of_ts is not None and end is not None and end >= as_of_ts and start > as_of_ts - timedelta(days=1):
                continue
        out.append(rec)
    df = pd.DataFrame(out, columns=cols)
    if len(df):
        df["week"] = df["week"].astype("Int64")
        df["is_total"] = df["is_total"].astype(bool)
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def parse_financing_fees(rows: Sequence[Sequence[Any]], block: str = "Month") -> pd.DataFrame:
    key = "month" if _norm(block) == "MONTH" else "week"
    cols = [key, "hold_financing_fees_usd", "hold_delta_sales_usd", "total_usd"]
    h = _find_header_row(rows, block, min_count=1)
    if h is None:
        return pd.DataFrame(columns=cols)
    blocks = _blocks(rows, h, block)
    if not blocks:
        return pd.DataFrame(columns=cols)
    kc, fields = blocks[0]
    c_fees = _pick(fields, "HOLD FINANCING FEES", "FINANCING FEES")
    c_delta = _pick(fields, "HOLD DELTA SALES", "DELTA SALES")
    c_total = _pick(fields, "TOTAL")
    out = []
    for r in range(h + 1, len(rows)):
        n = to_int(_cell(rows, r, kc))
        if n is None:
            continue
        if key == "month" and not 1 <= n <= 12:
            continue
        fees = to_float(_cell(rows, r, c_fees)) if c_fees is not None else float("nan")
        delta = to_float(_cell(rows, r, c_delta)) if c_delta is not None else float("nan")
        total = to_float(_cell(rows, r, c_total)) if c_total is not None else float("nan")
        if total != total:
            total = (0 if fees != fees else fees) + (0 if delta != delta else delta)
        if all(x != x or x == 0 for x in (fees, delta, total)):
            continue
        out.append({key: n, "hold_financing_fees_usd": 0.0 if fees != fees else fees,
                    "hold_delta_sales_usd": 0.0 if delta != delta else delta, "total_usd": total})
    return pd.DataFrame(out, columns=cols)


def parse_generic_table(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Best-effort: the first row with >= 2 text cells followed by a row with a number
    becomes the header; subsequent rows until a blank row become data. Else empty."""
    rows = list(rows)
    for i, row in enumerate(rows):
        texts = [v for v in row if isinstance(v, str) and v.strip() and not v.startswith("http")]
        if len(texts) < 2 or i + 1 >= len(rows):
            continue
        nxt = rows[i + 1]
        if not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nxt):
            continue
        header = [(_norm(v).lower().replace(" ", "_") or f"col{c}") for c, v in enumerate(row)]
        data = []
        for r in rows[i + 1:]:
            if not any(not _is_blank(v) for v in r):
                break
            data.append([r[c] if c < len(r) else None for c in range(len(header))])
        return pd.DataFrame(data, columns=header)
    return pd.DataFrame()


_TRADE_COLUMNS = (
    ("date", ("DATE (UTC)", "DATE")),
    ("counterparty", ("COUNTERPARTY",)),
    ("side", ("SIDE",)),
    ("symbol", ("SYMBOL",)),
    ("buy_qty", ("BUY QTY",)),
    ("buy_asset", ("BUY ASSET",)),
    ("sell_qty", ("SELL QTY",)),
    ("sell_asset", ("SELL ASSET",)),
    ("price", ("PRICE",)),
    ("pnl_usd", ("PNL", "P&L")),
    ("currency", ("CURRENCY",)),
    ("bps", ("BPS", "A1 SPREAD (BPS)")),
    ("month", ("MONTH",)),
)
_TRADE_NUMERIC = ("buy_qty", "sell_qty", "price", "pnl_usd", "bps")


def parse_trades(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    cols = [c for c, _ in _TRADE_COLUMNS] + ["notional_usd"]
    h = None
    for i, row in enumerate(rows[:10]):
        names = {_norm(v) for v in row}
        if "COUNTERPARTY" in names and ("PNL" in names or "P&L" in names):
            h = i
            break
    if h is None:
        return pd.DataFrame(columns=cols)
    fields = {}
    for c, v in enumerate(rows[h]):
        name = _norm(v)
        if name and name not in fields:
            fields[name] = c
    colmap = {name: _pick(fields, *cands) for name, cands in _TRADE_COLUMNS}
    recs = []
    for r in range(h + 1, len(rows)):
        row = rows[r]
        if not row or all(_is_blank(v) for v in row):
            continue
        rec = {}
        for name, col in colmap.items():
            rec[name] = _cell(rows, r, col) if col is not None else None
        recs.append(rec)
    df = pd.DataFrame(recs, columns=[c for c, _ in _TRADE_COLUMNS])
    if not len(df):
        df["notional_usd"] = pd.Series(dtype=float)
        return df
    df["date"] = df["date"].map(parse_sheet_date)
    df = df[df["date"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in _TRADE_NUMERIC:
        df[c] = df[c].map(to_float).astype(float)
    df["month"] = df["month"].map(to_int).astype("Int64")
    for c in ("counterparty", "symbol", "buy_asset", "sell_asset", "currency"):
        df[c] = df[c].map(lambda v: None if _is_blank(v) else str(v).strip())
    df["side"] = df["side"].map(lambda v: None if _is_blank(v) else str(v).strip().upper())

    def notional(row):
        ba = (row["buy_asset"] or "").upper()
        sa = (row["sell_asset"] or "").upper()
        if ba in STABLE_ASSETS and row["buy_qty"] == row["buy_qty"]:
            return row["buy_qty"]
        if sa in STABLE_ASSETS and row["sell_qty"] == row["sell_qty"]:
            return row["sell_qty"]
        if row["bps"] == row["bps"] and row["bps"] and row["pnl_usd"] == row["pnl_usd"]:
            return abs(row["pnl_usd"] / row["bps"] * 1e4)
        return float("nan")
    df["notional_usd"] = df.apply(notional, axis=1).astype(float)
    return df.sort_values("date").reset_index(drop=True)


__all__ = [
    "A1MetricsSheet",
    "ADC_LOGIN_CMD",
    "SHEETS_SCOPE",
    "SheetsAccessError",
    "SheetsUnavailable",
    "clamp_a1_range",
    "parse_a1_range",
    "parse_financing_fees",
    "parse_generic_table",
    "parse_monthly_volume_pnl",
    "parse_sheet_date",
    "parse_trades",
    "parse_weekly_pnl",
    "to_float",
]
