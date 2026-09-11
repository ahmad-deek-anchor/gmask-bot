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
``A1 database`` (helper tab; only columns R:X are read, tab has no as-of cell)
    row 1  ``T:X`` headers ``Client Flow PNL | Non Client Flow PNL | Change in Total PNL |
           Change in Client Flow PNL | Change in non Client Flow PNL`` (R1:S9 hold an
           unrelated asset / volume list). Rows 2-224: undated history (R blank).
           From row 225 (2025-07-24) on, R = snapshot datetime string
           ``YYYY-MM-DD HH:MM:SS`` (~22:10 UTC; a Google serial number is accepted too),
           S = cumulative YTD total realised PnL (= T + U), T / U = cumulative YTD
           client-flow / non-client-flow realised PnL (reset to 0 on Jan 1), V / W / X =
           that day's realised total / client-flow / non-client-flow PnL. Some dates
           are missing (no row), one date has two snapshots, and the series carries
           offsetting artefact pairs (e.g. +$12.0M / -$12.0M on 2025-08-14/15) that the
           desk's weekly totals include net - see ``parse_client_flow``.
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
TAB_A1_DATABASE = "A1 database"     # daily realised PnL split client flow / non-client flow (cols R:X)

# generous ranges: the tabs are ~20 wide and < 60 rows; the blotter is read whole
RANGE_VOLUME_PNL = "A1:T40"
RANGE_WEEKLY_PNL = "A1:N80"
RANGE_FINANCING_FEES = "A1:L70"
RANGE_NONCLIENT_PNL = "A1:Z40"
RANGE_TRADES = "A:M"
RANGE_A1_DATABASE = "R:X"

# artefact detection in the A1 database daily series: a day whose |realised total| is at
# least ARTIFACT_MIN_ABS and whose neighbouring row offsets it to within ARTIFACT_OFFSET_TOL
# (relative) is an erroneous booking + reversal pair, not real daily PnL.
ARTIFACT_MIN_ABS = 1_000_000.0
ARTIFACT_OFFSET_TOL = 0.05

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
            "a1_database": TAB_A1_DATABASE,
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


    # ------------------------------------------------------------------ client vs non-client flow

    def client_flow_daily(self, start=None, end=None) -> pd.DataFrame:
        """Daily A1 realised PnL split into client flow and non-client (proprietary) flow,
        from ``'A1 database'!R:X`` (one cached read).

        Columns: date (datetime64, normalised), realized_total (V), realized_client (W),
        realized_nonclient (X), cum_total_ytd (S), cum_client_ytd (T), cum_nonclient_ytd
        (U), row_number (1-based sheet row), flagged_artifact (bool), artifact_pair (row
        number of the first leg of the offsetting pair, <NA> otherwise). Undated rows are
        dropped; rows are sorted by date then sheet row (one date can carry two
        snapshots). `start` / `end` (inclusive, anything parse_sheet_date accepts) filter
        by date. Artefact rows are **kept** with flagged_artifact=True so callers can
        exclude and disclose them. attrs: tab, range, data_as_of (the dashboard's - this
        tab has no as-of cell), first_row_date, last_row_date, n_rows_total.
        """
        tab = self.tabs["a1_database"]
        rows = self.get_range(tab, RANGE_A1_DATABASE)
        full = parse_client_flow(rows)
        df = full
        s_ts = parse_sheet_date(start) if start is not None else None
        e_ts = parse_sheet_date(end) if end is not None else None
        if s_ts is not None:
            df = df[df["date"] >= s_ts.normalize()]
        if e_ts is not None:
            df = df[df["date"] <= e_ts.normalize()]
        df = df.reset_index(drop=True)
        df.attrs["tab"] = tab
        df.attrs["range"] = RANGE_A1_DATABASE
        df.attrs["data_as_of"] = self.dashboard_as_of()
        df.attrs["first_row_date"] = full["date"].min().strftime("%Y-%m-%d") if len(full) else None
        df.attrs["last_row_date"] = full["date"].max().strftime("%Y-%m-%d") if len(full) else None
        df.attrs["n_rows_total"] = int(len(full))
        return df

    def client_flow_window(self, start, end, exclude_artifacts: bool = True) -> Dict[str, Any]:
        """Sums of the client / non-client realised PnL split over [start, end] with the
        bookkeeping a careful analyst would do by hand.

        Artefact policy (verified against the desk's 'Weekly PNL' tab, which includes the
        offsetting pairs net): when **both** legs of a flagged pair fall inside the window
        they are kept (they net to the residual real PnL of those two days) and disclosed;
        a leg whose partner lies outside the window is excluded when exclude_artifacts is
        True (it would otherwise swing the total by millions) and disclosed.

        Returns a dict: start, end, tab, range, data_as_of, first_row_date, last_row_date,
        rows (DataFrame of the rows used, artefact legs excluded), excluded (DataFrame),
        n_rows, n_days, sums {realized_total, realized_client, realized_nonclient}, sums_all_rows
        (same, nothing excluded), client_share / nonclient_share (fractions of total, None
        when total is 0), artifact_rows (list of dicts with an `action`), artifact_net
        (net of kept pairs), missing_days / missing_weekdays (calendar days in the window
        up to the last populated row with no sheet row, 'YYYY-MM-DD (Dow)'), not_populated
        (days in the window after the last populated row), incomplete_rows (rows lacking
        the W / X split), duplicate_dates, weekly_check (None or {week, start_date,
        end_date, a1_realized, a1_unrealized, difference} when the window equals a
        dashboard week), ytd_check (None or the sheet's cumulative S/T/U values at the
        last row of the window with differences vs the summed rows, for windows that
        start on Jan 1).
        """
        s_ts = parse_sheet_date(start)
        e_ts = parse_sheet_date(end)
        if s_ts is None or e_ts is None:
            raise ValueError(f"start / end must be dates, got {start!r} / {end!r}")
        s_ts, e_ts = s_ts.normalize(), e_ts.normalize()
        if e_ts < s_ts:
            s_ts, e_ts = e_ts, s_ts
        df = self.client_flow_daily(s_ts, e_ts)
        last_row = df.attrs.get("last_row_date")
        last_ts = pd.Timestamp(last_row) if last_row else None

        artifact_rows: List[Dict[str, Any]] = []
        drop_idx = []
        art_net = 0.0
        if len(df) and df["flagged_artifact"].any():
            flagged = df[df["flagged_artifact"]]
            in_window = set(int(r) for r in flagged["row_number"].tolist())
            for pair_id, legs in flagged.groupby("artifact_pair"):
                full_pair = client_flow_pair_complete(legs, in_window)
                for idx, leg in legs.iterrows():
                    if full_pair:
                        action = "kept - both legs in window, they net out"
                        art_net += float(leg["realized_total"]) if leg["realized_total"] == leg["realized_total"] else 0.0
                    elif exclude_artifacts:
                        action = "excluded - offsetting leg is outside the window"
                        drop_idx.append(idx)
                    else:
                        action = "kept - offsetting leg is outside the window (exclude_artifacts=False)"
                    artifact_rows.append({
                        "row_number": int(leg["row_number"]), "date": leg["date"].strftime("%Y-%m-%d"),
                        "realized_total": float(leg["realized_total"]), "realized_client": float(leg["realized_client"]),
                        "realized_nonclient": float(leg["realized_nonclient"]), "pair": int(pair_id), "action": action,
                    })
        excluded = df.loc[drop_idx] if drop_idx else df.iloc[0:0]
        used = df.drop(index=drop_idx) if drop_idx else df

        def sums(frame: pd.DataFrame) -> Dict[str, float]:
            return {c: float(frame[c].sum(skipna=True)) if len(frame) else 0.0
                    for c in ("realized_total", "realized_client", "realized_nonclient")}
        total = sums(used)
        tot = total["realized_total"]
        share_c = total["realized_client"] / tot if tot else None
        share_n = total["realized_nonclient"] / tot if tot else None

        # coverage
        upto = min(e_ts, last_ts) if last_ts is not None else e_ts
        missing: List[str] = []
        missing_wd: List[str] = []
        if upto >= s_ts:
            have = set(pd.DatetimeIndex(df["date"]).normalize()) if len(df) else set()
            for d in pd.date_range(s_ts, upto):
                if d not in have:
                    label = f"{d.strftime('%Y-%m-%d')} ({d.strftime('%a')})"
                    missing.append(label)
                    if d.weekday() < 5:
                        missing_wd.append(label)
        not_populated = [d.strftime("%Y-%m-%d") for d in pd.date_range(max(s_ts, upto + timedelta(days=1)), e_ts)] \
            if last_ts is not None and e_ts > last_ts else []
        incomplete = used[used["realized_client"].isna() | used["realized_nonclient"].isna()] if len(used) else used
        dup = df["date"][df["date"].duplicated()].dt.strftime("%Y-%m-%d").unique().tolist() if len(df) else []

        weekly_check = None
        try:
            wk = self.weekly_pnl(include_future=True)
            hit = wk[(~wk["is_total"]) & (wk["start_date"] == s_ts) & (wk["end_date"] == e_ts)]
            if len(hit):
                w = hit.iloc[0]
                a1r = float(w["a1_realized"]) if w["a1_realized"] == w["a1_realized"] else None
                weekly_check = {
                    "week": int(w["week"]), "start_date": s_ts.strftime("%Y-%m-%d"), "end_date": e_ts.strftime("%Y-%m-%d"),
                    "a1_realized": a1r,
                    "a1_unrealized": float(w["a1_unrealized"]) if w["a1_unrealized"] == w["a1_unrealized"] else None,
                    "difference": (tot - a1r) if a1r is not None else None,
                }
        except SheetsAccessError:
            raise
        except Exception as e:  # noqa: BLE001 - the cross-check is best effort
            logger.debug("weekly cross-check unavailable: %s", e)

        ytd_check = None
        if len(used) and s_ts.month == 1 and s_ts.day == 1 and e_ts.year == s_ts.year:
            last = used.sort_values(["date", "row_number"]).iloc[-1]
            if last["cum_total_ytd"] == last["cum_total_ytd"]:
                cc = float(last["cum_client_ytd"]) if last["cum_client_ytd"] == last["cum_client_ytd"] else float("nan")
                cn = float(last["cum_nonclient_ytd"]) if last["cum_nonclient_ytd"] == last["cum_nonclient_ytd"] else float("nan")
                ytd_check = {
                    "date": last["date"].strftime("%Y-%m-%d"), "row_number": int(last["row_number"]),
                    "cum_total": float(last["cum_total_ytd"]), "cum_client": cc, "cum_nonclient": cn,
                    "diff_total": tot - float(last["cum_total_ytd"]),
                    "diff_client": total["realized_client"] - cc, "diff_nonclient": total["realized_nonclient"] - cn,
                }

        return {
            "start": s_ts.strftime("%Y-%m-%d"), "end": e_ts.strftime("%Y-%m-%d"),
            "ytd_check": ytd_check,
            "tab": df.attrs.get("tab"), "range": df.attrs.get("range"), "data_as_of": df.attrs.get("data_as_of"),
            "first_row_date": df.attrs.get("first_row_date"), "last_row_date": last_row,
            "rows": used.reset_index(drop=True), "excluded": excluded.reset_index(drop=True),
            "n_rows": int(len(used)), "n_days": int(used["date"].nunique()) if len(used) else 0,
            "sums": total, "sums_all_rows": sums(df),
            "client_share": share_c, "nonclient_share": share_n,
            "artifact_rows": artifact_rows, "artifact_net": art_net if artifact_rows else 0.0,
            "missing_days": missing, "missing_weekdays": missing_wd, "not_populated": not_populated,
            "incomplete_rows": [{"row_number": int(r["row_number"]), "date": r["date"].strftime("%Y-%m-%d")}
                                for _, r in incomplete.iterrows()],
            "duplicate_dates": dup,
            "weekly_check": weekly_check,
        }


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


_CLIENT_FLOW_COLS = ["date", "realized_total", "realized_client", "realized_nonclient",
                     "cum_total_ytd", "cum_client_ytd", "cum_nonclient_ytd",
                     "row_number", "flagged_artifact", "artifact_pair"]


def parse_client_flow(rows: Sequence[Sequence[Any]], first_row: int = 1,
                      artifact_min_abs: float = ARTIFACT_MIN_ABS,
                      artifact_tol: float = ARTIFACT_OFFSET_TOL) -> pd.DataFrame:
    """``'A1 database'!R:X`` values -> typed daily frame (see A1MetricsSheet.client_flow_daily).

    Column offsets within the payload are fixed (R..X -> 0..6): R date, S cumulative
    total, T cumulative client, U cumulative non-client, V daily total, W daily client,
    X daily non-client. Rows whose R is not a date (header, the asset list, the undated
    history, blanks) are dropped; dates may be strings or Google serial numbers and are
    normalised to midnight. `first_row` is the sheet row of rows[0] (1 for an R:X read).

    Artefact flagging: a row with |V| >= artifact_min_abs whose neighbouring dated row
    (previous or next in sheet order) has the opposite sign and |V_i + V_j| <= artifact_tol
    * |V_i| is an offsetting booking pair; both legs get flagged_artifact=True and
    artifact_pair = the first leg's row number.
    """
    recs = []
    for i, row in enumerate(rows):
        raw_date = row[0] if row else None
        if _is_blank(raw_date):
            continue
        ts = parse_sheet_date(raw_date)
        if ts is None:
            continue
        vals = [to_float(row[c]) if c < len(row) else float("nan") for c in range(1, 7)]
        if all(v != v for v in vals):
            continue
        recs.append({
            "date": ts.normalize(),
            "realized_total": vals[3], "realized_client": vals[4], "realized_nonclient": vals[5],
            "cum_total_ytd": vals[0], "cum_client_ytd": vals[1], "cum_nonclient_ytd": vals[2],
            "row_number": first_row + i,
        })
    df = pd.DataFrame(recs, columns=_CLIENT_FLOW_COLS[:8])
    df["flagged_artifact"] = False
    df["artifact_pair"] = pd.array([None] * len(df), dtype="Int64")
    if not len(df):
        df["date"] = pd.to_datetime(df["date"])
        return df
    df = df.sort_values(["row_number"]).reset_index(drop=True)
    v = df["realized_total"].tolist()
    n = len(v)
    for i in range(n):
        vi = v[i]
        if vi != vi or abs(vi) < artifact_min_abs:
            continue
        for j in (i - 1, i + 1):
            if not 0 <= j < n:
                continue
            vj = v[j]
            if vj != vj or vi * vj >= 0:
                continue
            if abs(vi + vj) <= artifact_tol * abs(vi):
                pair = int(df.at[min(i, j), "row_number"])
                for k in (i, j):
                    df.at[k, "flagged_artifact"] = True
                    if pd.isna(df.at[k, "artifact_pair"]):
                        df.at[k, "artifact_pair"] = pair
                break
    df["date"] = pd.to_datetime(df["date"])
    df["flagged_artifact"] = df["flagged_artifact"].astype(bool)
    return df.sort_values(["date", "row_number"]).reset_index(drop=True)


def client_flow_pair_complete(legs: pd.DataFrame, in_window: set) -> bool:
    """True when a flagged pair has both of its legs inside the window (`in_window` is the
    set of row numbers present). A pair always has exactly two legs; a lone leg means the
    partner row lies outside the requested date range."""
    return len(legs) >= 2 and all(int(r) in in_window for r in legs["row_number"])


__all__ = [
    "A1MetricsSheet",
    "ADC_LOGIN_CMD",
    "ARTIFACT_MIN_ABS",
    "ARTIFACT_OFFSET_TOL",
    "SHEETS_SCOPE",
    "SheetsAccessError",
    "SheetsUnavailable",
    "clamp_a1_range",
    "parse_a1_range",
    "parse_client_flow",
    "parse_financing_fees",
    "parse_generic_table",
    "parse_monthly_volume_pnl",
    "parse_sheet_date",
    "parse_trades",
    "parse_weekly_pnl",
    "to_float",
]
