"""@tool functions over the spot desk's "A1 Metrics Dashboard" Google Sheet (read-only).

Scope: the **spot** business's booked PnL as the desk maintains it by hand
(owner Joao Luis, updated daily): HOLD (client spot trading - commissions on client
trades) vs A1 (A1 Ltd, the principal spot desk) monthly volume / PnL / take rate,
weekly realised / unrealised PnL, HOLD financing fees, the counterparty trade
blotter, and A1's daily realised PnL split into client flow vs non-client (proprietary)
flow ('A1 database' tab, columns R:X). This is a different source from the Haruko mark-to-market PnL of the OTC /
derivatives book in BigQuery (tools/desk_tools.py); the tools say so in every answer.

Every tool returns compact markdown, formats USD with ``$`` and thousands separators,
bps to two decimals, and always prints the tab's "Data as of" date plus the source
line ("A1 Metrics Dashboard sheet maintained by the desk"). Nothing here writes.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from langchain_core.tools import tool

from providers.gsheets import SheetsAccessError, SheetsUnavailable, clamp_a1_range, parse_a1_range, parse_sheet_date
from tools.desk_tools import _isna, _md_table, _num, _usd

logger = logging.getLogger(__name__)

SOURCE_NAME = "A1 Metrics Dashboard (Google Sheet maintained daily by the spot desk)"
MAX_RANGE_ROWS = 200
MAX_RANGE_COLS = 30
MAX_TOP_N = 50
MAX_WEEKS = 60
MAX_MONTHS = 24
MAX_DAILY_ROWS = 45          # per-day table above this many rows collapses to monthly subtotals
BY_CHOICES = ("counterparty", "symbol", "side")
WINDOW_SHORTCUTS = ("last week", "this week", "mtd", "ytd", "last month", "this month", "last N days", "yesterday",
                    "YYYY-MM-DD [YYYY-MM-DD]")

UNAVAILABLE = (
    "Spot desk PnL (A1 Metrics Dashboard sheet) is not available: Application Default "
    "Credentials with the Google Sheets scope are missing. Re-authenticate with "
    "`gcloud auth application-default login` including the spreadsheets.readonly scope. "
    "Market-data and BigQuery desk tools still work."
)


# ---------------------------------------------------------------------------
# access + error handling
# ---------------------------------------------------------------------------

def _get_sheet():
    """A1MetricsSheet or None (tests monkeypatch this)."""
    from providers.factory import get_a1_metrics_sheet
    return get_a1_metrics_sheet()


def _guarded(name: str, body: Callable) -> str:
    sheet = _get_sheet()
    if sheet is None:
        return UNAVAILABLE
    try:
        return body(sheet)
    except (SheetsAccessError, SheetsUnavailable) as e:
        return f"Cannot read the A1 Metrics Dashboard sheet: {e}"
    except Exception as e:  # noqa: BLE001 - surface to the model, never crash the agent
        logger.error("%s failed: %s", name, e, exc_info=logger.isEnabledFor(logging.DEBUG))
        return f"Error reading the A1 Metrics Dashboard sheet in {name}: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _bps(v) -> str:
    return "n/a" if _isna(v) else f"{float(v):,.2f} bps"


def _pct(v, decimals: int = 1) -> str:
    return "n/a" if _isna(v) else f"{float(v) * 100:.{decimals}f}%"


def _date(v) -> str:
    if _isna(v):
        return "n/a"
    return v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v)[:10]


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _month_name(m) -> str:
    try:
        return MONTH_NAMES[int(m) - 1]
    except (TypeError, ValueError, IndexError):
        return str(m)


def _source_lines(sheet, tab: str, as_of: Optional[str], extra: Optional[str] = None) -> List[str]:
    as_of_s = as_of or "n/a (tab has no 'Data as of' cell)"
    if not as_of:
        try:
            dash = sheet.dashboard_as_of()
        except Exception:  # noqa: BLE001
            dash = None
        if dash:
            as_of_s += f"; dashboard 'Volume & PNL' tab is as of {dash}"
    lines = [f"Data as of: **{as_of_s}**",
             f"Source: {SOURCE_NAME}, tab '{tab}' - booked spot-business PnL, not the Haruko "
             "mark-to-market PnL of the derivatives book (BigQuery)."]
    if extra:
        lines.append(extra)
    return lines


def _clip_int(v, lo: int, hi: int, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

@tool
def get_spot_pnl_summary(year: Optional[int] = None) -> str:
    """Spot desk monthly volume, PnL and take rate for HOLD (client spot trading), A1
    (A1 Ltd principal spot desk) and the combined total, with YTD totals, from the
    A1 Metrics Dashboard sheet. MTD = the latest populated month. `year` defaults to
    the current year; 2025 reads the archived '2025 Volume & PNL' tab. Use for
    "A1 vs HOLD spot PnL", "spot PnL MTD / YTD", "take rate", "monthly spot volume".
    This is the spot business's booked PnL - not the Haruko derivatives-book PnL.
    """
    def body(sheet) -> str:
        df = sheet.monthly_volume_pnl(year=year)
        tab = df.attrs.get("tab", "Volume & PNL")
        as_of = df.attrs.get("data_as_of")
        if df.empty:
            return "\n".join([f"No monthly rows found on tab '{tab}'"] + _source_lines(sheet, tab, as_of))
        months = df[~df["is_ytd"]]
        ytd = df[df["is_ytd"]].set_index("entity") if df["is_ytd"].any() else pd.DataFrame()
        latest_month = int(months["month"].max()) if len(months) else None
        yr = df["year"].dropna().iloc[0] if df["year"].notna().any() else (year or "")
        out = [f"**Spot desk volume & PnL {int(yr) if yr != '' else ''}** (USD; take rate in bps = PnL / volume x 10,000)"]
        out += _source_lines(sheet, tab, as_of)
        # MTD block
        if latest_month is not None:
            cur = months[months["month"] == latest_month].set_index("entity")
            out.append("")
            out.append(f"**MTD ({_month_name(latest_month)} {int(yr) if yr != '' else ''}, latest populated month):**")
            for ent in ("HOLD", "A1", "TOTAL"):
                if ent in cur.index:
                    r = cur.loc[ent]
                    out.append(f"- {ent}: PnL {_usd(r['pnl_usd'])} on volume {_usd(r['volume_usd'])} "
                               f"({_bps(r['take_rate_bps'])})")
        # YTD block
        if len(ytd):
            out.append("")
            out.append("**YTD totals:**")
            for ent in ("HOLD", "A1", "TOTAL"):
                if ent in ytd.index:
                    r = ytd.loc[ent]
                    line = f"- {ent}: PnL {_usd(r['pnl_usd'])} on volume {_usd(r['volume_usd'])} ({_bps(r['take_rate_bps'])})"
                    if ent == "TOTAL" and not _isna(r.get("target_pnl_usd")):
                        line += f"; target {_usd(r['target_pnl_usd'])}"
                        if not _isna(r.get("pct_of_target")):
                            line += f", {_pct(r['pct_of_target'])} vs target"
                    out.append(line)
            if "HOLD" in ytd.index and "A1" in ytd.index:
                h, a = ytd.loc["HOLD"], ytd.loc["A1"]
                tot = (0 if _isna(h["pnl_usd"]) else h["pnl_usd"]) + (0 if _isna(a["pnl_usd"]) else a["pnl_usd"])
                if tot:
                    out.append(f"- A1 share of YTD PnL: {a['pnl_usd'] / tot * 100:.1f}%; HOLD share: {h['pnl_usd'] / tot * 100:.1f}%")
        # monthly table
        if len(months):
            out.append("")
            piv = {}
            for _, r in months.iterrows():
                piv.setdefault(int(r["month"]), {})[r["entity"]] = r
            rows = []
            for m in sorted(piv):
                cells = [_month_name(m)]
                for ent in ("HOLD", "A1", "TOTAL"):
                    r = piv[m].get(ent)
                    if r is None:
                        cells += ["", "", ""]
                    else:
                        cells += [_usd(r["volume_usd"]), _usd(r["pnl_usd"]), _bps(r["take_rate_bps"])]
                rows.append(cells)
            out += _md_table(["Month", "HOLD vol", "HOLD PnL", "HOLD take", "A1 vol", "A1 PnL", "A1 take",
                              "Total vol", "Total PnL", "Total take"], rows)
            tot_rows = months[months["entity"] == "TOTAL"]
            if len(tot_rows) and tot_rows["cum_pnl_usd"].notna().any():
                last = tot_rows.sort_values("month").iloc[-1]
                out.append(f"Cumulative through {_month_name(last['month'])}: volume {_usd(last['cum_volume_usd'])}, "
                           f"PnL {_usd(last['cum_pnl_usd'])}"
                           + (f", target {_usd(last['target_pnl_usd'])} ({_pct(last['pct_of_target'])} vs target)"
                              if not _isna(last.get("target_pnl_usd")) else ""))
        return "\n".join(out)
    return _guarded("get_spot_pnl_summary", body)


@tool
def get_weekly_spot_pnl(weeks: int = 8) -> str:
    """Last N weeks of spot desk PnL from the A1 Metrics Dashboard sheet: HOLD PnL, A1
    realised PnL, A1 unrealised PnL approximation, A1 total and the HOLD + A1 combined
    figure, with each week's start / end dates (Fri-Thu weeks), plus the sheet's
    year-to-date totals. Default 8 weeks, max 60. Use for "spot PnL this week / last
    week", "weekly spot PnL trend", "A1 realised vs unrealised".
    """
    def body(sheet) -> str:
        n = _clip_int(weeks, 1, MAX_WEEKS, 8)
        df = sheet.weekly_pnl()
        tab = df.attrs.get("tab", "Weekly PNL")
        as_of = df.attrs.get("data_as_of")
        wk = df[~df["is_total"]].sort_values("week")
        if wk.empty:
            return "\n".join([f"No weekly rows found on tab '{tab}'"] + _source_lines(sheet, tab, as_of))
        last = wk.tail(n)
        out = [f"**Spot desk weekly PnL - last {len(last)} week(s)** (USD)"]
        out += _source_lines(sheet, tab, as_of)
        newest = last.iloc[-1]
        if as_of and not _isna(newest["end_date"]) and pd.Timestamp(as_of) <= newest["end_date"]:
            out.append(f"Note: week {int(newest['week'])} ({_date(newest['start_date'])} to {_date(newest['end_date'])}) "
                       "is in progress - partial week.")
        rows = []
        for _, r in last.iterrows():
            rows.append([str(int(r["week"])), f"{_date(r['start_date'])} to {_date(r['end_date'])}", _usd(r["hold_pnl"]),
                         _usd(r["a1_realized"]), _usd(r["a1_unrealized"]), _usd(r["a1_total"]), _usd(r["total"])])
        out.append("")
        out += _md_table(["Week", "Dates", "HOLD PnL", "A1 realised", "A1 unrealised (approx)", "A1 total", "HOLD + A1"], rows)
        out.append("")
        out.append(f"Sum over these {len(last)} week(s): HOLD {_usd(last['hold_pnl'].sum())}, "
                   f"A1 {_usd(last['a1_total'].sum())} (realised {_usd(last['a1_realized'].sum())}, "
                   f"unrealised {_usd(last['a1_unrealized'].sum())}), combined {_usd(last['total'].sum())}; "
                   f"weekly average combined {_usd(last['total'].mean())}.")
        tot = df[df["is_total"]]
        if len(tot):
            t = tot.iloc[0]
            out.append(f"Sheet YTD totals (all weeks): HOLD {_usd(t['hold_pnl'])}, A1 {_usd(t['a1_total'])} "
                       f"(realised {_usd(t['a1_realized'])}, unrealised {_usd(t['a1_unrealized'])}), combined {_usd(t['total'])}.")
        out.append("A1 unrealised is the desk's approximation of open inventory PnL; realised is booked.")
        return "\n".join(out)
    return _guarded("get_weekly_spot_pnl", body)


def _coverage_note(df: pd.DataFrame, days: Optional[int]) -> str:
    start, end = df.attrs.get("coverage_start"), df.attrs.get("coverage_end")
    note = f"Blotter coverage on this tab: {start or 'n/a'} to {end or 'n/a'}"
    if days is not None:
        note += f"; window = last {days} day(s) ending at the latest trade in the blotter ({end})"
    dash = None
    try:
        dash_as_of = df.attrs.get("dashboard_as_of")
        dash = dash_as_of
    except Exception:  # noqa: BLE001
        pass
    if end and dash and pd.Timestamp(dash) - pd.Timestamp(end) > timedelta(days=7):
        note += (f". CAVEAT: the trade-level blotter stops at {end} while the dashboard is as of {dash} - "
                 "the sheet has no trade-level counterparty data after that date; for current totals use "
                 "get_spot_pnl_summary / get_weekly_spot_pnl.")
    return note + "."


@tool
def get_counterparty_pnl(days: int = 30, top_n: int = 15, counterparty: Optional[str] = None,
                         by: str = "counterparty") -> str:
    """Spot PnL and volume grouped by counterparty (default), symbol or side from the
    A1 Metrics Dashboard trade blotter: trades, notional, PnL, PnL share, average bps
    (volume-weighted). `days` is counted back from the latest trade in the blotter -
    the tool prints the coverage window, which may lag the dashboard. `counterparty`
    filters to names containing that text (case-insensitive) and lists their trades by
    symbol. Use for "which counterparties made us the most spot PnL", "PnL by symbol",
    "average bps we charge X".
    """
    def body(sheet) -> str:
        n_days = None if days is None or int(days) <= 0 else int(days)
        top = _clip_int(top_n, 1, MAX_TOP_N, 15)
        key = (by or "counterparty").strip().lower()
        if key not in BY_CHOICES:
            return f"Unknown `by`={by!r}; choose one of {', '.join(BY_CHOICES)}."
        df = sheet.counterparty_trades(days=n_days)
        try:
            df.attrs["dashboard_as_of"] = sheet.dashboard_as_of()
        except Exception:  # noqa: BLE001
            df.attrs["dashboard_as_of"] = None
        tab = df.attrs.get("tab", "db")
        as_of = df.attrs.get("coverage_end")
        header = [f"**Spot PnL by {key}**" + (f" - last {n_days} day(s)" if n_days else " - full blotter")]
        header += _source_lines(sheet, tab, as_of, _coverage_note(df, n_days))
        if df.empty:
            return "\n".join(header + ["No trades in that window."])
        if counterparty:
            mask = df["counterparty"].fillna("").str.contains(counterparty, case=False, regex=False)
            df = df[mask]
            if df.empty:
                return "\n".join(header + [f"No trades for counterparties matching {counterparty!r} in that window."])
            names = sorted(df["counterparty"].dropna().unique())
            header.append(f"Filter: counterparty contains {counterparty!r} -> {', '.join(names[:5])}"
                          + (" ..." if len(names) > 5 else ""))
            key = "symbol" if key == "counterparty" else key
        pnl_total = df["pnl_usd"].sum(skipna=True)
        g = df.groupby(df[key].fillna("(blank)"))
        agg = pd.DataFrame({
            "trades": g.size(),
            "notional_usd": g["notional_usd"].sum(min_count=1),
            "pnl_usd": g["pnl_usd"].sum(min_count=1),
            "last_trade": g["date"].max(),
        })
        # volume-weighted average bps; falls back to simple mean when notional is missing
        def wavg(sub: pd.DataFrame) -> float:
            w = sub["notional_usd"].where(sub["bps"].notna())
            if w.notna().any() and w.sum() > 0:
                return float((sub["bps"].fillna(0) * w.fillna(0)).sum() / w.sum())
            return float(sub["bps"].mean()) if sub["bps"].notna().any() else float("nan")
        agg["avg_bps"] = pd.Series({k: wavg(sub) for k, sub in g})
        agg = agg.sort_values("pnl_usd", ascending=False)
        shown = agg.head(top)
        rows = []
        for name, r in shown.iterrows():
            share = (r["pnl_usd"] / pnl_total * 100) if pnl_total else float("nan")
            rows.append([str(name), f"{int(r['trades']):,}", _usd(r["notional_usd"]), _usd(r["pnl_usd"]),
                         "n/a" if _isna(share) else f"{share:.1f}%", _bps(r["avg_bps"]), _date(r["last_trade"])])
        out = header + ["", f"{len(df):,} trades, {agg.shape[0]} {key}(s); total PnL {_usd(pnl_total)} on notional "
                            f"{_usd(df['notional_usd'].sum(skipna=True))}; window {_date(df['date'].min())} to {_date(df['date'].max())}."]
        out += _md_table([key.capitalize(), "Trades", "Notional", "PnL", "PnL share", "Avg bps (vol-wtd)", "Last trade"], rows)
        if agg.shape[0] > top:
            rest = agg.iloc[top:]
            out.append(f"... {agg.shape[0] - top} more {key}(s) with combined PnL {_usd(rest['pnl_usd'].sum())}.")
        neg = agg[agg["pnl_usd"] < 0]
        if len(neg):
            out.append(f"{len(neg)} {key}(s) with negative PnL in the window (worst: {neg.index[-1]} {_usd(neg['pnl_usd'].iloc[-1])}).")
        return "\n".join(out)
    return _guarded("get_counterparty_pnl", body)


@tool
def get_financing_fees(months: int = 6) -> str:
    """HOLD financing fees from the A1 Metrics Dashboard sheet: monthly HOLD financing
    fees and HOLD delta sales (USD) for the last N populated months (default 6), the
    year total, and the weekly block where populated. Use for "financing fees",
    "HOLD financing revenue", "delta sales".
    """
    def body(sheet) -> str:
        n = _clip_int(months, 1, MAX_MONTHS, 6)
        df = sheet.financing_fees()
        tab = df.attrs.get("tab", "Financing Fees")
        as_of = df.attrs.get("data_as_of")
        out = ["**HOLD financing fees** (USD)"]
        out += _source_lines(sheet, tab, as_of)
        if df.empty:
            out.append("No populated months on this tab.")
            return "\n".join(out)
        last = df.sort_values("month").tail(n)
        rows = [[_month_name(r["month"]), _usd(r["hold_financing_fees_usd"]), _usd(r["hold_delta_sales_usd"]), _usd(r["total_usd"])]
                for _, r in last.iterrows()]
        out.append("")
        out += _md_table(["Month", "HOLD financing fees", "HOLD delta sales", "Total"], rows)
        out.append(f"Sum of shown months: fees {_usd(last['hold_financing_fees_usd'].sum())}, delta sales "
                   f"{_usd(last['hold_delta_sales_usd'].sum())}, total {_usd(last['total_usd'].sum())}. "
                   f"Year total (all populated months): {_usd(df['total_usd'].sum())}; latest populated month: "
                   f"{_month_name(df['month'].max())}.")
        try:
            wk = sheet.financing_fees_weekly()
        except Exception as e:  # noqa: BLE001
            wk = pd.DataFrame()
            logger.debug("weekly financing block unavailable: %s", e)
        if len(wk):
            w = wk.sort_values("week").tail(6)
            out.append("")
            out.append("Weekly block (populated weeks, last 6):")
            out += _md_table(["Week", "Fees", "Delta sales", "Total"],
                             [[str(int(r["week"])), _usd(r["hold_financing_fees_usd"]), _usd(r["hold_delta_sales_usd"]), _usd(r["total_usd"])]
                              for _, r in w.iterrows()])
        out.append("Months not yet booked are omitted; the tab is filled in by the desk after month end.")
        return "\n".join(out)
    return _guarded("get_financing_fees", body)


@tool
def get_nonclient_pnl(months: int = 6) -> str:
    """Non-client (proprietary / non-client-flow) spot PnL from the A1 Metrics Dashboard
    'Nonclient PNL' tab, last N months if the tab holds a table. The tab currently only
    links to a separate HOLD PNL spreadsheet; the tool says so and returns the link
    rather than inventing numbers.
    """
    def body(sheet) -> str:
        n = _clip_int(months, 1, MAX_MONTHS, 6)
        df = sheet.nonclient_pnl()
        tab = df.attrs.get("tab", "Nonclient PNL")
        as_of = df.attrs.get("data_as_of")
        out = ["**Non-client spot PnL**"]
        out += _source_lines(sheet, tab, as_of)
        title, link = df.attrs.get("title"), df.attrs.get("link")
        if df.empty:
            out.append(f"The '{tab}' tab is not populated in this sheet" + (f" (it is titled '{title}'" if title else "")
                       + (f" and points to a separate spreadsheet: {link})" if link else (")" if title else "."))
                       + " No non-client PnL figures are available through this tool; the agent has no access to the linked sheet.")
            return "\n".join(out)
        show = df.tail(n)
        headers = [str(c) for c in show.columns]
        money = [any(k in h.lower() for k in ("pnl", "p&l", "usd", "fee", "volume", "total", "sales")) for h in headers]
        rows = [[(_usd(v) if money[i] else _raw_cell(v)) if isinstance(v, (int, float)) and not isinstance(v, bool)
                 else ("" if v is None else str(v)) for i, v in enumerate(r)]
                for r in show.itertuples(index=False)]
        out.append("")
        out += _md_table(headers, rows)
        if link:
            out.append(f"Linked sheet: {link}")
        return "\n".join(out)
    return _guarded("get_nonclient_pnl", body)


# ---------------------------------------------------------------------------
# client flow vs non-client flow (A1 database tab)
# ---------------------------------------------------------------------------

_LAST_N_DAYS_RE = re.compile(r"^(?:last|past|previous|trailing)\s+(\d{1,3})\s*(?:d|day|days)$")
_N_DAYS_RE = re.compile(r"^(\d{1,3})\s*(?:d|day|days)$")


def _week_bounds(anchor: pd.Timestamp, weeks: Optional[pd.DataFrame], offset: int = 0) -> Tuple[pd.Timestamp, pd.Timestamp, Optional[int]]:
    """(start, end, week_no) of the dashboard week containing `anchor`, shifted by `offset`
    weeks (-1 = last week). Uses the 'Weekly PNL' start / end dates when available, else
    the desk's Friday-Thursday convention."""
    if weeks is not None and len(weeks):
        wk = weeks[(~weeks["is_total"]) & weeks["start_date"].notna() & weeks["end_date"].notna()].sort_values("start_date")
        hit = wk[(wk["start_date"] <= anchor) & (wk["end_date"] >= anchor)]
        if len(hit):
            pos = wk.index.get_loc(hit.index[0]) + offset
            if 0 <= pos < len(wk):
                w = wk.iloc[pos]
                return w["start_date"].normalize(), w["end_date"].normalize(), int(w["week"])
    # Fri-Thu fallback
    start = anchor.normalize() - timedelta(days=(anchor.weekday() - 4) % 7) + timedelta(days=7 * offset)
    return start, start + timedelta(days=6), None


def resolve_client_flow_window(start_date: Optional[str], end_date: Optional[str], anchor: pd.Timestamp,
                               weeks: Optional[pd.DataFrame] = None) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
    """Turn the tool's date arguments into (start, end, label).

    `anchor` is the sheet's data-as-of date (never the wall clock). Shortcuts (case-
    insensitive, in start_date): 'last week' / 'this week' (dashboard weeks), 'mtd' /
    'month to date', 'ytd' / 'year to date', 'last month', 'this month', 'last N days'
    (N days ending on the anchor), 'yesterday', 'today'. Otherwise start_date / end_date
    are dates (ISO or anything the sheet parsers accept); a missing end_date means the
    anchor for open-ended words and the start itself for a single date.
    """
    key = re.sub(r"\s+", " ", (start_date or "").strip().lower())
    a = anchor.normalize()
    if key in ("last week", "previous week", "prior week"):
        s_, e_, wno = _week_bounds(a, weeks, -1)
        return s_, e_, f"last week{f' = dashboard week {wno}' if wno else ''}"
    if key in ("this week", "current week", "week to date", "wtd"):
        s_, e_, wno = _week_bounds(a, weeks, 0)
        return s_, e_, f"this week{f' = dashboard week {wno}' if wno else ''} (in progress)"
    if key in ("mtd", "month to date", "month-to-date", "this month", "current month"):
        return a.replace(day=1), a, "month to date"
    if key in ("ytd", "year to date", "year-to-date", "this year"):
        return a.replace(month=1, day=1), a, "year to date"
    if key in ("last month", "previous month", "prior month"):
        first_this = a.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        return last_prev.replace(day=1), last_prev, "last month"
    if key in ("yesterday",):
        d = a - timedelta(days=1)
        return d, d, "yesterday"
    if key in ("today", "latest"):
        return a, a, "latest day"
    m = _LAST_N_DAYS_RE.match(key) or _N_DAYS_RE.match(key)
    if m:
        n = max(1, min(int(m.group(1)), 400))
        return a - timedelta(days=n - 1), a, f"last {n} days"
    s_ts = parse_sheet_date(start_date) if start_date else None
    if s_ts is None:
        raise ValueError(f"Could not understand start_date={start_date!r}. Use an ISO date (YYYY-MM-DD) or one of: "
                         + ", ".join(WINDOW_SHORTCUTS) + ".")
    s_ts = s_ts.normalize()
    if end_date:
        e_key = re.sub(r"\s+", " ", str(end_date).strip().lower())
        e_ts = a if e_key in ("today", "now", "latest", "asof", "as of") else parse_sheet_date(end_date)
        if e_ts is None:
            raise ValueError(f"Could not understand end_date={end_date!r}; use an ISO date (YYYY-MM-DD).")
        e_ts = e_ts.normalize()
    else:
        e_ts = s_ts
    if e_ts < s_ts:
        s_ts, e_ts = e_ts, s_ts
    label = s_ts.strftime("%Y-%m-%d") if s_ts == e_ts else f"{s_ts.strftime('%Y-%m-%d')} to {e_ts.strftime('%Y-%m-%d')}"
    return s_ts, e_ts, label


def _share(v: Optional[float]) -> str:
    return "n/a" if v is None or _isna(v) else f"{v * 100:.1f}%"


def _client_flow_markdown(sheet, res: Dict, label: str, include_daily: bool) -> str:
    tab = res["tab"]
    sums = res["sums"]
    rows: pd.DataFrame = res["rows"]
    out = ["**A1 spot realised PnL - client flow vs non-client (proprietary) flow**"]
    coverage = f"{res['n_rows']} populated day-row(s)"
    if rows is not None and len(rows):
        coverage += f", sheet rows {int(rows['row_number'].min())}-{int(rows['row_number'].max())}"
    span = f"{res['start']} to {res['end']}" if res["start"] != res["end"] else res["start"]
    out.append(f"Window: {span}" + (f" ({label})" if label and label != span else "") + f" - {coverage}.")
    as_of = res.get("data_as_of")
    out.append(f"Data as of: **{as_of or 'n/a'}** (dashboard date; last populated row in '{tab}' is {res.get('last_row_date') or 'n/a'})")
    out.append(f"Source: {SOURCE_NAME}, tab '{tab}' columns R:X (R date, V daily realised total, W client flow, "
               "X non-client flow) - booked spot PnL of A1 Ltd, not the Haruko derivatives book (BigQuery).")

    if not res["n_rows"]:
        out.append("")
        out.append("No populated rows in that window" + (f"; the tab covers {res.get('first_row_date')} to {res.get('last_row_date')}."
                                                          if res.get("first_row_date") else "."))
        if res["not_populated"]:
            out.append(f"Days after the last populated row: {res['not_populated'][0]} to {res['not_populated'][-1]}.")
        if res["artifact_rows"]:
            out.append("Excluded artefact rows: " + "; ".join(
                f"row {a['row_number']} {a['date']} V {_usd(a['realized_total'])}" for a in res["artifact_rows"]))
        return "\n".join(out)

    tot = sums["realized_total"]
    out.append("")
    out.append(f"**Totals:** realised {_usd(tot)} = client flow {_usd(sums['realized_client'])} ({_share(res['client_share'])})"
               f" + non-client flow {_usd(sums['realized_nonclient'])} ({_share(res['nonclient_share'])})"
               + (f" over {res['n_days']} day(s)" if res["n_days"] else ""))
    if res["client_share"] is not None and (res["client_share"] < 0 or res["nonclient_share"] < 0):
        out.append("(one leg is negative, so the percentage shares are of the net total and exceed 100% / go negative.)")
    resid = tot - sums["realized_client"] - sums["realized_nonclient"]
    if abs(resid) > 1.0:
        out.append(f"Note: V does not equal W + X over this window (difference {_usd(resid)}) - "
                   + (f"{len(res['incomplete_rows'])} row(s) lack the W/X split: "
                      + ", ".join(f"row {r['row_number']} ({r['date']})" for r in res["incomplete_rows"])
                      if res["incomplete_rows"] else "sheet rows carry an inconsistency") + ".")
    elif res["incomplete_rows"]:
        out.append("Rows lacking the W/X split: " + ", ".join(f"row {r['row_number']} ({r['date']})" for r in res["incomplete_rows"]) + ".")

    if include_daily:
        out.append("")
        if len(rows) <= MAX_DAILY_ROWS:
            table = []
            for _, r in rows.iterrows():
                flag = " *" if r["flagged_artifact"] else ""
                table.append([r["date"].strftime("%Y-%m-%d") + flag, str(int(r["row_number"])), _usd(r["realized_total"]),
                              _usd(r["realized_client"]), _usd(r["realized_nonclient"])])
            out += _md_table(["Date", "Row", "Realised total (V)", "Client flow (W)", "Non-client (X)"], table)
            if rows["flagged_artifact"].any():
                out.append("\\* offsetting artefact pair - see below.")
        else:
            g = rows.groupby(rows["date"].dt.to_period("M"))
            table = [[str(p), str(int(sub["date"].nunique())), _usd(sub["realized_total"].sum()), _usd(sub["realized_client"].sum()),
                      _usd(sub["realized_nonclient"].sum())] for p, sub in g]
            out += _md_table(["Month", "Days", "Realised total (V)", "Client flow (W)", "Non-client (X)"], table)
            out.append(f"({len(rows)} day-rows collapsed to monthly subtotals; ask for a shorter window for the per-day table.)")

    notes = []
    if res["missing_days"]:
        md = res["missing_days"]
        shown = ", ".join(md[:12]) + (f" ... ({len(md)} in total)" if len(md) > 12 else "")
        notes.append(f"Missing days (no row in the sheet for these dates in the window; the tab normally has weekends too): {shown}.")
    if res["not_populated"]:
        np_ = res["not_populated"]
        notes.append(f"Not yet populated: {np_[0]}" + (f" to {np_[-1]}" if len(np_) > 1 else "") + " (after the last row).")
    if res["duplicate_dates"]:
        notes.append(f"Two snapshot rows on {', '.join(res['duplicate_dates'])} - both summed (intraday increments).")
    for a in res["artifact_rows"]:
        notes.append(f"Artefact row {a['row_number']} ({a['date']}): V {_usd(a['realized_total'])}, W {_usd(a['realized_client'])}, "
                     f"X {_usd(a['realized_nonclient'])} - {a['action']}.")
    if res["artifact_rows"] and any(a["action"].startswith("kept") for a in res["artifact_rows"]):
        notes.append(f"The kept pair(s) net to {_usd(res['artifact_net'])} (the desk's weekly totals carry them the same way); "
                     "the per-day figures on those dates are not meaningful.")
    if res["excluded"] is not None and len(res["excluded"]):
        ex = res["sums_all_rows"]
        notes.append(f"Including the excluded leg(s) the raw sums would be V {_usd(ex['realized_total'])}, "
                     f"W {_usd(ex['realized_client'])}, X {_usd(ex['realized_nonclient'])}.")
    wc = res.get("weekly_check")
    if wc:
        if wc["a1_realized"] is None:
            notes.append(f"Weekly cross-check: 'Weekly PNL' week {wc['week']} has no A1 Realized figure yet.")
        else:
            diff = wc["difference"]
            verdict = "matches" if abs(diff) < 1.0 else ("within rounding" if abs(diff) < 5 else "DIFFERS")
            notes.append(f"Weekly cross-check: 'Weekly PNL' tab week {wc['week']} ({wc['start_date']} to {wc['end_date']}) "
                         f"A1 Realized PNL {_usd(wc['a1_realized'])} - {verdict} (difference {_usd(diff)})"
                         + (f"; that tab's unrealised approximation for the week is {_usd(wc['a1_unrealized'])}, not included here"
                            if wc.get("a1_unrealized") is not None else "") + ".")
    yc = res.get("ytd_check")
    if yc:
        notes.append(f"YTD cross-check: the sheet's own cumulative YTD columns (S/T/U) at {yc['date']} read total "
                     f"{_usd(yc['cum_total'])}, client {_usd(yc['cum_client'])}, non-client {_usd(yc['cum_nonclient'])} - "
                     f"differences vs the summed daily rows: {_usd(yc['diff_total'])} / {_usd(yc['diff_client'])} / "
                     f"{_usd(yc['diff_nonclient'])}"
                     + (" (small: timing / missing rows in a hand-maintained sheet)" if max(abs(yc['diff_total']), abs(yc['diff_client']), abs(yc['diff_nonclient'])) < 0.05 * max(1.0, abs(yc['cum_total'])) else " (LARGE - flag to the desk)") + ".")
    if notes:
        out.append("")
        out += notes
    out.append("")
    out.append("Caveats: realised PnL only - the unrealised (open inventory) approximation is excluded; A1 Ltd spot only - "
               "HOLD (client commissions) is not included; the split is the desk's own client-flow / non-client-flow "
               "attribution in a hand-maintained sheet - quote the as-of date and do not extrapolate beyond the last populated row.")
    return "\n".join(out)


@tool
def get_a1_client_flow_split(start_date: str, end_date: Optional[str] = None, include_daily: bool = True) -> str:
    """A1 spot desk REALISED PnL split between client flow and non-client (proprietary /
    prop / house) flow for a date window, from the A1 Metrics Dashboard sheet ('A1
    database' tab, columns R:X: per-day realised total V = client flow W + non-client
    flow X). USE THIS for any question about client vs non-client / proprietary / prop
    flow, flow attribution, or "how much of A1's PnL came from clients" - never read
    raw cells for that. `start_date`: ISO date or a shortcut - "last week", "this week"
    (dashboard Fri-Thu weeks), "mtd", "ytd", "last month", "last N days", "yesterday" -
    resolved against the sheet's data-as-of date. `end_date`: ISO date (defaults to the
    start date for a single day; ignored for shortcuts). Returns the per-day table (or
    monthly subtotals for long windows; include_daily=False to skip), totals with % split,
    missing days, offsetting artefact rows (disclosed / excluded), a cross-check against
    the 'Weekly PNL' tab when the window is a dashboard week, and caveats (realised
    only, unrealised excluded; A1 only, HOLD not included).
    """
    def body(sheet) -> str:
        as_of = None
        try:
            as_of = sheet.dashboard_as_of()
        except (SheetsAccessError, SheetsUnavailable):
            raise
        except Exception:  # noqa: BLE001
            as_of = None
        weeks = None
        try:
            weeks = sheet.weekly_pnl(include_future=True)
        except (SheetsAccessError, SheetsUnavailable):
            raise
        except Exception as e:  # noqa: BLE001
            logger.debug("weekly tab unavailable for window resolution: %s", e)
        anchor = pd.Timestamp(as_of) if as_of else None
        if anchor is None:
            probe = sheet.client_flow_daily()
            last = probe.attrs.get("last_row_date")
            anchor = pd.Timestamp(last) if last else pd.Timestamp.today().normalize()
        try:
            s_ts, e_ts, label = resolve_client_flow_window(start_date, end_date, anchor, weeks)
        except ValueError as e:
            return str(e)
        res = sheet.client_flow_window(s_ts, e_ts, exclude_artifacts=True)
        return _client_flow_markdown(sheet, res, label, bool(include_daily))
    return _guarded("get_a1_client_flow_split", body)


@tool
def list_a1_dashboard_tabs() -> str:
    """List the tabs of the A1 Metrics Dashboard sheet (name, size) and which tools read
    them. Use before read_a1_dashboard_range to find a tab name.
    """
    def body(sheet) -> str:
        tabs = sheet.list_tabs()
        used = {
            sheet.tabs["volume_pnl"]: "get_spot_pnl_summary",
            f"2025 {sheet.tabs['volume_pnl']}": "get_spot_pnl_summary(year=2025)",
            sheet.tabs["weekly_pnl"]: "get_weekly_spot_pnl",
            sheet.tabs["financing_fees"]: "get_financing_fees",
            sheet.tabs["nonclient_pnl"]: "get_nonclient_pnl",
            sheet.tabs["trades"]: "get_counterparty_pnl (trade blotter)",
            sheet.tabs.get("a1_database", "A1 database"): "get_a1_client_flow_split (cols R:X)",
        }
        out = [f"**{sheet.title or 'A1 Metrics Dashboard'}** - {len(tabs)} tab(s)"]
        out += _source_lines(sheet, sheet.tabs["volume_pnl"], sheet.dashboard_as_of())
        out.append("")
        out += _md_table(["Tab", "Rows x cols", "Read by"],
                         [[t["title"], f"{t.get('rows') or '?'} x {t.get('cols') or '?'}", used.get(t["title"], "")] for t in tabs])
        out.append("Other tabs are helper / database tabs; read_a1_dashboard_range(tab, a1_range) shows any of them "
                   f"(capped at {MAX_RANGE_ROWS} rows x {MAX_RANGE_COLS} columns).")
        return "\n".join(out)
    return _guarded("list_a1_dashboard_tabs", body)


@tool
def read_a1_dashboard_range(tab: str, a1_range: str = "A1:T40") -> str:
    """Escape hatch: raw cell values of one range of the A1 Metrics Dashboard sheet as a
    markdown table (read-only; capped at 200 rows x 30 columns - larger ranges are
    clamped). `tab` is the tab name (see list_a1_dashboard_tabs), `a1_range` like
    "A1:T40" or "A:M". Prefer the typed tools; use this for tabs they do not cover
    (e.g. 'Client Metrics', 'A1 Deltas', 'Hold Deltas').
    """
    def body(sheet) -> str:
        if not tab or not str(tab).strip():
            return "tab is required (use list_a1_dashboard_tabs)."
        tab_name = str(tab).strip()
        inner_tab, rng = parse_a1_range(a1_range or "A1:T40")
        if inner_tab and inner_tab != tab_name:
            tab_name = inner_tab
        rng, clamped = clamp_a1_range(rng, MAX_RANGE_ROWS, MAX_RANGE_COLS)
        values = sheet.get_range(tab_name, rng)
        as_of = sheet.data_as_of(tab_name)          # separate (cached) A1:F5 read - the range may not include row 1
        out = [f"**'{tab_name}'!{rng}**" + (f" (clamped to {MAX_RANGE_ROWS} x {MAX_RANGE_COLS})" if clamped else "")]
        out += _source_lines(sheet, tab_name, as_of)
        if not values:
            out.append("(empty range)")
            return "\n".join(out)
        width = min(max(len(r) for r in values), MAX_RANGE_COLS)
        headers = [""] + [_col_letter(i, rng) for i in range(width)]
        rows = []
        start_row = int(rng.split(":")[0].lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ") or 1)
        for i, r in enumerate(values[:MAX_RANGE_ROWS]):
            rows.append([str(start_row + i)] + [_raw_cell(r[c] if c < len(r) else None) for c in range(width)])
        out.append("")
        out += _md_table(headers, rows)
        out.append(f"{min(len(values), MAX_RANGE_ROWS)} row(s) x {width} column(s); numbers are unformatted sheet values.")
        return "\n".join(out)
    return _guarded("read_a1_dashboard_range", body)


def _col_letter(offset: int, rng: str) -> str:
    from providers.gsheets import col_to_index, index_to_col
    first = rng.split(":")[0].rstrip("0123456789") or "A"
    return index_to_col(col_to_index(first) + offset)


def _raw_cell(v) -> str:
    if v is None or (isinstance(v, str) and not v.strip()):
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 1 else f"{v:,.2f}"
    s = str(v).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= 60 else s[:57] + "..."


def get_sheet_tools() -> list:
    """All A1 Metrics Dashboard tools, in registration order."""
    return [
        get_spot_pnl_summary,
        get_weekly_spot_pnl,
        get_counterparty_pnl,
        get_financing_fees,
        get_nonclient_pnl,
        get_a1_client_flow_split,
        list_a1_dashboard_tabs,
        read_a1_dashboard_range,
    ]


SHEET_TOOL_NAMES = [t.name for t in get_sheet_tools()]

__all__ = [
    "SHEET_TOOL_NAMES",
    "SOURCE_NAME",
    "UNAVAILABLE",
    "get_a1_client_flow_split",
    "get_counterparty_pnl",
    "get_financing_fees",
    "get_nonclient_pnl",
    "get_sheet_tools",
    "get_spot_pnl_summary",
    "get_weekly_spot_pnl",
    "list_a1_dashboard_tabs",
    "read_a1_dashboard_range",
    "resolve_client_flow_window",
]
