"""@tool functions over the spot desk's "A1 Metrics Dashboard" Google Sheet (read-only).

Scope: the **spot** business's booked PnL as the desk maintains it by hand
(owner Joao Luis, updated daily): HOLD (client spot trading - commissions on client
trades) vs A1 (A1 Ltd, the principal spot desk) monthly volume / PnL / take rate,
weekly realised / unrealised PnL, HOLD financing fees, and the counterparty trade
blotter. This is a different source from the Haruko mark-to-market PnL of the OTC /
derivatives book in BigQuery (tools/desk_tools.py); the tools say so in every answer.

Every tool returns compact markdown, formats USD with ``$`` and thousands separators,
bps to two decimals, and always prints the tab's "Data as of" date plus the source
line ("A1 Metrics Dashboard sheet maintained by the desk"). Nothing here writes.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable, List, Optional, Sequence

import pandas as pd
from langchain_core.tools import tool

from providers.gsheets import SheetsAccessError, SheetsUnavailable, clamp_a1_range, parse_a1_range
from tools.desk_tools import _isna, _md_table, _num, _usd

logger = logging.getLogger(__name__)

SOURCE_NAME = "A1 Metrics Dashboard (Google Sheet maintained daily by the spot desk)"
MAX_RANGE_ROWS = 200
MAX_RANGE_COLS = 30
MAX_TOP_N = 50
MAX_WEEKS = 60
MAX_MONTHS = 24
BY_CHOICES = ("counterparty", "symbol", "side")

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
        list_a1_dashboard_tabs,
        read_a1_dashboard_range,
    ]


SHEET_TOOL_NAMES = [t.name for t in get_sheet_tools()]

__all__ = [
    "SHEET_TOOL_NAMES",
    "SOURCE_NAME",
    "UNAVAILABLE",
    "get_counterparty_pnl",
    "get_financing_fees",
    "get_nonclient_pnl",
    "get_sheet_tools",
    "get_spot_pnl_summary",
    "get_weekly_spot_pnl",
    "list_a1_dashboard_tabs",
    "read_a1_dashboard_range",
]
