"""@tool functions over FRED (providers/fred.py): the desk's traditional-finance macro read.

    get_macro_snapshot(groups)          dashboard: rates, curve, dollar, oil, equity indices, VIX, credit, inflation, labour
    get_fred_series(series_id, days)    one series' recent history with changes
    search_fred(text, limit)            find a FRED series id

FRED publishes with a one-day lag and only on business days; every number here is quoted with
its observation date, never as "now". Equity / ETF tickers, commodity futures curves, FX pairs
and GICS-style sector data need a vendor we do not have yet; the tools say so when asked.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import pandas as pd
from langchain_core.tools import tool

from providers.fred import GROUPS, MACRO_SERIES, FredError
from tools.desk_tools import _md_table

logger = logging.getLogger(__name__)

SOURCE = "FRED (Federal Reserve Bank of St. Louis)"
MAX_ROWS = 45
UNAVAILABLE = ("Macro data (FRED) is not available: no API key. Create a free key at "
               "https://fred.stlouisfed.org/docs/api/api_key.html and store it as secret fred_api_key in "
               "anchorage-trading-solutions (or set env FRED_API_KEY). Crypto market data and desk tools still work.")
NOT_COVERED = ("Not covered: individual equity or ETF prices, commodity futures curves, FX pairs and GICS sector data "
               "need a traditional-finance vendor we do not have yet; FRED gives indices, rates, spreads and macro series.")


def _get_fred():
    """FredProvider or None (tests monkeypatch this)."""
    from providers.factory import get_fred_provider
    return get_fred_provider()


def _guarded(name: str, body: Callable) -> str:
    prov = _get_fred()
    if prov is None:
        return UNAVAILABLE
    try:
        return body(prov)
    except FredError as e:
        return f"FRED error in {name}: {e}"
    except Exception as e:  # noqa: BLE001
        logger.error("%s failed: %s", name, e, exc_info=logger.isEnabledFor(logging.DEBUG))
        return f"Error calling FRED in {name}: {type(e).__name__}: {e}"


def _day(t) -> str:
    return "n/a" if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d")


def _val(v, units: str) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    v = float(v)
    if units in ("%", "pp"):
        return f"{v:.2f}{'%' if units == '%' else ' pp'}"
    if units.startswith("$"):
        return f"${v:,.2f}"
    return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"


def _chg(v, units: str) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    v = float(v)
    if units in ("%", "pp"):
        return f"{v * 100:+.0f} bp"
    if units.startswith("$"):
        return f"{v:+,.2f}"
    return f"{v:+,.2f}" if abs(v) < 1000 else f"{v:+,.0f}"


def _pct_chg(new, old) -> str:
    try:
        if old is None or pd.isna(old) or float(old) == 0 or new is None or pd.isna(new):
            return "n/a"
        return f"{(float(new) / float(old) - 1) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "n/a"


@tool("get_macro_snapshot")
def get_macro_snapshot(groups: str = "") -> str:
    """Traditional-finance macro dashboard from FRED: US Treasury yields (2y, 10y, 30y), the 10y-2y curve, 10y real yield, SOFR and fed funds, the broad dollar index, WTI and Brent, S&P 500, Nasdaq and Dow, VIX, high-yield and investment-grade credit spreads, breakeven inflation, CPI and unemployment, each with its observation date and the change versus the prior print, one week and one month.

    Use for "where are yields / the dollar / oil today", "how is macro looking", "risk-off in
    TradFi?", "rates backdrop for crypto". All values are the latest FRED observation
    (published with a one-day lag, business days only), not live quotes.

    Args:
        groups: Optional comma-separated subset: rates, fx, commodities, equities, volatility, credit, inflation, labour.

    Returns:
        Markdown tables per group with value, date, change vs prior print / 1w / 1m.
    """
    def body(prov) -> str:
        want = [g.strip().lower() for g in (groups or "").split(",") if g.strip()]
        bad = [g for g in want if g not in GROUPS]
        if bad:
            return f"Unknown group(s) {', '.join(bad)}; choose from {', '.join(GROUPS)}."
        ids = [s for s, (_, _, g) in MACRO_SERIES.items() if not want or g in want]
        df = prov.latest(ids)
        out = [f"### Macro snapshot ({SOURCE})"]
        ok = df[df["error"].isna()]
        if len(ok):
            out.append(f"- latest observation dates {_day(ok['date'].min())} to {_day(ok['date'].max())}; changes are vs the "
                       "prior print, ~1 week and ~1 month earlier (bp for rates and spreads)")
        for g in GROUPS:
            sub = df[df["group"] == g]
            if sub.empty:
                continue
            out.append("")
            out.append(f"**{g.capitalize()}**")
            rows = []
            for _, r in sub.iterrows():
                if r["error"]:
                    rows.append([f"{r['name']} ({r['series_id']})", "n/a", "n/a", "n/a", "n/a", "n/a", f"error: {r['error']}"])
                    continue
                u = r["units"]
                rows.append([f"{r['name']} ({r['series_id']})", _val(r["value"], u), _day(r["date"]), _chg(r["chg_1"], u),
                             _chg(r["chg_1w"], u), _chg(r["chg_1m"], u),
                             "" if u in ("%", "pp") else f"1w {_pct_chg(r['value'], r['value_1w'])}, 1m {_pct_chg(r['value'], r['value_1m'])}"])
            out += _md_table(["Series", "Value", "As of", "vs prior", "1w", "1m", "% change"], rows)
        out.append(f"_Source: {SOURCE}, daily / monthly series, one-day publication lag; quote the observation date. {NOT_COVERED}_")
        return "\n".join(out)
    return _guarded("get_macro_snapshot", body)


@tool("get_fred_series")
def get_fred_series(series_id: str, days: int = 90) -> str:
    """Recent history of one FRED series (any FRED id: DGS10, DTWEXBGS, VIXCLS, DCOILWTICO, SP500, BAMLH0A0HYM2, CPIAUCSL, M2SL, WALCL ...): latest value with date, change over the window, high / low, and a table of observations (monthly series show every print; daily series the last 45).

    Use for "10-year yield over the last month", "how has the dollar index moved", "Fed balance
    sheet trend" (WALCL), "M2". Use search_fred to find an id.

    Args:
        series_id: FRED series id (case-insensitive).
        days: Window in calendar days (default 90, max 3650).

    Returns:
        Markdown summary and observation table with the series title, units and frequency.
    """
    def body(prov) -> str:
        sid = (series_id or "").strip().upper()
        if not sid:
            return "Give a FRED series id (use search_fred to find one)."
        n = max(7, min(int(days), 3650))
        info = prov.series_info(sid)
        df = prov.observations(sid, days=n)
        title = info.get("title") or sid
        units = info.get("units_short") or info.get("units") or ""
        u = MACRO_SERIES.get(sid, (None, units, None))[1] or units
        out = [f"### {title} ({sid}), last {n} days - {SOURCE}",
               f"- units: {units or 'n/a'}; frequency: {info.get('frequency') or 'n/a'}"
               + (f"; seasonal adjustment: {info.get('seasonal_adjustment')}" if info.get("seasonal_adjustment") else "")
               + f"; FRED last updated {info.get('last_updated', '')[:16] or 'n/a'}"]
        if df.empty:
            out.append(f"No observations in the last {n} days (series ends {info.get('observation_end') or 'n/a'}).")
            return "\n".join(out)
        first, last = df.iloc[0], df.iloc[-1]
        out.append(f"- latest {_val(last['value'], u)} as of {_day(last['date'])}; {_chg(last['value'] - first['value'], u)} vs "
                   f"{_day(first['date'])} ({_val(first['value'], u)})" + ("" if u in ("%", "pp") else f", {_pct_chg(last['value'], first['value'])}"))
        out.append(f"- window high {_val(df['value'].max(), u)} on {_day(df.loc[df['value'].idxmax(), 'date'])}, "
                   f"low {_val(df['value'].min(), u)} on {_day(df.loc[df['value'].idxmin(), 'date'])}; {len(df)} observations")
        shown = df if len(df) <= MAX_ROWS else df.iloc[-MAX_ROWS:]
        out.append("")
        out += _md_table(["Date", "Value"], [[_day(r["date"]), _val(r["value"], u)] for _, r in shown.iterrows()])
        if len(shown) < len(df):
            out.append(f"(table shows the last {len(shown)} of {len(df)} observations)")
        out.append(f"_Source: {SOURCE}; observation dates, one-day publication lag, business days only._")
        return "\n".join(out)
    return _guarded("get_fred_series", body)


@tool("search_fred")
def search_fred(text: str, limit: int = 10) -> str:
    """Search FRED for a series id by keywords (e.g. 'fed balance sheet', 'mortgage rate 30 year', 'euro dollar exchange rate', 'gold price'), sorted by popularity, with units, frequency and last update.

    Args:
        text: Search words.
        limit: Max results (default 10, max 25).

    Returns:
        Markdown table of candidate series to pass to get_fred_series.
    """
    def body(prov) -> str:
        q = (text or "").strip()
        if not q:
            return "Give some search words."
        df = prov.search(q, limit=limit)
        if df.empty:
            return f"FRED has no series matching '{q}'."
        out = [f"### FRED series matching '{q}' ({len(df)} shown, by popularity)"]
        out += _md_table(["Id", "Title", "Units", "Freq", "Last updated", "Ends"],
                         [[r["id"], r["title"], r["units"] or "", r["frequency"] or "", r["last_updated"] or "", r["observation_end"] or ""]
                          for _, r in df.iterrows()])
        out.append("_Pass an id to get_fred_series. Discontinued series show an old 'Ends' date._")
        return "\n".join(out)
    return _guarded("search_fred", body)


def get_macro_tools() -> list:
    """FRED macro tools, in registration order."""
    return [get_macro_snapshot, get_fred_series, search_fred]


MACRO_TOOL_NAMES = [t.name for t in get_macro_tools()]

__all__ = ["MACRO_TOOL_NAMES", "NOT_COVERED", "UNAVAILABLE", "get_fred_series", "get_macro_snapshot", "get_macro_tools", "search_fred"]
