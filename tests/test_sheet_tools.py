"""tools/sheet_tools.py with a fake A1MetricsSheet. No network, no GCP.

The fake serves the same canned tabs as tests/test_gsheets.py through the real
A1MetricsSheet parsing code, so these tests cover the whole path from `values`
payload to markdown.
"""

from __future__ import annotations

import pandas as pd
import pytest

from providers.gsheets import A1MetricsSheet, SheetsAccessError, SheetsUnavailable
from tests.test_gsheets import META, TABS, FakeClock, FakeCreds, FakeSession
from tools import sheet_tools as st
from tools.sheet_tools import (
    MAX_RANGE_COLS,
    MAX_RANGE_ROWS,
    SHEET_TOOL_NAMES,
    UNAVAILABLE,
    get_a1_client_flow_split,
    get_counterparty_pnl,
    get_financing_fees,
    get_nonclient_pnl,
    get_sheet_tools,
    get_spot_pnl_summary,
    get_weekly_spot_pnl,
    list_a1_dashboard_tabs,
    read_a1_dashboard_range,
    resolve_client_flow_window,
)


def make_sheet(session=None):
    return A1MetricsSheet("sheet-id-123", "quota-proj", cache_ttl_s=300, session=session or FakeSession(),
                          credentials=FakeCreds(), clock=FakeClock())


@pytest.fixture
def sheet(monkeypatch):
    s = make_sheet()
    monkeypatch.setattr(st, "_get_sheet", lambda: s)
    return s


@pytest.fixture
def no_sheet(monkeypatch):
    monkeypatch.setattr(st, "_get_sheet", lambda: None)


SOURCE_MARKER = "A1 Metrics Dashboard (Google Sheet maintained daily by the spot desk)"


# ----------------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------------

def test_tool_registry():
    tools = get_sheet_tools()
    assert SHEET_TOOL_NAMES == [
        "get_spot_pnl_summary", "get_weekly_spot_pnl", "get_counterparty_pnl", "get_financing_fees",
        "get_nonclient_pnl", "get_a1_client_flow_split", "list_a1_dashboard_tabs", "read_a1_dashboard_range",
    ]
    assert all(t.description for t in tools)
    for t in tools:
        assert "A1 Metrics Dashboard" in t.description
    from tools.desk_tools import DESK_TOOL_NAMES
    from tools.chat_tools import get_chat_tools
    assert not set(SHEET_TOOL_NAMES) & set(DESK_TOOL_NAMES)
    assert not set(SHEET_TOOL_NAMES) & {t.name for t in get_chat_tools()}


def test_prompt_mentions_sheet_tools():
    import chat
    prompt = chat.load_system_prompt()
    assert "Spot desk PnL (A1 Metrics Dashboard sheet)" in prompt
    for name in SHEET_TOOL_NAMES:
        assert name in prompt
    assert "HOLD" in prompt and "A1" in prompt and "Haruko" in prompt


# ----------------------------------------------------------------------------
# unavailable / errors
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda: get_spot_pnl_summary.invoke({}),
    lambda: get_weekly_spot_pnl.invoke({}),
    lambda: get_counterparty_pnl.invoke({}),
    lambda: get_financing_fees.invoke({}),
    lambda: get_nonclient_pnl.invoke({}),
    lambda: list_a1_dashboard_tabs.invoke({}),
    lambda: read_a1_dashboard_range.invoke({"tab": "db", "a1_range": "A1:B2"}),
])
def test_tools_report_unavailable(no_sheet, call):
    assert call() == UNAVAILABLE


def test_403_is_relayed_not_raised(monkeypatch):
    s = make_sheet(FakeSession(status_override=403))
    monkeypatch.setattr(st, "_get_sheet", lambda: s)
    out = get_spot_pnl_summary.invoke({})
    assert out.startswith("Cannot read the A1 Metrics Dashboard sheet")
    assert "403" in out and "spreadsheets.readonly" in out


def test_404_is_relayed(monkeypatch):
    s = make_sheet(FakeSession(status_override=404))
    monkeypatch.setattr(st, "_get_sheet", lambda: s)
    out = get_weekly_spot_pnl.invoke({"weeks": 4})
    assert "404" in out and "not shared" in out


def test_unexpected_exception_is_reported(monkeypatch):
    class Broken:
        def monthly_volume_pnl(self, year=None):
            raise KeyError("boom")
    monkeypatch.setattr(st, "_get_sheet", lambda: Broken())
    out = get_spot_pnl_summary.invoke({})
    assert out.startswith("Error reading the A1 Metrics Dashboard sheet in get_spot_pnl_summary: KeyError")


def test_sheets_unavailable_exception_is_relayed(monkeypatch):
    class NoCreds:
        def weekly_pnl(self, include_future=False):
            raise SheetsUnavailable("no ADC; run gcloud auth application-default login")
    monkeypatch.setattr(st, "_get_sheet", lambda: NoCreds())
    out = get_weekly_spot_pnl.invoke({})
    assert out.startswith("Cannot read the A1 Metrics Dashboard sheet: no ADC")


# ----------------------------------------------------------------------------
# get_spot_pnl_summary
# ----------------------------------------------------------------------------

def test_spot_pnl_summary_shape(sheet):
    out = get_spot_pnl_summary.invoke({})
    assert out.startswith("**Spot desk volume & PnL 2026**")
    assert "Data as of: **2026-09-11**" in out and SOURCE_MARKER in out and "tab 'Volume & PNL'" in out
    assert "not the Haruko mark-to-market PnL" in out
    # MTD = latest populated month (Feb in the canned tab)
    assert "**MTD (Feb 2026, latest populated month):**" in out
    assert "- HOLD: PnL $320,000 on volume $400,000,000 (8.00 bps)" in out
    assert "- A1: PnL $1,200,000 on volume $1,500,000,000 (8.00 bps)" in out       # back-filled take rate
    # YTD block with target
    assert "- A1: PnL $2,600,000 on volume $3,500,000,000 (7.43 bps)" in out
    assert "- TOTAL: PnL $3,420,000 on volume $4,900,000,000 (6.98 bps); target $60,000,000, -94.3% vs target" in out
    assert "A1 share of YTD PnL: 76.0%; HOLD share: 24.0%" in out
    # monthly table: only populated months, blank months omitted
    assert "| Jan | $1,000,000,000 | $500,000 | 5.00 bps | $2,000,000,000 | $1,400,000 | 7.00 bps | $3,000,000,000 | $1,900,000 | 6.33 bps |" in out
    assert "| Mar |" not in out and "| Apr |" not in out
    assert "Cumulative through Feb: volume $4,900,000,000, PnL $3,420,000, target $10,000,000 (-65.8% vs target)" in out


def test_spot_pnl_summary_2025_archive(sheet):
    out = get_spot_pnl_summary.invoke({"year": 2025})
    assert "**Spot desk volume & PnL 2025**" in out
    assert "Data as of: **2025-12-31**" in out and "tab '2025 Volume & PNL'" in out
    assert "MTD (Dec 2025" in out
    assert "- TOTAL: PnL $28,800,000 on volume $30,000,000,000 (9.60 bps); target $25,000,000, 15.2% vs target" in out


def test_spot_pnl_summary_empty_tab(monkeypatch):
    tabs = dict(TABS); tabs["Volume & PNL"] = [["Data as of ", "2026-09-11"], ["nothing"]]
    s = make_sheet(FakeSession(tabs=tabs))
    monkeypatch.setattr(st, "_get_sheet", lambda: s)
    out = get_spot_pnl_summary.invoke({})
    assert out.startswith("No monthly rows found on tab 'Volume & PNL'") and "2026-09-11" in out


# ----------------------------------------------------------------------------
# get_weekly_spot_pnl
# ----------------------------------------------------------------------------

def test_weekly_spot_pnl_shape(sheet):
    out = get_weekly_spot_pnl.invoke({"weeks": 3})
    assert out.startswith("**Spot desk weekly PnL - last 3 week(s)** (USD)")
    assert "Data as of: **2026-09-11**" in out and "tab 'Weekly PNL'" in out
    assert "| Week | Dates | HOLD PnL | A1 realised | A1 unrealised (approx) | A1 total | HOLD + A1 |" in out
    assert "| 36 | 2026-09-04 to 2026-09-10 | $3,800 | $290,000 | $285,000 | $575,000 | $578,800 |" in out
    assert "| 35 | 2026-08-28 to 2026-09-03 | $12,000 | $1,400,000 | $240,000 | $1,640,000 | $1,652,000 |" in out
    assert "| 2 |" in out and "| 1 |" not in out.split("| Week |")[1]
    assert "| 37 |" not in out                                            # future / in-progress zero week dropped
    assert "Sum over these 3 week(s): HOLD $495,800, A1 $2,257,000 (realised $1,710,000, unrealised $547,000), combined $2,752,800" in out
    assert "Sheet YTD totals (all weeks): HOLD $765,800, A1 $2,522,000 (realised $1,900,000, unrealised $622,000), combined $3,287,801." in out  # $765,800.5 rounds half-to-even
    assert "partial week" not in out                                        # week 36 ended before the as-of date


def test_weekly_spot_pnl_clips_and_defaults(sheet):
    out = get_weekly_spot_pnl.invoke({"weeks": 999})
    assert "last 4 week(s)" in out           # only 4 populated weeks exist
    out = get_weekly_spot_pnl.invoke({"weeks": 0})
    assert "last 1 week(s)" in out
    out = get_weekly_spot_pnl.invoke({})
    assert "last 4 week(s)" in out


def test_weekly_spot_pnl_partial_week_note(monkeypatch):
    tabs = dict(TABS)
    rows = [list(r) for r in TABS["Weekly PNL"]]
    rows[0] = ["Data as of ", "2026-09-08"]       # as-of falls inside week 36 -> partial
    tabs["Weekly PNL"] = rows
    s = make_sheet(FakeSession(tabs=tabs))
    monkeypatch.setattr(st, "_get_sheet", lambda: s)
    out = get_weekly_spot_pnl.invoke({"weeks": 2})
    assert "week 36 (2026-09-04 to 2026-09-10) is in progress - partial week" in out


# ----------------------------------------------------------------------------
# get_counterparty_pnl
# ----------------------------------------------------------------------------

def test_counterparty_pnl_by_counterparty(sheet):
    out = get_counterparty_pnl.invoke({"days": 30, "top_n": 15})
    assert out.startswith("**Spot PnL by counterparty** - last 30 day(s)")
    assert "Data as of: **2024-12-31**" in out and "tab 'db'" in out
    assert "Blotter coverage on this tab: 2024-06-01 to 2024-12-31; window = last 30 day(s) ending at the latest trade in the blotter (2024-12-31)" in out
    assert "CAVEAT: the trade-level blotter stops at 2024-12-31 while the dashboard is as of 2026-09-11" in out
    assert "| Counterparty | Trades | Notional | PnL | PnL share | Avg bps (vol-wtd) | Last trade |" in out
    # sorted by PnL desc: Gamma 2,800 > Alpha 640 > Beta 333 > Delta 0
    assert "| Gamma Technologies Inc. | 1 | $1,000,000 | $2,800 | 74.2% | 28.08 bps | 2024-12-30 |" in out
    assert out.index("| Gamma Technologies Inc. |") < out.index("| Alpha Fund LLC |") < out.index("| Beta Growth LP |") < out.index("| Delta Investments Inc. [A1] |")
    assert "| Alpha Fund LLC | 1 | $32,000 | $640.00 | 17.0% | 200.00 bps | 2024-12-31 |" in out   # < $1,000 -> 2 decimals
    assert "| Delta Investments Inc. [A1] | 1 | $6,000 | $0 | 0.0% | 0.00 bps | 2024-12-31 |" in out
    assert "4 trades, 4 counterparty(s); total PnL $3,773" in out
    assert "window 2024-12-30 to 2024-12-31" in out


def test_counterparty_pnl_by_symbol_and_side(sheet):
    out = get_counterparty_pnl.invoke({"days": 0, "by": "symbol"})           # days<=0 -> whole blotter
    assert "**Spot PnL by symbol** - full blotter" in out
    assert "| Symbol | Trades |" in out
    assert "| BTC/USD | 1 |" in out and "| ETH/USD | 1 | $50,000 | n/a |" in out   # blank PnL stays n/a
    assert "6 trades, 6 symbol(s)" in out
    out = get_counterparty_pnl.invoke({"days": 365, "by": "side", "top_n": 1})
    assert "| Side | Trades |" in out
    assert "| BUY | 4 |" in out and "... 1 more side(s)" in out
    # volume-weighted average bps for BUY: (640*32000 + 333*... ) -- check the weighting differs from the simple mean
    buy_line = next(l for l in out.splitlines() if l.startswith("| BUY |"))
    assert "bps" in buy_line and "87.00 bps" not in buy_line               # simple mean of (200, 90, 28.08, 30) = 87.02


def test_counterparty_pnl_filter_and_bad_by(sheet):
    out = get_counterparty_pnl.invoke({"days": 365, "counterparty": "alpha"})
    assert "Filter: counterparty contains 'alpha' -> Alpha Fund LLC" in out
    assert "| Symbol | Trades |" in out                                     # filtered view groups by symbol
    assert "| BTC/USD | 1 |" in out and "| ETH/USD | 1 |" in out
    out = get_counterparty_pnl.invoke({"days": 365, "counterparty": "nobody"})
    assert "No trades for counterparties matching 'nobody'" in out
    out = get_counterparty_pnl.invoke({"by": "venue"})
    assert out.startswith("Unknown `by`='venue'")


def test_counterparty_pnl_empty_blotter(monkeypatch):
    tabs = dict(TABS); tabs["db"] = [TABS["db"][0]]
    s = make_sheet(FakeSession(tabs=tabs))
    monkeypatch.setattr(st, "_get_sheet", lambda: s)
    out = get_counterparty_pnl.invoke({})
    assert "No trades in that window." in out and "Blotter coverage on this tab: n/a to n/a" in out


# ----------------------------------------------------------------------------
# get_financing_fees / get_nonclient_pnl
# ----------------------------------------------------------------------------

def test_financing_fees_shape(sheet):
    out = get_financing_fees.invoke({"months": 2})
    assert out.startswith("**HOLD financing fees** (USD)")
    assert "Data as of: **n/a (tab has no 'Data as of' cell); dashboard 'Volume & PNL' tab is as of 2026-09-11**" in out
    assert "tab 'Financing Fees'" in out
    assert "| Feb | $60,000 | $0 | $60,000 |" in out and "| Mar | $45,000 | $1,000 | $46,000 |" in out
    assert "| Jan |" not in out
    assert "Sum of shown months: fees $105,000, delta sales $1,000, total $106,000. Year total (all populated months): $146,000; latest populated month: Mar." in out
    assert "Weekly block (populated weeks, last 6):" in out and "| 9 | $60,000 | $0 | $60,000 |" in out


def test_financing_fees_empty(monkeypatch):
    tabs = dict(TABS); tabs["Financing Fees"] = [TABS["Financing Fees"][0]]
    s = make_sheet(FakeSession(tabs=tabs))
    monkeypatch.setattr(st, "_get_sheet", lambda: s)
    assert "No populated months on this tab." in get_financing_fees.invoke({})


def test_nonclient_pnl_unpopulated_tab(sheet):
    out = get_nonclient_pnl.invoke({})
    assert out.startswith("**Non-client spot PnL**")
    assert "tab 'Nonclient PNL'" in out
    assert "is not populated in this sheet (it is titled 'HOLD PNL' and points to a separate spreadsheet: https://docs.google.com/spreadsheets/d/other-sheet/" in out
    assert "no access to the linked sheet" in out


def test_nonclient_pnl_with_table(monkeypatch):
    tabs = dict(TABS)
    tabs["Nonclient PNL"] = [["Data as of ", "2026-09-11"], ["Month", "Non Client PNL", "Notes"], [1, 1_000.5, "a"], [2, -200.0, ""], [3, 300.0, ""]]
    s = make_sheet(FakeSession(tabs=tabs))
    monkeypatch.setattr(st, "_get_sheet", lambda: s)
    out = get_nonclient_pnl.invoke({"months": 2})
    assert "Data as of: **2026-09-11**" in out
    assert "| month | non_client_pnl | notes |" in out
    assert "| 2 | -$200.00 |  |" in out and "| 3 | $300.00 |  |" in out and "| 1 |" not in out   # month column is not money


# ----------------------------------------------------------------------------
# list_a1_dashboard_tabs / read_a1_dashboard_range
# ----------------------------------------------------------------------------

def test_list_tabs(sheet):
    out = list_a1_dashboard_tabs.invoke({})
    assert out.startswith("**A1 Metrics Dashboard** - 7 tab(s)")
    assert "get_a1_client_flow_split" in out
    assert "| Volume & PNL | 1000 x 29 | get_spot_pnl_summary |" in out
    assert "| 2025 Volume & PNL | 1000 x 29 | get_spot_pnl_summary(year=2025) |" in out
    assert "| db | 2972 x 26 | get_counterparty_pnl (trade blotter) |" in out
    assert f"capped at {MAX_RANGE_ROWS} rows x {MAX_RANGE_COLS} columns" in out


def test_read_range_basic(sheet):
    out = read_a1_dashboard_range.invoke({"tab": "Volume & PNL", "a1_range": "L3:S5"})
    assert out.startswith("**'Volume & PNL'!L3:S5**") and "(clamped" not in out
    assert "Data as of: **2026-09-11**" in out                            # as-of found in the served rows
    header = next(l for l in out.splitlines() if l.startswith("|  |"))
    assert header.startswith("|  | L | M | N |")
    assert "| 3 | Data as of  | 2026-09-11 |" in out                        # fake serves the whole tab; row numbering starts at 3
    assert "numbers are unformatted sheet values" in out


def test_read_range_cell_cap(sheet):
    sess = sheet._session
    out = read_a1_dashboard_range.invoke({"tab": "db", "a1_range": "A:M"})
    assert out.startswith("**'db'!A1:M200** (clamped to 200 x 30)")
    assert any(c["url"].endswith("%27db%27%21A1%3AM200") for c in sess.calls)   # the clamped range is what hits the API
    assert not any("A%3AM" in c["url"] or "AZ999" in c["url"] for c in sess.calls)  # the unclamped one never does
    out = read_a1_dashboard_range.invoke({"tab": "db", "a1_range": "A1:AZ999"})
    assert "**'db'!A1:AD200** (clamped to 200 x 30)" in out
    assert any(c["url"].endswith("%27db%27%21A1%3AAD200") for c in sess.calls)


def test_read_range_tab_in_range_and_validation(sheet):
    out = read_a1_dashboard_range.invoke({"tab": "Weekly PNL", "a1_range": "'Financing Fees'!A1:D3"})
    assert out.startswith("**'Financing Fees'!A1:D3**")                     # explicit tab inside the range wins
    assert read_a1_dashboard_range.invoke({"tab": "  ", "a1_range": "A1:B2"}).startswith("tab is required")
    out = read_a1_dashboard_range.invoke({"tab": "db", "a1_range": "!!"})
    assert out.startswith("Error reading the A1 Metrics Dashboard sheet in read_a1_dashboard_range: ValueError")
    out = read_a1_dashboard_range.invoke({"tab": "Nope", "a1_range": "A1:B2"})
    assert "Cannot read the A1 Metrics Dashboard sheet" in out and "400" in out


def test_read_range_empty(monkeypatch):
    tabs = dict(TABS); tabs["Empty"] = []
    s = make_sheet(FakeSession(tabs=tabs))
    monkeypatch.setattr(st, "_get_sheet", lambda: s)
    out = read_a1_dashboard_range.invoke({"tab": "Empty", "a1_range": "A1:B2"})
    assert "(empty range)" in out


def test_read_range_never_writes(sheet):
    read_a1_dashboard_range.invoke({"tab": "db", "a1_range": "A1:M5"})
    for call in sheet._session.calls:
        assert "/values/" in call["url"] or call["url"].endswith("sheet-id-123")
        assert ":append" not in call["url"] and ":batchUpdate" not in call["url"]

# ----------------------------------------------------------------------------
# get_a1_client_flow_split
# ----------------------------------------------------------------------------

ANCHOR = pd.Timestamp("2026-09-11")     # the canned dashboard's "Data as of" (a Friday)


def _weeks(sheet):
    return sheet.weekly_pnl(include_future=True)


def test_resolve_window_shortcuts_anchor_on_data_as_of(sheet):
    wk = _weeks(sheet)
    d = lambda s: pd.Timestamp(s)  # noqa: E731
    assert resolve_client_flow_window("last week", None, ANCHOR, wk) == (d("2026-09-04"), d("2026-09-10"), "last week = dashboard week 36")
    s_, e_, label = resolve_client_flow_window("This Week", None, ANCHOR, wk)
    assert (s_, e_) == (d("2026-09-11"), d("2026-09-17")) and label.startswith("this week = dashboard week 37")
    assert resolve_client_flow_window("mtd", None, ANCHOR, wk)[:2] == (d("2026-09-01"), d("2026-09-11"))
    assert resolve_client_flow_window("month to date", None, ANCHOR, wk)[2] == "month to date"
    assert resolve_client_flow_window("YTD", None, ANCHOR, wk) == (d("2026-01-01"), d("2026-09-11"), "year to date")
    assert resolve_client_flow_window("last month", None, ANCHOR, wk) == (d("2026-08-01"), d("2026-08-31"), "last month")
    assert resolve_client_flow_window("last 7 days", None, ANCHOR, wk) == (d("2026-09-05"), d("2026-09-11"), "last 7 days")
    assert resolve_client_flow_window("past 30 days", "ignored", ANCHOR, wk)[0] == d("2026-08-13")
    assert resolve_client_flow_window("yesterday", None, ANCHOR, wk) == (d("2026-09-10"), d("2026-09-10"), "yesterday")
    # explicit dates: single day, pair, reversed pair, 'today' as end
    assert resolve_client_flow_window("2026-09-04", None, ANCHOR, wk) == (d("2026-09-04"), d("2026-09-04"), "2026-09-04")
    assert resolve_client_flow_window("2026-09-04", "2026-09-10", ANCHOR, wk)[2] == "2026-09-04 to 2026-09-10"
    assert resolve_client_flow_window("2026-09-10", "2026-09-04", ANCHOR, wk)[:2] == (d("2026-09-04"), d("2026-09-10"))
    assert resolve_client_flow_window("Sep 4, 2026", "today", ANCHOR, wk)[:2] == (d("2026-09-04"), ANCHOR)
    # wall clock never used: a different anchor moves every shortcut
    other = pd.Timestamp("2026-03-15")
    assert resolve_client_flow_window("mtd", None, other, wk)[:2] == (d("2026-03-01"), other)
    with pytest.raises(ValueError, match="start_date"):
        resolve_client_flow_window("whenever", None, ANCHOR, wk)
    with pytest.raises(ValueError, match="end_date"):
        resolve_client_flow_window("2026-09-04", "later", ANCHOR, wk)


def test_resolve_window_week_fallback_without_weekly_tab():
    d = lambda s: pd.Timestamp(s)  # noqa: E731
    assert resolve_client_flow_window("last week", None, ANCHOR, None) == (d("2026-09-04"), d("2026-09-10"), "last week")
    assert resolve_client_flow_window("this week", None, pd.Timestamp("2026-09-09"), None)[:2] == (d("2026-09-04"), d("2026-09-10"))
    # an anchor outside the weekly tab's dates also falls back to Fri-Thu weeks
    assert resolve_client_flow_window("last week", None, pd.Timestamp("2027-06-16"), None)[:2] == (d("2027-06-04"), d("2027-06-10"))


def test_client_flow_split_explicit_week_markdown(sheet):
    out = get_a1_client_flow_split.invoke({"start_date": "2026-09-04", "end_date": "2026-09-10"})
    lines = out.split("\n")
    assert lines[0] == "**A1 spot realised PnL - client flow vs non-client (proprietary) flow**"
    assert lines[1] == "Window: 2026-09-04 to 2026-09-10 - 6 populated day-row(s), sheet rows 15-20."
    assert "Data as of: **2026-09-11**" in lines[2] and "last populated row in 'A1 database' is 2026-09-10" in lines[2]
    assert SOURCE_MARKER in out and "tab 'A1 database' columns R:X" in out and "not the Haruko" in out
    assert "**Totals:** realised $290,000 = client flow $65,000 (22.4%) + non-client flow $225,000 (77.6%) over 6 day(s)" in out
    assert "| Date | Row | Realised total (V) | Client flow (W) | Non-client (X) |" in out
    assert "| 2026-09-10 | 20 | $228,000 | -$18,000 | $246,000 |" in out
    assert "| 2026-09-05 | 15 | -$3,000 | $20,000 | -$23,000 |" in out
    assert "Missing days" in out and "2026-09-04 (Fri)" in out
    assert ("Weekly cross-check: 'Weekly PNL' tab week 36 (2026-09-04 to 2026-09-10) A1 Realized PNL $290,000 - matches "
            "(difference $0); that tab's unrealised approximation for the week is $285,000, not included here.") in out
    assert "Caveats: realised PnL only" in out and "HOLD (client commissions) is not included" in out
    assert "Artefact" not in out and "Not yet populated" not in out


def test_client_flow_split_shortcut_last_week_and_include_daily_false(sheet):
    out = get_a1_client_flow_split.invoke({"start_date": "last week", "include_daily": False})
    assert "Window: 2026-09-04 to 2026-09-10 (last week = dashboard week 36)" in out
    assert "$290,000" in out and "| Date |" not in out
    mtd = get_a1_client_flow_split.invoke({"start_date": "mtd"})
    assert "Window: 2026-09-01 to 2026-09-11 (month to date)" in mtd
    assert "Not yet populated: 2026-09-11 (after the last row)." in mtd
    assert "Two snapshot rows on 2026-09-01 - both summed" in mtd


def test_client_flow_split_discloses_kept_pair_mismatch_and_incomplete_rows(sheet):
    out = get_a1_client_flow_split.invoke({"start_date": "2026-08-28", "end_date": "2026-09-03"})
    assert "**Totals:** realised $797,000" in out
    assert "Note: V does not equal W + X over this window (difference $60,000) - 1 row(s) lack the W/X split: row 13 (2026-09-02)." in out
    assert "| 2026-08-30 * | 9 | $8,000,000 | $7,999,000 | $1,000 |" in out
    assert "Artefact row 9 (2026-08-30): V $8,000,000, W $7,999,000, X $1,000 - kept - both legs in window, they net out." in out
    assert "Artefact row 10 (2026-08-31): V -$7,990,000" in out
    assert "The kept pair(s) net to $10,000" in out
    assert "Weekly cross-check: 'Weekly PNL' tab week 35 (2026-08-28 to 2026-09-03) A1 Realized PNL $1,400,000 - DIFFERS (difference -$603,000)" in out
    assert "Including the excluded" not in out


def test_client_flow_split_excludes_lone_leg(sheet):
    out = get_a1_client_flow_split.invoke({"start_date": "2026-08-27", "end_date": "2026-08-30"})
    assert "**Totals:** realised $35,000 = client flow $25,000 (71.4%) + non-client flow $10,000 (28.6%) over 3 day(s)" in out
    assert "Artefact row 9 (2026-08-30): V $8,000,000, W $7,999,000, X $1,000 - excluded - offsetting leg is outside the window." in out
    assert "Including the excluded leg(s) the raw sums would be V $8,035,000, W $8,024,000, X $11,000." in out
    assert "| 2026-08-30" not in out and "Weekly cross-check" not in out


def test_client_flow_split_long_window_collapses_to_months(sheet, monkeypatch):
    monkeypatch.setattr(st, "MAX_DAILY_ROWS", 5)
    out = get_a1_client_flow_split.invoke({"start_date": "ytd"})
    assert "Window: 2026-01-01 to 2026-09-11 (year to date)" in out
    assert "| Month | Days | Realised total (V) | Client flow (W) | Non-client (X) |" in out
    assert "| 2026-08 | 5 |" in out and "| 2026-09 | 9 |" in out          # distinct dates (09-01 has two rows)
    assert "collapsed to monthly subtotals" in out
    assert "YTD cross-check: the sheet's own cumulative YTD columns (S/T/U) at 2026-09-10 read total $4,087,000, client $2,643,000, non-client $1,444,000" in out


def test_client_flow_split_single_day_empty_and_bad_input(sheet):
    one = get_a1_client_flow_split.invoke({"start_date": "2026-09-10"})
    assert "Window: 2026-09-10 - 1 populated day-row(s), sheet rows 20-20." in one
    assert "**Totals:** realised $228,000 = client flow -$18,000 (-7.9%) + non-client flow $246,000 (107.9%)" in one
    assert "one leg is negative" in one
    empty = get_a1_client_flow_split.invoke({"start_date": "2026-10-01", "end_date": "2026-10-03"})
    assert "No populated rows in that window; the tab covers 2026-08-27 to 2026-09-10." in empty
    assert "Days after the last populated row: 2026-10-01 to 2026-10-03." in empty
    bad = get_a1_client_flow_split.invoke({"start_date": "sometime"})
    assert bad.startswith("Could not understand start_date='sometime'") and "last week" in bad


def test_client_flow_split_unavailable_and_errors(no_sheet, monkeypatch):
    assert get_a1_client_flow_split.invoke({"start_date": "last week"}) == UNAVAILABLE
    s = make_sheet(session=FakeSession(status_override=403))
    monkeypatch.setattr(st, "_get_sheet", lambda: s)
    out = get_a1_client_flow_split.invoke({"start_date": "last week"})
    assert out.startswith("Cannot read the A1 Metrics Dashboard sheet") and "403" in out


def test_client_flow_split_docstring_routes_prop_flow_questions():
    doc = " ".join(get_a1_client_flow_split.description.split())
    for word in ("client flow", "non-client", "proprietary", "prop", "flow attribution", "REALISED", "A1 database", "last week", "mtd", "ytd"):
        assert word in doc, word
    assert "never read raw cells" in doc
