"""providers/gsheets.py with canned Sheets API payloads. No network, no GCP.

The `values` payloads mirror the real layouts observed on 2026-09-11 (merged group
headers, blank spacer columns, "Data as of" cell, future months / weeks blank or 0,
repeated week numbers at year end) with the numbers changed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import pytest

from providers import gsheets
from providers.gsheets import (
    A1MetricsSheet,
    SheetsAccessError,
    SheetsUnavailable,
    clamp_a1_range,
    parse_a1_range,
    parse_financing_fees,
    parse_generic_table,
    parse_monthly_volume_pnl,
    parse_sheet_date,
    parse_trades,
    parse_weekly_pnl,
    to_float,
)

SID = "sheet-id-123"

# ----------------------------------------------------------------------------
# canned tabs (real shapes, altered numbers)
# ----------------------------------------------------------------------------

VOLUME_PNL = [
    ["Data as of ", "2026-09-11"],
    ["Updated by Joao", "HOLD", "", "", "", "", "A1", "", "", "", "", "TOTAL"],
    ["", "Month", "Volume", "PNL", "Take Rate", "", "Month", "Volume", "PNL", "Take Rate", "",
     "Month", "Volume", "PNL", "Take Rate", "Cumulative Volume", "Cumulative PNL", "Target PNL", "% of Target"],
    ["", 1, 1_000_000_000, 500_000, 5.0, "", 1, 2_000_000_000, 1_400_000, 7.0, "",
     1, 3_000_000_000, 1_900_000, 6.333333, 3_000_000_000, 1_900_000, 5_000_000, -0.62],
    ["", 2, 400_000_000, 320_000, 8.0, "", 2, 1_500_000_000, 1_200_000, "", "",
     2, 1_900_000_000, 1_520_000, 8.0, 4_900_000_000, 3_420_000, 10_000_000, -0.658],
    ["", 3, "", 0, "", "", 3, 0, "", "", "", 3, "", "", "", "", "", 15_000_000],
    ["", 4, "", 0, "", "", 4, 0, "", "", "", 4, "", "", "", "", "", 20_000_000],
    ["", "Total YTD", 1_400_000_000, 820_000, 5.857143, "", "Total YTD", 3_500_000_000, 2_600_000, 7.428571, "",
     "Total YTD", 4_900_000_000, 3_420_000, 6.979592, 4_900_000_000, 3_420_000, 60_000_000, -0.943],
]

VOLUME_PNL_2025 = [
    ["Data as of 12/31/25"],
    ["", "HOLD", "", "", "", "", "A1", "", "", "", "", "TOTAL"],
    ["", "Month", "Volume", "PNL", "Take Rate", "", "Month", "Volume", "PNL", "Take Rate", "",
     "Month", "Volume", "PNL", "Take Rate", "Cumulative Volume", "Cumulative PNL", "Target PNL", "% of Target"],
    ["", 1, "", 900_000, "", "", 1, 600_000_000, 2_000_000, 33.3, "", 1, 600_000_000, 2_900_000, 48.3, 600_000_000, 2_900_000, 2_083_333, 0.39],
    ["", 12, 1_600_000_000, 1_200_000, 7.5, "", 12, 1_900_000_000, 800_000, 4.2, "", 12, 3_500_000_000, 2_000_000, 5.7, 30_000_000_000, 29_000_000, 25_000_000, 0.16],
    ["", "Total YTD", 12_000_000_000, 11_000_000, 9.17, "", "Total YTD", 18_000_000_000, 17_800_000, 9.89, "",
     "Total YTD", 30_000_000_000, 28_800_000, 9.6, 30_000_000_000, 28_800_000, 25_000_000, 0.152],
]

WEEKLY_PNL = [
    ["Data as of ", "2026-09-11"],
    ["Updated by Joao", "HOLD", "", "", "A1", "", "", "", "", "A1 + HOLD "],
    ["", "Week", "PNL", "", "Week", "Realized PNL", "Unrealized PNL Approximation", "Total", "", "Week", "PNL", "Start Date", "End Date"],
    ["", 1, 270_000.5, "", 1, 190_000.0, 75_000.25, 265_000.25, "", 1, 535_000.75, "Jan 1, 2026 (Thu)", "Jan 8, 2026 (Thu)"],
    ["", 2, 480_000.0, "", 2, 20_000.0, 22_000.0, 42_000.0, "", 2, 522_000.0, "Jan 9, 2026 (Fri)", "Jan 15, 2026 (Thu)"],
    ["", 35, 12_000.0, "", 35, 1_400_000.0, 240_000.0, 1_640_000.0, "", 35, 1_652_000.0, "Aug 28, 2026 (Fri)", "Sep 3, 2026 (Thu)"],
    ["", 36, 3_800.0, "", 36, 290_000.0, 285_000.0, 575_000.0, "", 36, 578_800.0, "Sep 4, 2026 (Fri)", "Sep 10, 2026 (Thu)"],
    ["", 37, 0, "", 37, 0, "", 0, "", 37, 0, "Sep 11, 2026 (Fri)", "Sep 17, 2026 (Thu)"],
    ["", 38, 0, "", 38, 0, "", 0, "", 38, 0, "Sep 18, 2026 (Fri)", "Sep 24, 2026 (Thu)"],
    ["", 49, 0, "", 49, 0, "", 0, "", 50, 0, "Dec 11, 2026 (Fri)", "Dec 17, 2026 (Thu)"],
    ["", 49, 0, "", 49, 0, "", 0, "", 50, 0, "Dec 18, 2026 (Fri)", "Dec 24, 2026 (Thu)"],
    ["", "Total", 765_800.5, "", "Total", 1_900_000.0, 622_000.25, 2_522_000.25, "", "Total", 3_287_800.75],
]

FINANCING_FEES = [
    ["Month", "HOLD Financing Fees", "HOLD Delta Sales", "Total", "", "", "Week", "HOLD Financing Fees", "HOLD Delta Sales", "Total"],
    [1, 40_000, "", 40_000, "", "", 1, 0, 0, 0],
    [2, 60_000, "", 60_000, "", "", 2, 0, 0, 0],
    [3, 45_000, 1_000, 46_000, "", "", 3, 40_000, 0, 40_000],
    [4, "", "", 0, "", "", 4, 0, 0, 0],
    [5, "", "", 0, "", "", 5, 0, 0, 0],
    ["", "", "", "", "", "", 9, 60_000, 0, 60_000],
    ["", "", "", "", "", "", 10, "", "", 0],
]

NONCLIENT_PNL = [
    ["", "HOLD PNL"],
    ["", "https://docs.google.com/spreadsheets/d/other-sheet/edit?gid=0#gid=0"],
]

TRADES = [
    ["Date (UTC)", "Counterparty", "Side", "Symbol", "Buy QTY", "Buy Asset", "Sell QTY", "Sell Asset", "Price", "PNL", "Currency", "bps", "Month"],
    ["2024-12-31 13:17:53", "Alpha Fund LLC", "BUY", "BTC/USD", 0.34, "BTC", 32_000.0, "USD", 94_000.0, 640.0, "USD", 200.0, 12],
    ["2024-12-31 16:40:01", "Beta Growth LP", "BUY", "XYZ/USD", 100_000, "XYZ", 37_000, "USD", 0.37, 333.0, "USD", 90.0, 12],
    ["2024-12-30 22:49:59", "Gamma Technologies Inc.", "BUY", "USDC.e/USD", 1_000_000.5, "USDC.e", 997_000.0, "USD", 0.997, 2_800.0, "USD", 28.08, 12],
    ["2024-12-31 4:57:50", "Delta Investments Inc. [A1]", "Sell", "WLD/USD", 6_000.0, "USD", 2_900.0, "WLD", 2.09, 0, "USD", 0, 12],
    ["2024-11-15 12:00:00", "Alpha Fund LLC", "SELL", "ETH/USD", 50_000.0, "USD", 20.0, "ETH", 2_500.0, "", "USD", 12.5, 11],
    ["2024-06-01 09:00:00", "Beta Growth LP", "BUY", "SOL/USD", 1_000, "SOL", 150_000, "USD", 150.0, 450.0, "USD", 30.0, 6],
    ["", "", "", "", "", "", "", "", "", "", "", "", ""],
    ["not a date", "Broken Row", "BUY", "BTC/USD", 1, "BTC", 1, "USD", 1, 1, "USD", 1, 1],
]

# 'A1 database'!R:X as the API returns it (the FakeSession serves this for any range on
# the tab): R1:S9 asset list + T:X headers, undated history rows (one artefact pair among
# them), then dated rows with one Google serial date, one missing day (2026-09-04), two
# snapshots on 2026-09-01, an offsetting pair on 08-30/08-31, a row without the W/X
# split (09-02), and trailing undated cumulative / total rows.
A1_DATABASE = [
    ["FIAT/STABLES", 5_348_119.1, "Client Flow PNL", "Non Client Flow PNL", "Change in Total PNL",
     "Change in Client Flow PNL", "Change in non Client Flow PNL"],
    ["BTC", 2_263_820, 4_035_722.69, 859_520.08],
    ["", "", 5_355_690.55, 1_179_441.43, 10_005_588.66, 10_001_359.63, 4_229.03],
    ["", "", 5_387_868.1, 1_178_685.72, -9_969_937.79, -9_969_182.08, -755.71],
    [],
    ["2026-08-27 22:10:24", 3_000_000.0, 1_800_000.0, 1_200_000.0, 10_000.0, 6_000.0, 4_000.0],
    [46262, 3_020_000.0, 1_815_000.0, 1_205_000.0, 20_000.0, 15_000.0, 5_000.0],           # serial = 2026-08-28
    ["2026-08-29 22:10:24", 3_025_000.0, 1_819_000.0, 1_206_000.0, 5_000.0, 4_000.0, 1_000.0],
    ["2026-08-30 22:10:24", 11_025_000.0, 9_818_000.0, 1_207_000.0, 8_000_000.0, 7_999_000.0, 1_000.0],
    ["2026-08-31 22:11:39", 3_035_000.0, 1_827_000.0, 1_208_000.0, -7_990_000.0, -7_991_000.0, 1_000.0],
    ["2026-09-01 8:01:34", 3_038_000.0, 1_830_000.0, 1_208_000.0, 3_000.0, 3_000.0, 0],
    ["2026-09-01 22:10:24", 3_037_000.0, 1_828_000.0, 1_209_000.0, -1_000.0, -2_000.0, 1_000.0],
    ["2026-09-02 22:10:25", 3_097_000.0, 1_868_000.0, 1_229_000.0, 60_000.0],
    ["2026-09-03 22:10:23", 3_797_000.0, 2_578_000.0, 1_219_000.0, 700_000.0, 710_000.0, -10_000.0],
    ["2026-09-05 22:10:24", 3_794_000, 2_598_000.0, 1_196_000.0, -3_000.0, 20_000.0, -23_000.0],
    ["2026-09-06 22:10:24", 3_798_000.0, 2_602_000.0, 1_196_000.0, 4_000.0, 4_000.0, 0],
    ["2026-09-07 22:10:23", 3_918_000.0, 2_721_000.0, 1_197_000.0, 120_000.0, 119_000.0, 1_000.0],
    ["2026-09-08 22:10:24", 3_852_000.0, 2_658_000.0, 1_194_000.0, -66_000.0, -63_000.0, -3_000.0],
    ["2026-09-09 22:10:25", 3_859_000.0, 2_661_000.0, 1_198_000.0, 7_000.0, 3_000.0, 4_000.0],
    ["2026-09-10 22:12:37", 4_087_000.0, 2_643_000.0, 1_444_000.0, 228_000.0, -18_000.0, 246_000.0],
    ["", 4_087_000.0, 2_643_000.0, 1_444_000.0],
    ["Total", 4_087_000.0, 2_643_000.0, 1_444_000.0, 1_087_000.0, 843_000.0, 244_000.0],
]

META = {
    "properties": {"title": "A1 Metrics Dashboard"},
    "sheets": [
        {"properties": {"sheetId": 1432313692, "title": "Volume & PNL", "gridProperties": {"rowCount": 1000, "columnCount": 29}}},
        {"properties": {"sheetId": 446825400, "title": "Weekly PNL", "gridProperties": {"rowCount": 1000, "columnCount": 25}}},
        {"properties": {"sheetId": 1918358161, "title": "2025 Volume & PNL", "gridProperties": {"rowCount": 1000, "columnCount": 29}}},
        {"properties": {"sheetId": 1929369812, "title": "Financing Fees", "gridProperties": {"rowCount": 974, "columnCount": 25}}},
        {"properties": {"sheetId": 1646520473, "title": "Nonclient PNL", "gridProperties": {"rowCount": 1000, "columnCount": 26}}},
        {"properties": {"sheetId": 200816853, "title": "db", "gridProperties": {"rowCount": 2972, "columnCount": 26}}},
        {"properties": {"sheetId": 1339723141, "title": "A1 database", "gridProperties": {"rowCount": 1000, "columnCount": 41}}},
    ],
}

TABS: Dict[str, List[List[Any]]] = {
    "Volume & PNL": VOLUME_PNL,
    "2025 Volume & PNL": VOLUME_PNL_2025,
    "Weekly PNL": WEEKLY_PNL,
    "Financing Fees": FINANCING_FEES,
    "Nonclient PNL": NONCLIENT_PNL,
    "db": TRADES,
    "A1 database": A1_DATABASE,
}


# ----------------------------------------------------------------------------
# fakes: credentials + requests.Session
# ----------------------------------------------------------------------------

class FakeCreds:
    def __init__(self, token="tok-1", expired=False, fail_refresh=False):
        self.token = token
        self.expired = expired
        self.fail_refresh = fail_refresh
        self.refreshes = 0

    def refresh(self, request):
        if self.fail_refresh:
            raise RuntimeError("invalid_grant: Token has been expired or revoked")
        self.refreshes += 1
        self.expired = False
        self.token = f"tok-{self.refreshes + 1}"


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Serves the canned tabs; records every request (url, params, headers)."""

    def __init__(self, tabs=TABS, meta=META, status_override=None):
        self.tabs = tabs
        self.meta = meta
        self.status_override = status_override   # int or callable(url) -> Optional[int]
        self.calls: List[Dict[str, Any]] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        override = self.status_override(url) if callable(self.status_override) else self.status_override
        if override:
            return FakeResponse(override, {"error": {"code": override, "message": f"fake {override}", "status": "X"}})
        from urllib.parse import unquote
        if "/values/" in url:
            rng = unquote(url.split("/values/", 1)[1])
            tab = rng.split("!", 1)[0].strip("'").replace("''", "'")
            if tab not in self.tabs:
                return FakeResponse(400, {"error": {"code": 400, "message": f"Unable to parse range: {rng}", "status": "INVALID_ARGUMENT"}})
            return FakeResponse(200, {"range": rng, "majorDimension": "ROWS", "values": self.tabs[tab]})
        return FakeResponse(200, self.meta)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_sheet(session=None, creds=None, clock=None, ttl=300, quota="quota-proj"):
    return A1MetricsSheet(SID, quota, cache_ttl_s=ttl, session=session or FakeSession(),
                          credentials=creds or FakeCreds(), clock=clock or FakeClock())


# ----------------------------------------------------------------------------
# cell helpers
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (1_234.5, 1_234.5), (7, 7.0), ("", None), (None, None), ("#N/A ()", None), ("-", None),
    ("$1,234.50", 1_234.5), ("5.882 bps", 5.882), ("12%", 12.0), ("-0.5453", -0.5453), (True, None),
    ("#VALUE! (Function YEAR ...)", None),
])
def test_to_float(raw, expected):
    got = to_float(raw)
    if expected is None:
        assert got != got  # NaN
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [
    ("Jan 1, 2026 (Thu)", "2026-01-01"), ("Sep 10, 2026 (Thu)", "2026-09-10"), ("2026-09-11", "2026-09-11"),
    ("12/31/25", "2025-12-31"), ("2024-02-12 20:05:00", "2024-02-12"), ("2024-12-31 4:57:50", "2024-12-31"),
    (46_000, "2025-12-09"), ("", None), (None, None), ("not a date", None), (1899, None),
])
def test_parse_sheet_date(raw, expected):
    ts = parse_sheet_date(raw)
    if expected is None:
        assert ts is None
    else:
        assert ts.strftime("%Y-%m-%d") == expected


def test_parse_a1_range_and_clamp():
    assert parse_a1_range("'Weekly PNL'!A1:M40") == ("Weekly PNL", "A1:M40")
    assert parse_a1_range("db!A:M") == ("db", "A:M")
    assert parse_a1_range("a1:b2") == (None, "A1:B2")
    with pytest.raises(ValueError):
        parse_a1_range("!!")
    assert clamp_a1_range("A1:T40", 200, 30) == ("A1:T40", False)
    assert clamp_a1_range("A1:AZ500", 200, 30) == ("A1:AD200", True)
    assert clamp_a1_range("A:M", 200, 30) == ("A1:M200", True)
    assert clamp_a1_range("C5:E", 200, 30) == ("C5:E204", True)
    assert clamp_a1_range("B7", 200, 30) == ("B7:B7", False)
    assert clamp_a1_range("E3:A1", 200, 30) == ("A1:E3", False)   # reversed corners


# ----------------------------------------------------------------------------
# parsers
# ----------------------------------------------------------------------------

def test_parse_monthly_volume_pnl_long_format():
    df = parse_monthly_volume_pnl(VOLUME_PNL, year=2026)
    assert set(df["entity"]) == {"HOLD", "A1", "TOTAL"}
    months = df[~df["is_ytd"]]
    # months 3 and 4 are blank / 0 in every group -> dropped
    assert sorted(months["month"].unique().tolist()) == [1, 2]
    hold1 = months[(months.entity == "HOLD") & (months.month == 1)].iloc[0]
    assert hold1["volume_usd"] == 1_000_000_000 and hold1["pnl_usd"] == 500_000 and hold1["take_rate_bps"] == 5.0
    assert (df["year"] == 2026).all()
    # spacer columns never become fields; TOTAL-only columns are NaN for HOLD / A1
    assert months[months.entity == "HOLD"]["cum_pnl_usd"].isna().all()
    tot1 = months[(months.entity == "TOTAL") & (months.month == 1)].iloc[0]
    assert tot1["cum_volume_usd"] == 3_000_000_000 and tot1["target_pnl_usd"] == 5_000_000
    assert tot1["pct_of_target"] == pytest.approx(-0.62)


def test_parse_monthly_ytd_row():
    df = parse_monthly_volume_pnl(VOLUME_PNL, year=2026)
    ytd = df[df["is_ytd"]].set_index("entity")
    assert len(ytd) == 3 and ytd["month"].isna().all()
    assert ytd.loc["A1", "pnl_usd"] == 2_600_000 and ytd.loc["TOTAL", "volume_usd"] == 4_900_000_000
    assert ytd.loc["TOTAL", "target_pnl_usd"] == 60_000_000


def test_parse_monthly_take_rate_backfilled_when_blank():
    df = parse_monthly_volume_pnl(VOLUME_PNL, year=2026)
    a1_feb = df[(df.entity == "A1") & (df.month == 2)].iloc[0]
    # sheet cell is '' -> pnl / volume * 1e4 = 1.2M / 1.5B * 1e4 = 8.0 bps
    assert a1_feb["take_rate_bps"] == pytest.approx(8.0)
    # sheet-provided take rate is kept verbatim (already in bps)
    hold_feb = df[(df.entity == "HOLD") & (df.month == 2)].iloc[0]
    assert hold_feb["take_rate_bps"] == 8.0


def test_parse_monthly_2025_layout_single_cell_as_of():
    df = parse_monthly_volume_pnl(VOLUME_PNL_2025, year=2025)
    hold_jan = df[(df.entity == "HOLD") & (df.month == 1)].iloc[0]
    assert hold_jan["volume_usd"] != hold_jan["volume_usd"]      # blank volume -> NaN, row kept (PnL present)
    assert hold_jan["pnl_usd"] == 900_000 and hold_jan["take_rate_bps"] != hold_jan["take_rate_bps"]
    assert df[df.is_ytd].set_index("entity").loc["TOTAL", "pnl_usd"] == 28_800_000


def test_parse_monthly_empty_and_unlabeled_groups():
    assert parse_monthly_volume_pnl([], year=2026).empty
    assert parse_monthly_volume_pnl([["Data as of", "2026-01-01"], ["nothing here"]]).empty
    # no group-label row at all -> conventional HOLD, A1, TOTAL order
    rows = [["Month", "Volume", "PNL", "Take Rate", "", "Month", "Volume", "PNL", "Take Rate", "", "Month", "Volume", "PNL", "Take Rate"],
            [1, 10.0, 1.0, 1000.0, "", 1, 20.0, 2.0, 1000.0, "", 1, 30.0, 3.0, 1000.0]]
    df = parse_monthly_volume_pnl(rows, year=2026)
    assert df["entity"].tolist() == ["HOLD", "A1", "TOTAL"]


def test_parse_weekly_pnl_dates_future_weeks_and_total():
    df = parse_weekly_pnl(WEEKLY_PNL, as_of="2026-09-11")
    weeks = df[~df["is_total"]]
    # week 37 starts on the as-of date itself (in progress, all zero) and later weeks are future -> dropped
    assert weeks["week"].tolist() == [1, 2, 35, 36]
    w1 = weeks.iloc[0]
    assert w1["start_date"] == pd.Timestamp("2026-01-01") and w1["end_date"] == pd.Timestamp("2026-01-08")
    assert w1["hold_pnl"] == 270_000.5 and w1["a1_realized"] == 190_000.0 and w1["a1_unrealized"] == 75_000.25
    assert w1["a1_total"] == pytest.approx(265_000.25) and w1["total"] == pytest.approx(535_000.75)
    assert str(weeks["start_date"].dtype).startswith("datetime64")
    tot = df[df["is_total"]]
    assert len(tot) == 1 and pd.isna(tot.iloc[0]["week"])
    assert tot.iloc[0]["total"] == pytest.approx(3_287_800.75) and tot.iloc[0]["a1_total"] == pytest.approx(2_522_000.25)


def test_parse_weekly_pnl_include_future_renumbers_duplicate_weeks():
    df = parse_weekly_pnl(WEEKLY_PNL, as_of="2026-09-11", include_future=True)
    weeks = df[~df["is_total"]]["week"].tolist()
    # sheet repeats week 50 twice at year end; parser keeps a strictly increasing sequence
    assert weeks == [1, 2, 35, 36, 37, 38, 50, 51]


def test_parse_weekly_pnl_without_as_of_drops_all_zero_weeks():
    df = parse_weekly_pnl(WEEKLY_PNL, as_of=None)
    assert df[~df["is_total"]]["week"].tolist() == [1, 2, 35, 36]


def test_parse_weekly_pnl_missing_group_falls_back():
    rows = [["", "HOLD"], ["", "Week", "PNL"], ["", 1, 10.0], ["", 2, 20.0], ["", "Total", 30.0]]
    df = parse_weekly_pnl(rows)
    weeks = df[~df.is_total]
    assert weeks["hold_pnl"].tolist() == [10.0, 20.0]
    assert weeks["total"].tolist() == [10.0, 20.0]        # total derived from HOLD when no combined block
    assert weeks["start_date"].isna().all()
    assert parse_weekly_pnl([]).empty


def test_parse_financing_fees_monthly_and_weekly_blocks():
    m = parse_financing_fees(FINANCING_FEES, block="Month")
    assert m["month"].tolist() == [1, 2, 3]                 # months 4-5 unpopulated -> dropped
    assert m["hold_financing_fees_usd"].tolist() == [40_000.0, 60_000.0, 45_000.0]
    assert m["hold_delta_sales_usd"].tolist() == [0.0, 0.0, 1_000.0]
    assert m["total_usd"].tolist() == [40_000.0, 60_000.0, 46_000.0]
    w = parse_financing_fees(FINANCING_FEES, block="Week")
    assert w["week"].tolist() == [3, 9] and w["total_usd"].tolist() == [40_000.0, 60_000.0]
    assert parse_financing_fees([], block="Month").empty


def test_parse_generic_table_nonclient_tab_is_empty():
    assert parse_generic_table(NONCLIENT_PNL).empty
    rows = [["Title only"], ["Month", "PNL", "Notes"], [1, 100.5, "x"], [2, 200.0, ""], [], [9, 9, 9]]
    df = parse_generic_table(rows)
    assert list(df.columns) == ["month", "pnl", "notes"] and df["pnl"].tolist() == [100.5, 200.0]


def test_parse_trades_typing():
    df = parse_trades(TRADES)
    assert len(df) == 6                                    # blank row + un-dated row dropped
    assert str(df["date"].dtype).startswith("datetime64")
    assert df["date"].is_monotonic_increasing
    assert df["pnl_usd"].dtype == float and df["bps"].dtype == float and df["buy_qty"].dtype == float
    assert str(df["month"].dtype) == "Int64"
    assert set(df["side"]) == {"BUY", "SELL"}               # 'Sell' normalised
    alpha_eth = df[(df.counterparty == "Alpha Fund LLC") & (df.symbol == "ETH/USD")].iloc[0]
    assert alpha_eth["pnl_usd"] != alpha_eth["pnl_usd"]     # '' -> NaN, not 0
    assert alpha_eth["notional_usd"] == 50_000.0            # USD is the buy leg on a SELL
    btc = df[df.symbol == "BTC/USD"].iloc[0]
    assert btc["notional_usd"] == 32_000.0 and btc["bps"] == 200.0
    usdc = df[df.symbol == "USDC.e/USD"].iloc[0]
    assert usdc["notional_usd"] == pytest.approx(1_000_000.5)   # stable buy leg preferred
    assert parse_trades([]).empty and parse_trades([["nothing"]]).empty


def test_parse_trades_notional_from_bps_when_no_stable_leg():
    rows = [TRADES[0], ["2024-05-05 10:00:00", "X", "BUY", "ETH/BTC", 10, "ETH", 0.5, "BTC", 0.05, 100.0, "USD", 10.0, 5]]
    df = parse_trades(rows)
    assert df.iloc[0]["notional_usd"] == pytest.approx(100.0 / 10.0 * 1e4)


# ----------------------------------------------------------------------------
# A1MetricsSheet: HTTP, auth headers, caching, typed accessors, errors
# ----------------------------------------------------------------------------

def test_request_headers_and_params():
    sess = FakeSession()
    sheet = make_sheet(session=sess)
    values = sheet.get_range("Volume & PNL", "A1:T40")
    assert values == VOLUME_PNL
    call = sess.calls[-1]
    assert call["url"] == f"{gsheets.SHEETS_API}/{SID}/values/%27Volume%20%26%20PNL%27%21A1%3AT40"
    assert call["headers"]["Authorization"] == "Bearer tok-1"
    assert call["headers"]["x-goog-user-project"] == "quota-proj"
    assert call["params"] == {"valueRenderOption": "UNFORMATTED_VALUE", "dateTimeRenderOption": "FORMATTED_STRING"}


def test_no_quota_project_header_when_unset():
    sess = FakeSession()
    sheet = make_sheet(session=sess, quota=None)
    sheet.list_tabs()
    assert "x-goog-user-project" not in sess.calls[-1]["headers"]


def test_expired_credentials_are_refreshed_once():
    creds = FakeCreds(token=None, expired=True)
    sess = FakeSession()
    sheet = make_sheet(session=sess, creds=creds)
    sheet.list_tabs()
    sheet.get_range("db", "A:M")
    assert creds.refreshes == 1
    assert sess.calls[-1]["headers"]["Authorization"] == "Bearer tok-2"


def test_401_triggers_refresh_and_retry():
    state = {"n": 0}

    def override(url):
        state["n"] += 1
        return 401 if state["n"] == 1 else None
    creds = FakeCreds()
    sess = FakeSession(status_override=override)
    sheet = make_sheet(session=sess, creds=creds)
    assert sheet.list_tabs()[0]["title"] == "Volume & PNL"
    assert creds.refreshes == 1 and len(sess.calls) == 2


def test_refresh_failure_is_sheets_unavailable():
    sheet = make_sheet(creds=FakeCreds(token=None, expired=True, fail_refresh=True))
    with pytest.raises(SheetsUnavailable) as ei:
        sheet.list_tabs()
    assert "gcloud auth application-default login" in str(ei.value)


def test_403_message_tells_operator_to_relogin_with_scope():
    sheet = make_sheet(session=FakeSession(status_override=403))
    with pytest.raises(SheetsAccessError) as ei:
        sheet.get_range("Volume & PNL", "A1:B2")
    msg = str(ei.value)
    assert ei.value.status == 403
    assert "403" in msg and "spreadsheets.readonly" in msg and "gcloud auth application-default login" in msg
    assert "quota-proj" in msg


def test_404_message_says_not_shared():
    sheet = make_sheet(session=FakeSession(status_override=404))
    with pytest.raises(SheetsAccessError) as ei:
        sheet.list_tabs()
    assert ei.value.status == 404
    assert "not found or is not shared" in str(ei.value) and SID in str(ei.value)


def test_429_and_other_errors():
    with pytest.raises(SheetsAccessError) as ei:
        make_sheet(session=FakeSession(status_override=429)).list_tabs()
    assert ei.value.status == 429 and "rate limit" in str(ei.value)
    with pytest.raises(SheetsAccessError) as ei:
        make_sheet(session=FakeSession(status_override=500)).list_tabs()
    assert ei.value.status == 500
    with pytest.raises(SheetsAccessError) as ei:
        make_sheet().get_range("No Such Tab", "A1:B2")
    assert ei.value.status == 400 and "Unable to parse range" in str(ei.value)


def test_cache_ttl_per_range():
    clock = FakeClock(1000.0)
    sess = FakeSession()
    sheet = make_sheet(session=sess, clock=clock, ttl=300)
    sheet.get_range("db", "A:M")
    sheet.get_range("db", "A:M")
    sheet.get_range("db", "a:m")            # case-insensitive key
    assert len(sess.calls) == 1
    sheet.get_range("Weekly PNL", "A1:N80")  # different range -> its own entry
    assert len(sess.calls) == 2
    clock.t += 299
    sheet.get_range("db", "A:M")
    assert len(sess.calls) == 2
    clock.t += 2                             # past the TTL
    sheet.get_range("db", "A:M")
    assert len(sess.calls) == 3
    sheet.clear_cache()
    sheet.get_range("db", "A:M")
    assert len(sess.calls) == 4


def test_cache_ttl_zero_disables_caching():
    sess = FakeSession()
    sheet = make_sheet(session=sess, ttl=0)
    sheet.list_tabs()
    sheet.list_tabs()
    assert len(sess.calls) == 2


def test_list_tabs_and_title():
    sheet = make_sheet()
    tabs = sheet.list_tabs()
    assert [t["title"] for t in tabs][:2] == ["Volume & PNL", "Weekly PNL"]
    assert tabs[0] == {"title": "Volume & PNL", "sheet_id": 1432313692, "rows": 1000, "cols": 29}
    assert sheet.title == "A1 Metrics Dashboard"
    assert sheet.url.endswith(f"/d/{SID}/edit")


def test_data_as_of_variants():
    sheet = make_sheet()
    assert sheet.data_as_of("Volume & PNL") == "2026-09-11"
    assert sheet.data_as_of("2025 Volume & PNL") == "2025-12-31"
    assert sheet.data_as_of("Financing Fees") is None
    assert sheet.data_as_of("x", rows=[["Data as of:", "", "Sep 10, 2026 (Thu)"]]) == "2026-09-10"
    assert sheet.data_as_of("x", rows=[["Data as of ", "garbage"]]) is None
    assert sheet.dashboard_as_of() == "2026-09-11"


def test_monthly_volume_pnl_accessor_and_year_routing():
    sess = FakeSession()
    sheet = make_sheet(session=sess)
    df = sheet.monthly_volume_pnl()
    assert df.attrs == {"tab": "Volume & PNL", "data_as_of": "2026-09-11"}
    assert (df["year"] == 2026).all()
    df26 = sheet.monthly_volume_pnl(year=2026)              # current year -> same tab, served from cache
    assert df26.attrs["tab"] == "Volume & PNL"
    df25 = sheet.monthly_volume_pnl(year=2025)
    assert df25.attrs == {"tab": "2025 Volume & PNL", "data_as_of": "2025-12-31"}
    assert (df25["year"] == 2025).all()
    with pytest.raises(SheetsAccessError):
        sheet.monthly_volume_pnl(year=2023)                  # tab does not exist -> API 400 surfaces


def test_weekly_pnl_accessor():
    df = make_sheet().weekly_pnl()
    assert df.attrs == {"tab": "Weekly PNL", "data_as_of": "2026-09-11"}
    assert df[~df.is_total]["week"].tolist() == [1, 2, 35, 36]
    assert len(make_sheet().weekly_pnl(include_future=True)) == 9


def test_financing_and_nonclient_accessors():
    sheet = make_sheet()
    f = sheet.financing_fees()
    assert f.attrs["tab"] == "Financing Fees" and f.attrs["data_as_of"] is None and f["month"].tolist() == [1, 2, 3]
    w = sheet.financing_fees_weekly()
    assert w["week"].tolist() == [3, 9]
    n = sheet.nonclient_pnl()
    assert n.empty and n.attrs["title"] == "HOLD PNL" and n.attrs["link"].startswith("https://docs.google.com/")


def test_counterparty_trades_accessor_days_relative_to_latest_trade():
    sheet = make_sheet()
    df = sheet.counterparty_trades()
    assert len(df) == 6
    assert df.attrs["coverage_start"] == "2024-06-01" and df.attrs["coverage_end"] == "2024-12-31"
    last30 = sheet.counterparty_trades(days=30)
    assert len(last30) == 4 and last30["date"].min() >= pd.Timestamp("2024-12-02")
    assert last30.attrs["coverage_end"] == "2024-12-31"      # coverage describes the whole blotter
    assert len(sheet.counterparty_trades(days=1)) == 3       # calendar day 2024-12-31 only; the 2024-12-30 22:49 trade is out


def test_counterparty_trades_days_is_calendar_based():
    sheet = make_sheet()
    d1 = sheet.counterparty_trades(days=1)
    assert set(d1["date"].dt.strftime("%Y-%m-%d")) == {"2024-12-31"}
    d2 = sheet.counterparty_trades(days=2)
    assert set(d2["date"].dt.strftime("%Y-%m-%d")) == {"2024-12-30", "2024-12-31"}


def test_from_config_reads_settings(monkeypatch):
    monkeypatch.setenv("A1_METRICS_SHEET_ID", "env-sheet")
    monkeypatch.setenv("GSHEETS_QUOTA_PROJECT", "env-quota")
    monkeypatch.setenv("GSHEETS_CACHE_TTL_S", "42")
    from utils.config import Config
    sheet = A1MetricsSheet.from_config(Config(), session=FakeSession(), credentials=FakeCreds())
    assert (sheet.spreadsheet_id, sheet.quota_project, sheet.cache_ttl_s) == ("env-sheet", "env-quota", 42)


def test_config_defaults(monkeypatch):
    for k in ("A1_METRICS_SHEET_ID", "GSHEETS_QUOTA_PROJECT", "GSHEETS_CACHE_TTL_S"):
        monkeypatch.delenv(k, raising=False)
    from utils.config import Config
    cfg = Config()
    assert cfg.A1_METRICS_SHEET_ID == "1BksNxC2QXHLjFJNCuv-GC9JOBHeb8EyoGwzTuNwqNSY"
    assert cfg.GSHEETS_QUOTA_PROJECT == "anchorage-corp-eng-playground"
    assert cfg.GSHEETS_CACHE_TTL_S == 300


def test_missing_adc_is_sheets_unavailable(monkeypatch):
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def boom(scopes=None):
        raise DefaultCredentialsError("no ADC")
    monkeypatch.setattr(google.auth, "default", boom)
    sheet = A1MetricsSheet(SID, "q", session=FakeSession())
    with pytest.raises(SheetsUnavailable) as ei:
        sheet.list_tabs()
    assert "spreadsheets.readonly" in str(ei.value)


def test_factory_returns_none_without_adc(monkeypatch):
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    from providers import factory

    def boom(scopes=None):
        raise DefaultCredentialsError("no ADC")
    monkeypatch.setattr(google.auth, "default", boom)
    factory.reset_a1_metrics_sheet()
    try:
        assert factory.get_a1_metrics_sheet() is None
        assert factory.get_a1_metrics_sheet() is None        # memoised
    finally:
        factory.reset_a1_metrics_sheet()


def test_factory_builds_sheet_with_adc(monkeypatch):
    import google.auth
    from providers import factory

    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (FakeCreds(), "proj"))
    factory.reset_a1_metrics_sheet()
    try:
        sheet = factory.get_a1_metrics_sheet()
        assert isinstance(sheet, A1MetricsSheet)
        assert factory.get_a1_metrics_sheet() is sheet
    finally:
        factory.reset_a1_metrics_sheet()


def test_constructor_requires_id():
    with pytest.raises(ValueError):
        A1MetricsSheet("", "q")

# ----------------------------------------------------------------------------
# client flow vs non-client flow ('A1 database'!R:X)
# ----------------------------------------------------------------------------

def test_parse_client_flow_dates_both_formats_and_drops_undated_rows():
    df = gsheets.parse_client_flow(A1_DATABASE)
    assert list(df.columns) == ["date", "realized_total", "realized_client", "realized_nonclient",
                                "cum_total_ytd", "cum_client_ytd", "cum_nonclient_ytd",
                                "row_number", "flagged_artifact", "artifact_pair"]
    assert len(df) == 15                                     # header, asset list, undated history, trailing rows dropped
    assert df["date"].dt.strftime("%Y-%m-%d").iloc[0] == "2026-08-27"
    serial = df[df["row_number"] == 7].iloc[0]
    assert serial["date"] == pd.Timestamp("2026-08-28") and serial["realized_total"] == 20_000.0
    assert df["date"].is_monotonic_increasing
    assert df["row_number"].tolist()[:3] == [6, 7, 8]
    assert df["date"].dt.hour.eq(0).all()                    # normalised to midnight
    # two snapshots on 09-01 keep both rows in sheet order
    sep1 = df[df["date"] == "2026-09-01"]
    assert sep1["row_number"].tolist() == [11, 12]
    # the row without W / X keeps V and has NaN split
    short = df[df["row_number"] == 13].iloc[0]
    assert short["realized_total"] == 60_000.0 and pd.isna(short["realized_client"]) and pd.isna(short["realized_nonclient"])
    assert short["cum_total_ytd"] == 3_097_000.0


def test_parse_client_flow_flags_offsetting_pair_only():
    df = gsheets.parse_client_flow(A1_DATABASE)
    flagged = df[df["flagged_artifact"]]
    assert flagged["row_number"].tolist() == [9, 10]          # the undated 3/4 pair is not in the frame
    assert flagged["artifact_pair"].tolist() == [9, 9]
    assert not df[df["row_number"] == 14]["flagged_artifact"].iloc[0]   # +700k has no offsetting neighbour
    # threshold: a big lone day is not an artefact
    rows = [["2026-01-01", 0, 0, 0, 5_000_000.0, 5_000_000.0, 0], ["2026-01-02", 0, 0, 0, 1_000.0, 1_000.0, 0]]
    assert not gsheets.parse_client_flow(rows)["flagged_artifact"].any()
    # tolerance: the offset must be within 5%
    rows = [["2026-01-01", 0, 0, 0, 5_000_000.0, 5_000_000.0, 0], ["2026-01-02", 0, 0, 0, -4_000_000.0, -4_000_000.0, 0]]
    assert not gsheets.parse_client_flow(rows)["flagged_artifact"].any()
    rows = [["2026-01-01", 0, 0, 0, 5_000_000.0, 5_000_000.0, 0], ["2026-01-02", 0, 0, 0, -4_900_000.0, -4_900_000.0, 0]]
    assert gsheets.parse_client_flow(rows)["flagged_artifact"].all()


def test_parse_client_flow_empty_and_first_row_offset():
    empty = gsheets.parse_client_flow([])
    assert empty.empty and "flagged_artifact" in empty.columns
    assert gsheets.parse_client_flow([["Client Flow PNL", "x"], ["not a date", 1, 2, 3, 4, 5, 6]]).empty
    df = gsheets.parse_client_flow(A1_DATABASE[5:7], first_row=6)
    assert df["row_number"].tolist() == [6, 7]


def test_client_flow_daily_accessor_filters_and_attrs():
    sess = FakeSession()
    sheet = make_sheet(session=sess)
    df = sheet.client_flow_daily()
    assert len(df) == 15
    assert df.attrs["tab"] == "A1 database" and df.attrs["range"] == "R:X"
    assert df.attrs["data_as_of"] == "2026-09-11"            # from the dashboard tab - this tab has no as-of cell
    assert df.attrs["first_row_date"] == "2026-08-27" and df.attrs["last_row_date"] == "2026-09-10"
    assert df.attrs["n_rows_total"] == 15
    sub = sheet.client_flow_daily("2026-09-05", "2026-09-10")
    assert len(sub) == 6 and sub["row_number"].tolist() == [15, 16, 17, 18, 19, 20]
    assert sub.attrs["last_row_date"] == "2026-09-10"        # attrs describe the whole tab
    only_start = sheet.client_flow_daily(start=pd.Timestamp("2026-09-09"))
    assert only_start["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-09-09", "2026-09-10"]
    # the R:X range is fetched once and cached
    rx_calls = [c for c in sess.calls if "R%3AX" in c["url"] or "R:X" in c["url"]]
    assert len(rx_calls) == 1


def test_client_flow_window_dashboard_week_matches_weekly_tab():
    sheet = make_sheet()
    res = sheet.client_flow_window("2026-09-04", "2026-09-10")
    assert res["start"] == "2026-09-04" and res["end"] == "2026-09-10"
    assert res["n_rows"] == 6 and res["n_days"] == 6
    assert res["sums"] == {"realized_total": 290_000.0, "realized_client": 65_000.0, "realized_nonclient": 225_000.0}
    assert res["sums_all_rows"] == res["sums"]
    assert abs(res["client_share"] - 65_000 / 290_000) < 1e-12 and abs(res["nonclient_share"] - 225_000 / 290_000) < 1e-12
    assert res["missing_days"] == ["2026-09-04 (Fri)"] and res["missing_weekdays"] == ["2026-09-04 (Fri)"]
    assert res["not_populated"] == [] and res["artifact_rows"] == [] and res["incomplete_rows"] == []
    assert res["duplicate_dates"] == []
    wc = res["weekly_check"]
    assert wc["week"] == 36 and wc["a1_realized"] == 290_000.0 and wc["difference"] == 0.0
    assert wc["a1_unrealized"] == 285_000.0
    assert res["ytd_check"] is None
    assert res["rows"]["row_number"].tolist() == [15, 16, 17, 18, 19, 20] and res["excluded"].empty


def test_client_flow_window_keeps_full_pair_and_reports_weekly_mismatch():
    sheet = make_sheet()
    res = sheet.client_flow_window("2026-08-28", "2026-09-03")
    # 20k + 5k + (8.0M - 7.99M) + 3k - 1k + 60k + 700k
    assert res["sums"]["realized_total"] == pytest.approx(797_000.0)
    assert res["sums"]["realized_client"] == pytest.approx(15_000 + 4_000 + 8_000 + 3_000 - 2_000 + 710_000)
    assert res["sums"]["realized_nonclient"] == pytest.approx(5_000 + 1_000 + 2_000 + 0 + 1_000 - 10_000)
    assert res["n_rows"] == 8 and res["n_days"] == 7                    # two snapshots on 09-01
    assert res["duplicate_dates"] == ["2026-09-01"]
    assert res["incomplete_rows"] == [{"row_number": 13, "date": "2026-09-02"}]
    assert [a["action"].split(" - ")[0] for a in res["artifact_rows"]] == ["kept", "kept"]
    assert [a["row_number"] for a in res["artifact_rows"]] == [9, 10]
    assert res["artifact_net"] == pytest.approx(10_000.0)
    assert res["excluded"].empty
    wc = res["weekly_check"]
    assert wc["week"] == 35 and wc["a1_realized"] == 1_400_000.0 and wc["difference"] == pytest.approx(-603_000.0)
    assert res["missing_days"] == []


def test_client_flow_window_excludes_lone_artifact_leg_and_discloses():
    sheet = make_sheet()
    res = sheet.client_flow_window("2026-08-27", "2026-08-30")
    assert res["sums"]["realized_total"] == 35_000.0                   # 10k + 20k + 5k, the +8M leg excluded
    assert res["sums_all_rows"]["realized_total"] == 8_035_000.0
    assert res["n_rows"] == 3 and len(res["excluded"]) == 1 and res["excluded"]["row_number"].iloc[0] == 9
    assert len(res["artifact_rows"]) == 1
    a = res["artifact_rows"][0]
    assert a["row_number"] == 9 and a["date"] == "2026-08-30" and a["realized_total"] == 8_000_000.0
    assert a["action"].startswith("excluded")
    assert res["weekly_check"] is None                                 # not a dashboard week
    kept = sheet.client_flow_window("2026-08-27", "2026-08-30", exclude_artifacts=False)
    assert kept["sums"]["realized_total"] == 8_035_000.0 and kept["excluded"].empty
    assert kept["artifact_rows"][0]["action"].startswith("kept")


def test_client_flow_window_future_days_and_empty_window():
    sheet = make_sheet()
    res = sheet.client_flow_window("2026-09-08", "2026-09-14")
    assert res["n_rows"] == 3 and res["missing_days"] == []
    assert res["not_populated"] == ["2026-09-11", "2026-09-12", "2026-09-13", "2026-09-14"]
    assert res["last_row_date"] == "2026-09-10"
    empty = sheet.client_flow_window("2026-10-01", "2026-10-05")
    assert empty["n_rows"] == 0 and empty["sums"] == {"realized_total": 0.0, "realized_client": 0.0, "realized_nonclient": 0.0}
    assert empty["client_share"] is None and empty["missing_days"] == []
    assert empty["not_populated"] == ["2026-10-0%d" % d for d in range(1, 6)]
    # reversed dates are swapped; non-dates raise
    swapped = sheet.client_flow_window("2026-09-10", "2026-09-04")
    assert swapped["start"] == "2026-09-04" and swapped["sums"]["realized_total"] == 290_000.0
    with pytest.raises(ValueError):
        sheet.client_flow_window("nope", "2026-09-04")


def test_client_flow_window_ytd_check_uses_cumulative_columns():
    sheet = make_sheet()
    res = sheet.client_flow_window("2026-01-01", "2026-09-10")
    yc = res["ytd_check"]
    assert yc["date"] == "2026-09-10" and yc["row_number"] == 20
    assert yc["cum_total"] == 4_087_000.0 and yc["cum_client"] == 2_643_000.0 and yc["cum_nonclient"] == 1_444_000.0
    assert yc["diff_total"] == pytest.approx(res["sums"]["realized_total"] - 4_087_000.0)
    # missing days are only counted up to the last populated row
    assert "2026-09-04 (Fri)" in res["missing_days"] and not any(d.startswith("2026-09-11") for d in res["missing_days"])
    assert res["not_populated"] == []


def test_client_flow_window_missing_days_split_weekdays():
    # window with a missing weekend day: drop 08-29 (Sat) via a trimmed payload
    tab = [r for r in A1_DATABASE if not (r and r[0] == "2026-08-29 22:10:24")]
    sheet = make_sheet(session=FakeSession(tabs={**TABS, "A1 database": tab}))
    res = sheet.client_flow_window("2026-08-27", "2026-09-05")
    assert res["missing_days"] == ["2026-08-29 (Sat)", "2026-09-04 (Fri)"]
    assert res["missing_weekdays"] == ["2026-09-04 (Fri)"]
