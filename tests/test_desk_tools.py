"""tools/desk_tools.py with a fake DeskBigQuery. No network, no GCP.

Canned frames are modelled on real LIMIT 3 samples taken on 2026-09-10 (numbers changed).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from providers.bigquery import DeskBigQuery, SQLGuardError
from tools import desk_tools as dt
from tools.desk_tools import (
    DESK_TOOL_NAMES,
    describe_desk_table,
    desk_symbol_to_token,
    get_desk_greeks_history,
    get_desk_pnl_history,
    get_desk_positions_by_symbol,
    get_desk_risk_snapshot,
    get_desk_tools,
    get_internal_price,
    get_open_orders,
    get_otc_derivatives_trades,
    get_perp_positions,
    list_desk_tables,
    query_desk_data,
)

UTC = timezone.utc
TS = pd.Timestamp("2026-09-10 19:33:02", tz="UTC")
EOD_TS = pd.Timestamp("2026-09-09 23:58:42", tz="UTC")


# ----------------------------------------------------------------------------
# canned frames (real column names, altered numbers)
# ----------------------------------------------------------------------------

def _snapshot_frame():
    rows = []
    for eid, delta, ytd, dq, pct in ((20, -60_000_000.0, -32_500_000.0, "High Invalid Pricer Rate", 13.96),
                                     (86, 21_000_000.0, 43_300_000.0, "Normal", 100.0)):
        rows.append({
            "position_timestamp": TS, "entity_id": eid, "position_count": 2492 if eid == 20 else 90, "venue_count": 12,
            "strategy_count": 1, "asset_count": 24, "symbol_count": 2210, "total_abs_size_usd": 13_900_000_000.0,
            "total_size_usd": 13_900_000_000.0, "total_equity_usd": -29_600_000.0, "total_portfolio_pnl": -430_869.30,
            "total_open_pnl": -500_156.32, "total_realised_pnl": 68_802.75, "total_week_to_date_pnl": -1_197_605.34,
            "total_month_to_date_pnl": -822_362.50, "total_quarter_to_date_pnl": -40_329_898.19, "total_year_to_date_pnl": ytd,
            "total_life_to_date_pnl": -29_695_685.93, "total_funding_pnl": 1614.58, "total_life_to_date_funding_pnl": 283_000.0,
            "total_fees": -1130.30, "total_life_to_date_fees": -2_681_678.0, "total_delta_usd": delta,
            "total_delta_adjusted_usd": -22_896_426.29, "total_abs_delta_usd": 4_589_154_411.0, "total_gamma_usd": 4_715_590.57,
            "total_gamma_percent_usd": 3_278_657.50, "total_vega": 102_337.66, "total_theta": -231_424.62, "total_theta_bs": -194_695.99,
            "total_vanna": 3_190_765.0, "total_charm": -17.0, "total_volga": 13_103.0,
            "portfolio_delta_risk_level": "High Delta Exposure", "portfolio_gamma_risk_level": "High Gamma Risk",
            "portfolio_vega_risk_level": "Medium Vega Risk", "large_delta_change_flag": eid == 86, "delta_change_severity": "Normal",
            "delta_adjusted_usd_change": 12_345_678.0, "large_ytd_pnl_change_flag": False, "ytd_pnl_change_severity": "Normal",
            "ytd_pnl_change": 1000.0, "large_theta_bs_hourly_change_flag": False, "large_gamma_percent_usd_hourly_change_flag": False,
            "valid_pricer_count": 348, "invalid_pricer_count": 2144, "valid_pricer_pct": pct, "data_quality_flag": dq,
            "hours_since_latest_calculation": 0, "total_week_to_date_volume_usd": 162_151_511.0,
            "total_month_to_date_volume_usd": 2_613_337_174.0,
        })
    return pd.DataFrame(rows)


def _portfolio_eod_frame(days=3):
    rows = []
    for i in range(days):
        d = date(2026, 9, 8 + i)
        for eid in (20, 86):
            rows.append({
                "as_of_date": d, "as_of_timestamp": pd.Timestamp(datetime(2026, 9, 8 + i, 23, 55, tzinfo=UTC)), "entity_id": eid,
                "total_portfolio_pnl": (i + 1) * 1000.0 * (1 if eid == 86 else -1), "total_week_to_date_pnl": 5000.0,
                "total_month_to_date_pnl": 9000.0, "total_year_to_date_pnl": 43_000_000.0 if eid == 86 else -32_000_000.0,
                "total_life_to_date_pnl": 1_000_000.0 + i * 1000.0, "total_equity_usd": 43_000_000.0,
                "total_abs_size_usd": 1_234_000_000.0, "total_funding_pnl": 0.0, "total_fees": -5.0,
                "data_quality_flag": "High Invalid Pricer Rate", "valid_pricer_pct": 27.78,
            })
    return pd.DataFrame(rows)


def _grouped_pnl_frame():
    rows = []
    for i, d in enumerate((date(2026, 9, 8), date(2026, 9, 9))):
        for grp, gross, pnl in (("DERIBIT", 2_991_189_706.0, -461_720.98), ("OTC", 7_071_204_152.0, -238_129.0),
                                ("other", 579_760_780.0, 119_822.0)):
            rows.append({"as_of_date": d, "grp": grp, "day_pnl": pnl * (i + 1), "gross_usd": gross, "ytd_pnl": 1000.0,
                         "n_positions": 10, "n_valid": 2})
    return pd.DataFrame(rows)


def _greeks_frame():
    rows = []
    for i in range(3):
        rows.append({
            "as_of_date": date(2026, 9, 8 + i), "as_of_timestamp": pd.Timestamp(datetime(2026, 9, 8 + i, 23, 56, tzinfo=UTC)),
            "entity_id": 20, "position_count": 2489, "total_abs_size_usd": 14_049_180_556.0,
            "total_delta_usd": -52_000_000.0 - i * 1_000_000, "total_delta_adjusted_usd": -10_000_000.0, "total_abs_delta_usd": 4e9,
            "total_gamma_usd": 6_750_003.47, "total_gamma_percent_usd": 5_926_975.04, "total_vega": 165_302.31,
            "total_theta": -468_441.55, "total_theta_bs": -280_015.03, "portfolio_delta_risk_level": "High Delta Exposure",
            "portfolio_gamma_risk_level": "High Gamma Risk", "portfolio_vega_risk_level": "Medium Vega Risk",
            "data_quality_flag": "Normal", "valid_pricer_pct": 100.0,
        })
    return pd.DataFrame(rows)


def _perps_frame():
    return pd.DataFrame([
        {"as_of_timestamp": EOD_TS, "entity_id": 20, "symbol": "BTC-PERPETUAL", "underlying_asset": "BTC", "venue": "DERIBIT",
         "position": 81_315_600.0, "size_coin": 1039.156, "size_usd": 81_315_600.0, "avg_px": 78_214.95, "mark_px": 78_251.55,
         "open_pnl": 38_049.69, "realised_pnl": -95.02, "total_pnl": 37_952.07, "funding_pnl": 0.0, "life_to_date_funding_pnl": -274_784.98,
         "year_to_date_pnl": 1_700_313.0, "delta_usd": 81_315_600.0, "maturity": pd.Timestamp("1970-01-01", tz="UTC"), "pricer_valid": True,
         "data_quality_flag": "Normal"},
        {"as_of_timestamp": EOD_TS, "entity_id": 20, "symbol": "BTCUSDT", "underlying_asset": "BTC", "venue": "BINANCE_EXCHANGE",
         "position": -401.066, "size_coin": 401.066, "size_usd": 31_384_035.74, "avg_px": 78_214.95, "mark_px": 78_251.55,
         "open_pnl": -14_674.21, "realised_pnl": 0.0, "total_pnl": -14_674.21, "funding_pnl": 0.0, "life_to_date_funding_pnl": 260_077.37,
         "year_to_date_pnl": -11_624_490.0, "delta_usd": -31_384_035.74, "maturity": pd.Timestamp("1970-01-01", tz="UTC"), "pricer_valid": True,
         "data_quality_flag": "Normal"},
        {"as_of_timestamp": EOD_TS, "entity_id": 20, "symbol": "BTC-25SEP26", "underlying_asset": "BTC", "venue": "DERIBIT",
         "position": 14_231_570.0, "size_coin": 181.87, "size_usd": 14_231_570.0, "avg_px": 78_322.52, "mark_px": 78_354.07,
         "open_pnl": 5725.67, "realised_pnl": 0.0, "total_pnl": 5731.09, "funding_pnl": 0.0, "life_to_date_funding_pnl": 0.0,
         "year_to_date_pnl": 0.0, "delta_usd": 14_212_948.77, "maturity": pd.Timestamp("2026-09-25 08:00", tz="UTC"), "pricer_valid": False,
         "data_quality_flag": "Pricer Invalid"},
        {"as_of_timestamp": EOD_TS, "entity_id": 20, "symbol": "HYPEUSDT", "underlying_asset": "HYPE", "venue": "BINANCE_EXCHANGE",
         "position": 4314.34, "size_coin": 4314.34, "size_usd": 360_721.97, "avg_px": 84.8, "mark_px": 83.61,
         "open_pnl": -5132.55, "realised_pnl": 0.0, "total_pnl": -5147.06, "funding_pnl": -14.51, "life_to_date_funding_pnl": 1436.78,
         "year_to_date_pnl": -81_233.59, "delta_usd": 360_721.97, "maturity": pd.Timestamp("1970-01-01", tz="UTC"), "pricer_valid": True,
         "data_quality_flag": "Normal"},
    ])


def _live_futures_frame():
    snap = pd.Timestamp("2026-09-10 19:40:48", tz="UTC")
    return pd.DataFrame([
        {"snapshot_timestamp": snap, "symbol": "BTCUSDT", "total_exchange_position": -393.036, "total_computed_position": -393.036,
         "reconciliation_status": "Fully Reconciled", "venues": "BINANCE_EXCHANGE, BYBIT"},
        {"snapshot_timestamp": snap, "symbol": "BTC-PERPETUAL", "total_exchange_position": 77_609_600.0, "total_computed_position": 77_609_600.0,
         "reconciliation_status": "Fully Reconciled", "venues": "DERIBIT"},
    ])


def _by_underlying_frame():
    return pd.DataFrame([
        {"underlying_asset": "BTC", "instrument_type": "OPTIONS", "n_positions": 2068, "gross_usd": 4_306_304_713.9, "delta_usd": -757_241_378.2,
         "gamma_usd": 6_788_427.5, "vega": 166_691.0, "theta": -467_475.9, "day_pnl": -341_073.6, "ytd_pnl": 14_773_765.5, "n_valid": 192, "as_of": EOD_TS},
        {"underlying_asset": "BTC", "instrument_type": "SPOT", "n_positions": 40, "gross_usd": 3_120_513_727.5, "delta_usd": 654_361_889.4,
         "gamma_usd": 0.0, "vega": 0.0, "theta": 0.0, "day_pnl": 305_722.1, "ytd_pnl": -32_132_594.2, "n_valid": 18, "as_of": EOD_TS},
        {"underlying_asset": "USDC", "instrument_type": "SPOT", "n_positions": 31, "gross_usd": 3_482_474_468.1, "delta_usd": 0.0,
         "gamma_usd": 0.0, "vega": 0.0, "theta": 0.0, "day_pnl": -30_702.9, "ytd_pnl": 41_069.1, "n_valid": 17, "as_of": EOD_TS},
        {"underlying_asset": "HYPE", "instrument_type": "FUTURES", "n_positions": 5, "gross_usd": 623_198.8, "delta_usd": 98_245.1,
         "gamma_usd": 0.0, "vega": 0.0, "theta": 0.0, "day_pnl": -1412.4, "ytd_pnl": -81_233.6, "n_valid": 2, "as_of": EOD_TS},
        {"underlying_asset": "PAXG", "instrument_type": "OPTIONS", "n_positions": 41, "gross_usd": 2_038_298.0, "delta_usd": 116_165.6,
         "gamma_usd": 77_414.6, "vega": 579.7, "theta": -439.8, "day_pnl": -150.2, "ytd_pnl": -11_277.4, "n_valid": 6, "as_of": EOD_TS},
    ])


def _btc_breakdown_frame():
    return pd.DataFrame([
        {"instrument_type": "OPTIONS", "venue": "DERIBIT", "n_positions": 973, "gross_usd": 2_860_477_548.0, "delta_usd": -345_261_540.0,
         "gamma_usd": -32_007_803.0, "vega": -492_451.0, "theta": 405_292.0, "day_pnl": -492_442.0, "ytd_pnl": -24_234_894.0, "n_valid": 90, "as_of": EOD_TS},
        {"instrument_type": "FUTURES", "venue": "BINANCE_EXCHANGE", "n_positions": 3, "gross_usd": 31_384_036.0, "delta_usd": -31_384_036.0,
         "gamma_usd": 0.0, "vega": 0.0, "theta": 0.0, "day_pnl": -14_674.0, "ytd_pnl": -11_624_490.0, "n_valid": 1, "as_of": EOD_TS},
    ])


def _btc_live_spot_frame():
    return pd.DataFrame([{
        "snapshot_timestamp": pd.Timestamp("2026-09-10 19:53:00", tz="UTC"), "symbol": "BTC", "position_type": "spot",
        "total_exchange_position": 17_982.6, "total_computed_position": 17_698.98, "total_position_delta_value": -21_990_262.0,
        "reconciliation_status": "Partially Reconciled", "venues": "ANCHORAGE, BINANCE_EXCHANGE, DERIBIT",
    }])


def _otc_trades_frame():
    return pd.DataFrame([
        {"exec_time": pd.Timestamp("2026-09-10 16:27:50", tz="UTC"), "status": "ACTIVE", "direction": "BUY", "product_type": "OPTION",
         "option_type": "CALL", "base_symbol": "BTC", "quote_symbol": "USD", "qty_base": 55.0, "notional_quote": 4_290_000.0, "strike": 78_000.0,
         "premium_per_unit": 0.05894, "premium_currency_type": "BASE", "settlement_type": "BASE",
         "expiration": pd.Timestamp("2026-11-12 08:00", tz="UTC"), "counterparty_name": "Example Trust LLC", "entity_name": "A1, LTD.",
         "package_indicator": None, "venue": None},
        {"exec_time": pd.Timestamp("2026-09-09 09:53:14", tz="UTC"), "status": "CANCELED", "direction": "SELL", "product_type": "OPTION",
         "option_type": "PUT", "base_symbol": "ETH", "quote_symbol": "USD", "qty_base": 150.0, "notional_quote": 330_000.0, "strike": 2200.0,
         "premium_per_unit": 5.7022, "premium_currency_type": "QUOTE", "settlement_type": "QUOTE",
         "expiration": pd.Timestamp("2026-09-16 08:00", tz="UTC"), "counterparty_name": "Example Fund Ltd", "entity_name": "A1, LTD.",
         "package_indicator": None, "venue": None},
    ])


def _otc_summary_frame():
    return pd.DataFrame([
        {"status": "ACTIVE", "option_type": "CALL", "base_symbol": "BTC", "n_trades": 138, "notional_quote": 1_039_423_000.0,
         "last_exec": pd.Timestamp("2026-09-10 16:27:50", tz="UTC")},
        {"status": "CANCELED", "option_type": "PUT", "base_symbol": "ETH", "n_trades": 11, "notional_quote": 42_264_489.0,
         "last_exec": pd.Timestamp("2026-09-09 09:53:14", tz="UTC")},
    ])


def _orders_frame():
    loaded = pd.Timestamp("2026-09-10 19:36:35", tz="UTC")
    return pd.DataFrame([
        {"loaded_at": loaded, "counterparty": "N/A", "order_id": "dfee5b94", "symbol": "XRP-USD", "side": "Sell", "strategy": None,
         "amount": 5000.0, "currency": "XRP", "limit_price": 200.0, "cum_qty": 0.0, "leaves_qty": 5000.0, "amount_left": 5000.0,
         "avg_px_all_in": None, "status": "Active", "hours_left": None, "start_time": None, "end_time": None, "is_multileg": False,
         "last_market": None},
        {"loaded_at": loaded, "counterparty": "N/A", "order_id": "32768e90", "symbol": "BTC-USD", "side": "Sell", "strategy": "Limit",
         "amount": 41.460491, "currency": "BTC", "limit_price": 99_099.0, "cum_qty": 0.0, "leaves_qty": 41.460491, "amount_left": 41.460491,
         "avg_px_all_in": None, "status": "Active", "hours_left": 12.5, "start_time": None, "end_time": "2026-09-11 08:00:00",
         "is_multileg": False, "last_market": None},
    ])


def _live_price_frame():
    return pd.DataFrame([{"canonical_symbol": "BTC", "price": 77_147.04, "provider_timestamp": pd.Timestamp("2026-09-10 19:32:03", tz="UTC"),
                          "source": "datastream_anchorage"}])


def _intraday_frame():
    rows = []
    for h, px in enumerate((77_000.0, 77_200.0, 77_500.0, 77_385.0)):
        rows.append({"hour": pd.Timestamp(datetime(2026, 9, 10, 15 + h, tzinfo=UTC)), "price": px, "high": px + 50, "low": px - 60,
                     "last_minute": pd.Timestamp(datetime(2026, 9, 10, 15 + h, 59, tzinfo=UTC)), "n_bars": 60})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# fake DeskBigQuery
# ----------------------------------------------------------------------------

class FakeBQ(DeskBigQuery):
    """Routes queries to canned frames by the table they reference; records SQL."""

    def __init__(self, frames=None, fail_on=None, empty=False):
        super().__init__(client=object(), data_project="anc-global-markets", allowed_datasets=("brokerage_a1", "pricing"),
                         max_rows=200, catalog_path="/nonexistent/bq_catalog.json")
        self.frames = frames or {}
        self.fail_on = fail_on or ()
        self.empty = empty
        self.sqls = []

    def query(self, sql):
        safe = self.validate(sql)
        self.sqls.append(safe)
        for key in self.fail_on:
            if key in safe:
                raise RuntimeError(f"boom on {key}")
        if self.empty:
            return pd.DataFrame()
        for key, frame in self.frames.items():
            if key in safe:
                df = frame() if callable(frame) else frame.copy()
                df.attrs["bytes_processed"] = 4096
                return df
        raise AssertionError(f"unexpected SQL in test: {safe[:200]}")

    # catalog pieces used by list/describe
    def catalog(self):
        return {"built_at": 0, "data_project": self.data_project, "datasets": {
            "brokerage_a1": {
                "fct_otc_haruko_pnl_by_venue_history": {"table": "brokerage_a1.fct_otc_haruko_pnl_by_venue_history", "type": "VIEW",
                                                        "description": "", "num_rows": None, "partitioning": None, "clustering": [],
                                                        "columns": [{"name": "venue", "type": "STRING", "description": ""},
                                                                    {"name": "total_pnl", "type": "FLOAT", "description": "day pnl"}]},
                "fct_otc_haruko_pnl_portfolio": {"table": "brokerage_a1.fct_otc_haruko_pnl_portfolio", "type": "TABLE", "description": "",
                                                 "num_rows": 65467, "num_bytes": 50_000_000, "modified": "2026-09-10T19:35:04+00:00",
                                                 "partitioning": "DAY on position_timestamp", "clustering": ["position_timestamp"],
                                                 "columns": [{"name": "position_timestamp", "type": "TIMESTAMP", "description": ""},
                                                             {"name": "total_delta_usd", "type": "FLOAT", "description": "USD delta"}]},
            },
            "pricing": {
                "current_price": {"table": "pricing.current_price", "type": "TABLE", "description": "latest px", "num_rows": 3378,
                                  "partitioning": None, "clustering": [], "columns": [{"name": "symbol", "type": "STRING", "description": ""}]},
            },
        }}


@pytest.fixture
def fake(monkeypatch):
    holder = {}

    def _install(**kw):
        bq = FakeBQ(**kw)
        holder["bq"] = bq
        monkeypatch.setattr(dt, "_get_bq", lambda: bq)
        return bq

    return _install


# ----------------------------------------------------------------------------
# tool registry / availability
# ----------------------------------------------------------------------------

def test_get_desk_tools_names_and_docs():
    names = [t.name for t in get_desk_tools()]
    assert names == [
        "get_desk_risk_snapshot", "get_desk_pnl_history", "get_desk_greeks_history", "get_perp_positions",
        "get_desk_positions_by_symbol", "get_otc_derivatives_trades", "get_open_orders", "get_internal_price",
        "list_desk_tables", "describe_desk_table", "query_desk_data",
    ]
    assert names == DESK_TOOL_NAMES
    for t in get_desk_tools():
        assert len(t.description) > 80, t.name
        assert any(k in t.description for k in ("Haruko", "BigQuery", "Talos", "pricing", "OTC")), t.name


def test_tools_report_unavailable_without_bigquery(monkeypatch):
    monkeypatch.setattr(dt, "_get_bq", lambda: None)
    out = get_desk_risk_snapshot.invoke({})
    assert "not available" in out and "Market-data tools still work" in out
    assert "not available" in query_desk_data.invoke({"sql": "SELECT 1"})


# ----------------------------------------------------------------------------
# symbol mapping
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("sym,tok", [
    ("BTC-PERPETUAL", "btc"), ("BTCUSDT", "btc"), ("BTC-USDT-SWAP", "btc"), ("BTC-25SEP26-80000-C", "btc"),
    ("BTCUSD-25SEP26-0800-79000-P-USD-PHYS", "btc"), ("BTC-PERP", "btc"), ("ETHUSD", "eth"), ("XAUT/USDT", "xaut"),
    ("USDG_SOLANA", "usdg"), ("hype", "hype"), ("PAXGUSDT", "paxg"), ("PYUSD", "pyusd"), ("USDC", "usdc"), ("USD", "usd"),
    ("", None), (None, None),
])
def test_desk_symbol_to_token(sym, tok):
    assert desk_symbol_to_token(sym) == tok


# ----------------------------------------------------------------------------
# get_desk_risk_snapshot
# ----------------------------------------------------------------------------

def test_risk_snapshot_formatting(fake):
    bq = fake(frames={"fct_otc_haruko_pnl_portfolio": _snapshot_frame})
    out = get_desk_risk_snapshot.invoke({})
    assert out.startswith("### Desk risk snapshot")
    assert "**A1 Ltd** - as of 2026-09-10 19:33 UTC" in out
    assert "**ADSD (Anchorage Digital Swap Dealer)**" in out
    assert "gross $13,900,000,000" in out and "equity -$29,600,000" in out
    assert "day -$430,869" in out and "YTD -$32,500,000" in out and "LTD -$29,695,686" in out
    assert "delta -$60,000,000" in out and "vega $102,338/vol pt" in out and "theta -$231,425/day" in out
    assert "delta 'High Delta Exposure'" in out
    # data-quality caveat only for the flagged entity
    assert "Data quality: High Invalid Pricer Rate; valid pricers 14.0% - CAVEAT" in out
    assert "Data quality: Normal; valid pricers 100.0%" in out
    assert "(348 valid / 2144 invalid)" in out
    # large change flag on entity 86
    assert "LARGE DELTA CHANGE (Normal, $12,345,678" in out
    assert "**Combined (2 entities)**" in out and "delta -$39,000,000" in out
    assert "Bytes scanned: 4.00 KB." in out
    # SQL hygiene: partition filter + guard applied
    assert "position_timestamp >= TIMESTAMP_SUB" in bq.sqls[0] and bq.sqls[0].rstrip().endswith("LIMIT 10")


def test_risk_snapshot_empty_and_error(fake):
    fake(empty=True)
    assert "No portfolio snapshot" in get_desk_risk_snapshot.invoke({})
    fake(fail_on=("fct_otc_haruko_pnl_portfolio",))
    out = get_desk_risk_snapshot.invoke({})
    assert out.startswith("Error querying desk data in get_desk_risk_snapshot") and "boom" in out


# ----------------------------------------------------------------------------
# get_desk_pnl_history
# ----------------------------------------------------------------------------

def test_pnl_history_portfolio(fake):
    bq = fake(frames={"fct_otc_haruko_pnl_portfolio_history_eod": lambda: _portfolio_eod_frame(3)})
    out = get_desk_pnl_history.invoke({"days": 10, "by": "portfolio"})
    assert out.startswith("### Desk PnL history, portfolio level (EOD, last 10 days, 2026-09-08 to 2026-09-10)")
    assert "**A1 Ltd** (3 days; latest EOD 2026-09-10 23:55 UTC)" in out
    assert "sum of day PnL -$6,000" in out          # -1000 -2000 -3000
    assert "sum of day PnL $6,000" in out           # ADSD
    assert "LTD PnL change $2,000 ($1,000,000 -> $1,002,000)" in out
    assert "| 2026-09-09 | -$2,000 |" in out
    assert "CAVEAT" in out
    assert "INTERVAL 10 DAY" in bq.sqls[0]


def test_pnl_history_clamps_days_and_rejects_bad_mode(fake):
    bq = fake(frames={"fct_otc_haruko_pnl_portfolio_history_eod": lambda: _portfolio_eod_frame(2)})
    get_desk_pnl_history.invoke({"days": 5000, "by": "portfolio"})
    assert "INTERVAL 90 DAY" in bq.sqls[-1]
    get_desk_pnl_history.invoke({"days": 1, "by": "portfolio"})
    assert "INTERVAL 7 DAY" in bq.sqls[-1]
    assert "Unknown breakdown" in get_desk_pnl_history.invoke({"days": 7, "by": "desk"})


def test_pnl_history_by_venue_pivot(fake):
    bq = fake(frames={"fct_otc_haruko_pnl_position_history_eod": _grouped_pnl_frame})
    out = get_desk_pnl_history.invoke({"days": 60, "by": "venue"})
    assert out.startswith("### Desk day PnL by venue (EOD, last 28 days, 2026-09-08 to 2026-09-09")
    assert "INTERVAL 28 DAY" in bq.sqls[0] and "COALESCE(p.venue, 'unknown')" in bq.sqls[0]
    # groups ordered by latest gross notional: OTC, DERIBIT, other
    assert "| date | OTC | DERIBIT | other | total |" in out
    assert "| 2026-09-09 | -$476,258 | -$923,442 | $239,644 | -$1,160,056 |" in out
    assert "| OTC | $7,071,204,152 | -$476,258 | $1,000 | 10 | 20% |" in out
    assert "Data quality: valid pricers 20.0% of positions at the latest EOD - CAVEAT" in out
    assert "COUNTIF(b.pricer_valid)" in bq.sqls[0]
    assert "Period total day PnL" in out
    out = get_desk_pnl_history.invoke({"days": 7, "by": "strategy"})
    assert "COALESCE(p.strategy_name, 'unassigned')" in bq.sqls[-1] and "by strategy" in out


def test_pnl_history_empty(fake):
    fake(empty=True)
    assert "No EOD portfolio rows" in get_desk_pnl_history.invoke({"days": 7})
    assert "No EOD position rows" in get_desk_pnl_history.invoke({"days": 7, "by": "venue"})


# ----------------------------------------------------------------------------
# get_desk_greeks_history
# ----------------------------------------------------------------------------

def test_greeks_history(fake):
    bq = fake(frames={"fct_otc_haruko_greeks_history_eod": _greeks_frame})
    out = get_desk_greeks_history.invoke({"days": 7})
    assert out.startswith("### Desk greeks history (EOD, last 7 days, 2026-09-08 to 2026-09-10)")
    assert "**A1 Ltd** (3 days; latest EOD 2026-09-10 23:56 UTC; 2,489 positions, gross notional $14,049,180,556)" in out
    assert "change over period: delta -$2,000,000, gamma $0, vega $0, theta $0" in out
    assert "| 2026-09-10 | -$54,000,000 | -$10,000,000 | $6,750,003 | $5,926,975 | $165,302 | -$468,442 | Normal |" in out
    assert "Data quality: Normal; valid pricers 100.0%" in out and "CAVEAT" not in out
    assert "INTERVAL 7 DAY" in bq.sqls[0]
    fake(empty=True)
    assert "No EOD greeks rows" in get_desk_greeks_history.invoke({"days": 7})


# ----------------------------------------------------------------------------
# get_perp_positions
# ----------------------------------------------------------------------------

def test_perp_positions(fake):
    bq = fake(frames={"fct_otc_haruko_pnl_position_history_eod": _perps_frame, "fct_otc_haruko_position_summary": _live_futures_frame})
    out = get_perp_positions.invoke({"top_n": 10})
    assert out.startswith("### Desk perp / futures positions (Haruko EOD 2026-09-09 23:58 UTC; top 4 by notional)")
    assert "4 positions shown (3 perpetual, 1 dated futures)" in out
    assert "gross notional $127,291,928" in out and "LTD funding -$13,271" in out
    assert "invalid pricers on 1 of 4 rows" in out
    assert "live exchange positions as of 2026-09-10 19:40 UTC" in out
    assert "| BTC-PERPETUAL | DERIBIT | A1 | perp | long | 1,039.16 | $81,315,600 | 78,214.95 | 78,251.55 | $38,050 | $37,952 | $0 | -$274,785 | $81,315,600 | 77,609,600 | ok |" in out
    assert "| BTCUSDT | BINANCE_EXCHANGE | A1 | perp | short | 401.066 | $31,384,036 |" in out and "| -393.036 | ok |" in out
    assert "| BTC-25SEP26 | DERIBIT | A1 | fut 2026-09-25 | long |" in out and "| - | INVALID |" in out
    assert "Token mapping - in the market-data universe" in out and "btc, hype" in out
    assert "instrument_type = 'FUTURES'" in bq.sqls[0] and "LIMIT 10" in bq.sqls[0]
    assert "position_type = 'futures'" in bq.sqls[1]


def test_perp_positions_live_overlay_optional_and_empty(fake):
    fake(frames={"fct_otc_haruko_pnl_position_history_eod": _perps_frame}, fail_on=("fct_otc_haruko_position_summary",))
    out = get_perp_positions.invoke({})
    assert "live position_summary unavailable: RuntimeError" in out and "| BTC-PERPETUAL |" in out
    fake(empty=True)
    assert "No open futures/perp positions" in get_perp_positions.invoke({})


# ----------------------------------------------------------------------------
# get_desk_positions_by_symbol
# ----------------------------------------------------------------------------

def test_positions_by_underlying(fake):
    bq = fake(frames={"fct_otc_haruko_pnl_position_history_eod": _by_underlying_frame})
    out = get_desk_positions_by_symbol.invoke({"top_n": 3})
    assert out.startswith("### Desk exposure by underlying (Haruko EOD 2026-09-09 23:58 UTC, all entities; top 3 of 4 underlyings")
    # BTC aggregated over OPTIONS + SPOT, mix column spot/opt/fut
    assert "| BTC | btc | $7,426,818,441 | -$102,879,489 | $6,788,428 | $166,691 | -$467,476 | -$35,352 | -$17,358,829 | 40/2068/0 | 10% |" in out
    # top 3 by gross notional: BTC, USDC, PAXG (HYPE at $623K is cut)
    assert "| USDC | usdc |" in out and "| PAXG | paxg |" in out and "| HYPE |" not in out
    assert "Stablecoin/fiat rows" in out
    # stablecoins excluded from the token hint; paxg reported as not covered
    assert "in the market-data universe (cross-check with get_zscore_signals / get_token_metrics): btc; not covered by the market-data tools: paxg" in out
    assert "GROUP BY p.underlying_asset, p.instrument_type" in bq.sqls[0] and "UPPER(p.underlying_asset)" not in bq.sqls[0]


def test_positions_for_symbol_with_live_spot(fake):
    bq = fake(frames={"fct_otc_haruko_pnl_position_history_eod": _btc_breakdown_frame,
                      "fct_otc_haruko_position_summary": _btc_live_spot_frame})
    out = get_desk_positions_by_symbol.invoke({"symbol": "BTC-PERP"})
    assert out.startswith("### Desk exposure in BTC (Haruko EOD 2026-09-09 23:58 UTC, all entities)")
    assert "total: 976 positions, gross notional $2,891,861,584, net delta -$376,645,576 (short)" in out
    assert "valid pricers 9.3%" in out
    assert "live reconciled position (2026-09-10 19:53 UTC): spot 17,698.98 BTC (delta value -$21,990,262, Partially Reconciled" in out
    assert "| OPTIONS | DERIBIT | 973 | $2,860,477,548 | -$345,261,540 |" in out and "| 90/973 |" in out
    assert "UPPER(p.underlying_asset) = 'BTC'" in bq.sqls[0] and "UPPER(symbol) = 'BTC'" in bq.sqls[1]
    assert "cross-check with get_zscore_signals" in out


def test_positions_by_symbol_bad_input_and_empty(fake):
    fake(empty=True)
    assert "No positions with underlying HYPE" in get_desk_positions_by_symbol.invoke({"symbol": "hype"})
    assert "No positions in the latest EOD" in get_desk_positions_by_symbol.invoke({})
    assert "unsupported characters" in get_desk_positions_by_symbol.invoke({"symbol": "btc'; DROP"}) \
        or "Could not interpret" in get_desk_positions_by_symbol.invoke({"symbol": "btc'; DROP"})


# ----------------------------------------------------------------------------
# get_otc_derivatives_trades
# ----------------------------------------------------------------------------

def test_otc_trades(fake):
    bq = fake(frames={"GROUP BY 1, 2, 3": _otc_summary_frame, "fct_otcderivatives_trades": _otc_trades_frame})
    out = get_otc_derivatives_trades.invoke({"days": 14, "top_n": 5})
    assert out.startswith("### OTC derivatives trades, last 14 days")
    assert "blotter latest exec 2026-09-10 16:27 UTC" in out
    assert "149 trades, notional $1,081,687,489 in the window; by status: ACTIVE 138 ($1,039,423,000); CANCELED 11 ($42,264,489)" in out
    assert "by underlying / type: BTC CALL 138 ($1,039,423,000); ETH PUT 11 ($42,264,489)" in out
    assert "| 2026-09-10 16:27 UTC | ACTIVE | BUY | BTC/USD CALL | 55 | $4,290,000 | 78,000 | 0.0589 BASE | 2026-11-12 | Example Trust LLC | A1, LTD. |" in out
    assert "| CANCELED | SELL | ETH/USD PUT | 150 | $330,000 | 2,200 | 5.7022 QUOTE |" in out
    assert "INTERVAL 14 DAY" in bq.sqls[0] and "LIMIT 5" in bq.sqls[0]
    assert "dim_otcderivatives_accounts" in bq.sqls[0] and "dim_otcderivatives_entities" in bq.sqls[0]
    fake(empty=True)
    assert "No OTC derivatives trades executed in the last 30 days" in get_otc_derivatives_trades.invoke({})


# ----------------------------------------------------------------------------
# get_open_orders
# ----------------------------------------------------------------------------

def test_open_orders(fake):
    fake(frames={"fct_a1_talos_open_orders_live": _orders_frame})
    out = get_open_orders.invoke({})
    assert out.startswith("### Open Talos orders (as of 2026-09-10 19:36 UTC; 2 orders)")
    assert "by symbol/side: BTC-USD sell x1; XRP-USD sell x1" in out
    assert "| XRP-USD | Sell | 5,000 XRP | 200.00 | 0 | 5,000 | - | Active | N/A | - | - |" in out
    assert "| BTC-USD | Sell | 41.4605 BTC | 99,099.00 | 0 | 41.4605 | Limit | Active | N/A | 12.5 | 2026-09-11 08:00:00 |" in out
    fake(empty=True)
    assert "No open Talos orders" in get_open_orders.invoke({})


# ----------------------------------------------------------------------------
# get_internal_price
# ----------------------------------------------------------------------------

def test_internal_price(fake):
    bq = fake(frames={"fct_current_asset_prices_live": _live_price_frame, "intraday_price": _intraday_frame})
    out = get_internal_price.invoke({"asset": "btc", "hours": 6})
    assert out.startswith("### Internal price: BTC")
    assert "- live: $77,147.04 (datastream_anchorage, 2026-09-10 19:32 UTC)" in out
    assert "first $77,000.00, last $77,385.00, change +0.50%, high $77,550.00, low $76,940.00" in out
    assert "| 2026-09-10 18:00 | $77,385.00 | $77,435.00 | $77,325.00 |" in out
    assert "UPPER(canonical_symbol) = 'BTC'" in bq.sqls[0]
    assert "base_symbol = 'BTC' AND quote_symbol = 'USD'" in bq.sqls[1] and "INTERVAL 6 HOUR" in bq.sqls[1]
    assert "in the market-data universe" in out and "btc" in out


def test_internal_price_validation_and_partial(fake):
    fake(frames={"fct_current_asset_prices_live": _live_price_frame, "intraday_price": pd.DataFrame})
    assert "Invalid asset symbol" in get_internal_price.invoke({"asset": "BTC; DROP"})
    out = get_internal_price.invoke({"asset": "BTC", "hours": 500})
    assert "- live: $77,147.04" in out and "no USD minute bars for BTC in the last 168 hours" in out
    fake(empty=True)
    assert "No internal price for XYZ" in get_internal_price.invoke({"asset": "xyz"})


# ----------------------------------------------------------------------------
# catalog tools
# ----------------------------------------------------------------------------

def test_list_desk_tables(fake):
    fake()
    out = list_desk_tables.invoke({"keyword": "venue"})
    assert "1 match 'venue'" in out
    assert "| brokerage_a1.fct_otc_haruko_pnl_by_venue_history | VIEW | n/a | - | 2 | venue |" in out
    out = list_desk_tables.invoke({})
    assert "3 match" in out and "| pricing.current_price | TABLE | 3,378 |" in out
    assert "No table in brokerage_a1, pricing matches 'zzz'" in list_desk_tables.invoke({"keyword": "zzz"})


def test_describe_desk_table(fake):
    fake()
    out = describe_desk_table.invoke({"table": "brokerage_a1.fct_otc_haruko_pnl_portfolio"})
    assert out.startswith("### anc-global-markets.brokerage_a1.fct_otc_haruko_pnl_portfolio (TABLE)")
    assert "rows 65,467; size 47.68 MB; modified 2026-09-10 19:35 UTC; partitioning DAY on position_timestamp; clustered by position_timestamp" in out
    assert "| total_delta_usd | FLOAT | USD delta |" in out
    assert "Filter on the partition column" in out
    out = describe_desk_table.invoke({"table": "hold.fct_hold_spot"})
    assert out.startswith("Rejected by the read-only guard") and "brokerage_a1, pricing" in out


# ----------------------------------------------------------------------------
# query_desk_data
# ----------------------------------------------------------------------------

def test_query_desk_data_guard_before_bigquery(fake):
    bq = fake()
    out = query_desk_data.invoke({"sql": "SELECT * FROM `anc-global-markets.hold.fct_hold_spot`"})
    assert out.startswith("Rejected by the read-only guard") and "'hold'" in out and bq.sqls == []
    out = query_desk_data.invoke({"sql": "DELETE FROM `anc-global-markets.brokerage_a1.x`"})
    assert "Rejected by the read-only guard" in out and bq.sqls == []


def test_query_desk_data_result_table_and_cap(fake):
    def frame():
        return pd.DataFrame({"as_of_date": [date(2026, 9, 10)] * 50, "entity_id": [20] * 50,
                             "total_year_to_date_pnl": [-32_239_478.92] * 50, "flag": ["x|y"] * 50})

    bq = fake(frames={"fct_otc_haruko_pnl_portfolio_history_eod": frame})
    out = query_desk_data.invoke({"sql": "SELECT as_of_date, entity_id, total_year_to_date_pnl, flag FROM "
                                         "`anc-global-markets.brokerage_a1.fct_otc_haruko_pnl_portfolio_history_eod` LIMIT 5000"})
    assert out.startswith("### Query result: 50 row(s), showing first 40")
    assert "| 2026-09-10 | 20 | -32,239,478.92 | x\\|y |" in out
    assert out.count("| 2026-09-10 |") == 40
    assert "Bytes scanned: 4.00 KB." in out
    assert bq.sqls[0].endswith("LIMIT 200")
    fake(empty=True)
    assert "Query returned no rows" in query_desk_data.invoke({"sql": "SELECT 1 FROM `anc-global-markets.pricing.current_price`"})


def test_query_desk_data_surfaces_bigquery_errors(fake):
    fake(fail_on=("current_price",))
    out = query_desk_data.invoke({"sql": "SELECT 1 FROM `anc-global-markets.pricing.current_price`"})
    assert out.startswith("Error querying desk data in query_desk_data: RuntimeError: boom")
