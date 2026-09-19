"""tools/snapshot_tools.py against the fake in-memory snapshot store. No network."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from tests.test_snapshot_daily import FakeSnapshotStore
from tools import snapshot_tools as st
from tools.snapshot_tools import (
    SNAPSHOT_TOOL_NAMES,
    UNAVAILABLE,
    compare_to_snapshot,
    get_snapshot_history,
    get_snapshot_tools,
    list_snapshot_metrics,
)

TODAY = date(2026, 9, 11)


def _seed(store: FakeSnapshotStore, days: int = 5, today: date = TODAY):
    """`days` daily snapshots ending today for a few metrics in every source."""
    for i in range(days):
        d = today - timedelta(days=days - 1 - i)
        delta = -50_000_000.0 + i * 1_000_000            # rises 1m/day
        store.put_snapshot(d, "haruko", "combined", "delta_usd", delta, json.dumps({"as_of": f"{d}T19:00:00"}))
        store.put_snapshot(d, "haruko", "20", "delta_usd", delta - 10e6, None)
        store.put_snapshot(d, "haruko", "combined", "valid_pricer_pct", 14.0 + i, None)
        store.put_snapshot(d, "haruko", "combined", "data_quality_flag", 0.0,
                           json.dumps({"flags": {"20": "High Invalid Pricer Rate", "86": "Normal"}}))
        store.put_snapshot(d, "signals", "btc", "funding_rate", 8.0 + i * 0.5, json.dumps({"as_of": str(d - timedelta(days=1)), "unit": "annualised %"}))
        store.put_snapshot(d, "signals", "btc", "spot_volume_z", -1.0 + i, None)
        store.put_snapshot(d, "signals", "eth", "price", 4_000.0 - i * 10, None)
        store.put_snapshot(d, "sheet", "TOTAL", "mtd_pnl_usd", 1_000_000.0 + i * 50_000, json.dumps({"month": 9}))
        store.put_snapshot(d, "sheet", "A1", "mtd_take_rate_bps", 7.0, None)
    return store


@pytest.fixture
def store(monkeypatch):
    s = _seed(FakeSnapshotStore())
    monkeypatch.setattr(st, "_get_store", lambda: s)
    monkeypatch.setattr(st, "datetime", _FixedDatetime)
    return s


class _FixedDatetime:
    """compare_to_snapshot uses datetime.now(timezone.utc).date() for the window."""
    @staticmethod
    def now(tz=None):
        from datetime import datetime as _dt
        return _dt(2026, 9, 11, 12, 0, tzinfo=tz)

    @staticmethod
    def strptime(s, fmt):
        from datetime import datetime as _dt
        return _dt.strptime(s, fmt)


@pytest.fixture
def no_store(monkeypatch):
    monkeypatch.setattr(st, "_get_store", lambda: None)


# ----------------------------------------------------------------------------
# registry / availability / argument validation
# ----------------------------------------------------------------------------

def test_registry():
    tools = get_snapshot_tools()
    assert SNAPSHOT_TOOL_NAMES == ["get_snapshot_history", "compare_to_snapshot", "list_snapshot_metrics"]
    for t in tools:
        assert t.description and "Args" in t.description or t.name == "list_snapshot_metrics"


def test_unavailable_store(no_store):
    assert get_snapshot_history.invoke({"source": "haruko", "metric": "delta_usd"}) == UNAVAILABLE
    assert compare_to_snapshot.invoke({"source": "haruko", "metric": "delta_usd", "date": "2026-09-01"}) == UNAVAILABLE
    assert list_snapshot_metrics.invoke({}) == UNAVAILABLE


def test_bad_arguments(store):
    assert get_snapshot_history.invoke({"source": "nope", "metric": "x"}).startswith("Unknown source 'nope'")
    assert "pass the token as `entity`" in get_snapshot_history.invoke({"source": "signals", "metric": "price"})
    assert compare_to_snapshot.invoke({"source": "haruko", "metric": "delta_usd", "date": "yesterday"}).startswith("Bad date")
    assert list_snapshot_metrics.invoke({"source": "x"}).startswith("Unknown source")


def test_store_error_is_reported_not_raised(store, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("locked")
    monkeypatch.setattr(store, "get_snapshot_series", boom)
    out = get_snapshot_history.invoke({"source": "haruko", "metric": "delta_usd"})
    assert out.startswith("Error reading snapshots in get_snapshot_history") and "locked" in out


# ----------------------------------------------------------------------------
# get_snapshot_history
# ----------------------------------------------------------------------------

def test_history_haruko_default_entity_combined(store):
    out = get_snapshot_history.invoke({"source": "haruko", "metric": "delta_usd"})
    assert out.startswith("**haruko / combined desk / delta_usd** - 5 daily snapshot(s), 2026-09-07 to 2026-09-11")
    assert "Latest (2026-09-11): -$46,000,000; first (2026-09-07): -$50,000,000; change +$4,000,000 (+8.0%)." in out
    assert "Min -$50,000,000 on 2026-09-07; max -$46,000,000 on 2026-09-11." in out
    assert "Latest row context: as_of=2026-09-11T19:00:00." in out
    assert "| Date | delta_usd |" in out and "| 2026-09-09 | -$48,000,000 |" in out


def test_history_entity_aliases_and_days(store):
    out = get_snapshot_history.invoke({"source": "haruko", "metric": "delta_usd", "entity": "A1", "days": 2})
    assert "**haruko / Derivs Risk (entity 20) / delta_usd** - 2 daily snapshot(s), 2026-09-10 to 2026-09-11" in out
    assert "-$56,000,000" in out
    assert get_snapshot_history.invoke({"source": "haruko", "metric": "delta_usd", "entity": "adsd"}).startswith(
        "No `haruko` snapshots for entity `86`")


def test_history_signals_and_sheet_formatting(store):
    out = get_snapshot_history.invoke({"source": "signals", "metric": "funding_rate", "entity": "BTC"})
    assert "**signals / BTC / funding_rate**" in out
    assert "Latest (2026-09-11): 10.00%; first (2026-09-07): 8.00%; change +2.00%." in out   # no relative % for rates
    assert "as_of=2026-09-10, unit=annualised %" in out
    out = get_snapshot_history.invoke({"source": "signals", "metric": "spot_volume_z", "entity": "btc"})
    assert "Latest (2026-09-11): +3.00; first (2026-09-07): -1.00; change +4.00." in out
    out = get_snapshot_history.invoke({"source": "sheet", "metric": "mtd_pnl_usd"})
    assert "**sheet / TOTAL / mtd_pnl_usd**" in out and "$1,200,000" in out and "(+20.0%)" in out
    out = get_snapshot_history.invoke({"source": "sheet", "metric": "mtd_take_rate_bps", "entity": "a1"})
    assert "7.00 bps" in out and "change +0.00 bps." in out


def test_history_data_quality_flag_table(store):
    out = get_snapshot_history.invoke({"source": "haruko", "metric": "data_quality_flag"})
    assert "| Date | Data quality |" in out and "| 2026-09-11 | 20: High Invalid Pricer Rate; 86: Normal |" in out
    assert "change" not in out


def test_history_no_data_hint(store):
    out = get_snapshot_history.invoke({"source": "sheet", "metric": "nope", "entity": "HOLD"})
    assert out == ("No `sheet` snapshots for entity `HOLD`, metric `nope` in the last 30 day(s). "
                   "Use list_snapshot_metrics('sheet') to see what has been captured.")


def test_series_normalisation_tolerates_row_shapes():
    class Obj:
        def __init__(self, d, v):
            self.snapshot_date, self.value, self.value_json = d, v, None

    rows = st._series(type("S", (), {"get_snapshot_series": staticmethod(lambda *a: [
        ("2026-09-02", 2, '{"a":1}'), {"snapshot_date": date(2026, 9, 1), "value": "1.5"}, Obj("2026-09-03", None), "garbage",
    ])})(), "haruko", "combined", "delta_usd", 30)
    assert [(str(d), v, j) for d, v, j in rows] == [("2026-09-01", 1.5, None), ("2026-09-02", 2.0, {"a": 1}), ("2026-09-03", None, None)]


# ----------------------------------------------------------------------------
# compare_to_snapshot
# ----------------------------------------------------------------------------

def test_compare_exact_date(store):
    out = compare_to_snapshot.invoke({"source": "haruko", "metric": "delta_usd", "date": "2026-09-08"})
    assert out.startswith("**haruko / combined desk / delta_usd** - latest 2026-09-11 vs 2026-09-08")
    assert "closest" not in out
    assert "| then | 2026-09-08 | -$49,000,000 |" in out and "| latest | 2026-09-11 | -$46,000,000 |" in out
    assert "Change: +$3,000,000 (+6.1%) over 3 day(s)." in out


def test_compare_falls_back_to_closest_earlier_snapshot(store):
    # remove the 09-09 row so 09-09 falls back to 09-08
    del store.rows[("2026-09-09", "haruko", "combined", "delta_usd")]
    out = compare_to_snapshot.invoke({"source": "haruko", "metric": "delta_usd", "date": "2026-09-09"})
    assert "latest 2026-09-11 vs 2026-09-08 (closest snapshot on or before 2026-09-09)" in out


def test_compare_before_history_and_same_day(store):
    out = compare_to_snapshot.invoke({"source": "signals", "metric": "price", "entity": "eth", "date": "2026-08-01"})
    assert out.startswith("No `signals` snapshot of `price` for ETH on or before 2026-08-01; earliest available is 2026-09-07 ($4,000)")
    out = compare_to_snapshot.invoke({"source": "signals", "metric": "price", "entity": "eth", "date": "2026-09-11"})
    assert "Only one snapshot in range (2026-09-11): $3,960. Nothing to compare yet." in out
    out = compare_to_snapshot.invoke({"source": "signals", "metric": "price", "entity": "eth", "date": "2026-09-10"})
    assert "Change: -$10.00 (-0.3%) over 1 day(s)." in out


def test_compare_data_quality_has_no_change_line(store):
    out = compare_to_snapshot.invoke({"source": "haruko", "metric": "data_quality_flag", "date": "2026-09-07"})
    assert "| then | 2026-09-07 | 20: High Invalid Pricer Rate; 86: Normal |" in out and "Change:" not in out


# ----------------------------------------------------------------------------
# list_snapshot_metrics
# ----------------------------------------------------------------------------

def test_list_metrics_all_sources(store):
    out = list_snapshot_metrics.invoke({})
    assert "**haruko** - latest snapshot: 2026-09-11" in out
    assert "- combined desk: data_quality_flag, delta_usd, valid_pricer_pct" in out
    assert "- Derivs Risk (entity 20): delta_usd" in out
    assert "**signals** - latest snapshot: 2026-09-11" in out
    assert "- tokens (2): btc, eth" in out and "- metrics (3): funding_rate, price, spot_volume_z" in out
    assert "**sheet**" in out and "- TOTAL: mtd_pnl_usd" in out and "- A1: mtd_take_rate_bps" in out


def test_list_metrics_single_source_and_empty(store):
    out = list_snapshot_metrics.invoke({"source": "sheet"})
    assert out.startswith("**sheet** - latest snapshot: 2026-09-11") and "haruko" not in out
    store.rows.clear()
    out = list_snapshot_metrics.invoke({"source": "haruko"})
    assert "latest snapshot: none yet" in out and "no snapshots captured yet" in out


def test_metrics_normalisation_shapes():
    class S:
        def list_snapshot_metrics(self, source):
            return ["a", ("e1", "m1"), {"entity": "e2", "metric": "m2"}, ["only"]]
    assert st._metrics(S(), "x") == {"*": ["a", "only"], "e1": ["m1"], "e2": ["m2"]}


def test_format_helpers():
    assert st._fmt("delta_usd", -1234.5) == "-$1,234" and st._fmt("day_pnl", -12.345) == "-$12.35"
    assert st._fmt("spot_volume_z", 1.234) == "+1.23"
    assert st._fmt("mtd_take_rate_bps", 6.5) == "6.50 bps"
    assert st._fmt("valid_pricer_pct", 14.0) == "14.00%"
    assert st._fmt("pcr_oi", 0.61234) == "0.612"
    assert st._fmt("price", None) == "n/a"
    assert st._fmt("data_quality_flag", 1.0) == "Normal" and st._fmt("data_quality_flag", 0.0) == "flagged"
    assert st._delta("delta_usd", 100.0, 150.0) == "+$50.00 (+50.0%)"
    assert st._delta("delta_usd", 1_000_000.0, 1_500_000.0) == "+$500,000 (+50.0%)"
    assert st._delta("delta_usd", 0.0, 150.0) == "+$150.00"
    assert st._delta("funding_rate", 8.0, 7.0) == "-1.00%"
    assert st._delta("price", None, 1.0) == "n/a"
