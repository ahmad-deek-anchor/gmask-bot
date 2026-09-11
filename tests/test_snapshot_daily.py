"""snapshot_daily.py with fake providers and a fake in-memory snapshot store.
No network, no GCP, no sqlite file.

The fake BigQuery reuses tests.test_desk_tools.FakeBQ (same canned portfolio frame as
the desk tools) and the fake sheet reuses tests.test_sheet_tools.make_sheet, so the
haruko / sheet captures run through the real query / parsing code.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

import snapshot_daily as sd
from snapshot_daily import (
    HARUKO_METRICS,
    Row,
    SourceResult,
    capture_haruko,
    capture_sheet,
    capture_signals,
    main,
    run,
)
from tests.test_desk_tools import FakeBQ, _snapshot_frame
from tests.test_sheet_tools import make_sheet

TODAY = date(2026, 9, 11)


# ----------------------------------------------------------------------------
# fake store implementing the spec'd snapshot API (shared with test_snapshot_tools)
# ----------------------------------------------------------------------------

class FakeSnapshotStore:
    """Dict keyed on the table's primary key (snapshot_date, source, entity, metric)."""

    def __init__(self, fail=False):
        self.rows = {}
        self.puts = 0
        self.fail = fail

    def put_snapshot(self, snapshot_date, source, entity, metric, value, value_json=None):
        if self.fail:
            raise RuntimeError("disk full")
        d = snapshot_date if isinstance(snapshot_date, str) else snapshot_date.isoformat()
        self.puts += 1
        self.rows[(d, source, entity, metric)] = {
            "snapshot_date": d, "source": source, "entity": entity, "metric": metric,
            "value": value, "value_json": value_json, "captured_at": datetime(2026, 9, 11, 23, 30).isoformat(),
        }

    def get_snapshot_series(self, source, entity, metric, days):
        latest = self.latest_snapshot_date(source)
        if latest is None:
            return []
        cutoff = (date.fromisoformat(latest) - timedelta(days=days - 1)).isoformat()
        out = [r for (d, s, e, m), r in self.rows.items()
               if s == source and e == entity and m == metric and d >= cutoff]
        return sorted(out, key=lambda r: r["snapshot_date"])

    def latest_snapshot_date(self, source):
        ds = [d for (d, s, _, _) in self.rows if s == source]
        return max(ds) if ds else None

    def list_snapshot_metrics(self, source):
        return sorted({(e, m) for (_, s, e, m) in self.rows if s == source})

    # test helpers
    def count(self, source=None):
        return sum(1 for (_, s, _, _) in self.rows if source is None or s == source)

    def get(self, source, entity, metric, d=TODAY):
        return self.rows[(d.isoformat(), source, entity, metric)]


# ----------------------------------------------------------------------------
# signals fixtures: canned fetch / calc so no provider is built
# ----------------------------------------------------------------------------

def fake_fetch(tokens, start, end, include_options=True):
    frames = {}
    for t in tokens:
        if t == "dead":
            continue                      # token without price data is skipped by the real fetch
        frames[t] = pd.DataFrame({"time": pd.date_range("2026-08-01", periods=40), "price": 1.0})
    return frames


def fake_calc(token_data, window=30):
    out = {}
    for token in token_data:
        metrics = {
            "spot_volume": {"value": 1_000_000.0, "z_score": 2.7, "is_outlier": True, "is_significant": True},
            "perp_oi": {"value": 5e9, "z_score": -0.4, "is_outlier": False, "is_significant": False},
            "total_liquidations": {"value": float("nan"), "z_score": float("nan"), "is_outlier": False, "is_significant": False},
            "funding_rate": {"value_annual_pct": 10.95, "value_8h_pct": 0.01},
            "price": {"value": 78_000.0, "pct_change_1d": -1.25},
        }
        if token == "btc":
            metrics["dvol_close"] = {"value": 48.2, "z_score": 1.1, "is_outlier": False, "is_significant": True}
            metrics["skew_25d_30d"] = {"value": -3.1, "change_7d": 0.8}
        out[token] = {"latest_date": "2026-09-10", "metrics": metrics, "has_outliers": True,
                      "has_significant_moves": True, "options_listed": token == "btc"}
    return out


def _by_metric(rows, entity):
    return {r.metric: r for r in rows if r.entity == entity}


# ----------------------------------------------------------------------------
# haruko
# ----------------------------------------------------------------------------

def test_capture_haruko_rows_and_combined():
    bq = FakeBQ(frames={"fct_otc_haruko_pnl_portfolio": _snapshot_frame})
    res = capture_haruko(bq)
    assert res.ok and res.calls == {"bigquery_queries": 1}
    # query shape: latest row per entity, all needed columns, read-only guard applied
    sql = bq.sqls[-1]
    assert "ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY position_timestamp DESC)" in sql
    assert "fct_otc_haruko_pnl_portfolio" in sql and "total_gamma_percent_usd" in sql

    a1, adsd, comb = _by_metric(res.rows, "20"), _by_metric(res.rows, "86"), _by_metric(res.rows, "combined")
    expected = set(HARUKO_METRICS) | {"valid_pricer_pct", "data_quality_flag"}
    assert set(a1) == set(adsd) == set(comb) == expected
    assert a1["delta_usd"].value == -60_000_000.0 and adsd["delta_usd"].value == 21_000_000.0
    assert comb["delta_usd"].value == -39_000_000.0
    assert comb["ytd_pnl"].value == pytest.approx(-32_500_000.0 + 43_300_000.0)
    assert comb["gross_notional"].value == 2 * 13_900_000_000.0
    assert a1["gamma_pct_usd"].value == 3_278_657.50 and a1["theta"].value == -231_424.62
    # data quality: numeric 1/0 plus the text in value_json
    assert a1["data_quality_flag"].value == 0.0 and adsd["data_quality_flag"].value == 1.0
    assert json.loads(a1["data_quality_flag"].value_json)["flag"] == "High Invalid Pricer Rate"
    cj = json.loads(comb["data_quality_flag"].value_json)
    assert cj["flags"] == {"20": "High Invalid Pricer Rate", "86": "Normal"} and comb["data_quality_flag"].value == 0.0
    # combined valid pricer pct is count-weighted (348 valid / 2144 invalid per entity in the fixture)
    assert comb["valid_pricer_pct"].value == pytest.approx(348 / (348 + 2144) * 100)
    assert json.loads(a1["delta_usd"].value_json)["as_of"].startswith("2026-09-10T19:33:02")
    assert len(res.rows) == 3 * len(expected)


def test_capture_haruko_single_entity_has_no_combined():
    bq = FakeBQ(frames={"fct_otc_haruko_pnl_portfolio": lambda: _snapshot_frame().iloc[:1]})
    res = capture_haruko(bq)
    assert {r.entity for r in res.rows} == {"20"}


def test_capture_haruko_errors(monkeypatch):
    with pytest.raises(RuntimeError, match="No portfolio snapshot"):
        capture_haruko(FakeBQ(empty=True))
    with pytest.raises(RuntimeError, match="boom"):
        capture_haruko(FakeBQ(fail_on=("fct_otc_haruko_pnl_portfolio",)))
    monkeypatch.setattr(sd, "_get_bq", lambda: None)      # no ADC / library
    with pytest.raises(RuntimeError, match="unavailable"):
        capture_haruko()


# ----------------------------------------------------------------------------
# signals
# ----------------------------------------------------------------------------

def test_capture_signals_metric_names():
    res = capture_signals(tokens=["BTC", "eth", "dead"], fetch=fake_fetch, calc=fake_calc)
    assert res.ok
    assert res.calls == {"tokens_requested": 3, "tokens_with_data": 2, "tokens_with_signals": 2}
    btc, eth = _by_metric(res.rows, "btc"), _by_metric(res.rows, "eth")
    assert set(eth) == {"spot_volume", "spot_volume_z", "perp_oi", "perp_oi_z",
                        "funding_rate", "price", "price_pct_change_1d"}
    # NaN value / z -> no row; options metrics only where present
    assert "total_liquidations" not in eth and "dvol_close" not in eth
    assert set(btc) == set(eth) | {"dvol_close", "dvol_close_z", "skew_25d_30d", "skew_25d_30d_chg7d"}
    assert btc["spot_volume"].value == 1_000_000.0 and btc["spot_volume_z"].value == 2.7
    assert btc["funding_rate"].value == 10.95 and btc["price_pct_change_1d"].value == -1.25
    assert btc["skew_25d_30d_chg7d"].value == 0.8
    vj = json.loads(btc["spot_volume"].value_json)
    assert vj == {"as_of": "2026-09-10", "z_score": 2.7, "is_outlier": True, "is_significant": True}
    assert json.loads(btc["funding_rate"].value_json)["unit"] == "annualised %"
    assert not any(r.entity == "dead" for r in res.rows)


def test_capture_signals_default_universe(monkeypatch):
    seen = {}

    def fetch(tokens, start, end, include_options=True):
        seen["tokens"] = list(tokens)
        seen["days"] = (end - start).days
        return {}

    res = capture_signals(fetch=fetch, calc=lambda d: {})
    from tools.metrics import FULL_TOKEN_UNIVERSE
    assert seen["tokens"] == FULL_TOKEN_UNIVERSE and seen["days"] == 45
    assert res.rows == [] and not res.ok


# ----------------------------------------------------------------------------
# sheet
# ----------------------------------------------------------------------------

def test_capture_sheet_mtd_ytd_week():
    res = capture_sheet(make_sheet(), today=TODAY)
    assert res.ok and res.calls == {"sheet_ranges": 2}
    hold, a1, tot = (_by_metric(res.rows, e) for e in ("HOLD", "A1", "TOTAL"))
    # months 1-2 populated in the fixture -> "current month" falls back to the latest populated (Feb)
    assert json.loads(hold["mtd_pnl_usd"].value_json)["month"] == 2
    assert set(hold) == {"mtd_volume_usd", "mtd_pnl_usd", "mtd_take_rate_bps",
                         "ytd_volume_usd", "ytd_pnl_usd", "ytd_take_rate_bps", "week_pnl_usd"}
    assert set(a1) == set(hold) | {"week_realized_pnl_usd", "week_unrealized_pnl_usd"}
    assert set(tot) == set(hold) | {"ytd_target_pnl_usd", "ytd_pct_of_target"}
    assert a1["ytd_pnl_usd"].value == 2_600_000 and tot["ytd_volume_usd"].value == 4_900_000_000
    assert tot["ytd_target_pnl_usd"].value == 60_000_000
    assert json.loads(tot["ytd_pnl_usd"].value_json)["basis"] == "sheet YTD row"
    # latest weekly row is week 36 (37 is in progress on the as-of date and dropped by the parser)
    wj = json.loads(tot["week_pnl_usd"].value_json)
    assert wj["week"] == 36 and wj["start_date"].startswith("2026-")
    assert a1["week_pnl_usd"].value == pytest.approx(a1["week_realized_pnl_usd"].value + a1["week_unrealized_pnl_usd"].value)
    assert tot["week_pnl_usd"].value == pytest.approx(hold["week_pnl_usd"].value + a1["week_pnl_usd"].value)


def test_capture_sheet_partial_and_total_failure():
    class Monthly:
        def monthly_volume_pnl(self):
            return make_sheet().monthly_volume_pnl()

        def weekly_pnl(self):
            raise RuntimeError("weekly tab missing")

    res = capture_sheet(Monthly(), today=TODAY)
    assert res.ok and not any(r.metric.startswith("week_") for r in res.rows)

    class Broken:
        def monthly_volume_pnl(self):
            raise RuntimeError("403 forbidden")

        def weekly_pnl(self):
            raise RuntimeError("403 forbidden")

    with pytest.raises(RuntimeError, match="monthly: RuntimeError: 403 forbidden; weekly"):
        capture_sheet(Broken(), today=TODAY)

def test_capture_sheet_unavailable(monkeypatch):
    monkeypatch.setattr(sd, "_get_sheet", lambda: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        capture_sheet()


# ----------------------------------------------------------------------------
# run(): isolation, idempotency, dry run, exit code
# ----------------------------------------------------------------------------

def _captures(**overrides):
    caps = {
        "haruko": lambda: capture_haruko(FakeBQ(frames={"fct_otc_haruko_pnl_portfolio": _snapshot_frame})),
        "signals": lambda: capture_signals(tokens=["btc", "eth"], fetch=fake_fetch, calc=fake_calc),
        "sheet": lambda: capture_sheet(make_sheet(), today=TODAY),
    }
    caps.update(overrides)
    return caps


def test_run_writes_all_sources_and_is_idempotent():
    store = FakeSnapshotStore()
    results = run(["haruko", "signals", "sheet"], TODAY, store=store, captures=_captures())
    assert all(r.error is None for r in results.values())
    counts = {s: store.count(s) for s in results}
    assert counts == {s: len(r.rows) for s, r in results.items()}
    assert counts["haruko"] == 36 and counts["signals"] == 7 + 11 and counts["sheet"] == 7 + 9 + 9
    assert all(r.written == len(r.rows) for r in results.values())
    assert store.get("haruko", "combined", "delta_usd")["value"] == -39_000_000.0
    assert store.get("signals", "btc", "spot_volume_z")["value"] == 2.7
    assert store.get("sheet", "TOTAL", "ytd_pnl_usd")["value"] == 3_420_000

    # second run: same PKs -> upsert, no growth
    before = store.count()
    puts_before = store.puts
    run(["haruko", "signals", "sheet"], TODAY, store=store, captures=_captures())
    assert store.count() == before and store.puts == 2 * puts_before
    # a different date adds rows
    run(["haruko"], date(2026, 9, 12), store=store, captures=_captures())
    assert store.count("haruko") == 2 * counts["haruko"]


def test_run_isolates_failing_sources():
    def boom():
        raise RuntimeError("bigquery down")

    store = FakeSnapshotStore()
    results = run(["haruko", "signals", "sheet"], TODAY, store=store, captures=_captures(haruko=boom))
    assert results["haruko"].error == "RuntimeError: bigquery down" and results["haruko"].rows == []
    assert results["signals"].error is None and results["sheet"].error is None
    assert store.count("haruko") == 0 and store.count("signals") > 0 and store.count("sheet") > 0

    # capture that returns nothing is a failure too
    results = run(["signals"], TODAY, store=store, captures={"signals": lambda: SourceResult("signals")})
    assert results["signals"].error == "no rows captured"

    # store failure is reported per source and does not raise
    results = run(["sheet"], TODAY, store=FakeSnapshotStore(fail=True), captures=_captures())
    assert results["sheet"].error.startswith("store write failed: RuntimeError: disk full")
    assert results["sheet"].written == 0


def test_run_dry_run_never_touches_store(monkeypatch):
    store = FakeSnapshotStore()
    monkeypatch.setattr(sd, "_get_store", lambda: (_ for _ in ()).throw(AssertionError("store built in dry run")))
    results = run(["haruko"], TODAY, store=None, dry_run=True, captures=_captures())
    assert results["haruko"].error is None and len(results["haruko"].rows) == 36 and results["haruko"].written == 0
    assert store.count() == 0


def test_main_cli(monkeypatch, capsys):
    store = FakeSnapshotStore()
    monkeypatch.setattr(sd, "_get_store", lambda: store)
    monkeypatch.setattr(sd, "CAPTURES", _captures())

    assert main(["--sources", "haruko,sheet", "--date", "2026-09-11"]) == 0
    out = capsys.readouterr().out
    assert "Snapshot 2026-09-11" in out and "- haruko: 36 rows, 36 written" in out and "- sheet: 25 rows" in out
    assert "calls bigquery_queries=1: ok" in out and store.count() == 36 + 25

    # dry run prints rows, writes nothing
    assert main(["--sources", "signals", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "(dry run)" in out and "btc        spot_volume_z" in out and store.count("signals") == 0

    # all sources failing -> exit 1; one failing -> 0
    def boom():
        raise RuntimeError("down")
    monkeypatch.setattr(sd, "CAPTURES", _captures(haruko=boom, sheet=boom))
    assert main(["--sources", "haruko,sheet"]) == 1
    assert "FAILED (RuntimeError: down)" in capsys.readouterr().out
    assert main(["--sources", "haruko,signals"]) == 0

    # argument validation
    assert main(["--sources", "nope"]) == 2
    assert main(["--date", "11/09/2026"]) == 2


def test_parse_date_default_is_utc_today():
    assert sd.parse_date(None) == sd.today_utc()
    assert sd.parse_date("2026-01-31") == date(2026, 1, 31)


def test_row_helpers():
    assert sd._num(float("nan")) is None and sd._num("x") is None and sd._num("3.5") == 3.5
    assert sd._num(pd.NA if hasattr(pd, "NA") else None) is None
    assert sd._iso(pd.Timestamp("2026-09-10 19:33:02", tz="UTC")).startswith("2026-09-10T19:33:02")
    assert sd._iso(pd.NaT) is None and sd._iso(None) is None
    assert Row("x", "m", 1.0).value_json is None
