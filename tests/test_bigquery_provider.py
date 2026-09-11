"""providers/bigquery.py: SQL guard, catalog cache and describe_table. No network, no GCP."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from providers import bigquery as bqmod
from providers.bigquery import (
    DeskBigQuery,
    SQLGuardError,
    format_bytes,
    referenced_tables,
    strip_sql_noise,
    validate_sql,
)

PROJECT = "anc-global-markets"
ALLOWED = ("brokerage_a1", "pricing")
T = f"`{PROJECT}.brokerage_a1.fct_otc_haruko_pnl_portfolio`"


def _v(sql, max_rows=200):
    return validate_sql(sql, PROJECT, ALLOWED, max_rows)


# ----------------------------------------------------------------------------
# guard: statement types
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    f"DELETE FROM {T} WHERE true",
    f"INSERT INTO {T} (a) VALUES (1)",
    f"CREATE TABLE {PROJECT}.brokerage_a1.x AS SELECT 1",
    f"UPDATE {T} SET a = 1",
    f"DROP TABLE {T}",
    f"MERGE {T} t USING x ON true WHEN MATCHED THEN DELETE",
    "DECLARE x INT64; SELECT 1",
    "BEGIN SELECT 1; END",
    f"EXPORT DATA OPTIONS(uri='gs://x') AS SELECT * FROM {T}",
    "",
    "   ",
])
def test_guard_rejects_non_select(sql):
    with pytest.raises(SQLGuardError):
        _v(sql)


def test_guard_rejects_multiple_statements():
    with pytest.raises(SQLGuardError, match="single statement"):
        _v(f"SELECT 1 FROM {T}; SELECT 2 FROM {T}")
    with pytest.raises(SQLGuardError):
        _v(f"SELECT 1 FROM {T}; DROP TABLE {T}")
    # one trailing semicolon is fine
    assert "LIMIT" in _v(f"SELECT 1 FROM {T};")


def test_guard_rejects_dml_hidden_after_select_keyword():
    with pytest.raises(SQLGuardError, match="not allowed"):
        _v(f"SELECT * FROM {T} WHERE x IN (SELECT 1) UNION ALL (DELETE FROM {T})")


def test_guard_ignores_keywords_inside_strings_and_comments():
    sql = (f"SELECT 'DELETE me' AS note, \"DROP\" AS other -- CREATE TABLE in a comment\n"
           f"/* INSERT INTO nothing */ FROM {T}")
    out = _v(sql)
    assert out.endswith("LIMIT 200")


def test_guard_allows_legit_functions_named_like_keywords():
    # REPEAT is a string function; FOR SYSTEM_TIME AS OF is read-only time travel
    sql = f"SELECT REPEAT('a', 3) FROM {T} FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)"
    assert "LIMIT 200" in _v(sql)


# ----------------------------------------------------------------------------
# guard: datasets / projects
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("sql,needle", [
    (f"SELECT * FROM `{PROJECT}.hold.fct_hold_spot`", "hold"),
    ("SELECT * FROM hold.fct_hold_spot", "hold"),
    (f"SELECT * FROM `{PROJECT}`.`crms`.`clients`", "crms"),
    (f"SELECT * FROM {T} t JOIN `{PROJECT}.lending.loans` l ON true", "lending"),
    (f"SELECT * FROM `{PROJECT}.marketdata.INFORMATION_SCHEMA.TABLES`", "marketdata"),
    ("SELECT * FROM `region-us`.INFORMATION_SCHEMA.JOBS", "INFORMATION_SCHEMA"),
    (f"SELECT * FROM `{PROJECT}.INFORMATION_SCHEMA.SCHEMATA`", "INFORMATION_SCHEMA"),
])
def test_guard_rejects_disallowed_datasets(sql, needle):
    with pytest.raises(SQLGuardError) as ei:
        _v(sql)
    msg = str(ei.value)
    assert needle in msg
    assert "brokerage_a1" in msg and "pricing" in msg  # lists the allowed datasets


def test_guard_rejects_other_projects():
    with pytest.raises(SQLGuardError, match="project 'other-proj'"):
        _v("SELECT * FROM `other-proj.brokerage_a1.t`")
    with pytest.raises(SQLGuardError, match="project"):
        # project-qualified ref hidden in a table function still caught
        _v(f"SELECT * FROM {T} WHERE x IN (SELECT y FROM some-other.brokerage_a1.z)")


@pytest.mark.parametrize("sql", [
    f"SELECT * FROM {T}",
    "SELECT * FROM brokerage_a1.fct_perps_positions",
    f"SELECT * FROM `{PROJECT}`.`pricing`.`current_price`",
    f"SELECT * FROM `{PROJECT}.brokerage_a1.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = 'x'",
    "SELECT * FROM pricing.INFORMATION_SCHEMA.TABLES",
    f"WITH a AS (SELECT * FROM {T}) SELECT a.x, b.y FROM a JOIN `{PROJECT}.pricing.current_price` b ON true",
    f"  -- leading comment\n  select 1 from {T}",
    f"(SELECT 1 FROM {T})",
])
def test_guard_accepts_allowed(sql):
    assert "LIMIT" in _v(sql)


def test_guard_ignores_ctes_aliases_and_struct_paths():
    sql = (f"WITH base AS (SELECT t.datastream_metadata.source_ts AS ts FROM {T} t) "
           "SELECT ts FROM base b JOIN base c ON b.ts = c.ts")
    assert "LIMIT 200" in _v(sql)


def test_referenced_tables_extraction():
    refs = referenced_tables(
        f"SELECT * FROM {T} p JOIN brokerage_a1.x y ON true, `{PROJECT}.pricing.z` /* other.ds.t */ WHERE a = 'q.r.s'"
    )
    assert refs == [f"{PROJECT}.brokerage_a1.fct_otc_haruko_pnl_portfolio", "brokerage_a1.x", f"{PROJECT}.pricing.z"]


def test_strip_sql_noise():
    out = strip_sql_noise("SELECT 'a;b' -- x; y\nFROM t /* ; */")
    assert out.strip() == "SELECT '' \nFROM t"
    assert ";" not in out  # literal and comment contents are gone


# ----------------------------------------------------------------------------
# guard: LIMIT
# ----------------------------------------------------------------------------

def test_limit_appended_when_missing():
    out = _v(f"SELECT * FROM {T}")
    assert out.endswith("\nLIMIT 200")


def test_limit_lowered_when_too_high():
    out = _v(f"SELECT * FROM {T} ORDER BY 1 LIMIT 5000")
    assert out.endswith("LIMIT 200") and "5000" not in out


def test_limit_kept_when_small_and_offset_preserved():
    out = _v(f"SELECT * FROM {T} LIMIT 5")
    assert out.endswith("LIMIT 5")
    out = _v(f"SELECT * FROM {T} LIMIT 999 OFFSET 10")
    assert out.endswith("LIMIT 200 OFFSET 10")


def test_limit_with_trailing_semicolon_and_comment():
    out = _v(f"SELECT * FROM {T} LIMIT 400; -- done")
    assert out.rstrip().endswith("LIMIT 200")
    out = _v(f"SELECT * FROM {T}\n-- trailing comment\n")
    assert out.endswith("LIMIT 200")


def test_inner_limit_does_not_count_as_outer():
    out = _v(f"SELECT * FROM (SELECT * FROM {T} LIMIT 10000) x")
    assert out.endswith("LIMIT 200") and "LIMIT 10000" in out


def test_custom_max_rows():
    out = _v(f"SELECT * FROM {T} LIMIT 50", max_rows=20)
    assert out.endswith("LIMIT 20")


# ----------------------------------------------------------------------------
# fake client for catalog / describe / query
# ----------------------------------------------------------------------------

class _Field(SimpleNamespace):
    pass


class _Table(SimpleNamespace):
    pass


def _field(name, ftype, desc="", mode="NULLABLE"):
    return _Field(name=name, field_type=ftype, description=desc, mode=mode)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.tables = {
            "brokerage_a1": {
                "fct_otc_haruko_greeks_history_eod": _Table(
                    table_type="TABLE", description="EOD greeks", num_rows=581, num_bytes=193448,
                    modified=datetime(2026, 9, 10, 0, 43, tzinfo=timezone.utc),
                    time_partitioning=SimpleNamespace(type_="DAY", field="as_of_date"), range_partitioning=None,
                    clustering_fields=None,
                    schema=[_field("as_of_date", "DATE", "snapshot date"), _field("entity_id", "INTEGER"),
                            _field("total_delta_usd", "FLOAT", "USD delta"), _field("data_quality_flag", "STRING")]),
                "fct_perps_positions": _Table(
                    table_type="VIEW", description="", num_rows=0, num_bytes=0, modified=None,
                    time_partitioning=None, range_partitioning=None, clustering_fields=["exchange", "symbol"],
                    schema=[_field("exchange", "STRING"), _field("symbol", "STRING"), _field("position_value", "BIGNUMERIC")]),
            },
            "pricing": {
                "current_price": _Table(
                    table_type="TABLE", description="latest price", num_rows=3378, num_bytes=1000, modified=None,
                    time_partitioning=None, range_partitioning=None, clustering_fields=None,
                    schema=[_field("symbol", "STRING"), _field("price", "BIGNUMERIC")]),
            },
        }

    def list_tables(self, dataset_ref):
        self.calls.append(("list_tables", dataset_ref))
        project, ds = dataset_ref.split(".")
        assert project == PROJECT
        return [SimpleNamespace(table_id=t, table_type=v.table_type) for t, v in self.tables[ds].items()]

    def get_table(self, ref):
        self.calls.append(("get_table", ref))
        project, ds, tid = ref.split(".")
        return self.tables[ds][tid]

    def query(self, sql, job_config=None, timeout=None):
        self.calls.append(("query", sql, job_config))
        return FakeJob(sql, job_config)


class FakeJob:
    def __init__(self, sql, job_config):
        self.sql = sql
        self.job_config = job_config
        self.total_bytes_processed = 12345

    def result(self, timeout=None):
        return self

    def to_dataframe(self, create_bqstorage_client=True):
        from decimal import Decimal
        return pd.DataFrame({"symbol": ["BTC"], "price": [Decimal("77067.80")], "n": [1]})


@pytest.fixture
def bq(tmp_path):
    clock = {"t": 1_000_000.0}
    client = FakeClient()
    inst = DeskBigQuery(client, data_project=PROJECT, billing_project="bill-proj", allowed_datasets=ALLOWED,
                        max_bytes_billed=2_000_000_000, max_rows=200, timeout_s=5,
                        catalog_path=tmp_path / "bq_catalog.json", now=lambda: clock["t"])
    inst._clock = clock
    return inst


def test_catalog_built_once_and_persisted(bq, tmp_path):
    cat = bq.catalog()
    assert set(cat["datasets"]) == set(ALLOWED)
    assert cat["datasets"]["brokerage_a1"]["fct_otc_haruko_greeks_history_eod"]["partitioning"] == "DAY on as_of_date"
    assert cat["datasets"]["brokerage_a1"]["fct_perps_positions"]["num_rows"] is None  # views have no row count
    assert cat["datasets"]["brokerage_a1"]["fct_perps_positions"]["clustering"] == ["exchange", "symbol"]
    n_calls = len(bq.client.calls)
    assert n_calls == 2 + 3  # 2 list_tables + 3 get_table

    # second call: memory cache, no client calls
    bq.catalog()
    assert len(bq.client.calls) == n_calls

    # a new instance reads the file, no client calls
    on_disk = json.loads((tmp_path / "bq_catalog.json").read_text())
    assert on_disk["data_project"] == PROJECT
    fresh = DeskBigQuery(FakeClient(), data_project=PROJECT, allowed_datasets=ALLOWED,
                         catalog_path=tmp_path / "bq_catalog.json", now=lambda: 1_000_000.0 + 3600)
    assert fresh.catalog()["built_at"] == cat["built_at"]
    assert fresh.client.calls == []


def test_catalog_refreshes_after_ttl_and_on_dataset_change(bq, tmp_path):
    bq.catalog()
    bq._clock["t"] += bqmod.CATALOG_TTL_S + 1
    bq.client.calls.clear()
    bq.catalog()
    assert any(c[0] == "list_tables" for c in bq.client.calls)

    other = DeskBigQuery(FakeClient(), data_project=PROJECT, allowed_datasets=("pricing",),
                         catalog_path=tmp_path / "bq_catalog.json", now=lambda: bq._clock["t"])
    cat = other.catalog()
    assert set(cat["datasets"]) == {"pricing"}


def test_list_tables_filters(bq):
    rows = bq.list_tables(keyword="delta")
    assert [r["table"] for r in rows] == ["brokerage_a1.fct_otc_haruko_greeks_history_eod"]
    assert rows[0]["matched_columns"] == ["total_delta_usd"]
    assert [r["table"] for r in bq.list_tables(dataset="pricing")] == ["pricing.current_price"]
    assert len(bq.list_tables()) == 3
    with pytest.raises(SQLGuardError):
        bq.list_tables(dataset="hold")


def test_describe_table(bq):
    info = bq.describe_table("brokerage_a1.fct_otc_haruko_greeks_history_eod")
    assert info["type"] == "TABLE" and info["num_rows"] == 581 and info["description"] == "EOD greeks"
    assert info["modified"].startswith("2026-09-10T00:43")
    assert [c["name"] for c in info["columns"]] == ["as_of_date", "entity_id", "total_delta_usd", "data_quality_flag"]
    assert info["columns"][2] == {"name": "total_delta_usd", "type": "FLOAT", "mode": "NULLABLE", "description": "USD delta"}
    # project prefix accepted; wrong dataset refused; bad shape refused
    assert bq.describe_table(f"{PROJECT}.pricing.current_price")["num_rows"] == 3378
    with pytest.raises(SQLGuardError):
        bq.describe_table("hold.fct_hold_spot")
    with pytest.raises(SQLGuardError):
        bq.describe_table("just_a_table")


def test_table_qualifier(bq):
    assert bq.table("brokerage_a1.x") == f"`{PROJECT}.brokerage_a1.x`"
    with pytest.raises(SQLGuardError):
        bq.table("hold.x")


def test_query_applies_guard_job_config_and_decimal_conversion(bq):
    df = bq.query("SELECT symbol, price FROM pricing.current_price")
    sql, cfg = bq.client.calls[-1][1], bq.client.calls[-1][2]
    assert sql.endswith("LIMIT 200")
    assert cfg.maximum_bytes_billed == 2_000_000_000 and cfg.use_legacy_sql is False and not cfg.dry_run
    assert df["price"].dtype == "float64" and df["price"].iloc[0] == pytest.approx(77067.80)
    assert df.attrs["bytes_processed"] == 12345 and bq.last_bytes_processed == 12345
    with pytest.raises(SQLGuardError):
        bq.query("SELECT * FROM hold.t")
    assert not any(c[0] == "query" and "hold" in c[1] for c in bq.client.calls)


def test_dry_run_sets_flag(bq):
    assert bq.dry_run("SELECT 1 FROM pricing.current_price") == 12345
    assert bq.client.calls[-1][2].dry_run is True


def test_format_bytes():
    assert format_bytes(None) == "n/a"
    assert format_bytes(512) == "512 B"
    assert format_bytes(2_000_000_000) == "1.86 GB"


def test_factory_is_thread_safe_under_parallel_tool_calls(monkeypatch):
    """The agent runs tool calls in threads; the first two callers must both see the instance."""
    import threading
    import time as _time

    import providers.factory as factory

    class Slow(DeskBigQuery):
        @property
        def client(self):
            _time.sleep(0.05)  # simulate credential resolution
            return object()

    monkeypatch.setattr(bqmod, "DeskBigQuery", Slow)
    factory.reset_desk_bigquery()
    results = []
    try:
        threads = [threading.Thread(target=lambda: results.append(factory.get_desk_bigquery())) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 4 and all(r is not None for r in results)
        assert len({id(r) for r in results}) == 1  # one shared instance
    finally:
        factory.reset_desk_bigquery()


def test_factory_returns_none_when_unavailable(monkeypatch):
    import providers.factory as factory

    class Boom(DeskBigQuery):
        @property
        def client(self):
            raise bqmod.BigQueryUnavailable("no creds")

    monkeypatch.setattr(bqmod, "DeskBigQuery", Boom)
    factory.reset_desk_bigquery()
    try:
        assert factory.get_desk_bigquery() is None
        assert factory.get_desk_bigquery() is None  # memoised
    finally:
        factory.reset_desk_bigquery()
