"""providers/snapshot_bq.py (BigQuery snapshot backend) with a fake client; backend selection;
snapshot_daily bulk writes; the sqlite -> BigQuery migration script. No network.

The fake client executes the backend's parameterised SQL on an in-memory sqlite table
(`@p` -> `:p`, the backticked table name -> `snapshots`) and applies MERGE as an upsert of the
@rows ARRAY<STRUCT> parameter, so the very same snapshot tests run over both backends and
must produce identical results.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import snapshot_daily as sd
from providers.memory_store import MemoryStore, SqliteSnapshotBackend
from providers.snapshot_bq import (
    COLUMNS,
    DEFAULT_TABLE,
    BigQuerySnapshotBackend,
    SnapshotBigQueryUnavailable,
    dedupe_rows,
    normalise_row,
)
from scripts import migrate_snapshots_to_bq as mig

TODAY = date(2026, 9, 11)
ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------
# fake BigQuery client
# ----------------------------------------------------------------------------

def _conv(v):
    if isinstance(v, datetime):
        if v.tzinfo is not None:
            v = v.astimezone(timezone.utc).replace(tzinfo=None)
        return v.replace(microsecond=0).isoformat(sep=" ")
    if isinstance(v, date):
        return v.isoformat()
    return v


class FakeJob:
    def __init__(self, rows, affected=None):
        self._rows = rows
        self.num_dml_affected_rows = affected

    def result(self, timeout=None):
        return iter(self._rows)


class FakeBQClient:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE snapshots (snapshot_date TEXT, source TEXT, entity TEXT, metric TEXT, value REAL, "
            "value_json TEXT, captured_at TEXT, PRIMARY KEY (snapshot_date, source, entity, metric))")
        self.calls: list = []          # (sql, params, job_config)
        self.closed = False

    def query(self, sql, job_config=None):
        params = list(job_config.query_parameters) if job_config is not None else []
        self.calls.append((sql, params, job_config))
        if sql.lstrip().upper().startswith("MERGE"):
            rows = next(p for p in params if p.name == "rows")
            n = 0
            for st in rows.values:
                v = st.struct_values
                self.conn.execute(
                    "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (snapshot_date, source, entity, metric) DO UPDATE SET value = excluded.value, "
                    "value_json = excluded.value_json, captured_at = excluded.captured_at",
                    tuple(_conv(v[c]) for c in COLUMNS))
                n += 1
            self.conn.commit()
            return FakeJob([], affected=n)
        sq = re.sub(r"`[^`]+`", "snapshots", sql)
        sq = re.sub(r"@(\w+)", r":\1", sq)
        return FakeJob(self.conn.execute(sq, {p.name: _conv(p.value) for p in params}).fetchall())

    def close(self):
        self.closed = True

    @property
    def merges(self):
        return [c for c in self.calls if c[0].lstrip().upper().startswith("MERGE")]


def bq_backend(**kw) -> BigQuerySnapshotBackend:
    kw.setdefault("table", "p.d.snapshots")
    kw.setdefault("project", "p")
    kw.setdefault("client", FakeBQClient())
    return BigQuerySnapshotBackend(**kw)


@pytest.fixture(params=["sqlite", "bigquery"])
def snap_store(request):
    if request.param == "sqlite":
        s = MemoryStore("sqlite:///:memory:", embedder=None, snapshot_backend="sqlite")
    else:
        s = MemoryStore("sqlite:///:memory:", embedder=None, snapshot_backend=bq_backend())
    yield s
    s.close()


# ----------------------------------------------------------------------------
# identical behaviour over both backends
# ----------------------------------------------------------------------------

def test_backend_identity(snap_store):
    assert snap_store.snapshot_backend_name in ("sqlite", "bigquery")
    assert type(snap_store.snapshot_backend).__name__ in ("SqliteSnapshotBackend", "BigQuerySnapshotBackend")


def test_snapshot_upsert_idempotent(snap_store):
    store = snap_store
    store.put_snapshot(date(2026, 9, 1), "haruko", "combined", "delta_usd", 1.5e6)
    store.put_snapshot("2026-09-01", "haruko", "combined", "delta_usd", 1.6e6, value_json={"flag": "Normal"})
    store.put_snapshot("2026-09-01", "haruko", "combined", "delta_usd", 1.6e6, value_json={"flag": "Normal"})
    assert store.count_snapshots("haruko") == 1
    row = store.get_snapshot("2026-09-01", "haruko", "combined", "delta_usd")
    assert row["value"] == 1.6e6 and row["value_json"] == {"flag": "Normal"}
    assert row["snapshot_date"] == date(2026, 9, 1) and row["captured_at"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", row["captured_at"])
    assert store.get_snapshot("2026-09-02", "haruko", "combined", "delta_usd") is None


def test_snapshot_series_and_listing(snap_store):
    store = snap_store
    today = date(2026, 9, 11)
    for i in range(10):
        d = today - timedelta(days=i)
        store.put_snapshot(d, "signals", "btc", "funding_rate_z", 0.1 * i)
        store.put_snapshot(d, "signals", "btc", "price", 100_000 + i)
    store.put_snapshot(today, "signals", "eth", "price", 4_000)
    store.put_snapshot(today, "haruko", "20", "day_pnl", None, value_json=[1, 2])
    store.put_snapshot(today, "haruko", "20", "nan", float("nan"))

    series = store.get_snapshot_series("signals", "btc", "funding_rate_z", days=5, end=today)
    assert [r["snapshot_date"] for r in series] == [today - timedelta(days=i) for i in range(4, -1, -1)]
    assert [round(r["value"], 1) for r in series] == [0.4, 0.3, 0.2, 0.1, 0.0]
    assert len(store.get_snapshot_series("signals", "btc", "funding_rate_z", days=None)) == 10
    assert store.get_snapshot_series("signals", "xxx", "price", days=None) == []

    assert store.latest_snapshot_date("signals") == today
    assert store.latest_snapshot_date("signals", entity="eth", metric="price") == today
    assert store.latest_snapshot_date("sheet") is None

    metrics = store.list_snapshot_metrics("signals")
    assert [(m["entity"], m["metric"], m["n"]) for m in metrics] == [
        ("btc", "funding_rate_z", 10), ("btc", "price", 10), ("eth", "price", 1)]
    assert metrics[0]["first_date"] == today - timedelta(days=9) and metrics[0]["last_date"] == today
    assert store.list_snapshot_metrics("signals", entity="eth") == [
        {"entity": "eth", "metric": "price", "first_date": today, "last_date": today, "n": 1}]
    assert store.list_snapshot_entities("signals") == ["btc", "eth"]
    assert store.list_snapshot_sources() == ["haruko", "signals"]
    row = store.get_snapshot(today, "haruko", "20", "day_pnl")
    assert row == {"snapshot_date": today, "value": None, "value_json": [1, 2], "captured_at": row["captured_at"]}
    assert store.get_snapshot(today, "haruko", "20", "nan")["value"] is None
    assert store.count_snapshots(snapshot_date=today) == 5
    assert store.count_snapshots() == 23


def test_put_snapshots_bulk_keeps_captured_at(snap_store):
    store = snap_store
    cap = datetime(2026, 9, 11, 17, 21, 3, tzinfo=timezone.utc)
    n = store.put_snapshots([
        {"snapshot_date": "2026-09-01", "source": "sheet", "entity": "TOTAL", "metric": "mtd_pnl", "value": 1,
         "captured_at": cap},
        {"snapshot_date": "2026-09-01", "source": "sheet", "entity": "TOTAL", "metric": "ytd_pnl", "value": 2,
         "value_json": {"m": 9}},
    ])
    assert n == 2 and store.count_snapshots("sheet") == 2
    assert store.get_snapshot("2026-09-01", "sheet", "TOTAL", "mtd_pnl")["captured_at"] == "2026-09-11 17:21:03"
    assert store.get_snapshot("2026-09-01", "sheet", "TOTAL", "ytd_pnl")["value_json"] == {"m": 9}


# ----------------------------------------------------------------------------
# BigQuery specifics: MERGE shape, batching, date filters, laziness
# ----------------------------------------------------------------------------

def _rows(n, source="signals", d=TODAY):
    return [{"snapshot_date": d, "source": source, "entity": "btc", "metric": f"m{i}", "value": float(i)}
            for i in range(n)]


def test_merge_sql_shape_batching_and_dedupe():
    client = FakeBQClient()
    b = bq_backend(client=client, batch_size=3, max_bytes_billed=12345)
    rows = _rows(7)
    rows.append({**rows[0], "value": 99.0})               # same key twice in one batch -> last wins
    assert b.put_snapshots(rows) == 8                      # rows given; 7 unique keys -> batches 3, 3, 1
    sizes = [len(next(p for p in c[1] if p.name == "rows").values) for c in client.merges]
    assert sizes == [3, 3, 1] and b.merges == 3
    sql, params, cfg = client.merges[0]
    assert sql.startswith("MERGE `p.d.snapshots` T")
    assert "USING (SELECT * FROM UNNEST(@rows)) S" in sql
    assert "T.snapshot_date = S.snapshot_date AND T.source = S.source" in sql
    assert "T.snapshot_date BETWEEN @min_date AND @max_date" in sql
    assert "WHEN MATCHED THEN UPDATE SET" in sql and "WHEN NOT MATCHED THEN INSERT" in sql
    assert [p.name for p in params] == ["rows", "min_date", "max_date"]
    assert params[1].type_ == "DATE" and params[1].value == TODAY and params[2].value == TODAY
    struct = params[0].values[0]
    assert list(struct.struct_types.items()) == [
        ("snapshot_date", "DATE"), ("source", "STRING"), ("entity", "STRING"), ("metric", "STRING"),
        ("value", "FLOAT64"), ("value_json", "STRING"), ("captured_at", "TIMESTAMP")]
    assert struct.struct_values["captured_at"].tzinfo is not None
    assert cfg.maximum_bytes_billed == 12345
    # idempotent semantics + last write wins
    assert b.count_snapshots("signals") == 7
    assert b.get_snapshot(TODAY, "signals", "btc", "m0")["value"] == 99.0
    b.put_snapshots(_rows(7))
    assert b.count_snapshots("signals") == 7 and b.get_snapshot(TODAY, "signals", "btc", "m0")["value"] == 0.0
    assert b.put_snapshots([]) == 0 and b.merges == 6


def test_dedupe_and_normalise_row():
    a = normalise_row({"snapshot_date": "2026-09-11", "source": "s", "entity": 20, "metric": "m",
                       "value": "nan", "value_json": {"a": 1}, "captured_at": "2026-09-11 01:02:03"})
    assert a["snapshot_date"] == TODAY and a["entity"] == "20" and a["value"] is None
    assert a["value_json"] == '{"a": 1}' and a["captured_at"] == datetime(2026, 9, 11, 1, 2, 3, tzinfo=timezone.utc)
    b = dict(a, value=1.0)
    assert dedupe_rows([a, b]) == [b] and dedupe_rows([b, a]) == [a]


def test_reads_filter_on_snapshot_date():
    client = FakeBQClient()
    b = bq_backend(client=client)
    b.put_snapshots(_rows(2))
    b.get_snapshot_series("signals", "btc", "m0", days=5, end=TODAY)
    sql, params, _ = client.calls[-1]
    assert "snapshot_date BETWEEN @start AND @end" in sql and sql.rstrip().endswith("ORDER BY snapshot_date ASC")
    by_name = {p.name: p for p in params}
    assert by_name["start"].value == TODAY - timedelta(days=4) and by_name["end"].value == TODAY
    assert by_name["start"].type_ == "DATE"
    b.get_snapshot_series("signals", "btc", "m0", days=None)
    assert "BETWEEN" not in client.calls[-1][0]
    b.get_snapshot(TODAY, "signals", "btc", "m0")
    sql, params, _ = client.calls[-1]
    assert "WHERE snapshot_date = @d" in sql and params[0].type_ == "DATE" and params[0].value == TODAY
    b.count_snapshots("signals", snapshot_date="2026-09-11")
    assert "snapshot_date = @d" in client.calls[-1][0]
    assert b.queries == 4


def test_lazy_client_and_env_defaults(monkeypatch):
    monkeypatch.delenv("SNAPSHOT_BQ_TABLE", raising=False)
    monkeypatch.delenv("BQ_BILLING_PROJECT", raising=False)
    b = BigQuerySnapshotBackend()
    assert b.table == DEFAULT_TABLE == "anchorage-corp-eng-playground.gmask_bot.snapshots"
    assert b.project == "anchorage-corp-eng-playground" and b._client is None
    assert "billed to anchorage-corp-eng-playground" in b.label
    monkeypatch.setenv("SNAPSHOT_BQ_TABLE", "`x.y.z`")
    monkeypatch.setenv("BQ_BILLING_PROJECT", "bill")
    b = BigQuerySnapshotBackend()
    assert b.table == "x.y.z" and b.project == "bill" and b._client is None

    from google.cloud import bigquery

    def boom(*a, **k):
        raise RuntimeError("no creds")

    monkeypatch.setattr(bigquery, "Client", boom)
    with pytest.raises(SnapshotBigQueryUnavailable):
        b.client
    client = FakeBQClient()
    b = bq_backend(client=client)
    b.close()
    assert client.closed and b._client is None


# ----------------------------------------------------------------------------
# selection via env / constructor
# ----------------------------------------------------------------------------

def test_backend_selection(monkeypatch):
    monkeypatch.setenv("SNAPSHOT_BACKEND", "bigquery")
    s = MemoryStore("sqlite:///:memory:", embedder=None)
    assert s.snapshot_backend_name == "bigquery" and isinstance(s.snapshot_backend, BigQuerySnapshotBackend)
    assert s.snapshot_backend._client is None           # nothing touched the network
    assert s.snapshot_backend.table == DEFAULT_TABLE
    s.close()

    monkeypatch.setenv("SNAPSHOT_BACKEND", "SQLite")
    s = MemoryStore("sqlite:///:memory:", embedder=None)
    assert isinstance(s.snapshot_backend, SqliteSnapshotBackend) and s.snapshot_backend_name == "sqlite"
    s.close()

    monkeypatch.setenv("SNAPSHOT_BACKEND", "dynamo")
    with pytest.raises(ValueError, match="SNAPSHOT_BACKEND"):
        MemoryStore("sqlite:///:memory:", embedder=None)
    with pytest.raises(ValueError):
        MemoryStore("sqlite:///:memory:", embedder=None, snapshot_backend="dynamo")

    s = MemoryStore("sqlite:///:memory:", embedder=None, snapshot_backend="bigquery")
    assert s.snapshot_backend_name == "bigquery"
    s.close()

    from utils.config import Config
    monkeypatch.setenv("SNAPSHOT_BACKEND", "bigquery")
    monkeypatch.delenv("SNAPSHOT_BQ_TABLE", raising=False)
    cfg = Config()
    assert cfg.SNAPSHOT_BACKEND == "bigquery" and cfg.SNAPSHOT_BQ_TABLE == DEFAULT_TABLE


def test_memories_stay_in_sqlite_with_bigquery_snapshots():
    s = MemoryStore("sqlite:///:memory:", embedder=None, snapshot_backend=bq_backend())
    mid = s.put("facts:shared", "desk prefers bps")
    assert s.get(mid).text == "desk prefers bps"
    assert s.snapshot_backend.client.calls == []        # no BigQuery traffic for memories
    s.close()


# ----------------------------------------------------------------------------
# snapshot_daily: bulk writes, signals-only path, lazy imports
# ----------------------------------------------------------------------------

class BulkStore:
    def __init__(self):
        self.batches = []

    def put_snapshots(self, rows):
        rows = list(rows)
        self.batches.append(rows)
        return len(rows)


def test_write_rows_prefers_bulk():
    store = BulkStore()
    rows = [sd.Row("btc", "price", 1.0, '{"a":1}'), sd.Row("eth", "price", None)]
    assert sd.write_rows(store, TODAY, "signals", rows) == 2
    assert len(store.batches) == 1 and store.batches[0][0] == {
        "snapshot_date": TODAY, "source": "signals", "entity": "btc", "metric": "price",
        "value": 1.0, "value_json": '{"a":1}'}

    from tests.test_snapshot_daily import FakeSnapshotStore
    legacy = FakeSnapshotStore()
    assert sd.write_rows(legacy, TODAY, "signals", rows) == 2 and legacy.puts == 2


def test_run_signals_only_into_bigquery():
    from tests.test_snapshot_daily import fake_calc, fake_fetch

    client = FakeBQClient()
    store = MemoryStore("sqlite:///:memory:", embedder=None, snapshot_backend=bq_backend(client=client))
    caps = {"signals": lambda: sd.capture_signals(tokens=["btc", "eth"], fetch=fake_fetch, calc=fake_calc)}
    results = sd.run(["signals"], TODAY, store=store, captures=caps)
    res = results["signals"]
    assert res.error is None and res.written == len(res.rows) == 18
    assert len(client.merges) == 1                          # one MERGE for the whole source
    assert store.count_snapshots("signals", snapshot_date=TODAY) == 18
    assert store.get_snapshot(TODAY, "signals", "btc", "spot_volume_z")["value"] == 2.7

    sd.run(["signals"], TODAY, store=store, captures=caps)  # rerun: idempotent
    assert len(client.merges) == 2 and store.count_snapshots("signals") == 18
    store.close()


def test_snapshot_daily_imports_only_stdlib_at_module_level():
    tree = ast.parse((ROOT / "snapshot_daily.py").read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    forbidden = {"slack_bot", "slack_bolt", "slack_sdk", "providers", "tools", "utils", "pandas", "google", "dotenv"}
    assert not (names & forbidden), names & forbidden


def test_main_with_bigquery_store(monkeypatch, capsys):
    from tests.test_snapshot_daily import fake_calc, fake_fetch

    client = FakeBQClient()
    store = MemoryStore("sqlite:///:memory:", embedder=None, snapshot_backend=bq_backend(client=client))
    monkeypatch.setattr(sd, "_get_store", lambda: store)
    monkeypatch.setattr(sd, "CAPTURES", {
        "signals": lambda: sd.capture_signals(tokens=["btc"], fetch=fake_fetch, calc=fake_calc)})
    assert sd.main(["--sources", "signals", "--date", "2026-09-11"]) == 0
    out = capsys.readouterr().out
    assert "- signals: 11 rows, 11 written" in out and len(client.merges) == 1
    store.close()


# ----------------------------------------------------------------------------
# migration script
# ----------------------------------------------------------------------------

@pytest.fixture
def local_db(tmp_path):
    path = tmp_path / "memory.db"
    s = MemoryStore(f"sqlite:///{path}", embedder=None, snapshot_backend="sqlite")
    cap = datetime(2026, 9, 11, 17, 21, 3)
    s.put_snapshot(TODAY, "haruko", "combined", "delta_usd", -1e6, value_json={"as_of": "x"}, captured_at=cap)
    s.put_snapshot(TODAY, "haruko", "20", "delta_usd", -2e6, captured_at=cap)
    s.put_snapshot(TODAY, "signals", "btc", "price", 78_000.0, captured_at=cap)
    s.put_snapshot(TODAY - timedelta(days=1), "sheet", "TOTAL", "mtd_pnl_usd", None, value_json=[1])
    s.close()
    return path


def test_read_sqlite_rows(local_db):
    rows = mig.read_sqlite_rows(str(local_db))
    assert len(rows) == 4 and rows[0]["snapshot_date"] == "2026-09-10" and rows[0]["source"] == "sheet"
    btc = next(r for r in rows if r["entity"] == "btc")
    assert btc["captured_at"] == datetime(2026, 9, 11, 17, 21, 3, tzinfo=timezone.utc)
    assert btc["value"] == 78_000.0 and btc["value_json"] is None
    assert [r["source"] for r in mig.read_sqlite_rows(str(local_db), ["haruko"])] == ["haruko", "haruko"]
    with pytest.raises(FileNotFoundError):
        mig.read_sqlite_rows(str(local_db.parent / "missing.db"))


def test_migrate_and_main(local_db, capsys):
    client = FakeBQClient()
    backend = bq_backend(client=client, batch_size=100)
    rows = mig.read_sqlite_rows(str(local_db))
    report = mig.migrate(rows, backend, batch_size=1)
    assert report == {"haruko": {"local": 2, "sent": 2, "bigquery": 2},
                      "sheet": {"local": 1, "sent": 1, "bigquery": 1},
                      "signals": {"local": 1, "sent": 1, "bigquery": 1}}
    assert len(client.merges) == 4
    assert backend.get_snapshot(TODAY, "signals", "btc", "price")["captured_at"] == "2026-09-11 17:21:03"
    assert backend.get_snapshot(TODAY, "haruko", "combined", "delta_usd")["value_json"] == {"as_of": "x"}

    # rerun via main(): idempotent, prints counts
    assert mig.main(["--db", str(local_db), "--batch-size", "500"], backend_factory=lambda table: backend) == 0
    out = capsys.readouterr().out
    assert "- haruko: 2 local rows, 2 sent, 2 now in BigQuery" in out and "Total 4 rows" in out
    assert backend.count_snapshots() == 4

    # dry run writes nothing; source filter; missing db
    fresh = bq_backend()
    assert mig.main(["--db", str(local_db), "--dry-run"], backend_factory=lambda table: fresh) == 0
    assert "(dry run)" in capsys.readouterr().out and fresh.merges == 0
    assert mig.main(["--db", str(local_db), "--sources", "signals"], backend_factory=lambda table: fresh) == 0
    assert fresh.count_snapshots() == 1 and fresh.list_snapshot_sources() == ["signals"]
    assert mig.main(["--db", str(local_db.parent / "nope.db")]) == 2
    assert mig.main(["--db", str(local_db), "--sources", "nothing"]) == 0
    assert "No snapshot rows" in capsys.readouterr().out


# ----------------------------------------------------------------------------
# deploy artefacts
# ----------------------------------------------------------------------------

def test_deploy_script_and_timer_units():
    script = ROOT / "deploy" / "deploy_snapshot_job.sh"
    text = script.read_text()
    assert script.stat().st_mode & 0o111, "deploy script must be executable"
    assert "gcloud run jobs deploy" in text and "trading-signals-snapshot" in text
    assert "snapshot_daily.py|--sources|${SOURCES}" in text and 'SOURCES="${SOURCES:-signals,etf,cme}"' in text
    assert "SNAPSHOT_BACKEND=bigquery" in text and "gm-bot@" in text
    assert "30 23 * * *" in text and "jobs/${JOB}:run" in text
    unit = (ROOT / "deploy" / "trading-signals-snapshot.service").read_text()
    assert "--sources haruko,sheet" in unit and "EnvironmentFile=-" in unit
