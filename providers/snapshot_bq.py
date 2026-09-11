"""Daily snapshots in a shared BigQuery table (env ``SNAPSHOT_BACKEND=bigquery``).

Why: the developer machine sleeps, so the daily ``signals`` snapshot runs as a Cloud Run job
(``deploy/deploy_snapshot_job.sh``) and every reader - the local Slack bot, ``chat.py``, the
local ``haruko,sheet`` timer - shares one table instead of one SQLite file per machine:

    `anchorage-corp-eng-playground.gmask_bot.snapshots`
        snapshot_date DATE, source STRING, entity STRING, metric STRING,
        value FLOAT64, value_json STRING, captured_at TIMESTAMP
        partitioned by snapshot_date, clustered by (source, entity, metric)

``BigQuerySnapshotBackend`` implements exactly the snapshot API of
``providers.memory_store.SqliteSnapshotBackend`` (same signatures, same return shapes):

* **writes** - ``put_snapshots`` sends one parameterised ``MERGE`` per batch (default 500
  rows) with an ``ARRAY<STRUCT>`` query parameter unnested as the source, so re-running a
  day is idempotent on the (snapshot_date, source, entity, metric) key; duplicates inside a
  batch are collapsed first (BigQuery rejects a MERGE whose source matches one target row
  twice). ``put_snapshot`` is a one-row batch.
* **reads** - parameterised SELECTs filtered on ``snapshot_date`` wherever the call gives a
  window, so partition pruning keeps every scan to a few days of a tiny table.

The client (``google.cloud.bigquery.Client(project=BQ_BILLING_PROJECT)``) is created lazily
on first use - constructing the backend never touches the network - and every job carries
``maximum_bytes_billed`` as a cost guard. Credentials are Application Default Credentials:
the user locally, the ``gm-bot`` service account on Cloud Run.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional, Sequence

from providers.memory_store import (
    _iso,
    _loads,
    _to_date,
    normalise_snapshot_json,
    normalise_snapshot_value,
)

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "anchorage-corp-eng-playground.gmask_bot.snapshots"
DEFAULT_PROJECT = "anchorage-corp-eng-playground"
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_BYTES_BILLED = 1_000_000_000  # 1 GB; the whole table is a few MB
DEFAULT_TIMEOUT_S = 120

COLUMNS = ("snapshot_date", "source", "entity", "metric", "value", "value_json", "captured_at")
_STRUCT_TYPES = {"snapshot_date": "DATE", "source": "STRING", "entity": "STRING", "metric": "STRING",
                 "value": "FLOAT64", "value_json": "STRING", "captured_at": "TIMESTAMP"}

MERGE_SQL = """MERGE `{table}` T
USING (SELECT * FROM UNNEST(@rows)) S
ON T.snapshot_date = S.snapshot_date AND T.source = S.source
   AND T.entity = S.entity AND T.metric = S.metric
   AND T.snapshot_date BETWEEN @min_date AND @max_date
WHEN MATCHED THEN UPDATE SET
   value = S.value, value_json = S.value_json, captured_at = S.captured_at
WHEN NOT MATCHED THEN INSERT (snapshot_date, source, entity, metric, value, value_json, captured_at)
   VALUES (S.snapshot_date, S.source, S.entity, S.metric, S.value, S.value_json, S.captured_at)"""


class SnapshotBigQueryUnavailable(RuntimeError):
    """google-cloud-bigquery missing or no credentials."""


def _aware_utc(dt: Optional[datetime]) -> datetime:
    """captured_at as a tz-aware UTC datetime (naive input is taken as UTC)."""
    if dt is None:
        return datetime.now(timezone.utc)
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _captured_str(v) -> Optional[str]:
    """captured_at in the sqlite backend's text form ('YYYY-MM-DD HH:MM:SS', UTC)."""
    if v is None:
        return None
    if isinstance(v, datetime):
        if v.tzinfo is not None:
            v = v.astimezone(timezone.utc).replace(tzinfo=None)
        return _iso(v)
    return str(v)


def normalise_row(r: dict) -> dict:
    """One input row (as accepted by put_snapshots) -> typed column dict."""
    return {
        "snapshot_date": _to_date(r["snapshot_date"]),
        "source": str(r["source"]),
        "entity": str(r["entity"]),
        "metric": str(r["metric"]),
        "value": normalise_snapshot_value(r.get("value")),
        "value_json": normalise_snapshot_json(r.get("value_json")),
        "captured_at": _aware_utc(r.get("captured_at")),
    }


def dedupe_rows(rows: Sequence[dict]) -> list[dict]:
    """Last write wins for repeated (snapshot_date, source, entity, metric) keys; order kept."""
    seen: dict[tuple, int] = {}
    out: list[dict] = []
    for r in rows:
        key = (r["snapshot_date"], r["source"], r["entity"], r["metric"])
        if key in seen:
            out[seen[key]] = r
        else:
            seen[key] = len(out)
            out.append(r)
    return out


class BigQuerySnapshotBackend:
    name = "bigquery"

    def __init__(self, table: str | None = None, project: str | None = None, client: Any = None,
                 batch_size: int = DEFAULT_BATCH_SIZE, max_bytes_billed: int = DEFAULT_MAX_BYTES_BILLED,
                 timeout_s: float | None = DEFAULT_TIMEOUT_S):
        self.table = (table or os.getenv("SNAPSHOT_BQ_TABLE") or DEFAULT_TABLE).strip().strip("`")
        self.project = project or os.getenv("BQ_BILLING_PROJECT") or DEFAULT_PROJECT
        self.batch_size = max(1, int(batch_size))
        self.max_bytes_billed = int(max_bytes_billed)
        self.timeout_s = timeout_s
        self._client = client
        self._lock = threading.Lock()
        self.merges = 0          # MERGE jobs issued (diagnostics / tests)
        self.queries = 0         # SELECT jobs issued

    @property
    def label(self) -> str:
        return f"{self.table}, jobs billed to {self.project}"

    # -- client / jobs --------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        from google.cloud import bigquery
                    except ImportError as e:  # pragma: no cover - environment
                        raise SnapshotBigQueryUnavailable("google-cloud-bigquery is not installed") from e
                    try:
                        self._client = bigquery.Client(project=self.project)
                    except Exception as e:  # DefaultCredentialsError etc.
                        raise SnapshotBigQueryUnavailable(f"BigQuery client could not be created: {e}") from e
        return self._client

    def _job_config(self, params: Sequence):
        from google.cloud import bigquery

        return bigquery.QueryJobConfig(query_parameters=list(params), maximum_bytes_billed=self.max_bytes_billed)

    def _run(self, sql: str, params: Sequence = ()):
        """Run one parameterised job; returns (rows, job)."""
        job = self.client.query(sql, job_config=self._job_config(params))
        rows = list(job.result(timeout=self.timeout_s))
        return rows, job

    @staticmethod
    def _p(name: str, type_: str, value):
        from google.cloud import bigquery

        return bigquery.ScalarQueryParameter(name, type_, value)

    @staticmethod
    def _struct(row: dict):
        from google.cloud import bigquery

        return bigquery.StructQueryParameter(
            "row", *[bigquery.ScalarQueryParameter(c, _STRUCT_TYPES[c], row[c]) for c in COLUMNS])

    # -- writes ---------------------------------------------------------------

    def put_snapshot(self, snapshot_date, source: str, entity: str, metric: str, value,
                     value_json=None, captured_at: datetime | None = None) -> None:
        self.put_snapshots([{"snapshot_date": snapshot_date, "source": source, "entity": entity,
                             "metric": metric, "value": value, "value_json": value_json,
                             "captured_at": captured_at}])

    def put_snapshots(self, rows: Iterable[dict]) -> int:
        """Upsert all rows with one MERGE per ``batch_size`` chunk. Returns the number of rows given."""
        from google.cloud import bigquery

        given = [normalise_row(r) for r in rows]
        if not given:
            return 0
        unique = dedupe_rows(given)
        if len(unique) != len(given):
            logger.info("Snapshot batch: %d rows collapsed to %d unique keys", len(given), len(unique))
        for i in range(0, len(unique), self.batch_size):
            chunk = unique[i:i + self.batch_size]
            dates = [r["snapshot_date"] for r in chunk]
            params = [
                bigquery.ArrayQueryParameter("rows", "STRUCT", [self._struct(r) for r in chunk]),
                self._p("min_date", "DATE", min(dates)),
                self._p("max_date", "DATE", max(dates)),
            ]
            _, job = self._run(MERGE_SQL.format(table=self.table), params)
            self.merges += 1
            affected = getattr(job, "num_dml_affected_rows", None)
            logger.debug("MERGE %d rows into %s (affected: %s)", len(chunk), self.table, affected)
        return len(given)

    # -- reads ----------------------------------------------------------------

    def _select(self, sql: str, params: Sequence = ()) -> list:
        rows, _ = self._run(sql, params)
        self.queries += 1
        return rows

    @staticmethod
    def _snap_row(r) -> dict:
        return {"snapshot_date": _to_date(r[0]), "value": None if r[1] is None else float(r[1]),
                "value_json": _loads(r[2]) if r[2] else None, "captured_at": _captured_str(r[3])}

    def get_snapshot(self, snapshot_date, source: str, entity: str, metric: str) -> Optional[dict]:
        rows = self._select(
            f"SELECT snapshot_date, value, value_json, captured_at FROM `{self.table}` "
            "WHERE snapshot_date = @d AND source = @source AND entity = @entity AND metric = @metric LIMIT 1",
            [self._p("d", "DATE", _to_date(snapshot_date)), self._p("source", "STRING", source),
             self._p("entity", "STRING", str(entity)), self._p("metric", "STRING", metric)])
        return self._snap_row(rows[0]) if rows else None

    def get_snapshot_series(self, source: str, entity: str, metric: str, days: int | None = 30,
                            end: date | None = None) -> list[dict]:
        sql = (f"SELECT snapshot_date, value, value_json, captured_at FROM `{self.table}` "
               "WHERE source = @source AND entity = @entity AND metric = @metric")
        params = [self._p("source", "STRING", source), self._p("entity", "STRING", str(entity)),
                  self._p("metric", "STRING", metric)]
        if days is not None:
            end_d = end or datetime.now(timezone.utc).date()
            start = end_d - timedelta(days=int(days) - 1)
            sql += " AND snapshot_date BETWEEN @start AND @end"
            params += [self._p("start", "DATE", start), self._p("end", "DATE", end_d)]
        sql += " ORDER BY snapshot_date ASC"
        return [self._snap_row(r) for r in self._select(sql, params)]

    def latest_snapshot_date(self, source: str, entity: str | None = None, metric: str | None = None) -> Optional[date]:
        sql = f"SELECT MAX(snapshot_date) FROM `{self.table}` WHERE source = @source"
        params = [self._p("source", "STRING", source)]
        if entity is not None:
            sql += " AND entity = @entity"
            params.append(self._p("entity", "STRING", str(entity)))
        if metric is not None:
            sql += " AND metric = @metric"
            params.append(self._p("metric", "STRING", metric))
        rows = self._select(sql, params)
        return _to_date(rows[0][0]) if rows and rows[0][0] else None

    def list_snapshot_metrics(self, source: str, entity: str | None = None) -> list[dict]:
        sql = (f"SELECT entity, metric, MIN(snapshot_date), MAX(snapshot_date), COUNT(*) FROM `{self.table}` "
               "WHERE source = @source")
        params = [self._p("source", "STRING", source)]
        if entity is not None:
            sql += " AND entity = @entity"
            params.append(self._p("entity", "STRING", str(entity)))
        sql += " GROUP BY entity, metric ORDER BY entity, metric"
        return [{"entity": r[0], "metric": r[1], "first_date": _to_date(r[2]), "last_date": _to_date(r[3]),
                 "n": int(r[4])} for r in self._select(sql, params)]

    def list_snapshot_entities(self, source: str) -> list[str]:
        return [r[0] for r in self._select(
            f"SELECT DISTINCT entity FROM `{self.table}` WHERE source = @source ORDER BY entity",
            [self._p("source", "STRING", source)])]

    def list_snapshot_sources(self) -> list[str]:
        return [r[0] for r in self._select(f"SELECT DISTINCT source FROM `{self.table}` ORDER BY source")]

    def count_snapshots(self, source: str | None = None, snapshot_date=None) -> int:
        sql, params = f"SELECT COUNT(*) FROM `{self.table}` WHERE 1 = 1", []
        if source is not None:
            sql += " AND source = @source"
            params.append(self._p("source", "STRING", source))
        if snapshot_date is not None:
            sql += " AND snapshot_date = @d"
            params.append(self._p("d", "DATE", _to_date(snapshot_date)))
        return int(self._select(sql, params)[0][0])

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["BigQuerySnapshotBackend", "SnapshotBigQueryUnavailable", "MERGE_SQL", "COLUMNS",
           "DEFAULT_TABLE", "normalise_row", "dedupe_rows"]
