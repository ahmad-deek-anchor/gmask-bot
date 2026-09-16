"""Persistent store for the dynamic access rules (who holds which role, which channels the bot
answers in, runtime settings). Rows are ``(kind, id) -> value``:

    kind "user"     id = Slack user id       value = role (viewer | desk | lead | admin | none)
    kind "channel"  id = Slack channel id    value = "allowed" | "confidential"
    kind "setting"  id = "default_role" | "paused"   value = role | "1" / "0"

Two backends behind the same three methods, chosen by ``ACCESS_BACKEND`` (defaults to the
snapshot backend, so Cloud Run uses BigQuery and a laptop uses sqlite):

    sqlite     data/access.db (``ACCESS_DB_PATH``)                       - single machine
    bigquery   anchorage-corp-eng-playground.gmask_bot.access (``ACCESS_BQ_TABLE``) - the bot's
               service account already has dataEditor on gmask_bot; the table is created on first use.

Writes are rare (an admin command) and reads are cached by access.policy.AccessControl, so
BigQuery latency is irrelevant here.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

KINDS = ("user", "channel", "setting")
DEFAULT_DB_PATH = "data/access.db"
DEFAULT_BQ_TABLE = "anchorage-corp-eng-playground.gmask_bot.access"
DEFAULT_BQ_PROJECT = "anchorage-corp-eng-playground"


class AccessStoreUnavailable(RuntimeError):
    """The backend cannot be opened (missing library, credentials, table)."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AccessStore:
    """Interface. ``list_rules`` returns dicts with kind, id, value, added_by, updated_at."""

    name = "abstract"

    def list_rules(self) -> list[dict]:
        raise NotImplementedError

    def put_rule(self, kind: str, rule_id: str, value: str, added_by: str) -> None:
        raise NotImplementedError

    def delete_rule(self, kind: str, rule_id: str) -> bool:
        raise NotImplementedError

    @property
    def label(self) -> str:
        return self.name


class MemoryAccessStore(AccessStore):
    """In-memory rules (tests, and the fail-open fallback when the real store cannot be opened)."""

    name = "memory"

    def __init__(self, rules: Optional[list[dict]] = None):
        self._rows: dict[tuple[str, str], dict] = {}
        for r in rules or []:
            self.put_rule(r["kind"], r["id"], r["value"], r.get("added_by", ""))

    def list_rules(self) -> list[dict]:
        return [dict(v) for v in self._rows.values()]

    def put_rule(self, kind: str, rule_id: str, value: str, added_by: str) -> None:
        self._rows[(kind, rule_id)] = {"kind": kind, "id": rule_id, "value": value, "added_by": added_by, "updated_at": _now()}

    def delete_rule(self, kind: str, rule_id: str) -> bool:
        return self._rows.pop((kind, rule_id), None) is not None


class SqliteAccessStore(AccessStore):
    name = "sqlite"

    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("ACCESS_DB_PATH", DEFAULT_DB_PATH)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS access_rules (kind TEXT NOT NULL, id TEXT NOT NULL, value TEXT NOT NULL, "
                               "added_by TEXT, updated_at TEXT, PRIMARY KEY (kind, id))")
            self._conn.commit()

    @property
    def label(self) -> str:
        return f"sqlite {self.path}"

    def list_rules(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT kind, id, value, added_by, updated_at FROM access_rules").fetchall()
        return [{"kind": k, "id": i, "value": v, "added_by": a, "updated_at": u} for k, i, v, a, u in rows]

    def put_rule(self, kind: str, rule_id: str, value: str, added_by: str) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO access_rules (kind, id, value, added_by, updated_at) VALUES (?, ?, ?, ?, ?) "
                               "ON CONFLICT(kind, id) DO UPDATE SET value = excluded.value, added_by = excluded.added_by, "
                               "updated_at = excluded.updated_at", (kind, rule_id, value, added_by, _now()))
            self._conn.commit()

    def delete_rule(self, kind: str, rule_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM access_rules WHERE kind = ? AND id = ?", (kind, rule_id))
            self._conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class BigQueryAccessStore(AccessStore):
    """Rules in one small BigQuery table; created on first use. Parameterised statements only."""

    name = "bigquery"

    def __init__(self, table: str | None = None, project: str | None = None, client: Any = None, timeout_s: float = 60.0):
        self.table = (table or os.getenv("ACCESS_BQ_TABLE") or DEFAULT_BQ_TABLE).strip().strip("`")
        self.project = project or os.getenv("BQ_BILLING_PROJECT") or DEFAULT_BQ_PROJECT
        self.timeout_s = timeout_s
        self._client = client
        self._lock = threading.Lock()
        self._ready = False

    @property
    def label(self) -> str:
        return f"bigquery {self.table}"

    @property
    def client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        from google.cloud import bigquery
                    except ImportError as e:  # pragma: no cover - environment
                        raise AccessStoreUnavailable("google-cloud-bigquery is not installed") from e
                    try:
                        self._client = bigquery.Client(project=self.project)
                    except Exception as e:
                        raise AccessStoreUnavailable(f"BigQuery client could not be created: {e}") from e
        return self._client

    def _params(self, values: dict):
        from google.cloud import bigquery
        return [bigquery.ScalarQueryParameter(k, "STRING", v) for k, v in values.items()]

    def _run(self, sql: str, values: Optional[dict] = None):
        from google.cloud import bigquery
        cfg = bigquery.QueryJobConfig(query_parameters=self._params(values or {}))
        job = self.client.query(sql, job_config=cfg)
        return list(job.result(timeout=self.timeout_s))

    def ensure_table(self) -> None:
        if self._ready:
            return
        self._run(f"CREATE TABLE IF NOT EXISTS `{self.table}` (kind STRING NOT NULL, id STRING NOT NULL, value STRING NOT NULL, "
                  "added_by STRING, updated_at TIMESTAMP)")
        self._ready = True

    def list_rules(self) -> list[dict]:
        self.ensure_table()
        rows = self._run(f"SELECT kind, id, value, added_by, CAST(updated_at AS STRING) AS updated_at FROM `{self.table}`")
        return [{"kind": r["kind"], "id": r["id"], "value": r["value"], "added_by": r["added_by"], "updated_at": r["updated_at"]}
                for r in rows]

    def put_rule(self, kind: str, rule_id: str, value: str, added_by: str) -> None:
        self.ensure_table()
        self._run(f"MERGE `{self.table}` t USING (SELECT @kind AS kind, @id AS id) s ON t.kind = s.kind AND t.id = s.id "
                  "WHEN MATCHED THEN UPDATE SET value = @value, added_by = @added_by, updated_at = CURRENT_TIMESTAMP() "
                  "WHEN NOT MATCHED THEN INSERT (kind, id, value, added_by, updated_at) VALUES (@kind, @id, @value, @added_by, CURRENT_TIMESTAMP())",
                  {"kind": kind, "id": rule_id, "value": value, "added_by": added_by})

    def delete_rule(self, kind: str, rule_id: str) -> bool:
        self.ensure_table()
        before = self._run(f"SELECT COUNT(*) AS n FROM `{self.table}` WHERE kind = @kind AND id = @id", {"kind": kind, "id": rule_id})
        n = int(before[0]["n"]) if before else 0
        if n:
            self._run(f"DELETE FROM `{self.table}` WHERE kind = @kind AND id = @id", {"kind": kind, "id": rule_id})
        return n > 0


def open_access_store(backend: str | None = None) -> AccessStore:
    """Backend from ``ACCESS_BACKEND`` (else ``SNAPSHOT_BACKEND``, else sqlite). Raises AccessStoreUnavailable."""
    name = (backend or os.getenv("ACCESS_BACKEND") or os.getenv("SNAPSHOT_BACKEND") or "sqlite").strip().lower()
    if name == "bigquery":
        store = BigQueryAccessStore()
        try:
            store.ensure_table()
        except AccessStoreUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            raise AccessStoreUnavailable(f"BigQuery access table unusable: {e}") from e
        return store
    if name == "memory":
        return MemoryAccessStore()
    if name != "sqlite":
        logger.warning("Unknown ACCESS_BACKEND %r; using sqlite", name)
    try:
        return SqliteAccessStore()
    except Exception as e:  # noqa: BLE001
        raise AccessStoreUnavailable(f"sqlite access store unusable: {e}") from e


__all__ = ["AccessStore", "AccessStoreUnavailable", "BigQueryAccessStore", "KINDS", "MemoryAccessStore",
           "SqliteAccessStore", "open_access_store"]
