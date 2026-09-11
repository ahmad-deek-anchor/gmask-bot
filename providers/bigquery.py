"""Read-only access to the Global Markets desk data in BigQuery.

``DeskBigQuery`` wraps ``google.cloud.bigquery.Client`` with the guard rails the
chat agent needs before it is allowed to run SQL written by an LLM:

* **SELECT / WITH only.** DML, DDL, scripting statements and multi-statement
  scripts are rejected before anything reaches BigQuery.
* **Dataset allow-list.** Every table referenced (``project.dataset.table``,
  ``dataset.table``, with or without backticks, ``INFORMATION_SCHEMA`` views
  included) must live in one of ``allowed_datasets`` of ``data_project``.
  The caller has table-data access on exactly those datasets; everything else
  in the project is denied and referencing it would only produce a 403.
* **Row cap.** A trailing ``LIMIT`` is appended or lowered to ``max_rows``.
* **Cost cap.** ``maximum_bytes_billed`` is set on every job (default 20 GB; the
  largest sanctioned query, Carson Levy's EOW derivatives PnL script, scans ~15 GB)
  and a ``dry_run()`` is available to estimate before running.
* **DECLARE scripts (narrow allowance, not for LLM SQL).** ``query_script`` /
  ``validate_declare_script`` accept a script made of ``DECLARE name TYPE DEFAULT
  <literal>;`` statements followed by exactly one SELECT / WITH query. The query
  part goes through the very same ``validate_sql`` (read-only, dataset allow-list,
  LIMIT), the DECLAREs may only carry plain literals (no sub-queries, no
  expressions). Only project code (providers/haruko_eod.py) calls it; the
  ``query_desk_data`` tool keeps using the strict single-statement ``query``.

Access model (verified 2026-09-10): the user has ``bigquery.tables.getData`` on
``anc-global-markets.brokerage_a1`` and ``anc-global-markets.pricing`` only, and
no ``bigquery.jobs.create`` in that project, so jobs are submitted to - and
billed to - ``billing_project`` (``anchorage-corp-eng-playground`` by default).
Authentication is Application Default Credentials, the same as Vertex.

Table metadata (schemas, row counts, partitioning) is fetched through the table
API, which works with the same grants, and cached in ``data/bq_catalog.json``
for 24 hours so ``list_tables`` / ``describe_table`` do not hit the API on every
chat turn.

``google.cloud.bigquery`` is imported lazily so the rest of the project (and the
test-suite) never needs it installed or authenticated.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

CATALOG_TTL_S = 24 * 3600

# Statement types / scripting keywords that are never allowed, matched as whole
# words on the comment- and string-stripped SQL.
_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "DROP", "ALTER", "TRUNCATE",
    "GRANT", "REVOKE", "DECLARE", "EXECUTE", "CALL", "BEGIN", "COMMIT", "ROLLBACK",
    "EXPORT", "LOAD",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
# SET is only a problem as a statement (SET x = ...); it is legal inside MERGE/UPDATE
# (already banned) and never appears in a SELECT except as the start of a script.
_SET_STATEMENT_RE = re.compile(r"^\s*SET\b", re.IGNORECASE)

_IDENT = r"[A-Za-z0-9_\-]+"
# Table references after FROM / JOIN (backticks already removed), 2-4 dotted parts.
_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(" + _IDENT + r"(?:\." + _IDENT + r"){1,3})\b", re.IGNORECASE
)
# Any 3-part reference whose first part looks like a GCP project id (contains a
# hyphen) anywhere in the text - catches refs hidden in table functions etc.
_PROJECT_REF_RE = re.compile(r"\b(" + _IDENT + r"-" + _IDENT + r")\.(" + _IDENT + r")\.(" + _IDENT + r")\b")
_TRAILING_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)(\s+OFFSET\s+\d+)?\s*$", re.IGNORECASE)


# DECLARE statements tolerated by validate_declare_script: a plain literal default only.
_DECLARE_TYPES = r"(?:TIME|DATE|DATETIME|TIMESTAMP|STRING|INT64|FLOAT64|NUMERIC|BIGNUMERIC|BOOL)"
_DECLARE_LITERAL = (
    r"(?:(?:TIME|DATE|DATETIME|TIMESTAMP)\s*)?'[^'\\;]*'"   # (typed) string literal, no ; or escapes
    r"|-?\d+(?:\.\d+)?"                                    # number
    r"|TRUE|FALSE"
)
_DECLARE_STMT_RE = re.compile(
    r"^\s*DECLARE\s+([A-Za-z_][A-Za-z0-9_]*)\s+(" + _DECLARE_TYPES + r")\s+DEFAULT\s+(" + _DECLARE_LITERAL + r")\s*;",
    re.IGNORECASE,
)
_DECLARE_START_RE = re.compile(r"^\s*DECLARE\b", re.IGNORECASE)


class SQLGuardError(ValueError):
    """The SQL was rejected by the read-only guard (never sent to BigQuery)."""


class BigQueryUnavailable(RuntimeError):
    """The client library or credentials are not available."""


# ---------------------------------------------------------------------------
# SQL guard (pure functions, no client needed)
# ---------------------------------------------------------------------------

def strip_sql_noise(sql: str) -> str:
    """Remove comments and string literals so keyword / table scanning is reliable."""
    out: List[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        two = sql[i:i + 2]
        if two == "--":
            j = sql.find("\n", i)
            i = n if j < 0 else j
            continue
        if two == "/*":
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
            continue
        if ch in ("'", '"'):
            # string literal (handles r'' / b'' prefixes implicitly - the prefix stays)
            j = i + 1
            while j < n:
                if sql[j] == "\\":
                    j += 2
                    continue
                if sql[j] == ch:
                    break
                j += 1
            out.append("''")
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _split_ref(ref: str) -> Tuple[Optional[str], Optional[str]]:
    """(project, dataset) for a dotted table reference; project None when omitted.

    Handles ``project.dataset.table``, ``dataset.table``,
    ``project.dataset.INFORMATION_SCHEMA.VIEW`` and ``dataset.INFORMATION_SCHEMA.VIEW``.
    Single-part names (CTEs, aliases) return (None, None) and are ignored.
    """
    parts = ref.split(".")
    upper = [p.upper() for p in parts]
    if "INFORMATION_SCHEMA" in upper:
        k = upper.index("INFORMATION_SCHEMA")
        if k == 0:
            return None, "INFORMATION_SCHEMA"  # project-level schema: not allowed
        dataset = parts[k - 1]
        project = parts[k - 2] if k >= 2 else None
        return project, dataset
    if len(parts) == 1:
        return None, None
    if len(parts) == 2:
        return None, parts[0]
    if len(parts) == 3:
        return parts[0], parts[1]
    raise SQLGuardError(f"Cannot parse table reference '{ref}'.")


def referenced_tables(sql: str) -> List[str]:
    """Dotted table references found after FROM / JOIN and any project-qualified ref."""
    text = strip_sql_noise(sql).replace("`", "")
    refs = [m.group(1) for m in _TABLE_REF_RE.finditer(text)]
    refs += [".".join(m.groups()) for m in _PROJECT_REF_RE.finditer(text)]
    seen: List[str] = []
    for r in refs:
        r = r.rstrip(".")
        if r and r not in seen:
            seen.append(r)
    return seen


def validate_sql(sql: str, data_project: str, allowed_datasets: Sequence[str], max_rows: int) -> str:
    """Return a safe version of ``sql`` (trailing LIMIT enforced) or raise SQLGuardError."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLGuardError("Empty SQL.")
    clean = strip_sql_noise(sql).strip()
    clean = clean.rstrip(";").strip()  # one trailing semicolon is tolerated
    if ";" in clean:
        raise SQLGuardError("Only a single statement is allowed (found ';').")
    first = re.match(r"^\s*\(*\s*(\w+)", clean)
    if not first or first.group(1).upper() not in ("SELECT", "WITH"):
        raise SQLGuardError("Only read-only SELECT / WITH queries are allowed.")
    if _SET_STATEMENT_RE.match(clean):
        raise SQLGuardError("Scripting statements are not allowed.")
    bad = _FORBIDDEN_RE.search(clean)
    if bad:
        raise SQLGuardError(f"Statement type '{bad.group(1).upper()}' is not allowed; read-only SELECT queries only.")

    allowed = {d.lower() for d in allowed_datasets}
    for ref in referenced_tables(clean):
        project, dataset = _split_ref(ref)
        if dataset is None:
            continue
        if project is not None and project.lower() != data_project.lower():
            raise SQLGuardError(
                f"Table '{ref}' is in project '{project}'; only '{data_project}' is accessible."
            )
        if dataset.lower() not in allowed:
            raise SQLGuardError(
                f"Dataset '{dataset}' (in '{ref}') is not accessible. Allowed datasets in "
                f"{data_project}: {', '.join(sorted(allowed))}."
            )

    # Row cap: lower an existing trailing LIMIT or append one.
    body = sql.strip().rstrip(";").rstrip()
    body = _strip_trailing_comments(body)
    m = _TRAILING_LIMIT_RE.search(body)
    if m:
        n = int(m.group(1))
        if n > max_rows:
            body = body[: m.start()] + f"LIMIT {max_rows}" + (m.group(2) or "")
    else:
        body = f"{body}\nLIMIT {max_rows}"
    return body


def split_declare_script(sql: str) -> Tuple[List[Tuple[str, str, str]], str]:
    """Split ``DECLARE a T DEFAULT lit; ... <query>`` into ([(name, type, literal)], query).

    Leading comments are skipped. Raises SQLGuardError when a DECLARE does not have
    the exact ``DECLARE name TYPE DEFAULT <literal>;`` shape (no sub-queries, no
    expressions, no untyped / default-less variables, no SET).
    """
    if not isinstance(sql, str) or not sql.strip():
        raise SQLGuardError("Empty SQL.")
    # drop leading comments so the header of sql/*.sql files is not mistaken for code
    rest = sql
    while True:
        stripped = rest.lstrip()
        if stripped.startswith("--"):
            nl = stripped.find("\n")
            rest = "" if nl < 0 else stripped[nl + 1:]
            continue
        if stripped.startswith("/*"):
            end = stripped.find("*/")
            if end < 0:
                raise SQLGuardError("Unterminated comment.")
            rest = stripped[end + 2:]
            continue
        rest = stripped
        break
    declares: List[Tuple[str, str, str]] = []
    while _DECLARE_START_RE.match(rest):
        m = _DECLARE_STMT_RE.match(rest)
        if not m:
            head = rest.strip().splitlines()[0][:80] if rest.strip() else ""
            raise SQLGuardError(
                "DECLARE statements must have the form 'DECLARE name TYPE DEFAULT <literal>;' "
                f"(plain literal only): {head!r}")
        declares.append((m.group(1), m.group(2).upper(), m.group(3)))
        rest = rest[m.end():]
    return declares, rest


def validate_declare_script(sql: str, data_project: str, allowed_datasets: Sequence[str], max_rows: int) -> str:
    """Guard for the narrow 'DECLARE literals + one SELECT/WITH' script shape.

    Every DECLARE must be ``DECLARE name TYPE DEFAULT <literal>;`` and the remainder
    must pass :func:`validate_sql` unchanged (single read-only SELECT / WITH, every
    table in an allowed dataset of ``data_project``, trailing LIMIT <= ``max_rows``).
    Returns the re-assembled script. Never weakens the dataset allow-list or the
    read-only rule: DML/DDL after the DECLAREs is refused exactly as before.
    """
    declares, body = split_declare_script(sql)
    if not body.strip():
        raise SQLGuardError("A DECLARE script must end with exactly one SELECT / WITH query.")
    safe_body = validate_sql(body, data_project, allowed_datasets, max_rows)
    if not declares:
        return safe_body
    names = [d[0].lower() for d in declares]
    if len(set(names)) != len(names):
        raise SQLGuardError("Duplicate DECLARE variable names.")
    header = "\n".join(f"DECLARE {n} {t} DEFAULT {lit};" for n, t, lit in declares)
    return f"{header}\n{safe_body}"


def _strip_trailing_comments(sql: str) -> str:
    """Drop trailing -- comments / whitespace so LIMIT detection sees the real end."""
    lines = sql.rstrip().splitlines()
    while lines and (not lines[-1].strip() or lines[-1].strip().startswith("--")):
        lines.pop()
    text = "\n".join(lines)
    # an inline trailing "-- comment" on the last line
    last_nl = text.rfind("\n")
    tail = text[last_nl + 1:]
    if "--" in tail and tail.count("'") % 2 == 0:
        text = text[: last_nl + 1] + tail[: tail.index("--")].rstrip()
    return text.rstrip()


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------

class DeskBigQuery:
    """Guarded, read-only BigQuery access to the desk datasets.

    ``client`` may be injected (tests use a fake); otherwise a
    ``google.cloud.bigquery.Client`` for ``billing_project`` is created lazily.
    """

    def __init__(
        self,
        client=None,
        *,
        data_project: str = "anc-global-markets",
        billing_project: str = "anchorage-corp-eng-playground",
        allowed_datasets: Iterable[str] = ("brokerage_a1", "pricing"),
        max_bytes_billed: int = 20_000_000_000,
        max_rows: int = 200,
        timeout_s: int = 60,
        catalog_path: str | Path = "data/bq_catalog.json",
        catalog_ttl_s: int = CATALOG_TTL_S,
        now=time.time,
    ):
        self._client = client
        self.data_project = data_project
        self.billing_project = billing_project
        self.allowed_datasets = tuple(allowed_datasets)
        self.max_bytes_billed = int(max_bytes_billed)
        self.max_rows = int(max_rows)
        self.timeout_s = int(timeout_s)
        self.catalog_path = Path(catalog_path)
        self.catalog_ttl_s = catalog_ttl_s
        self._now = now
        self._catalog: Optional[dict] = None
        self.last_bytes_processed: Optional[int] = None

    @classmethod
    def from_config(cls, cfg=None, client=None) -> "DeskBigQuery":
        if cfg is None:
            from utils.config import Config
            cfg = Config()
        return cls(
            client,
            data_project=cfg.BQ_DATA_PROJECT,
            billing_project=cfg.BQ_BILLING_PROJECT,
            allowed_datasets=cfg.BQ_ALLOWED_DATASETS,
            max_bytes_billed=cfg.BQ_MAX_BYTES_BILLED,
            max_rows=cfg.BQ_MAX_ROWS,
            timeout_s=cfg.BQ_TIMEOUT_S,
            catalog_path=cfg.BQ_CATALOG_PATH,
        )

    # -- client -------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            try:
                from google.cloud import bigquery
            except ImportError as e:  # pragma: no cover - depends on environment
                raise BigQueryUnavailable("google-cloud-bigquery is not installed") from e
            try:
                self._client = bigquery.Client(project=self.billing_project)
            except Exception as e:  # DefaultCredentialsError etc.
                raise BigQueryUnavailable(f"BigQuery client could not be created: {e}") from e
        return self._client

    def table(self, name: str) -> str:
        """Backticked fully-qualified name for ``dataset.table`` (or already qualified)."""
        parts = name.replace("`", "").split(".")
        if len(parts) == 2:
            parts = [self.data_project] + parts
        if len(parts) != 3:
            raise SQLGuardError(f"Expected dataset.table, got '{name}'.")
        if parts[1] not in self.allowed_datasets:
            raise SQLGuardError(
                f"Dataset '{parts[1]}' is not accessible. Allowed: {', '.join(self.allowed_datasets)}."
            )
        return "`" + ".".join(parts) + "`"

    # -- queries ------------------------------------------------------------

    def validate(self, sql: str) -> str:
        return validate_sql(sql, self.data_project, self.allowed_datasets, self.max_rows)

    def _job_config(self, dry_run: bool = False):
        from google.cloud import bigquery

        return bigquery.QueryJobConfig(
            maximum_bytes_billed=self.max_bytes_billed,
            use_legacy_sql=False,
            use_query_cache=True,
            dry_run=dry_run,
        )

    def dry_run(self, sql: str) -> int:
        """Bytes the (guarded) query would process. Never runs it."""
        safe = self.validate(sql)
        job = self.client.query(safe, job_config=self._job_config(dry_run=True))
        return int(job.total_bytes_processed or 0)

    def query(self, sql: str) -> pd.DataFrame:
        """Run a guarded read-only query and return a DataFrame (Decimal -> float).

        ``df.attrs['bytes_processed']``, ``df.attrs['cache_hit']`` and
        ``df.attrs['sql']`` are set for callers that want to report cost.
        """
        return self._run(self.validate(sql), self.timeout_s)

    def validate_script(self, sql: str, max_rows: Optional[int] = None) -> str:
        """Guard for project-owned 'DECLARE literals + one SELECT' scripts (see module doc)."""
        return validate_declare_script(sql, self.data_project, self.allowed_datasets,
                                       int(max_rows) if max_rows else self.max_rows)

    def query_script(self, sql: str, *, max_rows: Optional[int] = None,
                     timeout_s: Optional[int] = None) -> pd.DataFrame:
        """Run a ``DECLARE ... DEFAULT <literal>; ... SELECT`` script under the same guards.

        Same cost cap, dataset allow-list and read-only rule as :meth:`query`; the
        row cap (``max_rows``) and timeout may be raised per call because the
        sanctioned scripts return one row per day and take ~45 s. Not reachable
        from the LLM's ``query_desk_data`` tool.
        """
        safe = self.validate_script(sql, max_rows)
        return self._run(safe, int(timeout_s) if timeout_s else self.timeout_s)

    def _run(self, safe: str, timeout_s: int) -> pd.DataFrame:
        logger.debug("BigQuery query (billing=%s): %s", self.billing_project, safe)
        job = self.client.query(safe, job_config=self._job_config(), timeout=timeout_s)
        try:
            result = job.result(timeout=timeout_s)
        except Exception as e:  # surface permission problems clearly
            raise _friendly_error(e, self.billing_project, self.allowed_datasets) from e
        df = result.to_dataframe(create_bqstorage_client=False)
        df = _decimals_to_float(df)
        processed = getattr(job, "total_bytes_processed", None)
        self.last_bytes_processed = int(processed) if processed is not None else None
        df.attrs["bytes_processed"] = self.last_bytes_processed
        cache_hit = getattr(job, "cache_hit", None)
        df.attrs["cache_hit"] = bool(cache_hit) if cache_hit is not None else None
        df.attrs["sql"] = safe
        return df

    # -- metadata -----------------------------------------------------------

    def _table_info(self, dataset: str, table_id: str) -> dict:
        t = self.client.get_table(f"{self.data_project}.{dataset}.{table_id}")
        part = getattr(t, "time_partitioning", None)
        rng = getattr(t, "range_partitioning", None)
        partitioning = None
        if part is not None:
            partitioning = f"{getattr(part, 'type_', 'DAY')} on {getattr(part, 'field', None) or '_PARTITIONTIME'}"
        elif rng is not None:
            partitioning = f"RANGE on {getattr(rng, 'field', '?')}"
        modified = getattr(t, "modified", None)
        ttype = getattr(t, "table_type", None) or "TABLE"
        is_view = ttype.upper().endswith("VIEW")  # VIEW / MATERIALIZED_VIEW report no row count
        return {
            "table": f"{dataset}.{table_id}",
            "type": ttype,
            "description": getattr(t, "description", None) or "",
            "num_rows": int(t.num_rows) if getattr(t, "num_rows", None) is not None and not is_view else None,
            "num_bytes": int(t.num_bytes) if getattr(t, "num_bytes", None) is not None and not is_view else None,
            "modified": modified.isoformat() if isinstance(modified, datetime) else (str(modified) if modified else None),
            "partitioning": partitioning,
            "clustering": list(getattr(t, "clustering_fields", None) or []),
            "columns": [
                {"name": f.name, "type": f.field_type, "mode": getattr(f, "mode", None) or "NULLABLE",
                 "description": getattr(f, "description", None) or ""}
                for f in (t.schema or [])
            ],
        }

    def refresh_catalog(self) -> dict:
        """Rebuild the table catalog for every allowed dataset and persist it."""
        datasets: Dict[str, dict] = {}
        for ds in self.allowed_datasets:
            tables: Dict[str, dict] = {}
            for item in self.client.list_tables(f"{self.data_project}.{ds}"):
                tid = item.table_id
                try:
                    tables[tid] = self._table_info(ds, tid)
                except Exception as e:  # keep going; one bad table should not kill the catalog
                    logger.warning("describe %s.%s failed: %s", ds, tid, e)
                    tables[tid] = {"table": f"{ds}.{tid}", "type": getattr(item, "table_type", None),
                                   "description": "", "columns": [], "error": str(e)}
            datasets[ds] = tables
        catalog = {
            "built_at": self._now(),
            "built_at_iso": datetime.fromtimestamp(self._now(), tz=timezone.utc).isoformat(),
            "data_project": self.data_project,
            "datasets": datasets,
        }
        self._catalog = catalog
        try:
            self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
            self.catalog_path.write_text(json.dumps(catalog, indent=1, default=str))
        except OSError as e:
            logger.warning("could not write catalog to %s: %s", self.catalog_path, e)
        return catalog

    def catalog(self) -> dict:
        """Cached catalog (memory -> disk -> rebuild when missing/stale/mismatched)."""
        if self._catalog is None:
            try:
                self._catalog = json.loads(self.catalog_path.read_text())
            except (OSError, ValueError):
                self._catalog = None
        c = self._catalog
        stale = (
            c is None
            or c.get("data_project") != self.data_project
            or set(c.get("datasets", {})) != set(self.allowed_datasets)
            or (self._now() - float(c.get("built_at", 0))) > self.catalog_ttl_s
        )
        if stale:
            return self.refresh_catalog()
        return c

    def list_tables(self, dataset: Optional[str] = None, keyword: Optional[str] = None) -> List[dict]:
        """[{table, type, num_rows, description, columns(n)}] filtered by dataset / keyword.

        ``keyword`` matches the table name, its description or any column name.
        """
        cat = self.catalog()
        if dataset and dataset not in self.allowed_datasets:
            raise SQLGuardError(
                f"Dataset '{dataset}' is not accessible. Allowed: {', '.join(self.allowed_datasets)}."
            )
        kw = (keyword or "").strip().lower()
        out: List[dict] = []
        for ds, tables in cat.get("datasets", {}).items():
            if dataset and ds != dataset:
                continue
            for tid, info in sorted(tables.items()):
                cols = [c["name"] for c in info.get("columns", [])]
                hay = " ".join([tid, info.get("description") or ""] + cols).lower()
                if kw and kw not in hay:
                    continue
                out.append({
                    "table": f"{ds}.{tid}",
                    "type": info.get("type"),
                    "num_rows": info.get("num_rows"),
                    "partitioning": info.get("partitioning"),
                    "description": info.get("description") or "",
                    "n_columns": len(cols),
                    "matched_columns": [c for c in cols if kw and kw in c.lower()][:6] if kw else [],
                })
        return out

    def describe_table(self, name: str) -> dict:
        """Full metadata for ``dataset.table`` (from the catalog; live lookup if absent)."""
        parts = name.replace("`", "").split(".")
        if len(parts) == 3 and parts[0] == self.data_project:
            parts = parts[1:]
        if len(parts) != 2:
            raise SQLGuardError(f"Expected 'dataset.table', got '{name}'.")
        ds, tid = parts
        if ds not in self.allowed_datasets:
            raise SQLGuardError(
                f"Dataset '{ds}' is not accessible. Allowed: {', '.join(self.allowed_datasets)}."
            )
        info = self.catalog().get("datasets", {}).get(ds, {}).get(tid)
        if info is None or info.get("error"):
            info = self._table_info(ds, tid)
            if self._catalog is not None:
                self._catalog.setdefault("datasets", {}).setdefault(ds, {})[tid] = info
        return info


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _decimals_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """NUMERIC/BIGNUMERIC arrive as Decimal objects; convert to float64 for formatting."""
    for col in df.columns:
        s = df[col]
        if s.dtype == object:
            sample = s.dropna()
            if len(sample) and all(isinstance(v, Decimal) for v in sample.head(20)):
                df[col] = pd.to_numeric(s.map(lambda v: float(v) if isinstance(v, Decimal) else v), errors="coerce")
    return df


def _friendly_error(e: Exception, billing_project: str, allowed: Sequence[str]) -> Exception:
    msg = str(e).split("\n\nLocation:")[0].strip()  # drop the job id / location trailer
    if "bytes billed" in msg or "maximum_bytes_billed" in msg.lower():
        return RuntimeError(
            "Query exceeded the per-query cost cap (BQ_MAX_BYTES_BILLED). Add a partition/date "
            "filter, select fewer columns, or use a smaller table. " + msg
        )
    if "Access Denied" in msg or "403" in msg:
        return PermissionError(
            f"BigQuery access denied. Jobs run in '{billing_project}' (needs bigquery.jobUser there); "
            f"table data is readable only in datasets {', '.join(allowed)}. " + msg
        )
    return RuntimeError(msg)


def format_bytes(n: Optional[int]) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


__all__ = [
    "BigQueryUnavailable",
    "CATALOG_TTL_S",
    "DeskBigQuery",
    "SQLGuardError",
    "format_bytes",
    "referenced_tables",
    "split_declare_script",
    "strip_sql_noise",
    "validate_declare_script",
    "validate_sql",
]
