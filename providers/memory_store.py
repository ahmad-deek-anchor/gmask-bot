"""Persistent memory + daily data snapshots for the chat agent and the Slack bot.

One small SQLite database (default ``data/memory.db``; env ``MEMORY_DB_URL``) with two
tables and a tiny DAO. No LangGraph store classes, no ORM.

    memories(id TEXT PK, namespace, key, text, meta JSON, created_at, updated_at,
             expires_at NULL, embedding BLOB NULL)
        namespaces: facts:shared | prefs:<slack_user_id> | rules:<channel_id> | episodes:<slack_user_id>
    snapshots(snapshot_date DATE, source, entity, metric, value REAL, value_json TEXT, captured_at,
              PRIMARY KEY (snapshot_date, source, entity, metric))
        sources: haruko | signals | sheet (see snapshot_daily.py)

Search is semantic when an embedder is available (Vertex ``text-embedding-005`` by default,
embeddings kept as float32 bytes, cosine similarity in Python - the tables are small) and
falls back to keyword (BM25-ish) scoring when embeddings fail or are disabled
(``MEMORY_EMBEDDINGS=off``). Rules namespaces are always returned unfiltered.

Backends: ``sqlite:///path`` (default, fully supported) or ``postgresql://...`` (Cloud Run;
the same SQL through psycopg - needs ``pip install psycopg[binary]``, otherwise a clear
NotImplementedError is raised).

Snapshots have their own backend switch, env ``SNAPSHOT_BACKEND``: ``sqlite`` (default; the
``snapshots`` table above) or ``bigquery`` (providers/snapshot_bq.py: the shared table
``SNAPSHOT_BQ_TABLE`` that the Cloud Run job writes and the local bot reads). Memories
always stay in the local database. All snapshot methods return identical shapes on both.

Thread-safe: every statement runs under one lock on a single connection (the agent runs tool
calls in threads and the Slack bot summarises sessions in background tasks).
"""

from __future__ import annotations

import array
import concurrent.futures
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_DB_URL = "sqlite:///data/memory.db"
DEFAULT_SNAPSHOT_BACKEND = "sqlite"
SNAPSHOT_BACKENDS = ("sqlite", "bigquery")
DEFAULT_EMBED_MODEL = "text-embedding-005"
DEFAULT_EMBED_PROJECT = "anchorage-ai-development"
DEFAULT_EMBED_LOCATION = "us-central1"
DEFAULT_EPISODE_TTL_DAYS = 90

SHARED_FACTS = "facts:shared"


def prefs_namespace(user_id: str) -> str:
    return f"prefs:{user_id}"


def rules_namespace(channel_id: str) -> str:
    return f"rules:{channel_id}"


def episodes_namespace(user_id: str) -> str:
    return f"episodes:{user_id}"


def episode_ttl_days() -> int:
    raw = os.getenv("MEMORY_EPISODE_TTL_DAYS", "")
    try:
        return int(raw) if raw.strip() else DEFAULT_EPISODE_TTL_DAYS
    except ValueError:
        logger.warning("Ignoring non-integer MEMORY_EPISODE_TTL_DAYS=%r", raw)
        return DEFAULT_EPISODE_TTL_DAYS


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat(sep=" ")


def _parse_dt(s) -> Optional[datetime]:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s
    try:
        return datetime.fromisoformat(str(s))
    except ValueError:
        return None


def _to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

class HashEmbedder:
    """Deterministic, network-free embedder for tests: bag-of-words hashed into ``dim`` buckets."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.available = True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _tokenize(text):
                h = 0
                for ch in tok:
                    h = (h * 131 + ord(ch)) % (2 ** 31)
                vec[h % self.dim] += 1.0
            out.append(vec)
        return out


DEFAULT_EMBED_TIMEOUT_S = 20.0
EMBED_FAILURES_TO_TRIP = 3        # consecutive failures before the circuit breaker opens
EMBED_COOLDOWN_S = 600.0          # keyword-only for this long once it has opened


def embed_timeout_s() -> float:
    """Hard deadline for one embeddings call (env ``MEMORY_EMBED_TIMEOUT_S``, default 20 s)."""
    raw = os.getenv("MEMORY_EMBED_TIMEOUT_S", "")
    try:
        val = float(raw) if raw.strip() else DEFAULT_EMBED_TIMEOUT_S
    except ValueError:
        val = DEFAULT_EMBED_TIMEOUT_S
    return val if val > 0 else DEFAULT_EMBED_TIMEOUT_S


def call_with_deadline(fn: Callable[[], Any], timeout: float):
    """Run ``fn()`` on a daemon thread and wait at most ``timeout`` seconds for its result.

    Raises ``concurrent.futures.TimeoutError`` when the deadline passes; the thread is left
    to finish (or hang) on its own and, being a daemon, never blocks interpreter exit. This
    is what makes a hung embeddings call unable to block the caller - the Slack bot's event
    loop included - beyond the deadline.
    """
    fut: concurrent.futures.Future = concurrent.futures.Future()

    def runner():
        try:
            fut.set_result(fn())
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            fut.set_exception(e)

    threading.Thread(target=runner, name="memory-embed", daemon=True).start()
    return fut.result(timeout=timeout)


def _vertex_embeddings_client(model: str, project: str, location: str):
    import warnings

    from langchain_google_vertexai import VertexAIEmbeddings

    with warnings.catch_warnings():
        # langchain-google-vertexai 3.2 deprecates this class in favour of the GenAI package;
        # text-embedding-005 on Vertex still needs it.
        warnings.simplefilter("ignore")
        return VertexAIEmbeddings(model_name=model, project=project, location=location)


class VertexEmbedder:
    """Lazy ``langchain_google_vertexai.VertexAIEmbeddings`` behind a hard deadline and a
    circuit breaker.

    Every call (the one-off probe that builds the client and every ``embed_documents``) runs
    on a worker thread and is abandoned after ``timeout_s`` (env ``MEMORY_EMBED_TIMEOUT_S``,
    default 20 s). A timeout or error makes *that* call fall back to keyword search (returns
    ``None``); after ``EMBED_FAILURES_TO_TRIP`` consecutive failures embeddings are disabled
    for ``EMBED_COOLDOWN_S`` (10 min, logged once) and then retried. A missing dependency
    (ImportError) disables them for good.

    ``client_factory(model, project, location)`` and ``clock`` are injection points for tests.
    """

    def __init__(self, model: str | None = None, project: str | None = None, location: str | None = None,
                 timeout_s: float | None = None, client_factory: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.model = model or os.getenv("MEMORY_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self.project = project or os.getenv("MEMORY_EMBED_PROJECT") or os.getenv("VERTEX_PROJECT") or DEFAULT_EMBED_PROJECT
        self.location = location or os.getenv("MEMORY_EMBED_LOCATION") or DEFAULT_EMBED_LOCATION
        self.timeout_s = float(timeout_s) if timeout_s is not None else embed_timeout_s()
        self._client_factory = client_factory or _vertex_embeddings_client
        self._clock = clock
        self._client = None
        self._lock = threading.Lock()
        self.failures = 0                 # consecutive failures
        self.disabled_until = 0.0         # monotonic time the breaker closes again
        self.permanently_off = False      # ImportError: nothing to retry

    @property
    def available(self) -> bool:
        """False while the circuit breaker is open (or embeddings are permanently off)."""
        return not self.permanently_off and self._clock() >= self.disabled_until

    # -- bookkeeping -------------------------------------------------------

    def _succeeded(self) -> None:
        self.failures = 0

    def _failed(self, exc: BaseException, what: str) -> None:
        self.failures += 1
        detail = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}".rstrip(": ")
        if isinstance(exc, concurrent.futures.TimeoutError):
            detail = f"timed out after {self.timeout_s:g}s"
        if self.failures >= EMBED_FAILURES_TO_TRIP:
            self.disabled_until = self._clock() + EMBED_COOLDOWN_S
            self.failures = 0
            logger.warning("Memory embeddings disabled for %d min after %d consecutive failures (%s: %s); "
                           "keyword search until then", int(EMBED_COOLDOWN_S // 60), EMBED_FAILURES_TO_TRIP, what, detail)
        elif self.failures == 1:
            logger.warning("Memory embeddings %s failed (%s); falling back to keyword search for this call", what, detail)
        else:
            logger.debug("Memory embeddings %s failed again (%s); %d/%d", what, detail, self.failures, EMBED_FAILURES_TO_TRIP)

    # -- calls -------------------------------------------------------------

    def _ensure(self):
        """Return the live client, building and probing it under the deadline on first use."""
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                def make():
                    client = self._client_factory(self.model, self.project, self.location)
                    probe = client.embed_query("memory store probe")
                    if not probe or not isinstance(probe[0], float):
                        raise RuntimeError("empty embedding returned")
                    return client, len(probe)

                client, dim = call_with_deadline(make, self.timeout_s)
            except ImportError as e:
                self.permanently_off = True
                logger.warning("Memory embeddings unavailable (%s); keyword search only", e)
                raise
            self._client = client
            logger.info("Memory embeddings: Vertex %s (%s/%s, dim %d)", self.model, self.project, self.location, dim)
            return client

    def probe(self) -> bool:
        """Build + verify the client now (start-up warm-up, off the event loop). True when embeddings work."""
        if not self.available:
            return False
        try:
            self._ensure()
        except Exception as e:  # noqa: BLE001
            self._failed(e, "probe")
            return False
        self._succeeded()
        return True

    def embed(self, texts: Sequence[str]) -> Optional[list[list[float]]]:
        """Vectors for ``texts`` or ``None`` (keyword search for this call) - never blocks past the deadline."""
        if not self.available:
            return None
        try:
            client = self._ensure()
            vecs = call_with_deadline(lambda: client.embed_documents(list(texts)), self.timeout_s)
        except Exception as e:  # noqa: BLE001
            self._failed(e, "call")
            return None
        self._succeeded()
        return vecs


def default_embedder():
    """VertexEmbedder unless ``MEMORY_EMBEDDINGS`` is off/0/false/keyword."""
    mode = os.getenv("MEMORY_EMBEDDINGS", "vertex").strip().lower()
    if mode in ("off", "0", "false", "no", "none", "keyword"):
        return None
    return VertexEmbedder()


def _pack(vec: Sequence[float]) -> bytes:
    return array.array("f", [float(x) for x in vec]).tobytes()


def _unpack(blob) -> Optional[list[float]]:
    if not blob:
        return None
    a = array.array("f")
    a.frombytes(bytes(blob))
    return list(a)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Keyword scoring (BM25-ish; documents are a few hundred rows at most)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "i", "me", "my",
         "we", "you", "it", "that", "this", "with", "as", "at", "by", "do", "does", "what", "which", "how",
         "please", "about", "remember", "not", "no", "than", "from", "was", "were", "have", "has"}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP]


def keyword_scores(query: str, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 over ``docs`` for ``query``; scores normalised to [0, 1] by the best hit."""
    q = _tokenize(query)
    if not q or not docs:
        return [0.0] * len(docs)
    toks = [_tokenize(d) for d in docs]
    n = len(docs)
    avgdl = max(1.0, sum(len(t) for t in toks) / n)
    df: dict[str, int] = {}
    for t in toks:
        for term in set(t):
            df[term] = df.get(term, 0) + 1
    scores = []
    for t in toks:
        s = 0.0
        if t:
            counts: dict[str, int] = {}
            for term in t:
                counts[term] = counts.get(term, 0) + 1
            for term in q:
                # prefix match so "bps" hits "bps" and "prefer" hits "prefers"
                tf = sum(c for w, c in counts.items() if w == term or (len(term) >= 4 and w.startswith(term)))
                if not tf:
                    continue
                d = df.get(term, 0) or 1
                idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
                s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avgdl))
        scores.append(s)
    top = max(scores) if scores else 0.0
    return [s / top if top > 0 else 0.0 for s in scores]


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    id: str
    namespace: str
    key: Optional[str]
    text: str
    meta: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    score: float = 0.0

    @property
    def kind(self) -> str:
        return self.namespace.split(":", 1)[0]

    @property
    def owner(self) -> str:
        return self.namespace.split(":", 1)[1] if ":" in self.namespace else ""

    def short_id(self) -> str:
        return self.id[:8]

    def as_dict(self) -> dict:
        return {"id": self.id, "namespace": self.namespace, "key": self.key, "text": self.text,
                "meta": self.meta, "created_at": _iso(self.created_at) if self.created_at else None,
                "updated_at": _iso(self.updated_at) if self.updated_at else None,
                "expires_at": _iso(self.expires_at) if self.expires_at else None, "score": round(self.score, 3)}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

_SQLITE_DDL = [
    """CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        key TEXT,
        text TEXT NOT NULL,
        meta TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        expires_at TEXT,
        embedding BLOB
    )""",
    "CREATE INDEX IF NOT EXISTS ix_memories_ns ON memories(namespace)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_memories_ns_key ON memories(namespace, key)",
    """CREATE TABLE IF NOT EXISTS snapshots (
        snapshot_date TEXT NOT NULL,
        source TEXT NOT NULL,
        entity TEXT NOT NULL,
        metric TEXT NOT NULL,
        value REAL,
        value_json TEXT,
        captured_at TEXT NOT NULL,
        PRIMARY KEY (snapshot_date, source, entity, metric)
    )""",
    "CREATE INDEX IF NOT EXISTS ix_snapshots_series ON snapshots(source, entity, metric, snapshot_date)",
]

_PG_DDL = [
    """CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        key TEXT,
        text TEXT NOT NULL,
        meta TEXT,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        expires_at TIMESTAMP,
        embedding BYTEA
    )""",
    "CREATE INDEX IF NOT EXISTS ix_memories_ns ON memories(namespace)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_memories_ns_key ON memories(namespace, key)",
    """CREATE TABLE IF NOT EXISTS snapshots (
        snapshot_date DATE NOT NULL,
        source TEXT NOT NULL,
        entity TEXT NOT NULL,
        metric TEXT NOT NULL,
        value DOUBLE PRECISION,
        value_json TEXT,
        captured_at TIMESTAMP NOT NULL,
        PRIMARY KEY (snapshot_date, source, entity, metric)
    )""",
    "CREATE INDEX IF NOT EXISTS ix_snapshots_series ON snapshots(source, entity, metric, snapshot_date)",
]


class _Backend:
    """Minimal DB-API wrapper: one connection, one lock, ``?`` placeholders."""

    paramstyle_qmark = True

    def __init__(self, conn, ddl: Iterable[str]):
        self.conn = conn
        self.lock = threading.RLock()
        with self.lock:
            for stmt in ddl:
                self.conn.execute(stmt)
            self.conn.commit()

    def _sql(self, sql: str) -> str:
        return sql if self.paramstyle_qmark else sql.replace("?", "%s")

    def execute(self, sql: str, params: Sequence = ()) -> None:
        with self.lock:
            self.conn.execute(self._sql(sql), tuple(params))
            self.conn.commit()

    def executemany(self, sql: str, rows: Iterable[Sequence]) -> None:
        with self.lock:
            self.conn.executemany(self._sql(sql), [tuple(r) for r in rows])
            self.conn.commit()

    def fetchall(self, sql: str, params: Sequence = ()) -> list[tuple]:
        with self.lock:
            cur = self.conn.execute(self._sql(sql), tuple(params))
            return [tuple(r) for r in cur.fetchall()]

    def rowcount_execute(self, sql: str, params: Sequence = ()) -> int:
        with self.lock:
            cur = self.conn.execute(self._sql(sql), tuple(params))
            self.conn.commit()
            return cur.rowcount if cur.rowcount is not None else 0

    def close(self) -> None:
        with self.lock:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass


class _SqliteBackend(_Backend):
    def __init__(self, path: str):
        if path not in (":memory:", ""):
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path or ":memory:", check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL") if path not in (":memory:", "") else None
        conn.execute("PRAGMA busy_timeout=5000")
        super().__init__(conn, _SQLITE_DDL)


class _PostgresBackend(_Backend):
    paramstyle_qmark = False

    def __init__(self, url: str):
        try:
            import psycopg  # type: ignore
        except ImportError as e:
            raise NotImplementedError(
                "MEMORY_DB_URL points at PostgreSQL but the 'psycopg' driver is not installed. "
                "Run `pip install 'psycopg[binary]'` (Cloud Run: add it to requirements.txt), "
                "or use the default sqlite:///data/memory.db."
            ) from e
        conn = psycopg.connect(url, autocommit=True)
        super().__init__(conn, _PG_DDL)


def _open_backend(url: str) -> _Backend:
    url = (url or DEFAULT_DB_URL).strip()
    if url.startswith("sqlite:///"):
        return _SqliteBackend(url[len("sqlite:///"):])
    if url.startswith("sqlite://"):
        return _SqliteBackend(url[len("sqlite://"):] or ":memory:")
    if url.startswith(("postgresql://", "postgres://")):
        return _PostgresBackend(url)
    if "://" not in url:  # bare path
        return _SqliteBackend(url)
    raise NotImplementedError(f"Unsupported MEMORY_DB_URL scheme: {url.split('://', 1)[0]!r} "
                              "(use sqlite:///path or postgresql://...)")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MemoryStore:
    """Long-term memories (facts / prefs / rules / episodes) + daily snapshots.

    ``embedder``: object with ``embed(texts) -> list[vec] | None`` and an ``available`` flag.
    ``"auto"`` (default) picks :func:`default_embedder`; ``None`` disables embeddings (keyword search).
    """

    def __init__(self, url: str | None = None, embedder: Any = "auto", snapshot_backend: Any = None):
        self.url = url or os.getenv("MEMORY_DB_URL") or DEFAULT_DB_URL
        self._db = _open_backend(self.url)
        self._embedder = default_embedder() if embedder == "auto" else embedder
        self._search_lock = threading.Lock()
        self._snapshots = _open_snapshot_backend(snapshot_backend, self._db, self.url)
        logger.info("Memory store: %s (embeddings: %s)", self.url,
                    "keyword only" if self._embedder is None else type(self._embedder).__name__)
        logger.info("Snapshot backend: %s (%s)", self.snapshot_backend_name,
                    getattr(self._snapshots, "label", "") or type(self._snapshots).__name__)

    # -- embeddings ------------------------------------------------------

    @property
    def embeddings_enabled(self) -> bool:
        return self._embedder is not None and getattr(self._embedder, "available", True)

    def warm(self) -> bool:
        """Run the embedder's probe now (call from a worker thread at start-up so the first
        request never pays for - or hangs on - it). Returns whether embeddings are usable."""
        probe = getattr(self._embedder, "probe", None)
        if callable(probe):
            try:
                return bool(probe())
            except Exception as e:  # noqa: BLE001
                logger.warning("memory store warm-up failed: %s", e)
                return False
        return self.embeddings_enabled

    def _embed(self, texts: Sequence[str]) -> Optional[list[list[float]]]:
        if self._embedder is None:
            return None
        try:
            vecs = self._embedder.embed(texts)
        except Exception as e:  # noqa: BLE001
            logger.warning("Embedder failed (%s); keyword search only for this call", e)
            return None
        if not vecs or len(vecs) != len(texts):
            return None
        return vecs

    # -- memories: write -------------------------------------------------

    def put(self, namespace: str, text: str, meta: dict | None = None, key: str | None = None,
            ttl_days: int | float | None = None) -> str:
        """Insert (or replace, when ``key`` is given and exists in the namespace). Returns the id."""
        if not namespace or ":" not in namespace:
            raise ValueError(f"namespace must look like 'kind:owner', got {namespace!r}")
        text = (text or "").strip()
        if not text:
            raise ValueError("memory text is empty")
        now = _utcnow()
        expires = _iso(now + timedelta(days=float(ttl_days))) if ttl_days else None
        vecs = self._embed([text])
        blob = _pack(vecs[0]) if vecs else None
        meta_json = json.dumps(meta or {}, default=str)

        existing = None
        if key is not None:
            rows = self._db.fetchall("SELECT id, created_at FROM memories WHERE namespace = ? AND key = ?",
                                     (namespace, key))
            existing = rows[0] if rows else None
        if existing:
            mem_id = existing[0]
            self._db.execute(
                "UPDATE memories SET text = ?, meta = ?, updated_at = ?, expires_at = ?, embedding = ? WHERE id = ?",
                (text, meta_json, _iso(now), expires, blob, mem_id))
        else:
            mem_id = uuid.uuid4().hex
            self._db.execute(
                "INSERT INTO memories (id, namespace, key, text, meta, created_at, updated_at, expires_at, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mem_id, namespace, key, text, meta_json, _iso(now), _iso(now), expires, blob))
        return mem_id

    def delete(self, mem_id: str) -> bool:
        """Delete by full id or unique prefix (>= 6 chars). True when a row went away."""
        if not mem_id:
            return False
        rows = self._db.fetchall("SELECT id FROM memories WHERE id = ?", (mem_id,))
        if not rows and len(mem_id) >= 6:
            rows = self._db.fetchall("SELECT id FROM memories WHERE id LIKE ?", (mem_id + "%",))
            if len(rows) != 1:
                return False
        if not rows:
            return False
        return self._db.rowcount_execute("DELETE FROM memories WHERE id = ?", (rows[0][0],)) > 0

    def delete_namespace(self, namespace: str) -> int:
        return self._db.rowcount_execute("DELETE FROM memories WHERE namespace = ?", (namespace,))

    def purge_expired(self, now: datetime | None = None) -> int:
        now = now or _utcnow()
        return self._db.rowcount_execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                                         (_iso(now),))

    # -- memories: read --------------------------------------------------

    def _row(self, r: tuple, with_embedding: bool = False):
        mem = Memory(id=r[0], namespace=r[1], key=r[2], text=r[3], meta=_loads(r[4]),
                     created_at=_parse_dt(r[5]), updated_at=_parse_dt(r[6]), expires_at=_parse_dt(r[7]))
        return (mem, _unpack(r[8])) if with_embedding else mem

    _COLS = "id, namespace, key, text, meta, created_at, updated_at, expires_at, embedding"

    def get(self, mem_id: str) -> Optional[Memory]:
        rows = self._db.fetchall(f"SELECT {self._COLS} FROM memories WHERE id = ?", (mem_id,))
        if not rows and len(mem_id or "") >= 6:
            rows = self._db.fetchall(f"SELECT {self._COLS} FROM memories WHERE id LIKE ?", (mem_id + "%",))
            if len(rows) != 1:
                return None
        return self._row(rows[0]) if rows else None

    def list(self, namespace: str, include_expired: bool = False) -> list[Memory]:
        rows = self._db.fetchall(
            f"SELECT {self._COLS} FROM memories WHERE namespace = ? ORDER BY updated_at DESC, created_at DESC",
            (namespace,))
        out = [self._row(r) for r in rows]
        if not include_expired:
            now = _utcnow()
            out = [m for m in out if m.expires_at is None or m.expires_at > now]
        return out

    def count(self, namespace: str) -> int:
        return len(self.list(namespace))

    def search(self, namespaces: Sequence[str] | str, query: str, k: int = 6,
               min_score: float = 0.05) -> list[Memory]:
        """Top-``k`` memories across ``namespaces`` for ``query`` (semantic when embeddings exist for
        both sides, else keyword). Rules namespaces (``rules:*``) are always returned in full,
        first, regardless of the query. Expired rows are skipped."""
        if isinstance(namespaces, str):
            namespaces = [namespaces]
        namespaces = [ns for ns in dict.fromkeys(namespaces) if ns]
        if not namespaces:
            return []
        marks = ",".join("?" for _ in namespaces)
        rows = self._db.fetchall(
            f"SELECT {self._COLS} FROM memories WHERE namespace IN ({marks}) ORDER BY updated_at DESC",
            tuple(namespaces))
        now = _utcnow()
        items = [self._row(r, with_embedding=True) for r in rows]
        items = [(m, e) for m, e in items if m.expires_at is None or m.expires_at > now]

        rules = [m for m, _ in items if m.kind == "rules"]
        for m in rules:
            m.score = 1.0
        cands = [(m, e) for m, e in items if m.kind != "rules"]
        if not cands or k <= 0:
            return rules

        query = (query or "").strip()
        if not query:
            ranked = cands[:k]
            return rules + [m for m, _ in ranked]

        kw = keyword_scores(query, [m.text for m, _ in cands])
        qvec = None
        if self.embeddings_enabled and any(e for _, e in cands):
            q = self._embed([query])
            qvec = q[0] if q else None

        scored = []
        for (m, emb), kscore in zip(cands, kw):
            if qvec is not None and emb is not None and len(emb) == len(qvec):
                sem = cosine(qvec, emb)
                # semantic dominates; keyword adds a little precision for exact terms
                score = 0.8 * max(sem, 0.0) + 0.2 * kscore
            else:
                score = kscore
            m.score = score
            scored.append(m)
        scored.sort(key=lambda m: (m.score, m.updated_at or datetime.min), reverse=True)
        keep = [m for m in scored if m.score >= min_score][:k]
        return rules + keep

    # -- snapshots (delegated to the selected backend; see SqliteSnapshotBackend) ---------

    @property
    def snapshot_backend(self):
        """The object that owns the snapshots table (SqliteSnapshotBackend or
        providers.snapshot_bq.BigQuerySnapshotBackend)."""
        return self._snapshots

    @property
    def snapshot_backend_name(self) -> str:
        return getattr(self._snapshots, "name", type(self._snapshots).__name__)

    def put_snapshot(self, snapshot_date, source: str, entity: str, metric: str, value,
                     value_json=None, captured_at: datetime | None = None) -> None:
        """Upsert one (date, source, entity, metric) cell. ``value`` float-able or None;
        ``value_json`` a JSON string or any JSON-serialisable object (dict/list) or None."""
        return self._snapshots.put_snapshot(snapshot_date, source, entity, metric, value,
                                            value_json=value_json, captured_at=captured_at)

    def put_snapshots(self, rows: Iterable[dict]) -> int:
        """Bulk upsert; each row: {snapshot_date, source, entity, metric, value, value_json?,
        captured_at?}. Returns the number of rows given."""
        return self._snapshots.put_snapshots(rows)

    def get_snapshot(self, snapshot_date, source: str, entity: str, metric: str) -> Optional[dict]:
        """One cell as {snapshot_date: date, value, value_json (parsed), captured_at} or None."""
        return self._snapshots.get_snapshot(snapshot_date, source, entity, metric)

    def get_snapshot_series(self, source: str, entity: str, metric: str, days: int | None = 30,
                            end: date | None = None) -> list[dict]:
        """Daily points ascending: [{snapshot_date: date, value: float|None, value_json: obj|None,
        captured_at: str}]. ``days`` counts back from ``end`` (default today UTC); None = all."""
        return self._snapshots.get_snapshot_series(source, entity, metric, days=days, end=end)

    def latest_snapshot_date(self, source: str, entity: str | None = None, metric: str | None = None) -> Optional[date]:
        return self._snapshots.latest_snapshot_date(source, entity=entity, metric=metric)

    def list_snapshot_metrics(self, source: str, entity: str | None = None) -> list[dict]:
        """[{entity, metric, first_date, last_date, n}] sorted by entity, metric."""
        return self._snapshots.list_snapshot_metrics(source, entity=entity)

    def list_snapshot_entities(self, source: str) -> list[str]:
        return self._snapshots.list_snapshot_entities(source)

    def list_snapshot_sources(self) -> list[str]:
        return self._snapshots.list_snapshot_sources()

    def count_snapshots(self, source: str | None = None, snapshot_date=None) -> int:
        return self._snapshots.count_snapshots(source=source, snapshot_date=snapshot_date)

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        try:
            self._snapshots.close()
        except Exception:  # noqa: BLE001
            pass
        self._db.close()


# ---------------------------------------------------------------------------
# Snapshot backends
# ---------------------------------------------------------------------------

def normalise_snapshot_value(value) -> Optional[float]:
    """float or None (None / NaN / inf / non-numeric all become None)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def normalise_snapshot_json(value_json) -> Optional[str]:
    """JSON text or None; non-string objects are serialised."""
    if value_json is None:
        return None
    return value_json if isinstance(value_json, str) else json.dumps(value_json, default=str)


def snapshot_row_dict(r: tuple) -> dict:
    """The row shape every backend returns for one snapshot cell."""
    return {"snapshot_date": _to_date(r[0]), "value": None if r[1] is None else float(r[1]),
            "value_json": _loads(r[2]) if r[2] else None, "captured_at": str(r[3]) if r[3] else None}


class SqliteSnapshotBackend:
    """Snapshots table inside the memory database (the original, default behaviour)."""

    name = "sqlite"

    def __init__(self, db: _Backend, label: str = ""):
        self._db = db
        self.label = label

    def put_snapshot(self, snapshot_date, source: str, entity: str, metric: str, value,
                     value_json=None, captured_at: datetime | None = None) -> None:
        d = _to_date(snapshot_date).isoformat()
        v = normalise_snapshot_value(value)
        vj = normalise_snapshot_json(value_json)
        if captured_at is not None and captured_at.tzinfo is not None:   # store naive UTC text
            captured_at = captured_at.astimezone(timezone.utc).replace(tzinfo=None)
        cap = _iso(captured_at or _utcnow())
        self._db.execute(
            "INSERT INTO snapshots (snapshot_date, source, entity, metric, value, value_json, captured_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (snapshot_date, source, entity, metric) DO UPDATE SET "
            "value = excluded.value, value_json = excluded.value_json, captured_at = excluded.captured_at",
            (d, source, str(entity), metric, v, vj, cap))

    def put_snapshots(self, rows: Iterable[dict]) -> int:
        n = 0
        for r in rows:
            self.put_snapshot(r["snapshot_date"], r["source"], r["entity"], r["metric"], r.get("value"),
                              r.get("value_json"), captured_at=r.get("captured_at"))
            n += 1
        return n

    def get_snapshot(self, snapshot_date, source: str, entity: str, metric: str) -> Optional[dict]:
        d = _to_date(snapshot_date).isoformat()
        rows = self._db.fetchall(
            "SELECT snapshot_date, value, value_json, captured_at FROM snapshots "
            "WHERE snapshot_date = ? AND source = ? AND entity = ? AND metric = ?",
            (d, source, str(entity), metric))
        return snapshot_row_dict(rows[0]) if rows else None

    def get_snapshot_series(self, source: str, entity: str, metric: str, days: int | None = 30,
                            end: date | None = None) -> list[dict]:
        params: list = [source, str(entity), metric]
        sql = ("SELECT snapshot_date, value, value_json, captured_at FROM snapshots "
               "WHERE source = ? AND entity = ? AND metric = ?")
        if days is not None:
            end_d = end or datetime.now(timezone.utc).date()
            start = end_d - timedelta(days=int(days) - 1)
            sql += " AND snapshot_date >= ? AND snapshot_date <= ?"
            params += [start.isoformat(), end_d.isoformat()]
        sql += " ORDER BY snapshot_date ASC"
        return [snapshot_row_dict(r) for r in self._db.fetchall(sql, params)]

    def latest_snapshot_date(self, source: str, entity: str | None = None, metric: str | None = None) -> Optional[date]:
        sql = "SELECT MAX(snapshot_date) FROM snapshots WHERE source = ?"
        params: list = [source]
        if entity is not None:
            sql += " AND entity = ?"
            params.append(str(entity))
        if metric is not None:
            sql += " AND metric = ?"
            params.append(metric)
        rows = self._db.fetchall(sql, params)
        return _to_date(rows[0][0]) if rows and rows[0][0] else None

    def list_snapshot_metrics(self, source: str, entity: str | None = None) -> list[dict]:
        sql = ("SELECT entity, metric, MIN(snapshot_date), MAX(snapshot_date), COUNT(*) FROM snapshots "
               "WHERE source = ?")
        params: list = [source]
        if entity is not None:
            sql += " AND entity = ?"
            params.append(str(entity))
        sql += " GROUP BY entity, metric ORDER BY entity, metric"
        return [{"entity": r[0], "metric": r[1], "first_date": _to_date(r[2]), "last_date": _to_date(r[3]),
                 "n": int(r[4])} for r in self._db.fetchall(sql, params)]

    def list_snapshot_entities(self, source: str) -> list[str]:
        return [r[0] for r in self._db.fetchall(
            "SELECT DISTINCT entity FROM snapshots WHERE source = ? ORDER BY entity", (source,))]

    def list_snapshot_sources(self) -> list[str]:
        return [r[0] for r in self._db.fetchall("SELECT DISTINCT source FROM snapshots ORDER BY source")]

    def count_snapshots(self, source: str | None = None, snapshot_date=None) -> int:
        sql, params = "SELECT COUNT(*) FROM snapshots WHERE 1 = 1", []
        if source is not None:
            sql += " AND source = ?"
            params.append(source)
        if snapshot_date is not None:
            sql += " AND snapshot_date = ?"
            params.append(_to_date(snapshot_date).isoformat())
        return int(self._db.fetchall(sql, params)[0][0])

    def close(self) -> None:  # the connection belongs to the MemoryStore
        return None


def snapshot_backend_from_env() -> str:
    raw = (os.getenv("SNAPSHOT_BACKEND") or DEFAULT_SNAPSHOT_BACKEND).strip().lower()
    if raw not in SNAPSHOT_BACKENDS:
        raise ValueError(f"Unsupported SNAPSHOT_BACKEND={raw!r}; choose from {', '.join(SNAPSHOT_BACKENDS)}")
    return raw


def _open_snapshot_backend(spec, db: _Backend, url: str):
    """``spec``: None (env SNAPSHOT_BACKEND), 'sqlite' | 'bigquery', or a ready backend object."""
    if spec is None:
        spec = snapshot_backend_from_env()
    if isinstance(spec, str):
        kind = spec.strip().lower()
        if kind == "sqlite":
            return SqliteSnapshotBackend(db, label=url)
        if kind == "bigquery":
            from providers.snapshot_bq import BigQuerySnapshotBackend

            return BigQuerySnapshotBackend()
        raise ValueError(f"Unsupported snapshot backend {spec!r}; choose from {', '.join(SNAPSHOT_BACKENDS)}")
    return spec


def _loads(s) -> Any:
    if s is None or s == "":
        return {}
    if not isinstance(s, (str, bytes)):
        return s
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return {"raw": s}


__all__ = [
    "Memory", "MemoryStore", "HashEmbedder", "VertexEmbedder", "default_embedder", "keyword_scores", "cosine",
    "call_with_deadline", "embed_timeout_s",
    "SHARED_FACTS", "prefs_namespace", "rules_namespace", "episodes_namespace", "episode_ttl_days",
    "DEFAULT_DB_URL", "DEFAULT_EPISODE_TTL_DAYS",
    "SqliteSnapshotBackend", "snapshot_backend_from_env", "snapshot_row_dict",
    "normalise_snapshot_value", "normalise_snapshot_json", "DEFAULT_SNAPSHOT_BACKEND", "SNAPSHOT_BACKENDS",
]
