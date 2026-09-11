"""providers/memory_store.py: memories (put/search/list/delete/ttl), namespace isolation, snapshots.
No network, no Vertex: HashEmbedder (deterministic) or keyword fallback, SQLite in tmp_path / memory."""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta

import pytest

from providers import factory
from providers.memory_store import (
    DEFAULT_DB_URL,
    HashEmbedder,
    Memory,
    MemoryStore,
    cosine,
    episode_ttl_days,
    episodes_namespace,
    keyword_scores,
    prefs_namespace,
    rules_namespace,
)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(f"sqlite:///{tmp_path / 'nested' / 'memory.db'}", embedder=HashEmbedder())
    yield s
    s.close()


@pytest.fixture
def kw_store():
    s = MemoryStore("sqlite:///:memory:", embedder=None)
    yield s
    s.close()


# ----------------------------------------------------------------------------
# memories
# ----------------------------------------------------------------------------

def test_put_list_get_delete(store, tmp_path):
    assert (tmp_path / "nested" / "memory.db").exists()
    mid = store.put("facts:shared", "Take rate is quoted in bps", meta={"by": "U1"})
    assert len(mid) == 32
    items = store.list("facts:shared")
    assert [m.text for m in items] == ["Take rate is quoted in bps"]
    assert items[0].meta == {"by": "U1"} and items[0].kind == "facts" and items[0].owner == "shared"
    assert isinstance(items[0].created_at, datetime)
    assert store.get(mid).text == "Take rate is quoted in bps"
    assert store.get(mid[:8]).id == mid  # prefix lookup
    assert store.delete(mid[:8]) is True
    assert store.delete(mid) is False
    assert store.list("facts:shared") == []


def test_put_validates(store):
    with pytest.raises(ValueError):
        store.put("nocolon", "x")
    with pytest.raises(ValueError):
        store.put("facts:shared", "   ")


def test_key_replaces_in_namespace(store):
    a = store.put("rules:C1", "answer in bullets", key="rule")
    b = store.put("rules:C1", "answer in tables", key="rule")
    assert a == b
    assert [m.text for m in store.list("rules:C1")] == ["answer in tables"]
    # same key in another namespace is a different row
    c = store.put("rules:C2", "other channel", key="rule")
    assert c != a


def test_ttl_purge(store):
    keep = store.put("episodes:U1", "recent chat", ttl_days=90)
    gone = store.put("episodes:U1", "old chat", ttl_days=-1)  # already expired
    assert [m.id for m in store.list("episodes:U1")] == [keep]  # expired hidden from list
    assert len(store.list("episodes:U1", include_expired=True)) == 2
    assert store.search(["episodes:U1"], "chat", k=5) and all(m.id != gone for m in store.search(["episodes:U1"], "chat"))
    assert store.purge_expired() == 1
    assert store.purge_expired() == 0
    assert len(store.list("episodes:U1", include_expired=True)) == 1
    # future purge time removes the rest
    assert store.purge_expired(now=datetime.now() + timedelta(days=100)) == 1


def test_search_semantic_and_namespace_isolation(store):
    store.put("facts:shared", "Take rate is quoted in bps on the desk")
    store.put("facts:shared", "Deribit lists options for btc eth sol hype")
    store.put(prefs_namespace("U1"), "prefers bps not percent")
    store.put(prefs_namespace("U2"), "U2 secret preference about percent and bps")
    store.put(episodes_namespace("U2"), "U2 asked about bps take rate yesterday")
    store.put(rules_namespace("C1"), "Keep answers to 3 bullets", key="rule")
    store.put(rules_namespace("C2"), "Other channel rule", key="rule")

    found = store.search(["facts:shared", prefs_namespace("U1"), episodes_namespace("U1"), rules_namespace("C1")],
                         "what units for the take rate, bps or percent?", k=6)
    texts = [m.text for m in found]
    assert texts[0] == "Keep answers to 3 bullets"          # channel rule first, unfiltered
    assert "Take rate is quoted in bps on the desk" in texts
    assert "prefers bps not percent" in texts
    assert not any("U2" in t for t in texts)                 # never another user's prefs/episodes
    assert "Other channel rule" not in texts
    assert all(m.score > 0 for m in found)
    assert found[0].score == 1.0


def test_search_keyword_fallback_and_min_score(kw_store):
    kw_store.put("facts:shared", "Take rate is quoted in bps")
    kw_store.put("facts:shared", "ETH options only on Deribit")
    kw_store.put("facts:shared", "Weekend volume is compared with prior weekends")
    found = kw_store.search("facts:shared", "take rate bps", k=6)
    assert [m.text for m in found] == ["Take rate is quoted in bps"]
    assert found[0].score == 1.0
    assert kw_store.search("facts:shared", "zzz nothing", k=6) == []
    # empty query -> most recent items
    assert len(kw_store.search("facts:shared", "", k=2)) == 2
    assert kw_store.search([], "x") == []


def test_search_mixed_embeddings(store):
    """Rows stored while embeddings were down (no vector) still rank by keyword."""
    store.put("facts:shared", "Amberdata funding is annualised percent")
    store._embedder = None  # embeddings go down
    store.put("facts:shared", "ray has no liquidation rows on Amberdata")
    store._embedder = HashEmbedder()
    found = store.search("facts:shared", "ray liquidation rows", k=2)
    assert found and found[0].text.startswith("ray has no")


def test_embedder_failure_degrades(kw_store):
    class Boom:
        available = True

        def embed(self, texts):
            raise RuntimeError("vertex down")

    kw_store._embedder = Boom()
    mid = kw_store.put("facts:shared", "stored without a vector")
    assert kw_store.get(mid) is not None
    assert [m.text for m in kw_store.search("facts:shared", "vector", k=3)] == ["stored without a vector"]


def test_delete_namespace(store):
    store.put(prefs_namespace("U1"), "a")
    store.put(prefs_namespace("U1"), "b")
    store.put(prefs_namespace("U2"), "c")
    assert store.delete_namespace(prefs_namespace("U1")) == 2
    assert store.list(prefs_namespace("U1")) == []
    assert [m.text for m in store.list(prefs_namespace("U2"))] == ["c"]


def test_keyword_scores_and_cosine():
    scores = keyword_scores("bps take rate", ["take rate in bps", "unrelated text", ""])
    assert scores[0] == 1.0 and scores[1] == 0.0 and scores[2] == 0.0
    assert keyword_scores("", ["a"]) == [0.0]
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([], [1]) == 0.0


def test_hash_embedder_deterministic():
    e = HashEmbedder(dim=16)
    a, b = e.embed(["prefers bps", "prefers bps"])
    assert a == b and len(a) == 16 and sum(a) == 2.0


def test_memory_as_dict_and_short_id():
    m = Memory(id="abcdef1234567890", namespace="prefs:U1", key=None, text="t", created_at=datetime(2026, 9, 1))
    d = m.as_dict()
    assert d["created_at"] == "2026-09-01 00:00:00" and d["updated_at"] is None
    assert m.short_id() == "abcdef12"


def test_thread_safety(store):
    errors = []

    def worker(i):
        try:
            for j in range(20):
                store.put("facts:shared", f"fact {i}-{j}")
                store.search("facts:shared", f"fact {i}", k=3)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.count("facts:shared") == 80


# ----------------------------------------------------------------------------
# snapshots
# ----------------------------------------------------------------------------

def test_snapshot_upsert_idempotent(store):
    store.put_snapshot(date(2026, 9, 1), "haruko", "combined", "delta_usd", 1.5e6)
    store.put_snapshot("2026-09-01", "haruko", "combined", "delta_usd", 1.6e6, value_json={"flag": "Normal"})
    store.put_snapshot("2026-09-01", "haruko", "combined", "delta_usd", 1.6e6, value_json={"flag": "Normal"})
    assert store.count_snapshots("haruko") == 1
    row = store.get_snapshot("2026-09-01", "haruko", "combined", "delta_usd")
    assert row["value"] == 1.6e6 and row["value_json"] == {"flag": "Normal"}
    assert row["snapshot_date"] == date(2026, 9, 1) and row["captured_at"]
    assert store.get_snapshot("2026-09-02", "haruko", "combined", "delta_usd") is None


def test_snapshot_series_and_listing(store):
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
    assert series[-1]["value"] == 0.0 and series[0]["value"] == pytest.approx(0.4)
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
    assert store.get_snapshot(today, "haruko", "20", "day_pnl") == {
        "snapshot_date": today, "value": None, "value_json": [1, 2],
        "captured_at": store.get_snapshot(today, "haruko", "20", "day_pnl")["captured_at"]}
    assert store.get_snapshot(today, "haruko", "20", "nan")["value"] is None
    assert store.count_snapshots(snapshot_date=today) == 5


def test_put_snapshots_bulk(store):
    n = store.put_snapshots([
        {"snapshot_date": "2026-09-01", "source": "sheet", "entity": "TOTAL", "metric": "mtd_pnl", "value": 1},
        {"snapshot_date": "2026-09-01", "source": "sheet", "entity": "TOTAL", "metric": "ytd_pnl", "value": 2,
         "value_json": {"asof": "2026-09-01"}},
    ])
    assert n == 2 and store.count_snapshots("sheet") == 2


# ----------------------------------------------------------------------------
# backend selection / factory
# ----------------------------------------------------------------------------

def test_url_parsing(tmp_path, monkeypatch):
    s = MemoryStore(str(tmp_path / "bare.db"), embedder=None)  # bare path
    assert (tmp_path / "bare.db").exists()
    s.close()
    monkeypatch.setenv("MEMORY_DB_URL", f"sqlite:///{tmp_path / 'env.db'}")
    s = MemoryStore(embedder=None)
    assert s.url.endswith("env.db") and (tmp_path / "env.db").exists()
    s.close()
    monkeypatch.delenv("MEMORY_DB_URL")
    assert DEFAULT_DB_URL == "sqlite:///data/memory.db"
    with pytest.raises(NotImplementedError):
        MemoryStore("mysql://x", embedder=None)


def test_postgres_requires_psycopg(monkeypatch):
    import builtins
    import importlib

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "psycopg":
            raise ImportError("no psycopg")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(NotImplementedError) as ei:
        MemoryStore("postgresql://u:p@h/db", embedder=None)
    assert "psycopg" in str(ei.value)


def test_default_embedder_off(monkeypatch):
    from providers.memory_store import default_embedder

    monkeypatch.setenv("MEMORY_EMBEDDINGS", "off")
    assert default_embedder() is None
    monkeypatch.setenv("MEMORY_EMBEDDINGS", "vertex")
    e = default_embedder()
    assert type(e).__name__ == "VertexEmbedder" and e.model == "text-embedding-005"
    assert e.location == "us-central1" and e.project == "anchorage-ai-development"


def test_vertex_embedder_degrades_without_network(monkeypatch, caplog):
    """Client construction failing (no ADC) -> keyword search per call; the breaker opens after 3."""
    import sys
    import types

    from providers.memory_store import VertexEmbedder

    fake = types.ModuleType("langchain_google_vertexai")

    class BadEmbeddings:
        def __init__(self, **kw):
            raise RuntimeError("no ADC")

    fake.VertexAIEmbeddings = BadEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_google_vertexai", fake)
    e = VertexEmbedder(timeout_s=5)
    with caplog.at_level("WARNING"):
        assert e.embed(["x"]) is None
        assert e.available is True  # one failure: this call fell back, embeddings still enabled
        assert e.embed(["y"]) is None
        assert e.embed(["z"]) is None
    assert e.available is False  # 3 consecutive failures -> disabled for the cool-down
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("falling back to keyword search" in m for m in msgs) == 1  # first failure only
    assert sum("disabled for 10 min" in m for m in msgs) == 1
    store = MemoryStore("sqlite:///:memory:", embedder=e)
    assert store.embeddings_enabled is False
    store.put("facts:shared", "keyword only fact")
    assert [m.text for m in store.search("facts:shared", "keyword fact")] == ["keyword only fact"]


class _FakeEmbeddings:
    """Stand-in for VertexAIEmbeddings: optional sleep before answering, call counting."""

    def __init__(self, sleep_s: float = 0.0, dim: int = 4):
        self.sleep_s, self.dim, self.calls = sleep_s, dim, 0

    def _vec(self, text):
        return [float(len(text)), 1.0, 0.0, 0.0][: self.dim]

    def embed_query(self, text):
        self.calls += 1
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return self._vec(text)

    def embed_documents(self, texts):
        self.calls += 1
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return [self._vec(t) for t in texts]


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_vertex_embedder_happy_path_runs_off_caller_thread():
    from providers.memory_store import VertexEmbedder

    fake = _FakeEmbeddings()
    seen = []

    def factory(model, project, location):
        seen.append((model, project, location, threading.current_thread().name))
        return fake

    e = VertexEmbedder(model="m", project="p", location="l", timeout_s=5, client_factory=factory)
    assert e.probe() is True
    assert e.embed(["ab", "abc"]) == [[2.0, 1.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0]]
    assert seen == [("m", "p", "l", "memory-embed")]  # built under the deadline on the worker thread
    assert fake.calls == 2 and e.available is True and e.failures == 0


def test_vertex_embedder_deadline_never_blocks_caller(caplog):
    """A hung embeddings call returns None within the deadline and the caller keeps going."""
    from providers.memory_store import VertexEmbedder

    fake = _FakeEmbeddings(sleep_s=3.0)
    e = VertexEmbedder(timeout_s=0.2, client_factory=lambda *a: fake)
    t0 = time.monotonic()
    with caplog.at_level("WARNING"):
        assert e.embed(["slow"]) is None
    assert time.monotonic() - t0 < 1.5
    assert e.failures == 1 and e.available is True
    assert any("timed out after 0.2s" in r.getMessage() and "keyword search for this call" in r.getMessage()
               for r in caplog.records)


def test_vertex_embedder_circuit_breaker_opens_and_recovers(caplog):
    from providers.memory_store import EMBED_COOLDOWN_S, VertexEmbedder

    clock = _Clock()
    good = _FakeEmbeddings()
    calls = {"n": 0}

    def flaky_documents(texts):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RuntimeError("503 backend")
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    good.embed_documents = flaky_documents
    e = VertexEmbedder(timeout_s=5, client_factory=lambda *a: good, clock=clock)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert e.embed(["x"]) is None
        assert e.available is False
        assert e.embed(["x"]) is None  # short-circuited: no call while the breaker is open
    assert calls["n"] == 3
    assert sum("disabled for 10 min" in r.getMessage() for r in caplog.records) == 1
    clock.t += EMBED_COOLDOWN_S + 1
    assert e.available is True
    assert e.embed(["x"]) == [[1.0, 0.0, 0.0, 0.0]]  # recovered after the cool-down
    assert e.failures == 0


def test_vertex_embedder_import_error_is_permanent(caplog):
    from providers.memory_store import VertexEmbedder

    def no_module(*a):
        raise ImportError("no langchain_google_vertexai")

    e = VertexEmbedder(timeout_s=5, client_factory=no_module)
    with caplog.at_level("WARNING"):
        assert e.probe() is False
    assert e.available is False and e.permanently_off is True
    assert e.embed(["x"]) is None


def test_embed_timeout_env(monkeypatch):
    from providers.memory_store import embed_timeout_s

    monkeypatch.delenv("MEMORY_EMBED_TIMEOUT_S", raising=False)
    assert embed_timeout_s() == 20.0
    monkeypatch.setenv("MEMORY_EMBED_TIMEOUT_S", "7.5")
    assert embed_timeout_s() == 7.5
    monkeypatch.setenv("MEMORY_EMBED_TIMEOUT_S", "-1")
    assert embed_timeout_s() == 20.0
    monkeypatch.setenv("MEMORY_EMBED_TIMEOUT_S", "later")
    assert embed_timeout_s() == 20.0


def test_store_warm_runs_probe_and_reports():
    from providers.memory_store import VertexEmbedder

    fake = _FakeEmbeddings()
    e = VertexEmbedder(timeout_s=5, client_factory=lambda *a: fake)
    store = MemoryStore("sqlite:///:memory:", embedder=e)
    assert store.warm() is True and fake.calls == 1
    assert MemoryStore("sqlite:///:memory:", embedder=None).warm() is False
    assert MemoryStore("sqlite:///:memory:", embedder=HashEmbedder()).warm() is True  # no probe: just the flag


def test_episode_ttl_env(monkeypatch):
    monkeypatch.delenv("MEMORY_EPISODE_TTL_DAYS", raising=False)
    assert episode_ttl_days() == 90
    monkeypatch.setenv("MEMORY_EPISODE_TTL_DAYS", "7")
    assert episode_ttl_days() == 7
    monkeypatch.setenv("MEMORY_EPISODE_TTL_DAYS", "x")
    assert episode_ttl_days() == 90


def test_factory_memoises_and_resets(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DB_URL", f"sqlite:///{tmp_path / 'f.db'}")
    monkeypatch.setenv("MEMORY_EMBEDDINGS", "off")
    factory.reset_memory_store()
    try:
        a = factory.get_memory_store()
        assert a is factory.get_memory_store()
        assert (tmp_path / "f.db").exists()
        custom = MemoryStore("sqlite:///:memory:", embedder=None)
        factory.set_memory_store(custom)
        assert factory.get_memory_store() is custom
    finally:
        factory.reset_memory_store()
    assert factory._memory_store is None
