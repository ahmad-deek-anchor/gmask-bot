"""tools/memory_tools.py + tools/context.py: remember / recall / forget / what_do_you_remember, channel
rules, the auto-recall block, the storage guard and the episodic summariser. No network, no Vertex."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from providers import factory
from providers.memory_store import HashEmbedder, MemoryStore, episodes_namespace, prefs_namespace, rules_namespace
from tools import context as ctx
from tools import memory_tools as mt
from tools.context import request_context
from tools.memory_tools import (
    MEMORY_CONTEXT_MAX_CHARS,
    MEMORY_TOOL_NAMES,
    build_context,
    clear_channel_rule,
    count_replies,
    forget,
    format_memories,
    get_memory_tools,
    guard_text,
    recall,
    record_episode,
    remember,
    set_channel_rule,
    summarise_session,
    transcript,
    what_do_you_remember,
)


@pytest.fixture
def store():
    s = MemoryStore("sqlite:///:memory:", embedder=HashEmbedder())
    factory.set_memory_store(s)
    yield s
    factory.reset_memory_store()


@pytest.fixture
def as_user():
    """as_user("U1", "C1") -> context manager setting the identity contextvars."""
    def _cm(user="U1", channel="C1", is_dm=False):
        return request_context(user, channel, is_dm=is_dm)
    return _cm


# ----------------------------------------------------------------------------
# context
# ----------------------------------------------------------------------------

def test_request_context_sets_and_restores():
    assert ctx.snapshot() == {"user_id": None, "channel_id": None, "is_dm": False}
    with request_context("U1", "C1", is_dm=True, memory_context="<memories>x</memories>"):
        assert ctx.snapshot() == {"user_id": "U1", "channel_id": "C1", "is_dm": True}
        assert ctx.current_memory_context.get() == "<memories>x</memories>"
        with request_context("U2", "C2"):
            assert ctx.current_user_id.get() == "U2"
        assert ctx.current_user_id.get() == "U1"
    assert ctx.snapshot() == {"user_id": None, "channel_id": None, "is_dm": False}
    assert ctx.current_memory_context.get() == ""


def test_tools_registered():
    tools = get_memory_tools()
    assert [t.name for t in tools] == MEMORY_TOOL_NAMES
    assert all(t.description for t in tools)


# ----------------------------------------------------------------------------
# remember / recall / forget / what_do_you_remember
# ----------------------------------------------------------------------------

def test_remember_shared_and_me(store, as_user):
    with as_user("U1", "C1"):
        out = remember.invoke({"text": "Take rate is quoted in bps", "scope": "shared"})
        assert out.startswith("Stored in shared desk facts (id ") and "Take rate is quoted in bps" in out
        out = remember.invoke({"text": "I prefer bps not percent", "scope": "me"})
        assert out.startswith("Stored in your preferences")
    assert [m.text for m in store.list("facts:shared")] == ["Take rate is quoted in bps"]
    prefs = store.list(prefs_namespace("U1"))
    assert [m.text for m in prefs] == ["I prefer bps not percent"]
    assert prefs[0].meta == {"by": "U1", "channel": "C1", "scope": "me"}


def test_remember_dm_defaults_to_me(store, as_user):
    with as_user("U1", "D1", is_dm=True):
        remember.invoke({"text": "I like tables", "scope": "shared"})
        remember.invoke({"text": "shared fact for everyone: sky is sky_sky on Coin Metrics", "scope": "shared"})
    assert [m.text for m in store.list(prefs_namespace("U1"))] == ["I like tables"]
    assert len(store.list("facts:shared")) == 1


def test_remember_edge_cases(store, as_user):
    with as_user("U1", "C1"):
        assert remember.invoke({"text": "  "}) == "Nothing to remember: empty text."
        assert remember.invoke({"text": "x", "scope": "bogus"}).startswith("Unknown scope")
        assert remember.invoke({"text": "token xoxb-1234567890-abcdefghijk", "scope": "shared"}).startswith("Not stored")
        out = remember.invoke({"text": "the desk position is $5,000,000 long BTC", "scope": "shared"})
        assert out.startswith("Stored") and "Caution" in out and "positions/PnL" in out and "$5,000,000" in out
        long = remember.invoke({"text": " ".join(["prefer bps"] * 80), "scope": "me"})
        assert long.endswith('..."') and len(long) < 700
    with request_context(None, "C1"):
        assert remember.invoke({"text": "pref", "scope": "me"}).startswith("Memory is unavailable")
    assert store.count("facts:shared") == 1


def test_recall_scoped_to_user_and_channel(store, as_user):
    store.put("facts:shared", "Take rate is quoted in bps")
    store.put(prefs_namespace("U1"), "prefers bps not percent")
    store.put(prefs_namespace("U2"), "U2 wants percent")
    store.put(episodes_namespace("U2"), "U2 discussed bps yesterday")
    store.put(rules_namespace("C1"), "three bullets max", key="rule")
    with as_user("U1", "C1"):
        out = recall.invoke({"query": "take rate bps", "k": 5})
    assert "Take rate is quoted in bps" in out and "prefers bps not percent" in out
    assert "channel rule" in out and "three bullets max" in out
    assert "U2" not in out
    assert "| id " in out and "score" in out
    # a channel rule is always returned, so the 'no match' text appears only in a channel without one
    with as_user("U1", "C9"):
        assert recall.invoke({"query": "zzzz qqqq"}) == "No matching memories."


def test_recall_with_rule_only(store, as_user):
    store.put(rules_namespace("C1"), "three bullets max", key="rule")
    with as_user("U1", "C1"):
        out = recall.invoke({"query": "zzzz qqqq"})
    assert "1 memory item(s)" in out and "three bullets max" in out


def test_forget_by_id_text_and_everything(store, as_user):
    a = store.put(prefs_namespace("U1"), "prefers bps not percent")
    store.put(prefs_namespace("U1"), "follows btc and eth")
    store.put(episodes_namespace("U1"), "U1 asked about funding")
    other = store.put(prefs_namespace("U2"), "U2 pref")
    shared = store.put("facts:shared", "Take rate in bps")

    with as_user("U1", "C1"):
        assert forget.invoke({"query_or_id": ""}).startswith("Nothing to forget")
        assert forget.invoke({"query_or_id": "x", "scope": "bogus"}).startswith("Unknown scope")
        out = forget.invoke({"query_or_id": a[:8]})
        assert out.startswith("Deleted your preference") and "prefers bps" in out
        # another user's memory by id: refused
        assert forget.invoke({"query_or_id": other}) == "That memory is not yours to delete from here."
        # shared fact needs scope=shared
        assert forget.invoke({"query_or_id": shared}) == "That memory is not yours to delete from here."
        assert forget.invoke({"query_or_id": "Take rate", "scope": "shared"}).startswith("Deleted shared fact")
        assert forget.invoke({"query_or_id": "everything", "scope": "shared"}).startswith("Refusing to wipe")
        out = forget.invoke({"query_or_id": "everything about me"})
        assert out.startswith("Deleted everything I had about you: 2 item(s)")
        assert forget.invoke({"query_or_id": "anything left"}).startswith("No memory matching")
    assert store.list(prefs_namespace("U1")) == [] and store.list(episodes_namespace("U1")) == []
    assert [m.text for m in store.list(prefs_namespace("U2"))] == ["U2 pref"]
    with request_context(None, "C1"):
        assert forget.invoke({"query_or_id": "x"}).startswith("Memory is unavailable")


def test_forget_by_semantic_match(store, as_user):
    store.put(prefs_namespace("U1"), "quote funding in basis points please")
    store.put(prefs_namespace("U1"), "follows solana ecosystem tokens")
    with as_user("U1", "C1"):
        out = forget.invoke({"query_or_id": "funding basis points"})
    assert out.startswith("Deleted your preference") and "basis points" in out
    assert [m.text for m in store.list(prefs_namespace("U1"))] == ["follows solana ecosystem tokens"]


def test_what_do_you_remember(store, as_user):
    store.put(prefs_namespace("U1"), "prefers bps")
    store.put(episodes_namespace("U1"), "U1 asked about BTC OI on 2026-09-10")
    store.put(prefs_namespace("U2"), "U2 pref")
    for i in range(12):
        store.put("facts:shared", f"shared fact {i}")
    store.put(rules_namespace("C1"), "three bullets", key="rule")
    with as_user("U1", "C1"):
        out = what_do_you_remember.invoke({})
    assert "Your preferences (1):" in out and "prefers bps" in out
    assert "Your recent conversations remembered (1, expire after 90 days):" in out
    assert "Shared desk facts: 12" in out and "... 2 more" in out
    assert "Standing instructions for this channel: three bullets" in out
    assert "U2" not in out
    assert "- id " in out and "2026-" in out
    with request_context(None, None):
        out = what_do_you_remember.invoke({})
    assert out.startswith("No user identity") and "Standing instructions" not in out


# ----------------------------------------------------------------------------
# channel rules
# ----------------------------------------------------------------------------

def test_channel_rule_set_replace_clear(store, as_user):
    with as_user("U1", "C1"):
        out = set_channel_rule.invoke({"text": "always quote funding in bps"})
        assert out == 'Standing instruction for this channel set: "always quote funding in bps"'
        out = set_channel_rule.invoke({"text": "three bullets max"})
        assert "replaced: \"always quote funding in bps\"" in out
        assert [m.text for m in store.list(rules_namespace("C1"))] == ["three bullets max"]
        assert set_channel_rule.invoke({"text": ""}) == "Empty rule; nothing set."
        assert set_channel_rule.invoke({"text": "api_key=abcdef123456"}).startswith("Not stored")
        assert clear_channel_rule.invoke({}) == "Channel rule cleared."
        assert clear_channel_rule.invoke({}) == "This channel had no standing instruction."
    with as_user("U1", "D1", is_dm=True):
        assert set_channel_rule.invoke({"text": "x"}).startswith("Channel rules can only be set in a channel")
        assert clear_channel_rule.invoke({}).startswith("No channel rule to clear")
    with request_context("U1", None):
        assert set_channel_rule.invoke({"text": "x"}) == "No channel on this request; cannot set a rule."


def test_channel_rule_injected_verbatim(store):
    store.put(rules_namespace("C1"), "Answer in exactly three bullets.", key="rule")
    block = build_context("U1", "C1", "anything at all", store=store)
    assert block.startswith("<memories>\nStanding instructions for this channel: Answer in exactly three bullets.")
    assert block.endswith("</memories>")
    assert build_context("U1", "C2", "anything", store=store) == ""


# ----------------------------------------------------------------------------
# auto-recall block
# ----------------------------------------------------------------------------

def test_build_context_scopes_and_format(store):
    store.put("facts:shared", "Take rate is quoted in bps")
    store.put(prefs_namespace("U1"), "prefers bps not percent")
    store.put(prefs_namespace("U2"), "U2 secret")
    store.put(episodes_namespace("U1"), "U1 asked about take rate on 2026-09-10")
    store.put(episodes_namespace("U2"), "U2 episode")
    block = build_context("U1", "C1", "take rate in bps?", store=store)
    assert block.startswith("<memories>\nRelevant memories (may be stale; verify numbers with tools):")
    assert "[shared fact, " in block and "[your preference, " in block and "[past conversation, " in block
    assert "U2" not in block
    assert build_context("U1", "C1", "nothing relevant zzz", store=store) == ""


def test_build_context_size_cap(store):
    for i in range(30):
        store.put("facts:shared", f"bps fact number {i}: " + "x" * 120)
    block = build_context("U1", "C1", "bps fact", store=store, k=30)
    assert len(block) <= MEMORY_CONTEXT_MAX_CHARS
    assert block.endswith("- ...\n</memories>")
    small = format_memories(store.list("facts:shared")[:1], max_chars=80)
    assert len(small) <= 80 + len("\n- ...")


def test_build_context_never_raises(monkeypatch):
    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("db locked")

    assert build_context("U1", "C1", "x", store=Broken()) == ""
    assert format_memories([]) == ""


# ----------------------------------------------------------------------------
# guard
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "slack token xoxb-123456789012-abcdefghij",
    "key sk-abcdefghijklmnopqrstuvwxyz",
    "AKIAABCDEFGHIJKLMNOP is the aws key",
    "AIzaSyA-abcdefghijklmnopqrstuvwxyz0123456",
    "-----BEGIN PRIVATE KEY-----",
    "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123",
    "password = hunter22x",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
    "a" * 40,
])
def test_guard_refuses_keys(text):
    ok, warning = guard_text(text)
    assert ok is False and "credential" in warning


@pytest.mark.parametrize("text,expect", [
    ("Take rate is quoted in bps", None),
    ("the desk PnL is up", "positions/PnL"),
    ("our BTC position is large", "positions/PnL"),
    ("client paid $2.5m for the hedge", "large dollar amount ($2,500,000)"),
    ("threshold is $500k", None),
    ("USD 3,000,000 notional", "large dollar amount"),
])
def test_guard_warns(text, expect):
    ok, warning = guard_text(text)
    assert ok is True
    if expect is None:
        assert warning is None
    else:
        assert expect in warning


# ----------------------------------------------------------------------------
# episodic summariser
# ----------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, reply="<@U1> asked about BTC funding; z was 2.6 as of 2026-09-10."):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.reply)


def _messages():
    return [
        HumanMessage(content="<@U1>: anything odd in btc funding?"),
        AIMessage(content="", tool_calls=[{"name": "get_zscore_signals", "args": {"tokens": ["btc"]}, "id": "c1"}]),
        ToolMessage(content="| BTC | funding_rate | 60% | 2.60 | OUTLIER |", tool_call_id="c1"),
        AIMessage(content=[{"type": "text", "text": "BTC funding z = 2.60 (OUTLIER) as of 2026-09-10."}]),
        HumanMessage(content="<@U1>: thanks"),
        AIMessage(content="Any time."),
    ]


def test_transcript_and_count_replies():
    t = transcript(_messages())
    assert t.splitlines() == [
        "USER: <@U1>: anything odd in btc funding?",
        "ASSISTANT (called tools: get_zscore_signals)",
        "ASSISTANT: BTC funding z = 2.60 (OUTLIER) as of 2026-09-10.",
        "USER: <@U1>: thanks",
        "ASSISTANT: Any time.",
    ]
    assert "| BTC |" not in t  # tool output omitted
    assert count_replies(_messages()) == 2
    assert transcript([HumanMessage(content="x" * 100)] * 5, max_chars=150).startswith("...")


def test_summarise_session_and_record_episode(store, monkeypatch):
    llm = FakeLLM()
    summary = summarise_session(_messages(), llm)
    assert summary.startswith("<@U1> asked about BTC funding")
    assert "Transcript:" in llm.prompts[0] and "USER: <@U1>: anything odd" in llm.prompts[0]
    assert summarise_session([AIMessage(content="hi")], llm) == ""  # no user turn -> nothing

    monkeypatch.setenv("MEMORY_EPISODE_TTL_DAYS", "45")
    mid = record_episode(store, "U1", "C1", "C1:U1:1000", _messages(), llm=llm, started=1_000.0, ended="2026-09-11 10:00")
    eps = store.list(episodes_namespace("U1"))
    assert len(eps) == 1 and eps[0].id == mid and eps[0].text == summary
    assert eps[0].meta["channel"] == "C1" and eps[0].meta["session_key"] == "C1:U1:1000"
    assert eps[0].meta["started"] == "1970-01-01 00:16 UTC" and eps[0].meta["ended"] == "2026-09-11 10:00"
    assert eps[0].meta["replies"] == 2
    assert (eps[0].expires_at - eps[0].created_at).days in (44, 45)
    # same session again -> replaces (key = session_key), no duplicate
    record_episode(store, "U1", "C1", "C1:U1:1000", _messages(), llm=FakeLLM("updated"))
    assert [m.text for m in store.list(episodes_namespace("U1"))] == ["updated"]


def test_record_episode_never_raises(store, caplog):
    class Boom:
        def invoke(self, p):
            raise RuntimeError("vertex down")

    with caplog.at_level("WARNING"):
        assert record_episode(store, "U1", "C1", "k", _messages(), llm=Boom()) is None
    assert any("episode summary failed" in r.getMessage() for r in caplog.records)
    assert record_episode(store, None, "C1", "k", _messages(), llm=FakeLLM()) is None
    assert record_episode(store, "U1", "C1", "k", [], llm=FakeLLM()) is None
    assert store.list(episodes_namespace("U1")) == []


def test_episode_llm_uses_small_max_tokens(monkeypatch):
    import sys
    import types

    calls = []
    fake = types.ModuleType("utils.llm")
    fake.get_llm = lambda **kw: calls.append(kw) or "llm"
    monkeypatch.setitem(sys.modules, "utils.llm", fake)
    assert mt.episode_llm() == "llm"
    assert calls == [{"max_tokens": 300}]
