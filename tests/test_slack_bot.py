"""slack_bot.py, notifiers.slack posting and run_signals --post-slack. No network, no Slack, no Vertex."""

from __future__ import annotations

import asyncio
import logging

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

import run_signals
import slack_bot
from notifiers import slack as notifier
from slack_bot import (
    ERROR_TEXT,
    PLACEHOLDER,
    TIMEOUT_TEXT,
    RecordingClient,
    answer,
    build_agent,
    chunk_text,
    clean,
    conversation_key,
    handle,
    should_handle_dm,
    to_mrkdwn,
)


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    import sys
    import types

    fake_llm = types.ModuleType("utils.llm")

    def _boom(*a, **k):
        raise AssertionError("utils.llm.get_llm() must not be called in tests")

    fake_llm.get_llm = _boom
    monkeypatch.setitem(sys.modules, "utils.llm", fake_llm)


@pytest.fixture(autouse=True)
def memory_store():
    """In-memory MemoryStore with the deterministic hash embedder; nothing is written to data/memory.db."""
    from providers import factory
    from providers.memory_store import HashEmbedder, MemoryStore

    store = MemoryStore("sqlite:///:memory:", embedder=HashEmbedder())
    factory.set_memory_store(store)
    yield store
    factory.reset_memory_store()


@pytest.fixture(autouse=True)
def no_real_episode_llm(monkeypatch):
    """Episode summaries never reach Vertex; tests that want summaries install a fake."""
    monkeypatch.setattr(slack_bot, "episode_llm",
                        lambda: pytest.fail("slack_bot.episode_llm() must be patched in tests"))
    slack_bot.BACKGROUND_TASKS.clear()


# ----------------------------------------------------------------------------
# clean()
# ----------------------------------------------------------------------------

def test_clean_strips_mentions():
    assert clean("<@U123ABC> hello there") == "hello there"
    assert clean("  <@UBOT>   what about <@U9> ?") == "what about  ?"
    assert clean("<@UBOT>") == ""
    assert clean(None) == ""
    assert clean("plain") == "plain"


# ----------------------------------------------------------------------------
# to_mrkdwn()
# ----------------------------------------------------------------------------

def test_to_mrkdwn_bold():
    assert to_mrkdwn("**BTC** is up, **ETH** flat") == "*BTC* is up, *ETH* flat"
    assert to_mrkdwn("already *slack* bold") == "already *slack* bold"


def test_to_mrkdwn_headers():
    out = to_mrkdwn("# Title\n## Sub **x** ##\n### deep\ntext # not header")
    assert out == "*Title*\n*Sub x*\n*deep*\ntext # not header"


def test_to_mrkdwn_bullets():
    out = to_mrkdwn("- a\n  - b\n* c\n+ d\n*bold* not a bullet\n-not a bullet")
    assert out == "• a\n  • b\n• c\n• d\n*bold* not a bullet\n-not a bullet"


def test_to_mrkdwn_links():
    assert to_mrkdwn("see [docs](https://x.y/z) now") == "see <https://x.y/z|docs> now"


def test_to_mrkdwn_pipe_table_becomes_code_block():
    md = ("Summary:\n"
          "| token | metric | **z** |\n"
          "|---|:---:|---:|\n"
          "| BTC | perp_oi | 3.10 |\n"
          "| ETH | `funding_rate` | -0.4 |\n"
          "after")
    out = to_mrkdwn(md)
    lines = out.splitlines()
    assert lines[0] == "Summary:"
    assert lines[1] == "```" and lines[-2] == "```" and lines[-1] == "after"
    body = lines[2:-2]
    assert len(body) == 3  # separator row dropped
    assert body[0].startswith("token  metric        z")
    assert body[1].startswith("BTC    perp_oi       3.10")
    assert "|" not in "\n".join(body) and "**" not in out and "`funding_rate`" not in out
    # columns aligned: every 'metric' cell starts at the same offset
    assert {l.index("perp_oi") if "perp_oi" in l else l.index("funding_rate") for l in body[1:]} == {7}


def test_to_mrkdwn_code_fence_untouched():
    md = "intro **b**\n```\n# not a header\n- not a bullet\n| a | b |\n**raw**\n```\n- tail"
    out = to_mrkdwn(md)
    assert out == "intro *b*\n```\n# not a header\n- not a bullet\n| a | b |\n**raw**\n```\n• tail"


def test_to_mrkdwn_empty():
    assert to_mrkdwn("") == ""
    assert to_mrkdwn(None) == ""


# ----------------------------------------------------------------------------
# chunk_text()
# ----------------------------------------------------------------------------

def test_chunk_text_short_is_single():
    assert chunk_text("hi", 100) == ["hi"]
    assert chunk_text("", 100) == [""]


def test_chunk_text_prefers_newlines_and_respects_limit():
    lines = [f"line {i:03d} " + "x" * 40 for i in range(120)]
    text = "\n".join(lines)
    chunks = chunk_text(text, 500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert all(c.startswith("line ") for c in chunks)  # cut at newlines
    assert "\n".join(chunks).splitlines() == lines  # nothing lost


def test_chunk_text_rebalances_code_fences():
    body = "\n".join(f"row {i}" for i in range(300))
    text = "head\n```\n" + body + "\n```\ntail"
    chunks = chunk_text(text, 400)
    assert all(len(c) <= 400 for c in chunks)
    for c in chunks:
        assert c.count("```") % 2 == 0, c  # every chunk balanced
    assert chunks[1].startswith("```\n")
    assert chunks[-1].rstrip().endswith("tail")


# ----------------------------------------------------------------------------
# handle() with fakes
# ----------------------------------------------------------------------------

class FakeAgent:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.thread_ids: list[str] = []
        self.inputs: list[str] = []

    async def ainvoke(self, inp, config):
        self.thread_ids.append(config["configurable"]["thread_id"])
        human = inp["messages"][0]
        self.inputs.append(human.content)
        return {"messages": [human, AIMessage(content=self.reply)]}


class SlowAgent(FakeAgent):
    async def ainvoke(self, inp, config):
        await asyncio.sleep(5)
        return await super().ainvoke(inp, config)


class BoomAgent(FakeAgent):
    async def ainvoke(self, inp, config):
        raise RuntimeError("vertex exploded")


def _mention(text="<@UBOT> anything odd in btc?", **extra):
    ev = {"type": "app_mention", "channel": "C1", "user": "U1", "ts": "100.1", "text": text}
    ev.update(extra)
    return ev


def test_handle_mention_placeholder_then_update():
    client, agent = RecordingClient(), FakeAgent("**BTC** perp OI z = 3.1 (OUTLIER)")
    asyncio.run(handle(_mention(), client, agent))

    assert client.posts == [{"channel": "C1", "text": PLACEHOLDER}]  # top-level, not threaded
    assert client.updates == [{"channel": "C1", "ts": client.posts and "1001.000000",
                               "text": "*BTC* perp OI z = 3.1 (OUTLIER)"}]
    assert client.reactions == [{"channel": "C1", "timestamp": "100.1", "name": "eyes"}]
    assert len(agent.thread_ids) == 1 and agent.thread_ids[0].startswith("C1:U")
    assert agent.inputs == ["<@U1>: anything odd in btc?"]


def test_handle_in_thread_keeps_thread_memory():
    client, agent = RecordingClient(), FakeAgent("second")
    asyncio.run(handle(_mention(thread_ts="90.5"), client, agent))
    assert client.posts[0]["thread_ts"] == "90.5"
    assert agent.thread_ids == ["C1:90.5"]


def test_handle_ignores_empty_text():
    client, agent = RecordingClient(), FakeAgent()
    asyncio.run(handle(_mention("<@UBOT>"), client, agent))
    assert client.posts == [] and client.updates == [] and agent.thread_ids == []


def test_handle_long_reply_is_chunked_into_thread():
    lines = [f"line {i:03d} " + "x" * 80 for i in range(200)]  # ~18k chars
    client, agent = RecordingClient(), FakeAgent("\n".join(lines))
    asyncio.run(handle(_mention(), client, agent))

    assert len(client.updates) == 1
    overflow = client.posts[1:]
    assert len(overflow) >= 4
    texts = client.final_texts()
    assert all(len(t) <= 3900 for t in texts)
    assert all(p["thread_ts"] == "1001.000000" for p in overflow)  # overflow threads under the bot's reply
    assert "\n".join(texts).splitlines() == lines


def test_handle_timeout_gives_warning_text(caplog):
    client = RecordingClient()
    with caplog.at_level(logging.WARNING, logger="slack_bot"):
        asyncio.run(handle(_mention(), client, SlowAgent(), timeout=0.05))
    assert client.updates[0]["text"] == TIMEOUT_TEXT
    assert any("timed out" in r.getMessage() for r in caplog.records)


def test_handle_exception_gives_friendly_text(caplog):
    client = RecordingClient()
    with caplog.at_level(logging.ERROR, logger="slack_bot"):
        asyncio.run(handle(_mention(), client, BoomAgent()))
    assert client.updates[0]["text"] == ERROR_TEXT
    assert any("agent failed" in r.getMessage() for r in caplog.records)
    assert "vertex exploded" not in client.updates[0]["text"]


def test_handle_reaction_failure_is_ignored():
    class NoReactions(RecordingClient):
        async def reactions_add(self, **kwargs):
            raise RuntimeError("missing_scope")

    client = NoReactions()
    asyncio.run(handle(_mention(), client, FakeAgent("fine")))
    assert client.updates[0]["text"] == "fine"


def test_handle_react_false_skips_reaction():
    client = RecordingClient()
    asyncio.run(handle(_mention(), client, FakeAgent("fine"), react=False))
    assert client.reactions == []


# ----------------------------------------------------------------------------
# DM filter + conversation keys
# ----------------------------------------------------------------------------

def test_should_handle_dm_filter():
    base = {"channel_type": "im", "user": "U1", "text": "hi", "channel": "D1", "ts": "1.0"}
    assert should_handle_dm(base)
    assert not should_handle_dm({**base, "channel_type": "channel"})
    assert not should_handle_dm({**base, "channel_type": "group"})
    assert not should_handle_dm({**base, "bot_id": "B1"})
    assert not should_handle_dm({**base, "subtype": "message_changed"})
    assert not should_handle_dm({**base, "subtype": "channel_join"})
    assert not should_handle_dm({k: v for k, v in base.items() if k != "user"})


@pytest.fixture(autouse=True)
def _isolated_sessions(tmp_path, monkeypatch):
    """Every test gets its own session file; nothing is written under data/."""
    store = slack_bot.SessionStore(path=str(tmp_path / "sessions.json"))
    monkeypatch.setattr(slack_bot, "_SESSIONS", store)
    return store


class _Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_sessions_top_level_mentions_share_one_conversation(tmp_path):
    clock = _Clock()
    store = slack_bot.SessionStore(path=str(tmp_path / "s.json"), idle_minutes=120, now=clock)
    k1, r1 = store.key_for({"channel": "C1", "user": "U1", "ts": "5.0"})
    clock.t += 600
    k2, r2 = store.key_for({"channel": "C1", "user": "U1", "ts": "6.0"})
    assert k1 == k2                      # same person, same channel, 10 min later -> same memory
    assert (r1, r2) == (None, None)      # replies go to the channel, not a thread
    # a follow-up inside either thread continues the same session
    assert store.key_for({"channel": "C1", "user": "U1", "ts": "7.0", "thread_ts": "5.0"}) == (k1, "5.0")
    assert store.key_for({"channel": "C1", "user": "U2", "ts": "8.0", "thread_ts": "6.0"}) == (k1, "6.0")


def test_sessions_idle_timeout_and_other_users(tmp_path):
    clock = _Clock()
    store = slack_bot.SessionStore(path=str(tmp_path / "s.json"), idle_minutes=120, now=clock)
    k1, _ = store.key_for({"channel": "C1", "user": "U1", "ts": "5.0"})
    k_other, _ = store.key_for({"channel": "C1", "user": "U2", "ts": "5.5"})
    assert k_other != k1                 # different person -> own conversation
    clock.t += 121 * 60
    k2, _ = store.key_for({"channel": "C1", "user": "U1", "ts": "6.0"})
    assert k2 != k1                      # silent for >120 min -> fresh conversation
    # unknown thread (e.g. after the file was lost) falls back to the thread key
    assert store.key_for({"channel": "C9", "user": "U1", "ts": "9.1", "thread_ts": "9.0"}) == ("C9:9.0", "9.0")


def test_sessions_dm_is_top_level_and_persisted(tmp_path):
    path = tmp_path / "s.json"
    clock = _Clock()
    store = slack_bot.SessionStore(path=str(path), now=clock)
    k1, reply = store.key_for({"channel": "D1", "user": "U1", "ts": "7.0", "channel_type": "im"})
    assert reply is None                 # DMs are answered top-level
    reloaded = slack_bot.SessionStore(path=str(path), now=clock)   # simulate a restart
    k2, _ = reloaded.key_for({"channel": "D1", "user": "U1", "ts": "8.0", "channel_type": "im"})
    assert k1 == k2


def test_sessions_reset(tmp_path):
    clock = _Clock()
    store = slack_bot.SessionStore(path=str(tmp_path / "s.json"), now=clock)
    k1, _ = store.key_for({"channel": "C1", "user": "U1", "ts": "5.0"})
    store.reset({"channel": "C1", "user": "U1"})
    clock.t += 1
    k2, _ = store.key_for({"channel": "C1", "user": "U1", "ts": "6.0"})
    assert k2 != k1


@pytest.mark.parametrize("text,expected", [
    ("reset", True), ("Reset!", True), ("new topic", True), ("start over.", True),
    ("reset the BTC view please", False), ("what is the ETH skew", False),
])
def test_is_reset(text, expected):
    assert slack_bot.is_reset(text) is expected


def test_handle_reset_command_does_not_call_agent(_isolated_sessions):
    client = RecordingClient()

    class BoomAgent:
        async def ainvoke(self, *a, **k):
            raise AssertionError("agent must not run on reset")

    ev = {"channel": "C1", "user": "U1", "ts": "5.0", "text": "<@UBOT> reset"}
    asyncio.run(handle(ev, client, BoomAgent(), react=False))
    assert client.posts and client.posts[0]["text"] == slack_bot.RESET_TEXT
    assert "thread_ts" not in client.posts[0]
    assert not client.updates


def test_conversation_key_wrapper_uses_store(_isolated_sessions):
    k, r = conversation_key({"channel": "C1", "user": "U1", "ts": "5.0"})
    assert k.startswith("C1:U1:") and r is None
    assert conversation_key({"channel": "C1", "user": "U1", "ts": "6.0", "thread_ts": "5.0"}) == (k, "5.0")


def test_handle_dm_top_level_replies_without_thread():
    ev = {"type": "message", "channel_type": "im", "channel": "D1", "user": "U1", "ts": "7.0", "text": "hello"}
    client, agent = RecordingClient(), FakeAgent("hey")
    asyncio.run(handle(ev, client, agent))
    assert "thread_ts" not in client.posts[0]
    assert len(agent.thread_ids) == 1 and agent.thread_ids[0].startswith("D1:U1:")
    assert client.updates[0]["text"] == "hey"


# ----------------------------------------------------------------------------
# Agent wiring: real graph, fake LLM, SQLite checkpointer
# ----------------------------------------------------------------------------

class ToolAwareFakeLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_build_agent_sqlite_memory_and_answer(tmp_path):
    db = tmp_path / "nested" / "bot.db"
    llm = ToolAwareFakeLLM(messages=iter([AIMessage(content="first"), AIMessage(content="second"),
                                          AIMessage(content=[{"type": "text", "text": "blocks"}])]))

    async def go():
        async with build_agent(db_path=str(db), llm=llm, tools=[], system_prompt="p") as agent:
            a = await answer(agent, "C1:1", "U1", "one")
            b = await answer(agent, "C1:1", "U1", "two")
            c = await answer(agent, "C1:2", "U1", "three")
            state = await agent.aget_state({"configurable": {"thread_id": "C1:1"}})
            return a, b, c, len(state.values["messages"])

    a, b, c, n = asyncio.run(go())
    assert (a, b, c) == ("first", "second", "blocks")
    assert n == 4  # two human + two ai turns persisted for thread C1:1
    assert db.exists()


def test_build_agent_default_db_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_DB", str(tmp_path / "env" / "b.db"))
    llm = ToolAwareFakeLLM(messages=iter([AIMessage(content="x")]))

    async def go():
        async with build_agent(llm=llm, tools=[], system_prompt="p") as agent:
            return await answer(agent, "t", "U", "q")

    assert asyncio.run(go()) == "x"
    assert (tmp_path / "env" / "b.db").exists()


def test_load_slack_prompt_appends_addendum():
    from datetime import date

    p = slack_bot.load_slack_prompt(today=date(2026, 9, 10))
    assert "Thursday 2026-09-10" in p
    assert "get_zscore_signals" in p  # base chat prompt
    assert "mrkdwn" in p and "current_time" in p  # addendum
    assert p.index("Global Markets") < p.index("Slack delivery rules")


def test_current_time_tool():
    out = slack_bot.current_time.invoke({"timezone": "UTC"})
    assert "UTC" in out and len(out.split()) >= 3
    assert slack_bot.current_time.invoke({"timezone": "Not/AZone"}).startswith("Unknown timezone")


# ----------------------------------------------------------------------------
# notifiers.post_via_bot
# ----------------------------------------------------------------------------

class FakeWebClient:
    def __init__(self, fail_at=None):
        self.calls: list[dict] = []
        self.fail_at = fail_at

    def chat_postMessage(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("channel_not_found")
        return {"ok": True, "ts": f"{len(self.calls)}.000"}


def test_post_via_bot_single_message():
    wc = FakeWebClient()
    assert notifier.post_via_bot("hello *there*", "C9", "xoxb-test", client=wc) is True
    assert wc.calls == [{"channel": "C9", "text": "hello *there*", "mrkdwn": True, "unfurl_links": False}]


def test_post_via_bot_chunks_thread_under_first():
    wc = FakeWebClient()
    text = "\n".join(f"row {i} " + "y" * 90 for i in range(150))  # ~14k chars
    assert notifier.post_via_bot(text, "C9", "xoxb-test", client=wc) is True
    assert len(wc.calls) >= 3
    assert "thread_ts" not in wc.calls[0]
    assert all(c["thread_ts"] == "1.000" for c in wc.calls[1:])
    assert all(len(c["text"]) <= notifier.CHUNK_CHARS for c in wc.calls)
    assert "\n".join(c["text"] for c in wc.calls) == text


def test_post_via_bot_into_existing_thread():
    wc = FakeWebClient()
    assert notifier.post_via_bot("reply", "C9", "xoxb-test", thread_ts="55.5", client=wc)
    assert wc.calls[0]["thread_ts"] == "55.5"


def test_post_via_bot_failure_returns_false():
    wc = FakeWebClient(fail_at=1)
    assert notifier.post_via_bot("x", "C9", "xoxb-test", client=wc) is False
    assert notifier.post_via_bot("x", "", "xoxb-test", client=FakeWebClient()) is False
    assert notifier.post_via_bot("x", "C9", None, client=FakeWebClient()) is False


def test_post_message_webhook_requires_env_only(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    import utils.secrets as secrets_mod

    monkeypatch.setattr(secrets_mod, "get_secret",
                        lambda *a, **k: pytest.fail("webhook must never consult Secret Manager"))
    called = []
    monkeypatch.setattr(notifier.requests, "post", lambda *a, **k: called.append(a) or pytest.fail("no post"))
    assert notifier.post_message("text") is False
    assert called == []


# ----------------------------------------------------------------------------
# run_signals --post-slack precedence
# ----------------------------------------------------------------------------

class Cfg:
    def __init__(self, channel=None, token=None, webhook=None):
        self.SLACK_CHANNEL_ID = channel
        self.SLACK_BOT_TOKEN = token
        self.SLACK_WEBHOOK_URL = webhook


@pytest.fixture
def slack_spies(monkeypatch):
    calls = {"bot": [], "webhook": []}
    monkeypatch.setattr(notifier, "post_via_bot",
                        lambda text, channel_id, token, **kw: calls["bot"].append((text, channel_id, token)) or True)
    monkeypatch.setattr(notifier, "post_message",
                        lambda text, webhook_url=None, **kw: calls["webhook"].append((text, webhook_url)) or True)
    return calls


def test_post_to_slack_prefers_bot(slack_spies, capsys):
    rc = run_signals.post_to_slack("## BTC\n**OI** spike", cfg=Cfg("C1", "xoxb", "https://hook"))
    assert rc == 0
    assert slack_spies["bot"] == [("*BTC*\n*OI* spike", "C1", "xoxb")]  # converted to mrkdwn
    assert slack_spies["webhook"] == []
    assert "via bot" in capsys.readouterr().out


def test_post_to_slack_channel_flag_overrides_config(slack_spies):
    run_signals.post_to_slack("x", channel_override="C_FLAG", cfg=Cfg("C_CFG", "xoxb"))
    assert slack_spies["bot"][0][1] == "C_FLAG"


def test_post_to_slack_falls_back_to_webhook(slack_spies):
    rc = run_signals.post_to_slack("x", cfg=Cfg(None, "xoxb", "https://hook"))
    assert rc == 0 and slack_spies["bot"] == []
    assert slack_spies["webhook"] == [("x", "https://hook")]


def test_post_to_slack_channel_without_token_uses_webhook(slack_spies, caplog):
    with caplog.at_level(logging.WARNING, logger="run_signals"):
        rc = run_signals.post_to_slack("x", cfg=Cfg("C1", None, "https://hook"))
    assert rc == 0 and slack_spies["webhook"] == [("x", "https://hook")]
    assert any("no bot token" in r.getMessage() for r in caplog.records)


def test_post_to_slack_skips_with_warning(slack_spies, caplog, capsys):
    with caplog.at_level(logging.WARNING, logger="run_signals"):
        rc = run_signals.post_to_slack("x", cfg=Cfg())
    assert rc == 0
    assert slack_spies == {"bot": [], "webhook": []}
    assert any(r.getMessage() == run_signals.SLACK_SKIP_WARNING for r in caplog.records)
    assert "trading_signals_slack_channel_id" in caplog.text
    assert "Slack post skipped" in capsys.readouterr().out


def test_post_to_slack_bot_failure_exit_code(monkeypatch):
    monkeypatch.setattr(notifier, "post_via_bot", lambda *a, **k: False)
    assert run_signals.post_to_slack("x", cfg=Cfg("C1", "xoxb")) == 2


def test_parse_args_slack_channel():
    args = run_signals.parse_args(["--post-slack", "--slack-channel", "C777"])
    assert args.post_slack and args.slack_channel == "C777"
    assert run_signals.parse_args([]).slack_channel is None


def test_slack_header_mentions_summary():
    h = run_signals._slack_header({"tokens_analyzed": 2, "tokens_with_outliers": ["btc"],
                                   "tokens_with_significant_moves": []})
    assert h.startswith("*Statistical Signals - ")
    assert "Tokens analyzed: 2" in h and "Outliers (|z| >= 2.5): BTC" in h and "Significant" in h


# ----------------------------------------------------------------------------
# Config: Slack secrets and env-only webhook
# ----------------------------------------------------------------------------

def test_config_slack_attributes_resolve_via_secrets(monkeypatch):
    import utils.secrets as secrets_mod
    from utils.config import Config

    for var in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_CHANNEL_ID", "SLACK_WEBHOOK_URL"):
        monkeypatch.delenv(var, raising=False)
    asked = []

    def fake_get_secret(name, project=None, env_var=None):
        asked.append((name, project, env_var))
        return None  # e.g. the channel-id secret has no versions yet

    monkeypatch.setattr(secrets_mod, "get_secret", fake_get_secret)
    cfg = Config()
    assert cfg.SLACK_CHANNEL_ID is None
    assert cfg.SLACK_BOT_TOKEN is None and cfg.SLACK_APP_TOKEN is None
    assert asked == [
        ("trading_signals_slack_channel_id", "anchorage-trading-solutions", "SLACK_CHANNEL_ID"),
        ("trading_signals_slack_bot_token", "anchorage-trading-solutions", "SLACK_BOT_TOKEN"),
        ("trading_signals_slack_app_token", "anchorage-trading-solutions", "SLACK_APP_TOKEN"),
    ]
    # the webhook is env-only: no Secret Manager call
    assert cfg.SLACK_WEBHOOK_URL is None
    assert len(asked) == 3


def test_config_slack_env_overrides(monkeypatch):
    import utils.secrets as secrets_mod
    from utils.config import Config

    monkeypatch.setattr(secrets_mod, "get_secret", lambda *a, **k: pytest.fail("env must win"))
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C42")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hook.example")
    cfg = Config()
    assert (cfg.SLACK_CHANNEL_ID, cfg.SLACK_BOT_TOKEN, cfg.SLACK_WEBHOOK_URL) == ("C42", "xoxb-env", "https://hook.example")


def test_followup_under_bots_reply_continues_session(_isolated_sessions):
    client, agent = RecordingClient(), FakeAgent("first")
    asyncio.run(handle(_mention(), client, agent, react=False))
    bot_ts = client.posts[0]["text"] == PLACEHOLDER and "1001.000000"
    key = agent.thread_ids[0]
    asyncio.run(handle(_mention("<@UBOT> and eth?", thread_ts=bot_ts, ts="101.0"), client, agent, react=False))
    assert agent.thread_ids == [key, key]           # same conversation
    assert client.posts[-1]["thread_ts"] == bot_ts   # answered inside that thread


def test_reply_in_thread_setting_restores_threading(monkeypatch, tmp_path):
    monkeypatch.setattr(slack_bot, "REPLY_IN_THREAD", True)
    store = slack_bot.SessionStore(path=str(tmp_path / "s.json"))
    monkeypatch.setattr(slack_bot, "_SESSIONS", store)
    client, agent = RecordingClient(), FakeAgent("threaded")
    asyncio.run(handle(_mention(), client, agent, react=False))
    assert client.posts[0]["thread_ts"] == "100.1"
    ev = {"channel": "C1", "user": "U1", "ts": "5.0", "text": "<@UBOT> reset"}
    asyncio.run(handle(ev, client, FakeAgent(), react=False))
    assert client.posts[-1]["thread_ts"] == "5.0"


# ----------------------------------------------------------------------------
# Long-term memory: recall + identity in answer(), episodic summaries
# ----------------------------------------------------------------------------

class RecordingFakeLLM(ToolAwareFakeLLM):
    def _generate(self, messages, *args, **kwargs):
        RECORDED.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


RECORDED: list = []


def test_answer_injects_scoped_memories_into_system_prompt(tmp_path, memory_store):
    from providers.memory_store import prefs_namespace, rules_namespace

    RECORDED.clear()
    memory_store.put("facts:shared", "Take rate is quoted in bps")
    memory_store.put(prefs_namespace("U1"), "prefers bps not percent")
    memory_store.put(prefs_namespace("U2"), "U2 SECRET PREF")
    memory_store.put(rules_namespace("C1"), "Three bullets max.", key="rule")
    llm = RecordingFakeLLM(messages=iter([AIMessage(content="12 bps")]))

    async def go():
        async with build_agent(db_path=str(tmp_path / "b.db"), llm=llm, tools=[], system_prompt="BASE") as agent:
            reply = await answer(agent, "C1:U1:1", "U1", "take rate in bps?", channel_id="C1")
            state = await agent.aget_state({"configurable": {"thread_id": "C1:U1:1"}})
            return reply, state.values["messages"]

    reply, messages = asyncio.run(go())
    assert reply == "12 bps"
    system = RECORDED[0][0].content
    assert system.startswith("BASE\n\n<memories>\nStanding instructions for this channel: Three bullets max.")
    assert "Take rate is quoted in bps" in system and "prefers bps not percent" in system
    assert "U2 SECRET PREF" not in system
    assert messages[0].content == "<@U1>: take rate in bps?"  # persisted message is clean


def test_answer_recall_failure_does_not_block_reply(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(slack_bot, "build_context", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")))
    llm = ToolAwareFakeLLM(messages=iter([AIMessage(content="fine")]))

    async def go():
        async with build_agent(db_path=str(tmp_path / "b.db"), llm=llm, tools=[], system_prompt="p") as agent:
            return await answer(agent, "t", "U1", "q", channel_id="C1")

    with caplog.at_level(logging.WARNING, logger="slack_bot"):
        assert asyncio.run(go()) == "fine"
    assert any("memory recall failed" in r.getMessage() for r in caplog.records)


def test_handle_sets_identity_for_tools_dm_and_channel(memory_store):
    from providers.memory_store import prefs_namespace, rules_namespace
    from tools import context as ctx
    from tools.memory_tools import remember, set_channel_rule

    class ToolAgent(FakeAgent):
        """Runs the memory tools with whatever identity the contextvars carry."""

        def __init__(self, call):
            super().__init__("ok")
            self.call = call
            self.seen = []

        async def ainvoke(self, inp, config):
            self.seen.append(ctx.snapshot())
            self.reply = self.call()
            return await super().ainvoke(inp, config)

    dm = {"type": "message", "channel_type": "im", "channel": "D1", "user": "U1", "ts": "7.0",
          "text": "remember I like bps"}
    agent = ToolAgent(lambda: remember.invoke({"text": "I like bps", "scope": "shared"}))
    asyncio.run(handle(dm, RecordingClient(), agent, react=False))
    assert agent.seen == [{"user_id": "U1", "channel_id": "D1", "is_dm": True}]
    assert [m.text for m in memory_store.list(prefs_namespace("U1"))] == ["I like bps"]  # DM -> personal
    assert memory_store.list("facts:shared") == []

    agent = ToolAgent(lambda: set_channel_rule.invoke({"text": "bullets only"}))
    asyncio.run(handle(_mention(), RecordingClient(), agent, react=False))
    assert agent.seen == [{"user_id": "U1", "channel_id": "C1", "is_dm": False}]
    assert [m.text for m in memory_store.list(rules_namespace("C1"))] == ["bullets only"]
    assert ctx.current_user_id.get() is None  # restored after the turn


class FakeEpisodeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=f"<@U1> discussed BTC funding (summary #{len(self.prompts)}).")


def _run_turns(tmp_path, monkeypatch, memory_store, n_turns, sessions, reset_after=False, ts0=100.0):
    """Drive n_turns channel mentions through handle() with a real agent + SQLite checkpointer."""
    ep = FakeEpisodeLLM()
    monkeypatch.setattr(slack_bot, "episode_llm", lambda: ep)
    llm = ToolAwareFakeLLM(messages=iter([AIMessage(content=f"reply {i}") for i in range(n_turns + 2)]))
    client = RecordingClient()

    async def go():
        async with build_agent(db_path=str(tmp_path / "bot.db"), llm=llm, tools=[], system_prompt="p") as agent:
            for i in range(n_turns):
                await handle(_mention(f"<@UBOT> question {i}", ts=f"{ts0 + i}"), client, agent, react=False,
                             sessions=sessions)
                await slack_bot.flush_background()
            if reset_after:
                await handle(_mention("<@UBOT> reset", ts="999.0"), client, agent, react=False, sessions=sessions)
                await slack_bot.flush_background()
            return agent

    asyncio.run(go())
    return ep, client


def test_episode_after_every_sixth_reply(tmp_path, monkeypatch, memory_store):
    from providers.memory_store import episodes_namespace

    sessions = slack_bot.SessionStore(path=str(tmp_path / "s.json"))
    ep, client = _run_turns(tmp_path, monkeypatch, memory_store, 7, sessions)
    assert len(ep.prompts) == 1                               # exactly one summary: after reply #6
    assert "USER: <@U1>: question 5" in ep.prompts[0] and "ASSISTANT: reply 5" in ep.prompts[0]
    eps = memory_store.list(episodes_namespace("U1"))
    assert len(eps) == 1 and eps[0].text.startswith("<@U1> discussed BTC funding")
    key = sessions.active["C1:U1"]["key"]
    assert eps[0].meta["session_key"] == key and eps[0].meta["channel"] == "C1" and eps[0].meta["replies"] == 6
    assert eps[0].expires_at is not None
    assert sessions.summarised_replies(key) == 6
    assert len(client.updates) == 7                           # every reply still delivered


def test_episode_on_reset_and_no_duplicate(tmp_path, monkeypatch, memory_store):
    from providers.memory_store import episodes_namespace

    sessions = slack_bot.SessionStore(path=str(tmp_path / "s.json"))
    ep, client = _run_turns(tmp_path, monkeypatch, memory_store, 2, sessions, reset_after=True)
    assert len(ep.prompts) == 1                               # reset closes the session -> one final summary
    assert client.posts[-1]["text"] == slack_bot.RESET_TEXT
    eps = memory_store.list(episodes_namespace("U1"))
    assert len(eps) == 1 and eps[0].meta["replies"] == 2
    assert sessions.active == {} and sessions.closed == []


def test_no_episode_when_nothing_new_since_last_summary(tmp_path, monkeypatch, memory_store):
    sessions = slack_bot.SessionStore(path=str(tmp_path / "s.json"))
    ep, _ = _run_turns(tmp_path, monkeypatch, memory_store, 6, sessions, reset_after=True)
    assert len(ep.prompts) == 1   # summarised at reply 6; reset right after adds nothing


def test_episode_on_idle_timeout_detected_on_next_message(tmp_path, monkeypatch, memory_store):
    from providers.memory_store import episodes_namespace

    clock = _Clock()
    sessions = slack_bot.SessionStore(path=str(tmp_path / "s.json"), idle_minutes=120, now=clock)
    ep = FakeEpisodeLLM()
    monkeypatch.setattr(slack_bot, "episode_llm", lambda: ep)
    llm = ToolAwareFakeLLM(messages=iter([AIMessage(content="a"), AIMessage(content="b")]))
    client = RecordingClient()

    async def go():
        async with build_agent(db_path=str(tmp_path / "bot.db"), llm=llm, tools=[], system_prompt="p") as agent:
            await handle(_mention("<@UBOT> first", ts="1.0"), client, agent, react=False, sessions=sessions)
            await slack_bot.flush_background()
            assert ep.prompts == []
            old_key = sessions.active["C1:U1"]["key"]
            clock.t += 121 * 60
            await handle(_mention("<@UBOT> hours later", ts="2.0"), client, agent, react=False, sessions=sessions)
            await slack_bot.flush_background()
            return old_key

    old_key = asyncio.run(go())
    assert len(ep.prompts) == 1 and "USER: <@U1>: first" in ep.prompts[0]
    eps = memory_store.list(episodes_namespace("U1"))
    assert len(eps) == 1 and eps[0].meta["session_key"] == old_key
    assert eps[0].meta["started"].endswith("UTC") and eps[0].meta["ended"].endswith("UTC")
    assert sessions.active["C1:U1"]["key"] != old_key


def test_expire_stale_at_startup(tmp_path, monkeypatch, memory_store):
    from providers.memory_store import episodes_namespace

    clock = _Clock()
    sessions = slack_bot.SessionStore(path=str(tmp_path / "s.json"), idle_minutes=120, now=clock)
    ep = FakeEpisodeLLM()
    monkeypatch.setattr(slack_bot, "episode_llm", lambda: ep)
    llm = ToolAwareFakeLLM(messages=iter([AIMessage(content="a")]))

    async def go():
        async with build_agent(db_path=str(tmp_path / "bot.db"), llm=llm, tools=[], system_prompt="p") as agent:
            await handle(_mention("<@UBOT> first", ts="1.0"), RecordingClient(), agent, react=False, sessions=sessions)
            await slack_bot.flush_background()
            # simulate a restart hours later: the session file is reloaded and stale sessions closed
            clock.t += 3 * 3600
            reloaded = slack_bot.SessionStore(path=str(tmp_path / "s.json"), idle_minutes=120, now=clock)
            stale = reloaded.expire_stale()
            assert [i["user"] for i in stale] == ["U1"] and reloaded.active == {}
            for info in stale:
                slack_bot.spawn_episode(agent, reloaded, info, final=True)
            await slack_bot.flush_background()
            return reloaded

    reloaded = asyncio.run(go())
    assert len(ep.prompts) == 1
    eps = memory_store.list(episodes_namespace("U1"))
    assert len(eps) == 1
    assert reloaded.summarised_replies(eps[0].meta["session_key"]) == 1
    # bookkeeping survives another reload
    again = slack_bot.SessionStore(path=str(tmp_path / "s.json"), idle_minutes=120, now=clock)
    assert again.episodes == reloaded.episodes
    assert again.expire_stale() == []


def test_episode_failure_is_logged_not_raised(tmp_path, monkeypatch, memory_store, caplog):
    class Boom:
        def invoke(self, p):
            raise RuntimeError("vertex down")

    monkeypatch.setattr(slack_bot, "episode_llm", lambda: Boom())
    sessions = slack_bot.SessionStore(path=str(tmp_path / "s.json"))
    llm = ToolAwareFakeLLM(messages=iter([AIMessage(content="a")]))
    client = RecordingClient()

    async def go():
        async with build_agent(db_path=str(tmp_path / "bot.db"), llm=llm, tools=[], system_prompt="p") as agent:
            await handle(_mention("<@UBOT> q", ts="1.0"), client, agent, react=False, sessions=sessions)
            await handle(_mention("<@UBOT> reset", ts="2.0"), client, agent, react=False, sessions=sessions)
            with caplog.at_level(logging.WARNING):
                await slack_bot.flush_background()

    asyncio.run(go())
    assert client.updates[0]["text"] == "a" and client.posts[-1]["text"] == slack_bot.RESET_TEXT
    assert any("episode summary failed" in r.getMessage() for r in caplog.records)


def test_fake_agent_without_state_skips_episodes(_isolated_sessions):
    """Agents without aget_state (the FakeAgent above) never trigger summaries or errors."""
    client, agent = RecordingClient(), FakeAgent("ok")
    for i in range(7):
        asyncio.run(handle(_mention(ts=f"{100 + i}"), client, agent, react=False))
    assert len(client.updates) == 7 and slack_bot.BACKGROUND_TASKS == set()


def test_session_store_persists_episode_bookkeeping(tmp_path):
    clock = _Clock()
    store = slack_bot.SessionStore(path=str(tmp_path / "s.json"), now=clock)
    k, _ = store.key_for({"channel": "C1", "user": "U1", "ts": "5.0"})
    store.mark_summarised(k, 6)
    assert store.summarised_replies(k) == 6 and store.summarised_replies("other") == 0
    store.reset({"channel": "C1", "user": "U1"})
    closed = store.drain_closed()
    assert len(closed) == 1 and closed[0]["key"] == k and closed[0]["user"] == "U1" and closed[0]["channel"] == "C1"
    assert closed[0]["started"] == float(int(clock.t)) and closed[0]["last"] == clock.t
    assert store.drain_closed() == []
    reloaded = slack_bot.SessionStore(path=str(tmp_path / "s.json"), now=clock)
    assert reloaded.summarised_replies(k) == 6


def test_slack_prompt_mentions_memory_rules():
    from datetime import date

    p = slack_bot.load_slack_prompt(today=date(2026, 9, 10))
    assert "Memory in Slack" in p and "set_channel_rule" in p and "remember" in p
