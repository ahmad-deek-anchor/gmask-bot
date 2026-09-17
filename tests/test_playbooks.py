"""playbooks.py: parsing, matching, role gating, the Slack `playbooks` command and prompt injection. No network."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

import chat
import playbooks as pb
import slack_bot
from access.policy import AccessControl, Policy, reset_access_control, set_access_control
from access.store import MemoryAccessStore
from slack_bot import RecordingClient, handle
from tests.test_chat import RecordingFakeLLM, ToolAwareFakeLLM

ADMIN = "U0BP6EUTHA9"

SAMPLE = """---
name: test_note
description: A test playbook
triggers: ["test note", "note du jour"]
min_role: desk
---
Call `get_live_price("btc")` and reply with exactly one line.
"""


@pytest.fixture(autouse=True)
def fresh_playbooks(tmp_path, monkeypatch):
    (tmp_path / "test_note.md").write_text(SAMPLE)
    (tmp_path / "open.md").write_text("---\nname: open\ndescription: no role needed\ntriggers: [\"open sesame\"]\n---\nBody of open.\n")
    monkeypatch.setenv("PLAYBOOKS_DIR", str(tmp_path))
    pb.reset_playbooks()
    yield
    pb.reset_playbooks()


def test_parse_and_match():
    p = pb.parse_playbook(SAMPLE, path="/x/test_note.md")
    assert p.name == "test_note" and p.min_role == "desk" and p.triggers == ["test note", "note du jour"]
    assert p.body.strip().startswith("Call `get_live_price")
    assert p.matches("please run the TEST   note for me") and p.matches("Note du jour?") and not p.matches("test notes")
    block = p.prompt_block()
    assert block.startswith('<playbook name="test_note">') and block.endswith("</playbook>") and "exactly one line" in block
    bare = pb.parse_playbook("no front matter here", path="/x/plain_thing.md")
    assert bare.name == "plain_thing" and bare.triggers == ["plain thing"] and bare.min_role == "viewer"


def test_load_match_and_describe():
    names = [p.name for p in pb.playbooks()]
    assert names == ["open", "test_note"]
    assert pb.match("give me the test note").name == "test_note" and pb.match("btc price") is None
    assert pb.is_playbook_command("playbooks") and pb.is_playbook_command("Playbook list") and not pb.is_playbook_command("run the playbook")
    text = pb.describe("viewer")
    assert "*test_note*" in text and "needs role desk (not available to you)" in text and "`open sesame`" in text
    assert "(not available" not in pb.describe("admin")


def test_repo_daily_commentary_playbook_is_valid(monkeypatch):
    monkeypatch.delenv("PLAYBOOKS_DIR")
    pb.reset_playbooks()
    real = pb.playbooks()
    daily = next(p for p in real if p.name == "daily_commentary")
    assert daily.min_role == "viewer" and "daily commentary" in daily.triggers
    for tool_name in ("get_top_movers", "get_crypto_news", "get_zscore_signals", "get_cme_curve", "get_etf_flows",
                      "get_options_snapshot", "get_treasury_curve", "get_vix_history", "get_macro_snapshot"):
        assert tool_name in daily.body
    assert pb.match("please generate the daily commentary").name == "daily_commentary"


def test_with_playbook_gates_on_role():
    assert chat.with_playbook("", "btc price") == ""
    assert chat.with_playbook("MEM", "btc price") == "MEM"
    out = chat.with_playbook("MEM", "run the test note", role="desk")
    assert out.startswith("MEM\n\n<playbook name=\"test_note\">")
    assert chat.with_playbook("MEM", "run the test note", role="viewer") == "MEM"          # below min_role
    assert chat.with_playbook("", "open sesame", role="viewer").startswith('<playbook name="open">')
    assert chat.with_playbook("", "run the test note").startswith("<playbook")             # terminal: no role gate


def test_terminal_session_injects_playbook(monkeypatch):
    from tests import test_chat as tc
    tc.RECORDED.clear()
    llm = RecordingFakeLLM(messages=iter([AIMessage(content="one line")]))
    session = chat.ChatSession(llm=llm, tools=[], system_prompt="BASE", recall=False)
    answer, _ = session.ask("open sesame please")
    assert answer == "one line"
    system = tc.RECORDED[0][0].content
    assert system.startswith("BASE\n\n<playbook name=\"open\">") and "Body of open." in system
    assert session.last_memory_context.startswith("<playbook")


def test_slack_playbooks_command_and_role_gated_injection(tmp_path, monkeypatch):
    control = AccessControl(Policy.load(), MemoryAccessStore())
    control.set_channel("C1", "allowed", by=ADMIN)
    control.set_user("UDESK", "desk", by=ADMIN)
    set_access_control(control)
    monkeypatch.setattr(slack_bot, "_SESSIONS", slack_bot.SessionStore(path=str(tmp_path / "s.json")))
    try:
        seen = []

        class Agent:
            async def ainvoke(self, inp, config):
                from tools.context import current_memory_context
                seen.append(current_memory_context.get())
                return {"messages": [inp["messages"][0], AIMessage(content="ok")]}

        client = RecordingClient()
        asyncio.run(handle({"channel": "C1", "user": "U1", "ts": "1.0", "text": "<@UBOT> playbooks"}, client, Agent(), react=False))
        assert client.posts[-1]["text"].startswith("*Playbooks*") and "not available to you" in client.posts[-1]["text"]
        asyncio.run(handle({"channel": "C1", "user": "U1", "ts": "2.0", "text": "<@UBOT> run the test note"}, client, Agent(), react=False))
        assert "<playbook" not in (seen[-1] or "")                                          # viewer: gated out
        asyncio.run(handle({"channel": "C1", "user": "UDESK", "ts": "3.0", "text": "<@UBOT> run the test note"}, client, Agent(), react=False))
        assert '<playbook name="test_note">' in seen[-1]
        asyncio.run(handle({"channel": "C1", "user": "U1", "ts": "4.0", "text": "<@UBOT> open sesame"}, client, Agent(), react=False))
        assert '<playbook name="open">' in seen[-1]
    finally:
        reset_access_control()
