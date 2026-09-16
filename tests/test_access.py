"""access/: policy loading, decisions, tool gating, Slack admin commands and the handle() hook. No network."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import tool

import slack_bot
from access import policy as pol
from access.commands import HELP, is_access_command, run_access_command, strip_leading_mention
from access.policy import AccessControl, Policy, get_access_control, reset_access_control, set_access_control
from access.store import MemoryAccessStore, SqliteAccessStore
from slack_bot import RecordingClient, handle
from tools import context as ctx

ADMIN = "U0BP6EUTHA9"          # bootstrap admin in access/policy.yaml


@pytest.fixture
def policy():
    return Policy.load()


@pytest.fixture
def control(policy):
    return AccessControl(policy, MemoryAccessStore())


@pytest.fixture
def installed(control, monkeypatch):
    set_access_control(control)
    yield control
    reset_access_control()


def _decision(control, user="U1", channel="C1", is_dm=False):
    return control.decide(user, channel, is_dm)


# --- policy -------------------------------------------------------------------

def test_policy_loads_and_covers_every_registered_tool(policy):
    from chat import default_tools
    grouped = {t for tools in policy.tool_groups.values() for t in tools}
    registered = {t.name for t in default_tools()}
    assert registered <= grouped, f"tools without a group: {sorted(registered - grouped)}"
    assert grouped <= registered, f"policy names unknown tools: {sorted(grouped - registered)}"
    assert policy.default_role == "viewer" and ADMIN in policy.bootstrap_admins
    assert policy.roles["desk"]["inherits"] == "viewer" and policy.roles["admin"]["inherits"] == "lead"


def test_role_inheritance_and_lookups(policy):
    viewer, desk, lead, admin = (policy.tools_for(r) for r in ("viewer", "desk", "lead", "admin"))
    assert viewer < desk < lead == admin
    assert {"get_live_price", "get_crypto_news", "get_macro_snapshot", "get_snapshot_history", "remember"} <= viewer
    assert "get_desk_risk_snapshot" not in viewer and "get_desk_risk_snapshot" in desk
    assert "run_full_signals_analysis" in desk - viewer and "get_counterparty_pnl" in desk - viewer
    assert "query_desk_data" in lead - desk and "set_channel_rule" in lead - desk
    assert policy.group_of("get_cme_curve") == "cme" and policy.group_of("nonexistent") is None
    assert policy.role_needed_for_group("desk_risk") == "desk" and policy.role_needed_for_group("desk_raw") == "lead"
    assert policy.tools_for("none") == set()


def test_policy_validation_rejects_duplicates_and_unknowns():
    base = {"roles": {"viewer": {"tool_groups": ["a"]}}, "tool_groups": {"a": ["x"], "b": ["x"]}}
    with pytest.raises(ValueError, match="two groups"):
        Policy.from_dict(base)
    with pytest.raises(ValueError, match="unknown tool group"):
        Policy.from_dict({"roles": {"viewer": {"tool_groups": ["zzz"]}}, "tool_groups": {}})
    with pytest.raises(ValueError, match="unknown role"):
        Policy.from_dict({"roles": {"boss": {"tool_groups": []}}, "tool_groups": {}})


# --- decisions ------------------------------------------------------------------

def test_decide_roles_places_and_admin_exemption(control):
    d = _decision(control, "U1", "C1")
    assert d.role == "viewer" and not d.place_ok and d.reason == "channel:unlisted" and not d.allowed
    control.set_channel("C1", "allowed", by=ADMIN)
    d = _decision(control, "U1", "C1")
    assert d.place_ok and not d.place_confidential and d.allowed
    control.set_channel("C2", "confidential", by=ADMIN)
    assert _decision(control, "U1", "C2").place_confidential
    dm = _decision(control, "U1", "D1", is_dm=True)
    assert dm.place_ok and dm.place_confidential and dm.reason == "dm"
    a = _decision(control, ADMIN, "C9")
    assert a.is_admin and a.place_ok                                  # admins can act in unlisted channels
    control.set_user("U2", "desk", by=ADMIN)
    assert _decision(control, "U2", "C1").role == "desk"
    control.set_user("U3", "none", by=ADMIN)
    assert not _decision(control, "U3", "C1").allowed
    control.set_default_role("none", by=ADMIN)
    assert _decision(control, "U1", "C1").role == "none" and _decision(control, ADMIN, "C1").role == "admin"
    control.set_paused(True, by=ADMIN)
    assert not _decision(control, "U2", "C1").allowed and _decision(control, ADMIN, "C1").allowed
    with pytest.raises(ValueError):
        control.set_user("U4", "boss", by=ADMIN)


def test_rules_are_cached_then_invalidated(policy):
    clock = [1000.0]
    store = MemoryAccessStore()
    control = AccessControl(policy, store, now=lambda: clock[0])
    assert control.role_for("U1") == "viewer"
    store.put_rule("user", "U1", "desk", "x")            # behind the cache
    assert control.role_for("U1") == "viewer"
    clock[0] += policy.refresh_seconds + 1
    assert control.role_for("U1") == "desk"
    control.set_user("U1", "lead", by=ADMIN)             # admin ops invalidate immediately
    assert control.role_for("U1") == "lead"


def test_sqlite_store_roundtrip(tmp_path):
    store = SqliteAccessStore(str(tmp_path / "access.db"))
    store.put_rule("user", "U1", "desk", ADMIN)
    store.put_rule("user", "U1", "lead", ADMIN)
    store.put_rule("channel", "C1", "confidential", ADMIN)
    rules = {(r["kind"], r["id"]): r["value"] for r in store.list_rules()}
    assert rules == {("user", "U1"): "lead", ("channel", "C1"): "confidential"}
    assert store.delete_rule("user", "U1") and not store.delete_rule("user", "U1")
    again = SqliteAccessStore(str(tmp_path / "access.db"))
    assert [r["id"] for r in again.list_rules()] == ["C1"]


# --- tool checks --------------------------------------------------------------------

def test_check_tool_by_role_place_and_argument(control):
    control.set_channel("C1", "allowed", by=ADMIN)
    control.set_channel("CPRIV", "confidential", by=ADMIN)
    control.set_user("UDESK", "desk", by=ADMIN)
    control.set_user("ULEAD", "lead", by=ADMIN)
    viewer = _decision(control, "U1", "C1")
    desk_pub = _decision(control, "UDESK", "C1")
    desk_priv = _decision(control, "UDESK", "CPRIV")
    desk_dm = _decision(control, "UDESK", "D1", is_dm=True)
    lead_priv = _decision(control, "ULEAD", "CPRIV")

    assert control.check_tool("get_live_price", {}, viewer) is None
    assert control.check_tool("current_time", {}, viewer) is None                       # ungated utility
    denied = control.check_tool("get_desk_risk_snapshot", {}, viewer)
    assert denied.startswith("Not permitted: get_desk_risk_snapshot needs the desk role (you have viewer)")
    assert "confidential desk data" in control.check_tool("get_desk_risk_snapshot", {}, desk_pub)   # right role, public channel
    assert control.check_tool("get_desk_risk_snapshot", {}, desk_priv) is None
    assert control.check_tool("get_desk_risk_snapshot", {}, desk_dm) is None
    assert "needs the lead role" in control.check_tool("query_desk_data", {"sql": "select 1"}, desk_priv)
    assert control.check_tool("query_desk_data", {"sql": "select 1"}, lead_priv) is None
    # argument rules: memory scope and snapshot source
    assert control.check_tool("remember", {"text": "x", "scope": "me"}, viewer) is None
    assert "remember(scope=shared) needs the lead role" in control.check_tool("remember", {"text": "x", "scope": "shared"}, viewer)
    assert control.check_tool("get_snapshot_history", {"source": "signals", "metric": "price"}, viewer) is None
    assert "needs the desk role" in control.check_tool("get_snapshot_history", {"source": "haruko", "metric": "delta_usd"}, viewer)
    assert "confidential" in control.check_tool("get_snapshot_history", {"source": "sheet", "metric": "mtd_pnl_usd"}, desk_pub)
    assert control.check_tool("get_snapshot_history", {"source": "sheet", "metric": "mtd_pnl_usd"}, desk_priv) is None


def test_gate_tools_uses_request_context(control):
    calls = []

    @tool("get_desk_risk_snapshot")
    def fake_desk(entity: str = "combined") -> str:
        """fake"""
        calls.append(entity)
        return "delta -39m"

    @tool("get_live_price")
    def fake_live(token: str) -> str:
        """fake"""
        return f"{token} 75000"

    @tool("current_time")
    def fake_time() -> str:
        """fake"""
        return "now"

    control.set_channel("C1", "allowed", by=ADMIN)
    control.set_channel("CPRIV", "confidential", by=ADMIN)
    control.set_user("UDESK", "desk", by=ADMIN)
    gated = {t.name: t for t in control.gate_tools([fake_desk, fake_live, fake_time])}
    assert set(gated) == {"get_desk_risk_snapshot", "get_live_price", "current_time"}
    assert gated["get_live_price"].description == fake_live.description and gated["get_live_price"].args == fake_live.args
    with ctx.request_context("U1", "C1"):
        assert gated["get_live_price"].invoke({"token": "btc"}) == "btc 75000"
        assert gated["get_desk_risk_snapshot"].invoke({}).startswith("Not permitted")
        assert gated["current_time"].invoke({}) == "now"
    with ctx.request_context("UDESK", "C1"):
        assert "confidential" in gated["get_desk_risk_snapshot"].invoke({})
    with ctx.request_context("UDESK", "CPRIV"):
        assert gated["get_desk_risk_snapshot"].invoke({}) == "delta -39m"
    with ctx.request_context("UDESK", "D1", is_dm=True):
        assert gated["get_desk_risk_snapshot"].invoke({"entity": "20"}) == "delta -39m"
    assert calls == ["combined", "20"]
    # the async path is gated too
    async def go():
        with ctx.request_context("U1", "C1"):
            return await gated["get_desk_risk_snapshot"].ainvoke({})
    assert asyncio.run(go()).startswith("Not permitted")


# --- commands -------------------------------------------------------------------------

def test_command_parsing_helpers():
    assert is_access_command("access help") and is_access_command("Permissions list") and is_access_command("perms whoami")
    assert not is_access_command("what is access to CME data like") and not is_access_command("btc price")
    assert strip_leading_mention("<@UBOT> access add user <@U2> desk") == "access add user <@U2> desk"


def test_admin_commands_mutate_rules(control):
    admin = _decision(control, ADMIN, "C1")
    out = run_access_command("<@UBOT> access add user <@U2> desk", admin, control)
    assert out == "Done: <@U2> is now *desk*." and control.role_for("U2") == "desk"
    assert "unknown role 'boss'" in run_access_command("access add user <@U2> boss", admin, control)
    assert "Usage" in run_access_command("access add user <@U2>", admin, control)
    out = run_access_command("access add channel here", admin, control)
    assert out.startswith("Done: the bot now answers in <#C1> (allowed).") and control.channel_mode("C1") == "allowed"
    out = run_access_command("access add channel <#CPRIV|gm-desk> confidential", admin, control)
    assert "<#CPRIV> (confidential)" in out and control.channel_mode("CPRIV") == "confidential"
    out = run_access_command("access add channel confidential", admin, control)     # 'here' implied
    assert control.channel_mode("C1") == "confidential"
    assert run_access_command("access remove channel <#CPRIV>", admin, control).startswith("Done") and control.channel_mode("CPRIV") is None
    assert "was not enabled" in run_access_command("access remove channel <#CPRIV>", admin, control)
    assert run_access_command("access default none", admin, control).startswith("Done") and control.default_role == "none"
    assert run_access_command("access pause", admin, control).startswith("Paused") and control.paused
    assert run_access_command("access resume", admin, control).startswith("Resumed") and not control.paused
    listing = run_access_command("access list", admin, control)
    assert "• desk: <@U2>" in listing and "channel <#C1>: confidential" in listing and f"<@{ADMIN}>" in listing
    assert run_access_command("access remove user <@U2>", admin, control).startswith("Done") and control.role_for("U2") == "none"
    assert run_access_command("access help", admin, control) == HELP
    assert "did not understand" in run_access_command("access frobnicate", admin, control)


def test_non_admin_commands(control):
    control.set_channel("C1", "allowed", by=ADMIN)
    viewer = _decision(control, "U1", "C1")
    who = run_access_command("access whoami", viewer, control)
    assert "role *viewer*" in who and "channel <#C1> (allowed)" in who
    assert run_access_command("access add user <@U2> desk", viewer, control).startswith("Only admins")
    assert run_access_command("access list", viewer, control).startswith("Only leads and admins")
    control.set_user("UL", "lead", by=ADMIN)
    assert "*Access rules*" in run_access_command("access list", _decision(control, "UL", "C1"), control)


# --- handle() hook --------------------------------------------------------------------

class _Agent:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, inp, config):
        from langchain_core.messages import AIMessage
        self.calls += 1
        return {"messages": [inp["messages"][0], AIMessage(content="answer")]}


def _ev(user="U1", channel="C1", text="<@UBOT> btc price?", **extra):
    ev = {"type": "app_mention", "channel": channel, "user": user, "ts": "100.1", "text": text}
    ev.update(extra)
    return ev


def test_handle_denies_unlisted_channel_then_admin_enables_it(installed, tmp_path, monkeypatch):
    monkeypatch.setattr(slack_bot, "_SESSIONS", slack_bot.SessionStore(path=str(tmp_path / "s.json")))
    client, agent = RecordingClient(), _Agent()
    asyncio.run(handle(_ev(), client, agent, react=False))
    assert agent.calls == 0 and client.posts[-1]["text"] == installed.policy.message("denied_place") and not client.updates
    asyncio.run(handle(_ev(user=ADMIN, text="<@UBOT> access add channel here"), client, agent, react=False))
    assert client.posts[-1]["text"].startswith("Done: the bot now answers in <#C1>") and agent.calls == 0
    asyncio.run(handle(_ev(), client, agent, react=False))
    assert agent.calls == 1 and client.updates[-1]["text"] == "answer"
    asyncio.run(handle(_ev(user="U1", text="<@UBOT> access whoami"), client, agent, react=False))
    assert "role *viewer*" in client.posts[-1]["text"] and agent.calls == 1
    asyncio.run(handle(_ev(user=ADMIN, text="<@UBOT> access pause"), client, agent, react=False))
    asyncio.run(handle(_ev(), client, agent, react=False))
    assert client.posts[-1]["text"] == installed.policy.message("paused") and agent.calls == 1
    asyncio.run(handle(_ev(user=ADMIN), client, agent, react=False))
    assert agent.calls == 2                                          # admins keep working while paused


def test_handle_dm_allowed_by_default_and_none_role_denied(installed, tmp_path, monkeypatch):
    monkeypatch.setattr(slack_bot, "_SESSIONS", slack_bot.SessionStore(path=str(tmp_path / "s.json")))
    client, agent = RecordingClient(), _Agent()
    asyncio.run(handle(_ev(channel="D1", channel_type="im"), client, agent, react=False))
    assert agent.calls == 1
    installed.set_user("U1", "none", by=ADMIN)
    asyncio.run(handle(_ev(channel="D1", channel_type="im"), client, agent, react=False))
    assert agent.calls == 1 and client.posts[-1]["text"] == installed.policy.message("denied_role")


def test_get_access_control_off_switch(monkeypatch):
    reset_access_control()
    monkeypatch.setenv("ACCESS_CONTROL", "off")
    assert get_access_control() is None
    reset_access_control()
    monkeypatch.setenv("ACCESS_CONTROL", "on")
    monkeypatch.setenv("ACCESS_BACKEND", "memory")
    control = get_access_control()
    assert control is not None and control.role_for(ADMIN) == "admin" and control.store.name == "memory"
    reset_access_control()


def test_build_agent_gates_tools(monkeypatch, tmp_path, installed):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class LLM(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    seen = {}
    real = slack_bot.build_chat_agent

    def spy(llm=None, tools=None, system_prompt=None, checkpointer=None):
        seen["tools"] = tools
        return real(llm=llm, tools=tools, system_prompt=system_prompt, checkpointer=checkpointer)

    monkeypatch.setattr(slack_bot, "build_chat_agent", spy)
    monkeypatch.setattr(slack_bot, "load_slack_prompt", lambda today=None: "p")

    async def go():
        async with slack_bot.build_agent(db_path=str(tmp_path / "a.db"), llm=LLM(messages=iter([AIMessage(content="x")]))):
            pass
    asyncio.run(go())
    names = {t.name for t in seen["tools"]}
    assert "get_desk_risk_snapshot" in names and "current_time" in names
    desk = next(t for t in seen["tools"] if t.name == "get_desk_risk_snapshot")
    with ctx.request_context("U1", "C1"):
        assert desk.invoke({}).startswith("Not permitted")          # wrapped, not the raw tool
