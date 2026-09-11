"""Agent + workflow tests with a fake LLM. No network, no GCP, no Vertex."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agents.events import FinalEvent, TokenEvent
from agents.trading_agent import TradingAgentLangGraph, _message_text
from workflows.signals_workflow import SignalsWorkflow, run_signals_analysis


class ToolAwareFakeLLM(FakeMessagesListChatModel):
    """FakeMessagesListChatModel that accepts bind_tools() so it can drive a ReAct agent."""

    def bind_tools(self, tools, **kwargs):  # noqa: D401
        return self


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """Guard: any accidental call to utils.llm.get_llm fails loudly instead of hitting Vertex."""
    import sys
    import types

    fake_mod = types.ModuleType("utils.llm")

    def _boom(*a, **k):
        raise AssertionError("utils.llm.get_llm() must not be called in tests")

    fake_mod.get_llm = _boom
    monkeypatch.setitem(sys.modules, "utils.llm", fake_mod)


# ----------------------------------------------------------------------------
# TradingAgentLangGraph
# ----------------------------------------------------------------------------

def test_agent_returns_llm_answer_without_tool_calls():
    llm = ToolAwareFakeLLM(responses=[AIMessage(content="BTC perp OI is an outlier at z=2.9.")])
    agent = TradingAgentLangGraph(llm=llm, tools=[])
    answer = agent.ask("Any outliers?")
    assert answer == "BTC perp OI is an outlier at z=2.9."


def test_agent_uses_injected_tools_and_system_prompt():
    from langchain_core.tools import tool

    calls = []

    @tool("echo_tool")
    def echo_tool(text: str) -> str:
        """Echo text back."""
        calls.append(text)
        return f"echo:{text}"

    llm = ToolAwareFakeLLM(responses=[
        AIMessage(content="", tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "call_1"}]),
        AIMessage(content="Tool said echo:hi"),
    ])
    agent = TradingAgentLangGraph(llm=llm, tools=[echo_tool])
    answer = agent.ask("run the tool")
    assert calls == ["hi"]
    assert answer == "Tool said echo:hi"
    assert "z-score" in agent.system_prompt.lower() or "z score" in agent.system_prompt.lower()


def test_agent_ask_reports_errors_instead_of_raising():
    class Broken:
        def bind_tools(self, *a, **k):
            raise RuntimeError("no model")

    agent = TradingAgentLangGraph(llm=Broken(), tools=[])
    assert agent.ask("q").startswith("Error:")


def test_message_text_handles_content_blocks():
    assert _message_text(AIMessage(content="plain")) == "plain"
    blocks = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "tool_use", "id": "x"}, {"type": "text", "text": "b"}])
    assert _message_text(blocks) == "ab"


# ----------------------------------------------------------------------------
# SignalsWorkflow with fake provider + fake LLM
# ----------------------------------------------------------------------------

def _synthetic_token_data(spike: bool) -> pd.DataFrame:
    """45 days of flat-ish metrics; when `spike`, the last day's perp OI is 10x."""
    rng = np.random.default_rng(0)
    days = 45
    time = pd.date_range(end=datetime(2026, 9, 10), periods=days, freq="D")
    df = pd.DataFrame({
        "time": time,
        "price": 100 + rng.normal(0, 0.5, days),
        "spot_volume": 1_000_000 + rng.normal(0, 20_000, days),
        "perp_volume": 2_000_000 + rng.normal(0, 40_000, days),
        "perp_oi": 5_000_000 + rng.normal(0, 50_000, days),
        "funding_rate": np.full(days, 0.08),
        "long_liquidations": 10_000 + rng.normal(0, 500, days),
        "short_liquidations": 10_000 + rng.normal(0, 500, days),
    })
    df["total_liquidations"] = df["long_liquidations"] + df["short_liquidations"]
    for col in ("spot_volume", "perp_volume", "perp_oi", "total_liquidations"):
        df.loc[df.index[-1], col] = df[col].iloc[:-1].median()  # calm by construction
    df["price_pct_change"] = df["price"].pct_change() * 100
    if spike:
        df.loc[df.index[-1], "perp_oi"] *= 10
    return df


class _Resp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content="**Key Signal**: OI spike"):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Resp(self.content)


@pytest.fixture
def fake_fetch(monkeypatch):
    import workflows.signals_workflow as wf

    def _fetch(tokens, start_date, end_date, **kw):
        return {t: _synthetic_token_data(spike=(t == "btc")) for t in tokens}

    monkeypatch.setattr(wf, "fetch_token_metrics", _fetch)


def test_workflow_streams_token_event_only_for_significant_tokens(fake_fetch):
    llm = FakeLLM()
    events = list(SignalsWorkflow(llm=llm).analyze_stream(tokens=["btc", "eth"]))

    token_events = [e for e in events if isinstance(e, TokenEvent)]
    finals = [e for e in events if isinstance(e, FinalEvent)]

    assert len(finals) == 1
    assert [e.token for e in token_events] == ["btc"]
    assert token_events[0].analysis_text == "**Key Signal**: OI spike"
    assert token_events[0].signals["metrics"]["perp_oi"]["is_outlier"] is True
    assert finals[0].summary["tokens_analyzed"] == 2
    assert finals[0].summary["tokens_with_outliers"] == ["btc"]
    assert "eth" in finals[0].all_signals
    # one LLM call per significant token, with system + human message
    assert len(llm.calls) == 1 and len(llm.calls[0]) == 2
    assert "BTC" in llm.calls[0][1].content and "perp_oi" in llm.calls[0][1].content


def test_workflow_analyze_combines_markdown_and_stats(fake_fetch):
    result = SignalsWorkflow(llm=FakeLLM("analysis body")).analyze(tokens=["btc", "eth"])
    assert result["analysis"].startswith("## BTC")
    assert "analysis body" in result["analysis"]
    assert set(result) == {"analysis", "stats"}
    assert result["stats"]["summary"]["tokens_with_significant_moves"] == ["btc"]
    assert result["stats"]["outlier_threshold"] == 2.5


def test_workflow_handles_llm_content_blocks(fake_fetch):
    llm = FakeLLM([{"type": "text", "text": "block text"}])
    result = SignalsWorkflow(llm=llm).analyze(tokens=["btc"])
    assert "block text" in result["analysis"]


def test_workflow_llm_failure_is_captured_per_token(fake_fetch):
    class Exploding:
        def invoke(self, messages):
            raise RuntimeError("vertex down")

    events = list(SignalsWorkflow(llm=Exploding()).analyze_stream(tokens=["btc"]))
    te = next(e for e in events if isinstance(e, TokenEvent))
    assert te.analysis_text.startswith("<error: RuntimeError")
    assert any(isinstance(e, FinalEvent) for e in events)


def test_workflow_no_data_yields_empty_final(monkeypatch):
    import workflows.signals_workflow as wf
    monkeypatch.setattr(wf, "fetch_token_metrics", lambda *a, **k: {})
    events = list(SignalsWorkflow(llm=FakeLLM()).analyze_stream(tokens=["btc"]))
    assert len(events) == 1 and isinstance(events[0], FinalEvent)
    assert events[0].summary["tokens_analyzed"] == 0
    result = SignalsWorkflow(llm=FakeLLM()).analyze(tokens=["btc"])
    assert "No significant statistical anomalies" in result["analysis"]


def test_run_signals_analysis_uses_workflow(monkeypatch, fake_fetch):
    import workflows.signals_workflow as wf

    monkeypatch.setattr(wf.SignalsWorkflow, "_get_llm", lambda self: FakeLLM("via wrapper"))
    result = run_signals_analysis(tokens=["btc"], lookback_days=45)
    assert "via wrapper" in result["analysis"]
    assert result["stats"]["lookback_window"] == 30


def test_workflow_analyze_all_writes_up_calm_tokens_and_marks_missing_metrics(fake_fetch, monkeypatch):
    import workflows.signals_workflow as wf

    def _fetch(tokens, start_date, end_date, **kw):
        out = {t: _synthetic_token_data(spike=False) for t in tokens}
        out["eth"] = out["eth"].drop(columns=["perp_oi", "long_liquidations", "short_liquidations",
                                              "total_liquidations"])  # derivatives provider had none
        return out

    monkeypatch.setattr(wf, "fetch_token_metrics", _fetch)
    llm = FakeLLM("calm write-up")
    events = list(SignalsWorkflow(llm=llm).analyze_stream(tokens=["btc", "eth"], analyze_all=True))
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    assert [e.token for e in token_events] == ["btc", "eth"]
    assert next(e for e in events if isinstance(e, FinalEvent)).summary["tokens_with_significant_moves"] == []

    eth_prompt = llm.calls[1][1].content
    assert "perp_oi: not available" in eth_prompt
    assert "total_liquidations: not available" in eth_prompt
    assert "nan" not in eth_prompt.lower()
    # default behaviour unchanged: nothing significant -> no LLM call
    assert list(e for e in SignalsWorkflow(llm=FakeLLM()).analyze_stream(tokens=["eth"])
                if isinstance(e, TokenEvent)) == []


def test_cli_tokens_implies_analyze_all():
    import run_signals

    a = run_signals.parse_args(["--tokens", "btc", "eth"])
    assert a.tokens == ["btc", "eth"] and not a.significant_only
    b = run_signals.parse_args(["--tokens", "btc", "--significant-only"])
    assert b.significant_only
    c = run_signals.parse_args(["--all"])
    assert c.analyze_all and c.tokens is None


def test_cli_post_slack_without_webhook_is_a_noop(monkeypatch, capsys):
    """--post-slack with no webhook configured must log + skip: no HTTP, exit code 0."""
    import run_signals
    import notifiers.slack as slack
    import utils.config as config

    class _Cfg:
        SLACK_WEBHOOK_URL = None
        SLACK_CHANNEL_ID = None   # secret trading_signals_slack_channel_id has no versions
        SLACK_BOT_TOKEN = None

    monkeypatch.setattr(config, "Config", _Cfg)
    monkeypatch.setattr(slack.requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("posted")))

    class _WF(SignalsWorkflow):
        def __init__(self, *a, **k): ...
        def analyze_stream(self, **kw):
            yield FinalEvent(all_signals={}, summary={"tokens_analyzed": 0, "tokens_with_outliers": [],
                                                      "tokens_with_significant_moves": []})

    monkeypatch.setattr(run_signals, "SignalsWorkflow", _WF)
    rc = run_signals.main(["--tokens", "btc", "--post-slack"])
    assert rc == 0
    assert "Slack post skipped" in capsys.readouterr().out
