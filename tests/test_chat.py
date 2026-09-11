"""chat.py + tools/chat_tools.py tests. No network, no GCP, no Vertex."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import chat
import tools.metrics as metrics_mod
from tools import chat_tools
from tools.chat_tools import (
    MAX_TOKENS_PER_CALL,
    get_chat_tools,
    get_price_history,
    get_token_metrics,
    get_zscore_signals,
    list_token_universe,
    run_full_signals_analysis,
    validate_tokens,
)
from tools.metrics import FULL_TOKEN_UNIVERSE, TEST_TOKEN_UNIVERSE

END = datetime(2026, 9, 10)


@pytest.fixture(autouse=True)
def no_real_llm_or_provider(monkeypatch):
    """Any accidental Vertex / provider call fails loudly."""
    import sys
    import types

    fake_llm = types.ModuleType("utils.llm")

    def _boom(*a, **k):
        raise AssertionError("utils.llm.get_llm() must not be called in tests")

    fake_llm.get_llm = _boom
    monkeypatch.setitem(sys.modules, "utils.llm", fake_llm)
    monkeypatch.setattr(metrics_mod, "get_provider", lambda: (_ for _ in ()).throw(
        AssertionError("providers.factory.get_provider() must not be called in tests")))


# ----------------------------------------------------------------------------
# synthetic data
# ----------------------------------------------------------------------------

def _frame(days=45, seed=0, oi_spike=False, funding_spike=False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = pd.date_range(end=END, periods=days, freq="D")
    df = pd.DataFrame({
        "time": time,
        "price": 100 + rng.normal(0, 0.5, days),
        "spot_volume": 1_000_000 + rng.normal(0, 20_000, days),
        "perp_volume": 2_000_000 + rng.normal(0, 40_000, days),
        "perp_oi": 5_000_000 + rng.normal(0, 50_000, days),
        "funding_rate": 10.0 + rng.normal(0, 0.5, days),  # annualised %
        "long_liquidations": 10_000 + rng.normal(0, 500, days),
        "short_liquidations": 10_000 + rng.normal(0, 500, days),
    })
    df["total_liquidations"] = df["long_liquidations"] + df["short_liquidations"]
    for col in ("spot_volume", "perp_volume", "perp_oi", "total_liquidations", "funding_rate"):
        df.loc[df.index[-1], col] = df[col].iloc[:-1].median()
    df["price_pct_change"] = df["price"].pct_change() * 100
    if oi_spike:
        df.loc[df.index[-1], "perp_oi"] *= 10
    if funding_spike:
        df.loc[df.index[-1], "funding_rate"] = 60.0
    return df


@pytest.fixture
def fake_fetch(monkeypatch):
    calls = []

    def _fetch_token_metrics(tokens, start_date, end_date):
        calls.append(list(tokens))
        days = (end_date - start_date).days
        return {t: _frame(days=days, oi_spike=(t == "btc"), funding_spike=(t == "sol")) for t in tokens}

    def _fetch_price_history(token, days=30):
        calls.append(("price", token, days))
        df = _frame(days=days)[["time", "price"]].copy()
        return df

    monkeypatch.setattr(metrics_mod, "fetch_token_metrics", _fetch_token_metrics)
    monkeypatch.setattr(metrics_mod, "fetch_price_history", _fetch_price_history)
    return calls


# ----------------------------------------------------------------------------
# chat_tools
# ----------------------------------------------------------------------------

def test_validate_tokens_normalises_and_caps():
    valid, unknown, dropped = validate_tokens("BTC, eth sol nope")
    assert valid == ["btc", "eth", "sol"]
    assert unknown == ["nope"]
    assert dropped == []

    many = FULL_TOKEN_UNIVERSE[: MAX_TOKENS_PER_CALL + 3]
    valid, unknown, dropped = validate_tokens(many)
    assert len(valid) == MAX_TOKENS_PER_CALL
    assert dropped == many[MAX_TOKENS_PER_CALL:]
    assert unknown == []


def test_list_token_universe_tool():
    import json

    out = json.loads(list_token_universe.invoke({}))
    assert out["full_universe"] == FULL_TOKEN_UNIVERSE
    assert out["test_universe"] == TEST_TOKEN_UNIVERSE
    assert out["max_tokens_per_call"] == MAX_TOKENS_PER_CALL


def test_get_token_metrics_unknown_token(fake_fetch):
    out = get_token_metrics.invoke({"token": "doge"})
    assert "Unknown token 'doge'" in out
    assert "btc" in out  # lists supported symbols
    assert fake_fetch == []  # no fetch attempted


def test_get_token_metrics_table(fake_fetch):
    out = get_token_metrics.invoke({"token": "ETH", "days": 45})
    assert out.startswith("### ETH daily metrics")
    assert "45 rows" in out
    assert "Latest row: 2026-09-10" in out
    for col in ("price", "spot_volume", "perp_volume", "perp_oi", "funding_rate", "total_liquidations"):
        assert f"- {col}:" in out
    # funding is shown as a percentage, not USD
    funding_line = next(l for l in out.splitlines() if l.startswith("- funding_rate:"))
    assert funding_line.endswith("%") and "$" not in funding_line
    table_rows = [l for l in out.splitlines() if l.startswith("| 2026-")]
    assert len(table_rows) == 45
    assert fake_fetch == [["eth"]]


def test_get_token_metrics_clamps_days(fake_fetch):
    out = get_token_metrics.invoke({"token": "btc", "days": 1})
    assert "7 rows" in out  # MIN_DAYS
    out = get_token_metrics.invoke({"token": "btc", "days": 10_000})
    assert "120 rows" in out  # MAX_DAYS


def test_get_zscore_signals_table_shape_and_flags(fake_fetch):
    out = get_zscore_signals.invoke({"tokens": ["btc", "eth", "sol"], "days": 45})
    lines = out.splitlines()
    assert lines[0].startswith("### Z-score signals")
    assert "| token | metric | latest value | z-score | flag |" in lines

    rows = [l for l in lines if l.startswith("| BTC |") or l.startswith("| ETH |") or l.startswith("| SOL |")]
    # 3 tokens x 5 metrics (spot_volume, perp_volume, perp_oi, total_liquidations, funding_rate)
    assert len(rows) == 15
    metrics = {l.split("|")[2].strip() for l in rows}
    assert metrics == {"spot_volume", "perp_volume", "perp_oi", "total_liquidations", "funding_rate"}

    btc_oi = next(l for l in rows if l.startswith("| BTC | perp_oi"))
    assert "OUTLIER" in btc_oi
    sol_funding = next(l for l in rows if l.startswith("| SOL | funding_rate"))
    assert "OUTLIER" in sol_funding and "60.00%" in sol_funding
    eth_rows = [l for l in rows if l.startswith("| ETH |")]
    assert all("normal" in l for l in eth_rows)

    assert "**Outliers:** BTC, SOL" in out
    assert "**Price context**" in out
    assert fake_fetch == [["btc", "eth", "sol"]]


def test_get_zscore_signals_unknown_and_cap(fake_fetch):
    requested = FULL_TOKEN_UNIVERSE[: MAX_TOKENS_PER_CALL + 2] + ["fake"]
    out = get_zscore_signals.invoke({"tokens": requested})
    assert "Unknown token(s) ignored: FAKE" in out
    assert f"Token cap is {MAX_TOKENS_PER_CALL}" in out
    for t in FULL_TOKEN_UNIVERSE[MAX_TOKENS_PER_CALL: MAX_TOKENS_PER_CALL + 2]:
        assert t.upper() in out  # named as dropped
    assert fake_fetch == [FULL_TOKEN_UNIVERSE[:MAX_TOKENS_PER_CALL]]


def test_get_zscore_signals_no_valid_tokens(fake_fetch):
    out = get_zscore_signals.invoke({"tokens": ["xxx"]})
    assert "No valid tokens" in out and "XXX" in out
    assert fake_fetch == []


def test_get_zscore_signals_fetch_error_is_reported(monkeypatch):
    def _explode(*a, **k):
        raise RuntimeError("amberdata 503")

    monkeypatch.setattr(metrics_mod, "fetch_token_metrics", _explode)
    out = get_zscore_signals.invoke({"tokens": ["btc"]})
    assert out.startswith("Error computing z-scores") and "amberdata 503" in out


def test_get_price_history(fake_fetch):
    out = get_price_history.invoke({"token": "btc", "days": 10})
    assert out.startswith("### BTC spot price")
    assert "(10 days)" in out
    assert "- latest: $" in out and "- high: $" in out and "- period change:" in out
    assert len([l for l in out.splitlines() if l.startswith("| 2026-")]) == 10
    assert ("price", "btc", 10) in fake_fetch


def test_get_price_history_unknown_token(fake_fetch):
    assert "Unknown token 'abc'" in get_price_history.invoke({"token": "abc"})


def test_run_full_signals_analysis_uses_workflow(monkeypatch):
    import workflows.signals_workflow as wf

    seen = {}

    def _fake_run(tokens=None, lookback_days=45, **kw):
        seen["tokens"], seen["days"] = tokens, lookback_days
        return {"analysis": "## BTC\n\nOI spike", "stats": {"summary": {
            "tokens_analyzed": len(tokens), "tokens_with_outliers": ["btc"],
            "tokens_with_significant_moves": ["btc"]}}}

    monkeypatch.setattr(wf, "run_signals_analysis", _fake_run)

    out = run_full_signals_analysis.invoke({"tokens": ["btc", "eth", "zzz"], "days": 30})
    assert seen == {"tokens": ["btc", "eth"], "days": 30}
    assert "## BTC" in out and "OI spike" in out
    assert "Outliers: BTC" in out
    assert "Unknown token(s) ignored: ZZZ" in out

    out = run_full_signals_analysis.invoke({})
    assert seen["tokens"] == TEST_TOKEN_UNIVERSE


def test_get_chat_tools_names():
    names = [t.name for t in get_chat_tools()]
    assert names == [
        "list_token_universe", "get_token_metrics", "get_zscore_signals",
        "get_price_history", "run_full_signals_analysis", "get_multi_day_signals_tool",
        "get_options_snapshot", "get_vol_term_structure", "get_options_flow", "get_gamma_exposure",
    ]
    assert all(t.description for t in get_chat_tools())


def test_default_tools_include_desk_tools():
    from tools.desk_tools import DESK_TOOL_NAMES

    names = [t.name for t in chat.default_tools()]
    assert names[: len(get_chat_tools())] == [t.name for t in get_chat_tools()]
    assert names[len(get_chat_tools()):] == DESK_TOOL_NAMES
    assert len(names) == len(set(names))  # no duplicates


# ----------------------------------------------------------------------------
# chat.py: prompt, agent, session
# ----------------------------------------------------------------------------

def test_system_prompt_today_injection():
    prompt = chat.load_system_prompt(today=date(2026, 9, 10))
    assert "{today}" not in prompt
    assert "Thursday 2026-09-10" in prompt
    assert "Global Markets" in prompt
    assert "2.5" in prompt and "30-day" in prompt
    for tool_name in ("get_zscore_signals", "get_token_metrics", "get_price_history",
                      "run_full_signals_analysis", "list_token_universe",
                      "get_desk_risk_snapshot", "get_perp_positions", "query_desk_data"):
        assert tool_name in prompt
    assert "Desk data (BigQuery)" in prompt and "data_quality_flag" in prompt and "brokerage_a1" in prompt


def test_system_prompt_fallback_when_file_missing(tmp_path):
    prompt = chat.load_system_prompt(today=date(2026, 1, 2), path=tmp_path / "missing.md")
    assert "Friday 2026-01-02" in prompt and "{today}" not in prompt


class ToolAwareFakeLLM(GenericFakeChatModel):
    """GenericFakeChatModel that accepts bind_tools() so it can drive a ReAct agent."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_builds_and_returns_fake_answer():
    llm = ToolAwareFakeLLM(messages=iter([AIMessage(content="Nothing unusual in BTC today.")]))
    session = chat.ChatSession(llm=llm, tools=[], system_prompt="test prompt")
    answer, tools_called = session.ask("anything odd in btc?")
    assert answer == "Nothing unusual in BTC today."
    assert tools_called == []


def test_agent_history_persists_across_turns_and_reset():
    llm = ToolAwareFakeLLM(messages=iter([AIMessage(content="first"), AIMessage(content="second"),
                                          AIMessage(content="third")]))
    session = chat.ChatSession(llm=llm, tools=[], system_prompt="p")
    assert session.ask("one")[0] == "first"
    assert session.ask("two")[0] == "second"
    # human + ai per turn -> 4 messages in the thread
    assert session._history_len() == 4

    old = session.thread_id
    session.reset()
    assert session.thread_id != old
    assert session._history_len() == 0
    assert session.ask("three")[0] == "third"


def test_agent_reports_tool_calls():
    from langchain_core.tools import tool

    @tool("echo_tool")
    def echo_tool(text: str) -> str:
        """Echo text back."""
        return f"echo:{text}"

    llm = ToolAwareFakeLLM(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "echo_tool", "args": {"text": "hi"}, "id": "c1"}]),
        AIMessage(content=[{"type": "text", "text": "Tool said echo:hi"}]),
    ]))
    session = chat.ChatSession(llm=llm, tools=[echo_tool], system_prompt="p")
    answer, tools_called = session.ask("run it")
    assert tools_called == ["echo_tool"]
    assert answer == "Tool said echo:hi"


def test_message_text_handles_blocks():
    assert chat.message_text(AIMessage(content="plain")) == "plain"
    blocks = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "tool_use", "id": "x"},
                                {"type": "text", "text": "b"}])
    assert chat.message_text(blocks) == "ab"


def test_one_shot_cli(monkeypatch, capsys):
    llm = ToolAwareFakeLLM(messages=iter([AIMessage(content="**BTC** is calm.")]))
    real_build = chat.build_chat_agent
    monkeypatch.setattr(chat, "build_chat_agent",
                        lambda **kw: real_build(llm=llm, tools=[], system_prompt="p"))
    rc = chat.main(["-q", "is btc calm?"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "BTC" in out and "is calm" in out
