"""Tests for utils.llm (no network: ChatAnthropicVertex is faked)."""

import pytest

from utils import llm as llm_mod


class FakeChatAnthropicVertex:
    """Records constructor kwargs; never talks to Vertex."""

    created: list["FakeChatAnthropicVertex"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).created.append(self)


@pytest.fixture
def fake_vertex(monkeypatch):
    import langchain_google_vertexai.model_garden as mg

    FakeChatAnthropicVertex.created = []
    monkeypatch.setattr(mg, "ChatAnthropicVertex", FakeChatAnthropicVertex)
    for var in ("VERTEX_PROJECT", "VERTEX_LOCATION", "VERTEX_MODEL"):
        monkeypatch.delenv(var, raising=False)
    llm_mod.reset_llm_cache()
    yield FakeChatAnthropicVertex
    llm_mod.reset_llm_cache()


def test_defaults(fake_vertex):
    llm = llm_mod.get_llm()
    assert isinstance(llm, fake_vertex)
    assert llm.kwargs == {
        "model_name": "claude-sonnet-4-6",
        "project": "anchorage-ai-development",
        "location": "us-east5",
        "temperature": 0.0,
        "max_tokens": 4096,
        "timeout": 120.0,      # LLM_TIMEOUT_S default; anthropic's own default is 10 minutes
        "max_retries": 2,      # langchain attempt count: LLM_MAX_RETRIES (1) + the first try
    }


def test_timeout_and_retries_from_env(monkeypatch, fake_vertex):
    monkeypatch.setenv("LLM_TIMEOUT_S", "45")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    llm = llm_mod.get_llm()
    assert llm.kwargs["timeout"] == 45.0
    assert llm.kwargs["max_retries"] == 1  # zero retries = one attempt


def test_timeout_and_retries_ignore_garbage(monkeypatch, fake_vertex):
    monkeypatch.setenv("LLM_TIMEOUT_S", "soon")
    monkeypatch.setenv("LLM_MAX_RETRIES", "-3")
    llm = llm_mod.get_llm()
    assert llm.kwargs["timeout"] == 120.0
    assert llm.kwargs["max_retries"] == 2


def test_env_overrides(monkeypatch, fake_vertex):
    monkeypatch.setenv("VERTEX_PROJECT", "other-project")
    monkeypatch.setenv("VERTEX_LOCATION", "europe-west1")
    monkeypatch.setenv("VERTEX_MODEL", "claude-opus-4-6")
    llm = llm_mod.get_llm()
    assert llm.kwargs["project"] == "other-project"
    assert llm.kwargs["location"] == "europe-west1"
    assert llm.kwargs["model_name"] == "claude-opus-4-6"


def test_argument_overrides(monkeypatch, fake_vertex):
    monkeypatch.setenv("VERTEX_MODEL", "from-env")
    llm = llm_mod.get_llm(temperature=0.7, max_tokens=512, model="explicit-model")
    assert llm.kwargs["model_name"] == "explicit-model"  # arg beats env
    assert llm.kwargs["temperature"] == 0.7
    assert llm.kwargs["max_tokens"] == 512


def test_caching_identity(fake_vertex):
    a = llm_mod.get_llm()
    b = llm_mod.get_llm()
    c = llm_mod.get_llm(temperature=0.0, max_tokens=4096, model="claude-sonnet-4-6")
    assert a is b is c
    assert len(fake_vertex.created) == 1

    d = llm_mod.get_llm(temperature=0.5)
    e = llm_mod.get_llm(max_tokens=100)
    f = llm_mod.get_llm(model="another")
    assert len({id(x) for x in (a, d, e, f)}) == 4
    assert len(fake_vertex.created) == 4


def test_reset_llm_cache(fake_vertex):
    a = llm_mod.get_llm()
    llm_mod.reset_llm_cache()
    b = llm_mod.get_llm()
    assert a is not b
    assert len(fake_vertex.created) == 2


def test_real_class_accepts_kwargs_offline():
    """The real ChatAnthropicVertex must accept exactly the kwargs get_llm passes
    (constructing it does not hit the network)."""
    llm_mod.reset_llm_cache()
    try:
        llm = llm_mod.get_llm(temperature=0.2, max_tokens=33, model="claude-sonnet-4-6")
    finally:
        llm_mod.reset_llm_cache()
    assert type(llm).__name__ == "ChatAnthropicVertex"
    assert llm.model_name == "claude-sonnet-4-6"
    assert llm.temperature == 0.2
    assert llm.max_output_tokens == 33
    assert llm.timeout == 120.0
    assert llm.max_retries == 2
