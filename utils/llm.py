"""Shared LLM factory: Claude via Google Vertex AI Model Garden.

Usage:
    from utils.llm import get_llm
    llm = get_llm()                       # claude-sonnet-4-6, temperature 0, 4096 max tokens
    llm = get_llm(temperature=0.3, max_tokens=1024, model="claude-sonnet-4-6")

Authentication
--------------
There is NO API key. ``ChatAnthropicVertex`` authenticates with Google
Application Default Credentials (ADC). Locally run::

    gcloud auth application-default login
    gcloud config set project anchorage-ai-development   # optional

and make sure the account has the Vertex AI User role on the project. In
GCP-hosted environments the attached service account is used automatically.
No direct Anthropic API key or SDK client is used; auth is Google ADC only.

Configuration (environment / .env, loaded once at import via python-dotenv)
--------------------------------------------------------------------------
    VERTEX_PROJECT   GCP project hosting the Model Garden endpoint
                     (default: anchorage-ai-development)
    VERTEX_LOCATION  Vertex region (default: us-east5)
    VERTEX_MODEL     Model Garden model name (default: claude-sonnet-4-6)
    LLM_TIMEOUT_S    per-request HTTP timeout in seconds (default: 120; the anthropic
                     client default is 10 minutes, which let a hung call stall the
                     Slack bot for longer than its own 240 s request deadline)
    LLM_MAX_RETRIES  retries after a failed attempt on APIError / timeout / rate limit
                     (default: 1, i.e. two attempts; anthropic's default is 2 retries)

Instances are cached per (model, temperature, max_tokens) so repeated calls
share one client. ``langchain_google_vertexai`` is imported lazily inside
``get_llm`` so importing this module is cheap and does not need GCP.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from utils.net import prefer_ipv4  # noqa: E402

prefer_ipv4()

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

DEFAULT_PROJECT = "anchorage-ai-development"
DEFAULT_LOCATION = "us-east5"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 1

_cache: dict[tuple[str, float, int], "BaseChatModel"] = {}
_cache_lock = threading.Lock()


def vertex_settings() -> dict[str, str]:
    """Resolve project / location / model from the environment with defaults."""
    return {
        "project": os.environ.get("VERTEX_PROJECT") or DEFAULT_PROJECT,
        "location": os.environ.get("VERTEX_LOCATION") or DEFAULT_LOCATION,
        "model": os.environ.get("VERTEX_MODEL") or DEFAULT_MODEL,
    }


def _env_number(name: str, default: float, cast=float, minimum: float = 0.0):
    raw = os.environ.get(name, "")
    try:
        val = cast(raw) if raw.strip() else default
    except ValueError:
        return default
    return val if val >= minimum else default


def llm_timeout_s() -> float:
    """Per-request timeout for the model call (env ``LLM_TIMEOUT_S``, default 120 s)."""
    return _env_number("LLM_TIMEOUT_S", DEFAULT_TIMEOUT_S, float, 1.0)


def llm_max_retries() -> int:
    """Retries after the first failed attempt (env ``LLM_MAX_RETRIES``, default 1)."""
    return _env_number("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES, int, 0)


def get_llm(
    temperature: float = 0.0,
    max_tokens: int = 4096,
    model: str | None = None,
) -> "BaseChatModel":
    """Return a (cached) ``ChatAnthropicVertex`` chat model.

    Args:
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens per response.
        model: Model Garden model name; defaults to ``VERTEX_MODEL`` env or
            ``claude-sonnet-4-6``.

    Auth is Google ADC (see module docstring); no API key is read.
    """
    settings = vertex_settings()
    model_name = model or settings["model"]
    key = (model_name, float(temperature), int(max_tokens))

    with _cache_lock:
        llm = _cache.get(key)
        if llm is not None:
            return llm

    # ADC has no default project unless gcloud config / GOOGLE_CLOUD_PROJECT set one;
    # point it at the Vertex project so google.auth stops warning on every call.
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings["project"])

    # Lazy import: langchain_google_vertexai pulls in google-cloud-aiplatform.
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex

    # Bounded requests: ``timeout`` is passed to the underlying (Async)AnthropicVertex
    # client; ``max_retries`` here is langchain's tenacity ``stop_after_attempt`` count
    # (total attempts, the anthropic client itself is built with max_retries=0), so one
    # retry is max_retries=2. Without these the anthropic defaults apply: a 10-minute
    # timeout and 2 retries - up to 30 minutes for one hung call.
    llm = ChatAnthropicVertex(
        model_name=model_name,
        project=settings["project"],
        location=settings["location"],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=llm_timeout_s(),
        max_retries=llm_max_retries() + 1,
    )

    with _cache_lock:
        return _cache.setdefault(key, llm)


def reset_llm_cache() -> None:
    """Drop all cached LLM instances (for tests / config changes)."""
    with _cache_lock:
        _cache.clear()
