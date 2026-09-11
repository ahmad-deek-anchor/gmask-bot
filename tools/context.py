"""Per-request identity for tools, carried in contextvars.

The adapters set these around each agent turn (``slack_bot.handle``: the Slack user and
channel ids; ``chat.py``: user ``local``, channel ``terminal``) and the memory tools read
them, so the LLM never has to pass - or be trusted with - a user id. LangGraph copies the
current ``contextvars`` context into the threads / tasks that run tool and model nodes, and
``asyncio`` tasks inherit a copy at creation, so values set before ``invoke`` / ``ainvoke``
are visible inside the graph.

    from tools.context import request_context
    with request_context(user_id="U123", channel_id="C456", is_dm=False):
        agent.invoke(...)
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional

current_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_user_id", default=None)
current_channel_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_channel_id", default=None)
current_is_dm: contextvars.ContextVar[bool] = contextvars.ContextVar("current_is_dm", default=False)
# Auto-recalled memories for the current turn (read by chat.py's dynamic system prompt middleware).
current_memory_context: contextvars.ContextVar[str] = contextvars.ContextVar("current_memory_context", default="")


@contextmanager
def request_context(user_id: str | None, channel_id: str | None, is_dm: bool = False,
                    memory_context: str | None = None) -> Iterator[None]:
    """Set the identity contextvars for the duration of the block, then restore the old values."""
    tokens = [
        current_user_id.set(user_id),
        current_channel_id.set(channel_id),
        current_is_dm.set(bool(is_dm)),
    ]
    mem_token = current_memory_context.set(memory_context) if memory_context is not None else None
    try:
        yield
    finally:
        if mem_token is not None:
            current_memory_context.reset(mem_token)
        current_is_dm.reset(tokens[2])
        current_channel_id.reset(tokens[1])
        current_user_id.reset(tokens[0])


def snapshot() -> dict:
    """Current identity as a dict (for logging / tests)."""
    return {"user_id": current_user_id.get(), "channel_id": current_channel_id.get(), "is_dm": current_is_dm.get()}


__all__ = ["current_user_id", "current_channel_id", "current_is_dm", "current_memory_context",
           "request_context", "snapshot"]
