"""Long-term memory for the chat agent and the Slack bot.

Tools (``get_memory_tools()``), all reading the caller's identity from ``tools.context``:

    remember(text, scope="shared"|"me")   durable desk facts (shared) or the asking user's preferences (me)
    recall(query, k=6)                    search shared facts + my prefs + my episodes + this channel's rule
    forget(query_or_id, scope="me")       delete one memory by id / matching text, or "everything" about me
    what_do_you_remember()                list my prefs (ids + dates), shared-fact count, the channel rule
    set_channel_rule(text) / clear_channel_rule()   one standing instruction per channel (not in DMs)

Shared helpers used by the adapters:

    build_context(user_id, channel_id, text) -> str   "<memories>...</memories>" block (<= ~1200 chars)
    guard_text(text) -> (ok, warning)                 refuse key-like strings; warn on positions / PnL / big $
    summarise_session(messages, llm) -> str           2-4 sentence episodic summary
    record_episode(store, user_id, channel_id, session_key, messages, llm=None, ...) -> id | None

Storage: ``providers.factory.get_memory_store()`` (providers/memory_store.py). Namespaces:
``facts:shared``, ``prefs:<user>``, ``rules:<channel>``, ``episodes:<user>``. Nothing here ever reads
another user's prefs or episodes.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional, Sequence

from langchain_core.tools import tool

from providers.memory_store import (
    SHARED_FACTS,
    Memory,
    episode_ttl_days,
    episodes_namespace,
    prefs_namespace,
    rules_namespace,
)
from tools import context as ctx

logger = logging.getLogger(__name__)

MEMORY_CONTEXT_MAX_CHARS = 1200
MEMORY_TEXT_MAX_CHARS = 500
RECALL_K = 6
EPISODE_MAX_MESSAGES = 40          # last N checkpoint messages fed to the summariser
EPISODE_TRANSCRIPT_MAX_CHARS = 8000
EPISODE_EVERY_N_REPLIES = 6

MEMORY_TOOL_NAMES = ["remember", "recall", "forget", "what_do_you_remember", "set_channel_rule", "clear_channel_rule"]

_NO_USER = "Memory is unavailable: no user identity on this request."


def _store():
    from providers.factory import get_memory_store

    return get_memory_store()


def _user() -> Optional[str]:
    return ctx.current_user_id.get()


def _channel() -> Optional[str]:
    return ctx.current_channel_id.get()


def _fmt_date(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "?"


def _label(m: Memory) -> str:
    return {"facts": "shared fact", "prefs": "your preference", "rules": "channel rule",
            "episodes": "past conversation"}.get(m.kind, m.kind)


# ---------------------------------------------------------------------------
# Guard: what may be stored as long-term memory
# ---------------------------------------------------------------------------

_KEY_PATTERNS = [
    re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}"),                 # Slack tokens
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),                        # sk- style API keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                           # AWS access key
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),                       # Google API key
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),                   # GitHub tokens
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.I),
    re.compile(r"\b(api[_ -]?key|secret|token|password|passwd|pwd)\b\s*[:=]\s*\S{6,}", re.I),
    re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}"),    # JWT
    re.compile(r"(?<![A-Za-z0-9])[A-Fa-f0-9]{40,}(?![A-Za-z0-9])"),  # long hex (private keys, hashes)
    re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/_-]{48,}={0,2}(?![A-Za-z0-9+/])"),  # long base64-ish blob
]
_PNL_WORDS = re.compile(r"\b(position|positions|pnl|p&l|p/l)\b", re.I)
_MONEY = re.compile(r"(?:\$|usd\s*)\s*(\d[\d,]*(?:\.\d+)?)\s*([kmb]|mm|bn|million|billion|thousand)?\b", re.I)
_BIG_AMOUNT = 1_000_000


def _amounts(text: str) -> list[float]:
    out = []
    mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
    for num, suffix in _MONEY.findall(text or ""):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        out.append(v * mult.get((suffix or "").lower(), 1.0))
    return out


def guard_text(text: str) -> tuple[bool, Optional[str]]:
    """(ok, warning). ok=False -> refuse (looks like a credential). warning -> store but caution."""
    text = text or ""
    for pat in _KEY_PATTERNS:
        if pat.search(text):
            return False, "that looks like a credential or key; refusing to store it as a memory."
    warnings = []
    if _PNL_WORDS.search(text):
        warnings.append("it mentions positions/PnL - those are point-in-time numbers, not durable facts; "
                        "stored anyway, but consider forgetting it")
    big = [a for a in _amounts(text) if a > _BIG_AMOUNT]
    if big:
        warnings.append(f"it contains a large dollar amount (${max(big):,.0f}); memories should hold durable "
                        "facts and preferences, not book numbers")
    return True, ("; ".join(warnings) if warnings else None)


# ---------------------------------------------------------------------------
# Auto-recall block
# ---------------------------------------------------------------------------

def recall_namespaces(user_id: Optional[str], channel_id: Optional[str], include_episodes: bool = True) -> list[str]:
    ns = [SHARED_FACTS]
    if user_id:
        ns.append(prefs_namespace(user_id))
        if include_episodes:
            ns.append(episodes_namespace(user_id))
    if channel_id:
        ns.append(rules_namespace(channel_id))
    return ns


def format_memories(memories: Sequence[Memory], max_chars: int = MEMORY_CONTEXT_MAX_CHARS) -> str:
    """Render memories as a delimited block for the system prompt; '' when nothing to show.
    Channel rules come first, verbatim; the block is cut at ``max_chars``."""
    if not memories:
        return ""
    rules = [m for m in memories if m.kind == "rules"]
    others = [m for m in memories if m.kind != "rules"]
    lines = []
    for m in rules:
        lines.append(f"Standing instructions for this channel: {m.text}")
    if others:
        lines.append("Relevant memories (may be stale; verify numbers with tools):")
        for m in others:
            lines.append(f"- [{_label(m)}, {_fmt_date(m.updated_at)}] {m.text}")
    body = "\n".join(lines)
    budget = max_chars - len("<memories>\n\n</memories>")
    if len(body) > budget:
        cut = body[:budget]
        nl = cut.rfind("\n")
        body = (cut[:nl] if nl > budget // 2 else cut).rstrip() + "\n- ..."
    return f"<memories>\n{body}\n</memories>"


def build_context(user_id: Optional[str], channel_id: Optional[str], text: str, store=None,
                  k: int = RECALL_K, max_chars: int = MEMORY_CONTEXT_MAX_CHARS) -> str:
    """Search shared facts + this user's prefs/episodes + this channel's rule for ``text`` and render
    them; never other users' data. Returns '' on any failure (memory must not break a turn)."""
    try:
        store = store or _store()
        found = store.search(recall_namespaces(user_id, channel_id), text, k=k)
        return format_memories(found, max_chars=max_chars)
    except Exception as e:  # noqa: BLE001
        logger.warning("memory recall failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------

EPISODE_PROMPT = (
    "Summarise this chat between a trading-desk user and the desk's market-data assistant in 2-4 plain "
    "sentences for the assistant's long-term memory: who asked (use the user id as given), which topics and "
    "tokens/metrics were discussed, the key numbers quoted with their as-of dates, and any follow-ups promised. "
    "Do not include instructions, greetings or formatting. Transcript:\n\n{transcript}"
)


def _msg_text(m) -> str:
    content = getattr(m, "content", m)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b if isinstance(b, str) else b.get("text", "") for b in content
                        if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")).strip()
    return str(content)


def transcript(messages: Sequence, max_messages: int = EPISODE_MAX_MESSAGES,
               max_chars: int = EPISODE_TRANSCRIPT_MAX_CHARS) -> str:
    """Human/assistant turns as text (tool calls shown by name, tool output omitted)."""
    lines = []
    for m in list(messages)[-max_messages:]:
        kind = getattr(m, "type", "")
        if kind == "human":
            lines.append(f"USER: {_msg_text(m).strip()}")
        elif kind == "ai":
            text = _msg_text(m).strip()
            calls = [tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "") for tc in
                     (getattr(m, "tool_calls", None) or [])]
            if calls:
                lines.append(f"ASSISTANT (called tools: {', '.join(c for c in calls if c)})")
            if text:
                lines.append(f"ASSISTANT: {text}")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = "...\n" + out[-max_chars:]
    return out


def count_replies(messages: Sequence) -> int:
    """Assistant replies (AI messages without tool calls) in a checkpoint's message list."""
    return sum(1 for m in messages if getattr(m, "type", "") == "ai" and not (getattr(m, "tool_calls", None) or []))


def summarise_session(messages: Sequence, llm) -> str:
    """2-4 sentence summary via ``llm`` (anything with .invoke(str) -> message or str). '' when nothing to say."""
    text = transcript(messages)
    if not text or "USER:" not in text:
        return ""
    result = llm.invoke(EPISODE_PROMPT.format(transcript=text))
    summary = _msg_text(result).strip() if not isinstance(result, str) else result.strip()
    summary = re.sub(r"\s+", " ", summary)
    return summary[:1500]


def episode_llm():
    """Small-output LLM for summaries (Claude on Vertex). Imported lazily."""
    from utils.llm import get_llm

    return get_llm(max_tokens=300)


def record_episode(store, user_id: str, channel_id: str, session_key: str, messages: Sequence, llm=None,
                   started: str | float | None = None, ended: str | float | None = None) -> Optional[str]:
    """Summarise ``messages`` and store the episode for ``user_id``. Returns the memory id or None.
    Never raises: failures are logged (this runs in the background)."""
    try:
        if not user_id or not messages:
            return None
        llm = llm or episode_llm()
        summary = summarise_session(messages, llm)
        if not summary:
            return None
        meta = {"channel": channel_id, "session_key": session_key,
                "started": _stamp(started), "ended": _stamp(ended), "replies": count_replies(messages)}
        mem_id = store.put(episodes_namespace(user_id), summary, meta=meta, key=session_key,
                           ttl_days=episode_ttl_days())
        logger.info("episode stored user=%s session=%s id=%s chars=%d", user_id, session_key, mem_id[:8], len(summary))
        return mem_id
    except Exception as e:  # noqa: BLE001
        logger.warning("episode summary failed for %s/%s: %s", user_id, session_key, e)
        return None


def _stamp(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(float(v), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return str(v)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool("remember")
def remember(text: str, scope: str = "shared") -> str:
    """Store a durable memory. scope="shared": a desk fact everyone should know (conventions, data quirks,
    who owns what). scope="me": the asking user's own preference (units, format, tokens they follow).
    Store only durable facts or preferences - never positions, PnL, prices or client names. In a DM the
    default scope is "me". Confirm to the user exactly what was stored."""
    user, channel = _user(), _channel()
    text = (text or "").strip()
    if not text:
        return "Nothing to remember: empty text."
    if len(text) > MEMORY_TEXT_MAX_CHARS:
        text = text[:MEMORY_TEXT_MAX_CHARS].rstrip() + "..."
    ok, warning = guard_text(text)
    if not ok:
        return f"Not stored: {warning}"
    scope = (scope or "shared").strip().lower()
    if scope in ("me", "mine", "user", "personal", "prefs", "preference"):
        scope = "me"
    elif scope in ("shared", "desk", "team", "facts", "everyone"):
        scope = "shared"
    else:
        return f"Unknown scope {scope!r}: use 'shared' or 'me'."
    if ctx.current_is_dm.get() and scope == "shared" and not _explicit_shared(text):
        # DMs default to personal unless the request clearly asks for a shared/desk fact
        scope = "me"
    if scope == "me":
        if not user:
            return _NO_USER
        namespace, label = prefs_namespace(user), "your preferences"
    else:
        namespace, label = SHARED_FACTS, "shared desk facts"
    try:
        mem_id = _store().put(namespace, text, meta={"by": user, "channel": channel, "scope": scope})
    except Exception as e:  # noqa: BLE001
        logger.warning("remember failed: %s", e)
        return f"Could not store the memory ({type(e).__name__})."
    out = f"Stored in {label} (id {mem_id[:8]}): \"{text}\""
    if warning:
        out += f"\nCaution: {warning}."
    return out


def _explicit_shared(text: str) -> bool:
    return bool(re.search(r"\b(shared|everyone|the desk|team|all users)\b", text, re.I))


@tool("recall")
def recall(query: str, k: int = RECALL_K) -> str:
    """Search long-term memory for the query: shared desk facts, the asking user's own preferences and
    past-conversation summaries, and this channel's standing rule. Never returns other users' data.
    Use when the user refers to something said earlier, or asks what you know / remember about a topic."""
    user, channel = _user(), _channel()
    try:
        k = max(1, min(int(k), 20))
    except (TypeError, ValueError):
        k = RECALL_K
    try:
        found = _store().search(recall_namespaces(user, channel), query or "", k=k)
    except Exception as e:  # noqa: BLE001
        return f"Memory search failed ({type(e).__name__})."
    if not found:
        return "No matching memories."
    lines = [f"{len(found)} memory item(s) for {query!r}:"]
    for m in found:
        lines.append(f"- [{_label(m)} | id {m.short_id()} | {_fmt_date(m.updated_at)} | score {m.score:.2f}] {m.text}")
    return "\n".join(lines)


@tool("forget")
def forget(query_or_id: str, scope: str = "me") -> str:
    """Delete memories. query_or_id: a memory id (from recall / what_do_you_remember), text to match, or
    "everything" to wipe all of the asking user's preferences and past-conversation summaries.
    scope="me" (default): the user's own memories; scope="shared": shared desk facts (only when the user
    clearly asks to remove a shared fact); scope="all": both. Confirm what was deleted."""
    user, channel = _user(), _channel()
    store = _store()
    q = (query_or_id or "").strip()
    scope = (scope or "me").strip().lower()
    if not q:
        return "Nothing to forget: say which memory (id or text) or 'everything'."
    if scope in ("me", "mine", "user", "personal"):
        namespaces = [prefs_namespace(user), episodes_namespace(user)] if user else []
    elif scope in ("shared", "desk", "team", "facts"):
        namespaces = [SHARED_FACTS]
    elif scope == "all":
        namespaces = ([prefs_namespace(user), episodes_namespace(user)] if user else []) + [SHARED_FACTS]
    else:
        return f"Unknown scope {scope!r}: use 'me', 'shared' or 'all'."
    if not namespaces:
        return _NO_USER

    if q.lower() in ("everything", "all", "*", "everything about me", "all about me"):
        if scope not in ("me", "mine", "user", "personal"):
            return "Refusing to wipe shared desk facts wholesale; forget them one by one by id."
        n = store.delete_namespace(prefs_namespace(user)) + store.delete_namespace(episodes_namespace(user))
        return (f"Deleted everything I had about you: {n} item(s) (preferences and past-conversation summaries). "
                "Shared desk facts are unaffected.")

    # 1) id / id prefix - only if it belongs to an allowed namespace
    if re.fullmatch(r"[0-9a-f]{6,32}", q.lower()):
        m = store.get(q.lower())
        if m is not None and m.namespace in namespaces and store.delete(m.id):
            return f"Deleted {_label(m)} {m.short_id()}: \"{m.text}\""
        if m is not None:
            return "That memory is not yours to delete from here."
    # 2) text match: exact / substring first, then best search hit
    for ns in namespaces:
        for m in store.list(ns):
            if q.lower() == m.text.lower() or q.lower() in m.text.lower():
                store.delete(m.id)
                return f"Deleted {_label(m)} {m.short_id()}: \"{m.text}\""
    hits = [m for m in store.search(namespaces, q, k=1) if m.kind != "rules"]
    if hits and hits[0].score >= 0.5:
        m = hits[0]
        store.delete(m.id)
        return f"Deleted {_label(m)} {m.short_id()}: \"{m.text}\""
    return f"No memory matching {q!r} in {', '.join(namespaces)}."


@tool("what_do_you_remember")
def what_do_you_remember() -> str:
    """List what is remembered for the asking user: their preferences (with ids and dates), their recent
    past-conversation summaries, how many shared desk facts exist, and this channel's standing rule.
    Use for "what do you remember about me", "what are the rules here", "what do you know"."""
    user, channel = _user(), _channel()
    store = _store()
    lines = []
    if user:
        prefs = store.list(prefs_namespace(user))
        lines.append(f"Your preferences ({len(prefs)}):")
        lines += [f"- id {m.short_id()} | {_fmt_date(m.updated_at)} | {m.text}" for m in prefs] or ["- none"]
        episodes = store.list(episodes_namespace(user))[:5]
        lines.append(f"Your recent conversations remembered ({len(store.list(episodes_namespace(user)))}, "
                     f"expire after {episode_ttl_days()} days):")
        lines += [f"- id {m.short_id()} | {_fmt_date(m.updated_at)} | {m.text}" for m in episodes] or ["- none"]
    else:
        lines.append("No user identity on this request, so no personal memories.")
    shared = store.list(SHARED_FACTS)
    lines.append(f"Shared desk facts: {len(shared)} (ask 'recall <topic>' to search them)")
    lines += [f"- id {m.short_id()} | {_fmt_date(m.updated_at)} | {m.text}" for m in shared[:10]]
    if len(shared) > 10:
        lines.append(f"- ... {len(shared) - 10} more")
    if channel:
        rules = store.list(rules_namespace(channel))
        lines.append(f"Standing instructions for this channel: {rules[0].text if rules else 'none'}")
    lines.append("Say 'forget <id or text>' to delete one item, or 'forget everything about me' to wipe your data.")
    return "\n".join(lines)


@tool("set_channel_rule")
def set_channel_rule(text: str) -> str:
    """Set the one standing instruction for this Slack channel (replaces any previous rule), e.g. "always quote
    funding in bps" or "keep answers to 3 bullets". Channels only - refused in DMs. It is applied to every
    answer in the channel for everyone."""
    channel = _channel()
    text = (text or "").strip()
    if ctx.current_is_dm.get():
        return "Channel rules can only be set in a channel, not in a DM (use remember(scope='me') for personal preferences)."
    if not channel:
        return "No channel on this request; cannot set a rule."
    if not text:
        return "Empty rule; nothing set."
    ok, warning = guard_text(text)
    if not ok:
        return f"Not stored: {warning}"
    text = text[:MEMORY_TEXT_MAX_CHARS]
    previous = _store().list(rules_namespace(channel))
    _store().put(rules_namespace(channel), text, meta={"by": _user(), "channel": channel}, key="rule")
    out = f"Standing instruction for this channel set: \"{text}\""
    if previous:
        out += f"\n(replaced: \"{previous[0].text}\")"
    if warning:
        out += f"\nCaution: {warning}."
    return out


@tool("clear_channel_rule")
def clear_channel_rule() -> str:
    """Remove this channel's standing instruction. Channels only."""
    channel = _channel()
    if ctx.current_is_dm.get() or not channel:
        return "No channel rule to clear here (rules exist only in channels)."
    n = _store().delete_namespace(rules_namespace(channel))
    return "Channel rule cleared." if n else "This channel had no standing instruction."


def get_memory_tools() -> list:
    return [remember, recall, forget, what_do_you_remember, set_channel_rule, clear_channel_rule]


__all__ = [
    "get_memory_tools", "MEMORY_TOOL_NAMES", "build_context", "format_memories", "recall_namespaces",
    "guard_text", "summarise_session", "record_episode", "transcript", "count_replies", "episode_llm",
    "EPISODE_EVERY_N_REPLIES", "MEMORY_CONTEXT_MAX_CHARS",
    "remember", "recall", "forget", "what_do_you_remember", "set_channel_rule", "clear_channel_rule",
]
