"""Slack output: mrkdwn conversion, chunking, and two posting paths.

- ``post_via_bot`` - chat.postMessage with the bot token (preferred; threads long
  reports under the first chunk).
- ``post_message`` - legacy incoming webhook, used only when the env var
  SLACK_WEBHOOK_URL is set explicitly.

Nothing here is called unless the caller opts in (``run_signals.py --post-slack``
or the Slack bot itself). ``to_mrkdwn`` / ``chunk_text`` are pure functions shared
with ``slack_bot.py``.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# Slack rejects messages over ~40k chars and truncates the display well before
# that; 3900 keeps every chunk comfortably readable on a phone.
CHUNK_CHARS = 3900
MAX_TEXT_CHARS = 39_000
DEFAULT_TIMEOUT = 15

_FENCE_RE = re.compile(r"^\s*```")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(?=\S)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_TABLE_SEP_RE = re.compile(r"^\s*:?-{2,}:?\s*$")


# ---------------------------------------------------------------------------
# Markdown -> Slack mrkdwn
# ---------------------------------------------------------------------------

def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _split_cells(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells = [c.strip() for c in s.split("|")]
    # Bold / code markers would render literally inside a code block.
    return [re.sub(r"\*\*|`", "", c) for c in cells]


def _render_table(rows: List[str]) -> List[str]:
    """Pipe-table lines -> a ``` code block with space-aligned columns."""
    parsed = [_split_cells(r) for r in rows]
    parsed = [cells for cells in parsed if not all(_TABLE_SEP_RE.match(c or "-") for c in cells)]
    if not parsed:
        return []
    width = max(len(cells) for cells in parsed)
    parsed = [cells + [""] * (width - len(cells)) for cells in parsed]
    col_w = [max(len(row[i]) for row in parsed) for i in range(width)]
    lines = ["```"]
    for row in parsed:
        lines.append("  ".join(cell.ljust(col_w[i]) for i, cell in enumerate(row)).rstrip())
    lines.append("```")
    return lines


def _convert_line(line: str) -> str:
    m = _HEADER_RE.match(line)
    if m:
        title = re.sub(r"\*\*", "", m.group(1)).strip()
        return f"*{title}*"
    line = _BOLD_RE.sub(r"*\1*", line)
    line = _BULLET_RE.sub(r"\1• ", line)
    line = _LINK_RE.sub(r"<\2|\1>", line)
    return line


def to_mrkdwn(text: str) -> str:
    """Best-effort Markdown -> Slack mrkdwn.

    ``**x**`` -> ``*x*``; ``# Heading`` -> ``*Heading*``; ``- ``/``* `` bullets -> ``• ``;
    ``[t](url)`` -> ``<url|t>``; pipe tables -> aligned text inside a ``` block.
    Code fences are passed through untouched.
    """
    if not text:
        return text or ""
    out: List[str] = []
    table: List[str] = []
    in_fence = False

    def flush_table() -> None:
        if table:
            out.extend(_render_table(table))
            table.clear()

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            flush_table()
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if _is_table_row(line):
            table.append(line)
            continue
        flush_table()
        out.append(_convert_line(line))
    flush_table()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, limit: int = CHUNK_CHARS) -> List[str]:
    """Split ``text`` into pieces of at most ``limit`` chars.

    Prefers newline boundaries and keeps ``` fences balanced across pieces
    (closing an open fence at the cut and reopening it in the next piece).
    Always returns at least one element.
    """
    if limit <= 10:
        raise ValueError("limit too small")
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    rest = text
    open_fence = False
    while rest:
        prefix = "```\n" if open_fence else ""
        budget = limit - len(prefix) - (4 if open_fence or "```" in rest else 0)
        if len(rest) <= budget:
            chunks.append(prefix + rest)
            break
        cut = rest.rfind("\n", 0, budget)
        if cut < budget // 2:  # no useful newline: hard cut
            cut = budget
        piece, rest = rest[:cut], rest[cut:].lstrip("\n")
        fences = sum(1 for l in piece.splitlines() if _FENCE_RE.match(l))
        now_open = (fences % 2 == 1) != open_fence
        if now_open:
            piece = piece.rstrip("\n") + "\n```"
        chunks.append(prefix + piece)
        open_fence = now_open
    return chunks or [text]


# ---------------------------------------------------------------------------
# Posting
# ---------------------------------------------------------------------------

def post_via_bot(text: str, channel_id: str, token: str, thread_ts: Optional[str] = None,
                 client=None) -> bool:
    """Post ``text`` to ``channel_id`` with the bot token via chat.postMessage.

    Chunks at CHUNK_CHARS; the first chunk goes to the channel (or the given
    thread) and every further chunk is threaded under it. ``client`` may be any
    object with a ``chat_postMessage(**kwargs)`` method (tests). Never raises.
    """
    if not channel_id or not token:
        logger.error("post_via_bot: channel_id and token are required")
        return False
    if client is None:
        from slack_sdk import WebClient
        client = WebClient(token=token)

    chunks = chunk_text(text)
    ts = thread_ts
    for i, chunk in enumerate(chunks):
        kwargs = {"channel": channel_id, "text": chunk, "mrkdwn": True, "unfurl_links": False}
        if ts:
            kwargs["thread_ts"] = ts
        try:
            resp = client.chat_postMessage(**kwargs)
        except Exception as e:  # noqa: BLE001 - slack_sdk raises SlackApiError on non-ok
            logger.error(f"Slack chat.postMessage failed (chunk {i + 1}/{len(chunks)}): {e}")
            return False
        if i == 0 and ts is None:
            ts = resp.get("ts") if hasattr(resp, "get") else getattr(resp, "data", {}).get("ts")
    logger.info(f"Posted {len(chunks)} Slack message(s) to {channel_id}")
    return True


def _resolve_webhook(webhook_url: Optional[str]) -> Optional[str]:
    if webhook_url:
        return webhook_url
    from utils.config import Config
    return Config().SLACK_WEBHOOK_URL


def post_message(text: str, webhook_url: str | None = None, blocks: list | None = None) -> bool:
    """POST ``text`` (and optional Block Kit ``blocks``) to a Slack incoming webhook.

    The URL comes from the argument, else env SLACK_WEBHOOK_URL (never Secret Manager).
    Returns True on HTTP 2xx, False otherwise. Never raises.
    """
    url = _resolve_webhook(webhook_url)
    if not url:
        logger.error("Slack webhook URL not configured (env SLACK_WEBHOOK_URL)")
        return False

    if len(text) > MAX_TEXT_CHARS:
        text = text[: MAX_TEXT_CHARS - 20] + "\n... (truncated)"

    payload: dict = {"text": text}
    if blocks:
        payload["blocks"] = blocks

    try:
        resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        logger.error(f"Slack webhook request failed: {e}")
        return False

    if 200 <= resp.status_code < 300:
        return True
    logger.error(f"Slack webhook returned {resp.status_code}: {resp.text[:200]}")
    return False
