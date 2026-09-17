"""Playbooks: named recipes the agent follows when a message asks for them by phrase.

A playbook is a markdown file in ``prompts/playbooks/`` with a small YAML front matter:

    ---
    name: daily_commentary
    description: The desk's daily market note
    triggers: ["daily commentary", "daily market note"]
    min_role: viewer
    ---
    <instructions for the model: which tools to call, in what order, and the output template>

When an incoming message contains one of the trigger phrases (case-insensitive) and the sender's
role is at least ``min_role``, the playbook body is appended to that turn's system prompt (through
``tools.context.current_memory_context``, next to the recalled memories), so the model follows a
fixed recipe instead of improvising. The rest of the time the prompt is unchanged.

Files ship inside the container image like every other prompt (``COPY . .``), so editing a
playbook is a commit and a redeploy. ``PLAYBOOKS_DIR`` overrides the directory.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parent / "prompts" / "playbooks"
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
COMMAND_RE = re.compile(r"^\s*playbooks?\b", re.IGNORECASE)


@dataclass
class Playbook:
    name: str
    description: str
    triggers: list = field(default_factory=list)
    min_role: str = "viewer"
    body: str = ""
    path: Optional[str] = None

    def matches(self, text: str) -> bool:
        """True when a trigger phrase appears as whole words (case- and whitespace-insensitive)."""
        t = " ".join((text or "").lower().split())
        for trig in self.triggers:
            words = " ".join(str(trig).lower().split())
            if words and re.search(r"(?<!\w)" + re.escape(words) + r"(?!\w)", t):
                return True
        return False

    def prompt_block(self) -> str:
        return (f"<playbook name=\"{self.name}\">\nThe user asked for the '{self.name}' playbook. Follow these instructions "
                f"exactly for this reply; they take precedence over the general answer-shape rules but never over the data "
                f"rules (every number from a tool, dates and sources stated).\n\n{self.body.strip()}\n</playbook>")


def parse_playbook(text: str, path: Optional[str] = None) -> Playbook:
    m = FRONT_MATTER_RE.match(text)
    meta: dict = {}
    body = text
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = text[m.end():]
    name = str(meta.get("name") or (Path(path).stem if path else "playbook")).strip()
    triggers = meta.get("triggers") or [name.replace("_", " ")]
    if isinstance(triggers, str):
        triggers = [triggers]
    return Playbook(name=name, description=str(meta.get("description") or ""), triggers=[str(t) for t in triggers],
                    min_role=str(meta.get("min_role") or "viewer"), body=body, path=path)


def load_playbooks(directory: str | Path | None = None) -> list[Playbook]:
    d = Path(directory or os.getenv("PLAYBOOKS_DIR") or DEFAULT_DIR)
    out: list[Playbook] = []
    if not d.exists():
        return out
    for p in sorted(d.glob("*.md")):
        try:
            out.append(parse_playbook(p.read_text(), path=str(p)))
        except Exception as e:  # noqa: BLE001 - one bad file must not take the bot down
            logger.error("playbook %s unreadable: %s", p, e)
    return out


_cache: Optional[list[Playbook]] = None


def playbooks() -> list[Playbook]:
    """Loaded once per process (the files are part of the image)."""
    global _cache
    if _cache is None:
        _cache = load_playbooks()
        logger.info("playbooks loaded: %s", ", ".join(p.name for p in _cache) or "none")
    return _cache


def reset_playbooks() -> None:
    global _cache
    _cache = None


def match(text: str) -> Optional[Playbook]:
    """The first playbook whose trigger phrase appears in ``text``; None for ordinary questions."""
    for p in playbooks():
        if p.matches(text):
            return p
    return None


def is_playbook_command(text: str) -> bool:
    return bool(COMMAND_RE.match(text or ""))


def describe(role: Optional[str] = None) -> str:
    """Slack mrkdwn listing for the `playbooks` command."""
    from access.policy import role_rank
    items = playbooks()
    if not items:
        return "No playbooks are installed."
    lines = ["*Playbooks* - say the trigger phrase in a question to run one:"]
    for p in items:
        ok = role is None or role_rank(role) >= role_rank(p.min_role)
        lines.append(f"• *{p.name}* - {p.description or 'no description'} - say: " + " / ".join(f"`{t}`" for t in p.triggers)
                     + (f" - needs role {p.min_role}" if p.min_role != "viewer" else "") + ("" if ok else " (not available to you)"))
    return "\n".join(lines)


__all__ = ["Playbook", "describe", "is_playbook_command", "load_playbooks", "match", "parse_playbook", "playbooks", "reset_playbooks"]
