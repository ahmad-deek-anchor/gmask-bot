#!/usr/bin/env python
"""Terminal chat agent for the Global Markets crypto signals desk.

A LangGraph ReAct agent (Claude on Vertex AI via utils.llm.get_llm) with tools
over the z-score signal pipeline (tools/chat_tools.py) and the desk's BigQuery
data (tools/desk_tools.py: Haruko PnL/greeks/positions, perps, OTC derivatives
trades, Talos orders, internal prices). Conversation history is kept in memory
per session (MemorySaver checkpointer, one thread_id per session).

Usage
-----
    python chat.py                      # interactive REPL
    python chat.py -q "question"        # one-shot: answer and exit
    python chat.py -v                   # INFO logging

REPL commands: /reset (new conversation), /tokens (list universe), /quit | /exit | Ctrl-D.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "chat_assistant_prompt.md"

FALLBACK_SYSTEM_PROMPT = """Today is {today}.
You are the crypto market analyst for the Global Markets desk. Answer questions about
spot and perpetual-futures markets using the provided tools only; never guess data.
State the date range used, quote z-scores with flags (|z|>=2.5 outlier, >=1.0
significant, <1.0 normal), and be concise."""

RECURSION_LIMIT = 40


# ---------------------------------------------------------------------------
# prompt / agent construction
# ---------------------------------------------------------------------------

def load_system_prompt(today: Optional[date] = None, path: Path = SYSTEM_PROMPT_PATH) -> str:
    """Read prompts/chat_assistant_prompt.md and fill the {today} placeholder."""
    try:
        template = path.read_text()
    except FileNotFoundError:
        logger.warning("System prompt not found at %s; using built-in fallback", path)
        template = FALLBACK_SYSTEM_PROMPT
    today = today or date.today()
    return template.replace("{today}", today.strftime("%A %Y-%m-%d"))


def default_tools() -> list:
    """Market-data tools (tools/chat_tools.py) + desk BigQuery tools (tools/desk_tools.py).

    Registering a desk tool never touches BigQuery; the client is built lazily on the
    first call and the tool answers with a clear message when it is unavailable.
    """
    from tools.chat_tools import get_chat_tools
    from tools.desk_tools import get_desk_tools

    return get_chat_tools() + get_desk_tools()


def build_chat_agent(llm=None, tools: Optional[list] = None, system_prompt: Optional[str] = None,
                     checkpointer=None):
    """Compile the ReAct graph. All arguments are injectable so tests stay offline."""
    if llm is None:
        from utils.llm import get_llm
        llm = get_llm(max_tokens=4096)
    if tools is None:
        tools = default_tools()
    if system_prompt is None:
        system_prompt = load_system_prompt()
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    try:
        from langchain.agents import create_agent
        return create_agent(llm, tools=tools, system_prompt=system_prompt, checkpointer=checkpointer)
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(llm, tools=tools, prompt=system_prompt, checkpointer=checkpointer)


def message_text(message) -> str:
    """Plain text from a message whose content may be a string or content blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content
            if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")
        ).strip()
    return str(content)


class ChatSession:
    """One conversation thread over a compiled agent graph."""

    def __init__(self, agent=None, **agent_kwargs):
        self.agent = agent if agent is not None else build_chat_agent(**agent_kwargs)
        self.thread_id = self._new_thread_id()

    @staticmethod
    def _new_thread_id() -> str:
        return uuid.uuid4().hex

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": RECURSION_LIMIT}

    def reset(self) -> None:
        """Start a fresh conversation (new checkpoint thread)."""
        self.thread_id = self._new_thread_id()

    def _history_len(self) -> int:
        try:
            state = self.agent.get_state(self.config)
            return len((state.values or {}).get("messages", []))
        except Exception:  # no checkpointer / first turn
            return 0

    def ask(self, text: str) -> Tuple[str, List[str]]:
        """Run one turn. Returns (final assistant text, tool names called this turn)."""
        before = self._history_len()
        result = self.agent.invoke({"messages": [HumanMessage(content=text)]}, config=self.config)
        messages = result.get("messages", [])
        new = messages[before:] if before <= len(messages) else messages

        tools_called: List[str] = []
        for m in new:
            for tc in getattr(m, "tool_calls", None) or []:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name:
                    tools_called.append(name)

        answer = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "ai" and not (getattr(m, "tool_calls", None) or []):
                answer = message_text(m)
                break
        if not answer and messages:
            answer = message_text(messages[-1])
        return answer, tools_called


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def _print_answer(console, answer: str, tools_called: List[str]) -> None:
    from rich.markdown import Markdown

    if tools_called:
        console.print(f"[dim]tools: {', '.join(tools_called)}[/dim]")
    console.print(Markdown(answer or "_(no answer)_"))


def _print_tokens(console) -> None:
    from tools.metrics import FULL_TOKEN_UNIVERSE, TEST_TOKEN_UNIVERSE

    console.print(f"[bold]Universe ({len(FULL_TOKEN_UNIVERSE)}):[/bold] " + " ".join(FULL_TOKEN_UNIVERSE))
    console.print("[bold]Test set:[/bold] " + " ".join(TEST_TOKEN_UNIVERSE))


def run_turn(console, session: ChatSession, text: str) -> int:
    """Send one message, print the reply. Returns 0 on success, 1 on error."""
    try:
        with console.status("[dim]thinking...[/dim]", spinner="dots"):
            answer, tools_called = session.ask(text)
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
        return 1
    except Exception as e:
        logger.debug("turn failed", exc_info=True)
        console.print(f"[red]error:[/red] {type(e).__name__}: {e}")
        return 1
    _print_answer(console, answer, tools_called)
    return 0


def repl(console, session: ChatSession) -> int:
    console.print(
        "[bold]Global Markets signals chat[/bold] - Claude on Vertex AI. "
        "[dim]/reset  /tokens  /quit[/dim]"
    )
    while True:
        try:
            text = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        cmd = text.lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/reset":
            session.reset()
            console.print("[dim]conversation reset[/dim]")
            continue
        if cmd == "/tokens":
            _print_tokens(console)
            continue
        if cmd.startswith("/"):
            console.print("[dim]commands: /reset  /tokens  /quit[/dim]")
            continue
        run_turn(console, session, text)
    console.print("[dim]bye[/dim]")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal chat over the crypto z-score signal tools.")
    parser.add_argument("-q", "--question", metavar="TEXT", help="Ask one question, print the answer, exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="INFO logging (default WARNING)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "urllib3", "google", "cm_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not args.verbose:
        # "No project ID could be determined" - cosmetic; ADC still works (see README quota-project step)
        logging.getLogger("google.auth._default").setLevel(logging.ERROR)

    from rich.console import Console

    console = Console()
    try:
        with console.status("[dim]connecting to Vertex AI...[/dim]", spinner="dots"):
            session = ChatSession()
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        console.print(f"[red]failed to start agent:[/red] {type(e).__name__}: {e}")
        return 1

    if args.question:
        return run_turn(console, session, args.question)
    try:
        return repl(console, session)
    except KeyboardInterrupt:
        console.print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
