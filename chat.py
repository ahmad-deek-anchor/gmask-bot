#!/usr/bin/env python
"""Terminal chat agent for the Global Markets crypto signals desk.

A LangGraph ReAct agent (Claude on Vertex AI via utils.llm.get_llm) with tools
over the z-score signal pipeline (tools/chat_tools.py), the desk's BigQuery
data (tools/desk_tools.py: Haruko PnL/greeks/positions, perps, OTC derivatives
trades, Talos orders, internal prices), the spot desk PnL sheet, long-term memory
(tools/memory_tools.py) and daily data snapshots (tools/snapshot_tools.py, when
present). Conversation history is kept in memory per session (MemorySaver
checkpointer, one thread_id per session).

Long-term memory: before every turn the relevant shared facts, the user's own
preferences / past-conversation summaries and the channel rule are looked up in
providers/memory_store.py and appended to the *system prompt of that model call
only* (a `dynamic_prompt` middleware reading tools.context.current_memory_context),
so the checkpointed user message stays clean. The terminal identity is user
``local`` in channel ``terminal``; `/reset` and `/quit` summarise the session into
an episode.

Usage
-----
    python chat.py                      # interactive REPL
    python chat.py -q "question"        # one-shot: answer and exit
    python chat.py -v                   # INFO logging

REPL commands: /reset (new conversation), /tokens (list universe), /memory (what is remembered),
/quit | /exit | Ctrl-D.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import date, datetime
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
    """Market-data tools (tools/chat_tools.py) + desk BigQuery tools (tools/desk_tools.py)
    + spot desk PnL sheet tools (tools/sheet_tools.py) + long-term memory tools
    (tools/memory_tools.py) + daily snapshot tools (tools/snapshot_tools.py, optional).

    Registering a desk or sheet tool never touches BigQuery / Google Sheets; the clients
    are built lazily on the first call and the tools answer with a clear message when
    they are unavailable. The snapshot tools are skipped when their module is absent.
    """
    from tools.chat_tools import get_chat_tools
    from tools.desk_tools import get_desk_tools
    from tools.memory_tools import get_memory_tools
    from tools.sheet_tools import get_sheet_tools

    tools = get_chat_tools() + get_desk_tools() + get_sheet_tools() + get_memory_tools()
    try:
        from tools.snapshot_tools import get_snapshot_tools
    except ImportError:
        logger.debug("tools.snapshot_tools not available; snapshot tools not registered")
    else:
        tools += get_snapshot_tools()
    return tools


def memory_prompt_middleware(system_prompt: str):
    """`dynamic_prompt` middleware: system prompt + the current turn's <memories> block.

    The block is read from ``tools.context.current_memory_context`` (set by ChatSession.ask /
    slack_bot.answer) and reaches only the model call, never the checkpointed messages.
    Returns None when the middleware API is unavailable (older langchain)."""
    try:
        from langchain.agents.middleware import dynamic_prompt
    except ImportError:  # pragma: no cover - langchain < 1.0
        return None

    from tools.context import current_memory_context

    @dynamic_prompt
    def with_memories(request) -> str:
        block = current_memory_context.get()
        return f"{system_prompt.rstrip()}\n\n{block}" if block else system_prompt

    return with_memories


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
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(llm, tools=tools, prompt=system_prompt, checkpointer=checkpointer)
    middleware = memory_prompt_middleware(system_prompt)
    return create_agent(llm, tools=tools, system_prompt=system_prompt, checkpointer=checkpointer,
                        middleware=[middleware] if middleware is not None else ())


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


TERMINAL_USER = "local"
TERMINAL_CHANNEL = "terminal"


class ChatSession:
    """One conversation thread over a compiled agent graph.

    ``user_id`` / ``channel_id`` identify the caller to the memory tools (contextvars);
    ``memory_store`` defaults to providers.factory.get_memory_store(); ``episode_llm`` (a
    callable returning a small LLM) is used to summarise the session on reset / close.
    """

    def __init__(self, agent=None, user_id: str = TERMINAL_USER, channel_id: str = TERMINAL_CHANNEL,
                 memory_store=None, episode_llm=None, recall: bool = True, **agent_kwargs):
        self.agent = agent if agent is not None else build_chat_agent(**agent_kwargs)
        self.thread_id = self._new_thread_id()
        self.user_id = user_id
        self.channel_id = channel_id
        self._memory_store = memory_store
        self._episode_llm = episode_llm
        self.recall = recall
        self.started = datetime.now()
        self.last_memory_context = ""

    @staticmethod
    def _new_thread_id() -> str:
        return uuid.uuid4().hex

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": RECURSION_LIMIT}

    def reset(self, summarise: bool = False) -> None:
        """Start a fresh conversation (new checkpoint thread); optionally store an episode first."""
        if summarise:
            self.close()
        self.thread_id = self._new_thread_id()
        self.started = datetime.now()

    def _history(self) -> list:
        try:
            state = self.agent.get_state(self.config)
            return list((state.values or {}).get("messages", []))
        except Exception:  # no checkpointer / first turn
            return []

    def _history_len(self) -> int:
        return len(self._history())

    def memory_store(self):
        if self._memory_store is None:
            from providers.factory import get_memory_store

            self._memory_store = get_memory_store()
        return self._memory_store

    def close(self) -> Optional[str]:
        """Summarise this conversation into an episode (episodes:<user>). Returns the memory id or None.
        Nothing is stored when there was no assistant reply; failures are logged, never raised."""
        try:
            from tools.memory_tools import count_replies, record_episode

            messages = self._history()
            if count_replies(messages) == 0:
                return None
            llm = self._episode_llm() if self._episode_llm is not None else None
            return record_episode(self.memory_store(), self.user_id, self.channel_id, self.thread_id, messages,
                                  llm=llm, started=self.started.strftime("%Y-%m-%d %H:%M"),
                                  ended=datetime.now().strftime("%Y-%m-%d %H:%M"))
        except Exception as e:  # noqa: BLE001
            logger.warning("could not store the session summary: %s", e)
            return None

    def ask(self, text: str) -> Tuple[str, List[str]]:
        """Run one turn. Returns (final assistant text, tool names called this turn).

        Relevant memories are looked up first and injected into this turn's system prompt via
        the middleware (tools.context.current_memory_context); the identity contextvars let the
        memory tools know who is asking."""
        from tools.context import request_context
        from tools.memory_tools import build_context

        before = self._history_len()
        block = build_context(self.user_id, self.channel_id, text, store=self.memory_store()) if self.recall else ""
        self.last_memory_context = block
        with request_context(self.user_id, self.channel_id, is_dm=False, memory_context=block):
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


def _print_memory_line(console, session: ChatSession) -> None:
    block = getattr(session, "last_memory_context", "")
    if block:
        n = sum(1 for l in block.splitlines() if l.startswith("- ") or l.startswith("Standing instructions"))
        console.print(f"[dim]recall: {n} memory item(s) injected[/dim]")


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
    _print_memory_line(console, session)
    _print_answer(console, answer, tools_called)
    return 0


def _close_session(console, session: ChatSession) -> None:
    """Store the episodic summary of the conversation (terminal /reset and /quit)."""
    try:
        with console.status("[dim]saving conversation summary...[/dim]", spinner="dots"):
            mem_id = session.close()
    except Exception as e:  # noqa: BLE001
        logger.debug("session close failed", exc_info=True)
        console.print(f"[dim]summary not saved: {type(e).__name__}[/dim]")
        return
    if mem_id:
        console.print(f"[dim]conversation summary saved (episode {mem_id[:8]})[/dim]")


def repl(console, session: ChatSession) -> int:
    console.print(
        "[bold]Global Markets signals chat[/bold] - Claude on Vertex AI. "
        "[dim]/reset  /tokens  /memory  /quit[/dim]"
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
            _close_session(console, session)
            session.reset()
            console.print("[dim]conversation reset[/dim]")
            continue
        if cmd == "/tokens":
            _print_tokens(console)
            continue
        if cmd == "/memory":
            from tools.context import request_context
            from tools.memory_tools import what_do_you_remember

            with request_context(session.user_id, session.channel_id):
                console.print(what_do_you_remember.invoke({}))
            continue
        if cmd.startswith("/"):
            console.print("[dim]commands: /reset  /tokens  /memory  /quit[/dim]")
            continue
        run_turn(console, session, text)
    _close_session(console, session)
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
