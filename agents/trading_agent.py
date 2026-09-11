"""LangGraph ReAct trading agent over the statistical-signals tools.

The agent runs on Claude via Vertex AI (utils.llm.get_llm) and can call:
  - get_statistical_signals_tool   (tools/signals.py)
  - get_multi_day_signals_tool     (tools/signals.py)
  - get_token_price_history_tool   (tools/metrics.py)

Usage:
    from agents.trading_agent import TradingAgentLangGraph
    print(TradingAgentLangGraph().ask("Which tokens have perp OI outliers today?"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.md"

DEFAULT_SYSTEM_PROMPT = """You are an analyst for a crypto trading team.
Answer questions about spot and perpetual-futures markets using the provided
statistical-signal tools. Base every claim on tool output; be concise and specific."""


def _load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text()
    except FileNotFoundError:
        logger.warning(f"System prompt not found at {SYSTEM_PROMPT_PATH}, using default")
        return DEFAULT_SYSTEM_PROMPT


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content
            if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")
        )
    return str(content)


def build_agent(llm=None, tools: Optional[list] = None, system_prompt: Optional[str] = None):
    """Compile a ReAct agent graph. Injectable `llm`/`tools` keep tests offline."""
    if llm is None:
        from utils.llm import get_llm
        llm = get_llm()
    if tools is None:
        from tools import get_signal_tools
        tools = get_signal_tools()
    if system_prompt is None:
        system_prompt = _load_system_prompt()

    try:
        from langchain.agents import create_agent
        return create_agent(llm, tools=tools, system_prompt=system_prompt)
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(llm, tools=tools, prompt=system_prompt)


class TradingAgentLangGraph:
    """Question-answering agent over z-score signals and price history."""

    def __init__(self, llm=None, tools: Optional[list] = None):
        self._llm = llm
        self._tools = tools
        self.system_prompt = _load_system_prompt()
        self._agent_executor = None
        logger.info("TradingAgentLangGraph initialized")

    @property
    def llm(self):
        if self._llm is None:
            from utils.llm import get_llm
            self._llm = get_llm()
        return self._llm

    def _get_or_create_agent(self):
        if self._agent_executor is None:
            self._agent_executor = build_agent(
                llm=self.llm, tools=self._tools, system_prompt=self.system_prompt
            )
        return self._agent_executor

    def ask(self, question: str, verbose: bool = False) -> str:
        """Run the ReAct loop for one question and return the final answer text."""
        logger.info(f"ask(): {question[:120]}")
        try:
            agent = self._get_or_create_agent()
            state = agent.invoke({"messages": [HumanMessage(content=question)]})
            messages = state["messages"]
            answer = _message_text(messages[-1])

            if verbose:
                tool_calls = sum(1 for m in messages if getattr(m, "type", "") == "tool")
                print(f"DEBUG: {len(messages)} messages, {tool_calls} tool call(s), "
                      f"{len(answer)} chars in answer")
            return answer

        except Exception as e:
            logger.warning(f"Agent execution failed: {type(e).__name__}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return f"Error: {e}"

    def ask_with_data(self, question: str, verbose: bool = False) -> str:
        """Alias for ask() (kept for callers of the previous API)."""
        return self.ask(question, verbose)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Which tokens in the test universe show outlier z-scores today?"
    print(TradingAgentLangGraph().ask(q, verbose=True))
