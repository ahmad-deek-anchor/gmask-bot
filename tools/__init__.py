"""LangChain tools exposed to the agents.

Import the modules lazily via `get_signal_tools()` so that importing the
`tools` package does not construct a market-data provider.
"""


def get_signal_tools() -> list:
    """All @tool-decorated functions from tools.signals and tools.metrics."""
    from tools.metrics import get_token_price_history_tool
    from tools.signals import get_multi_day_signals_tool, get_statistical_signals_tool

    return [
        get_statistical_signals_tool,
        get_multi_day_signals_tool,
        get_token_price_history_tool,
    ]


__all__ = ["get_signal_tools"]
