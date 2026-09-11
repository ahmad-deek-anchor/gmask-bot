"""Event types yielded by SignalsWorkflow.analyze_stream()."""

from dataclasses import dataclass
from typing import Union


@dataclass
class TokenEvent:
    """Emitted as each significant token finishes LLM analysis."""
    token: str
    signals: dict           # single-token slice: calculate_statistical_signals()[token]
    analysis_text: str      # per-token LLM output; on error, begins with "<error:"


@dataclass
class FinalEvent:
    """Emitted once after all significant tokens have been yielded."""
    all_signals: dict       # full signals dict incl. calm tokens
    summary: dict           # tokens_analyzed, tokens_with_outliers, tokens_with_significant_moves


StreamEvent = Union[TokenEvent, FinalEvent]
