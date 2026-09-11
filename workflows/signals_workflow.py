"""
Statistical Signals Workflow.

Fetches daily market data through the configured provider, computes z-scores
against 30-day rolling medians, and asks Claude (via Vertex AI) to write a
short per-token analysis for every token with a significant move.

Pipeline
--------
1. fetch_token_metrics()            -> raw daily metrics per token
2. calculate_statistical_signals()  -> z-scores, outlier / significant flags
3. per-token LLM call                -> markdown analysis (TokenEvent)
4. FinalEvent with all signals + summary
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.events import FinalEvent, StreamEvent, TokenEvent
from tools.metrics import (
    FULL_TOKEN_UNIVERSE,
    INSIGNIFICANT_THRESHOLD,
    OUTLIER_THRESHOLD,
    TEST_TOKEN_UNIVERSE,
    Z_WINDOW,
    fetch_token_metrics,
)
from tools.signals import LEVEL_CHANGE_DAYS, OPTIONS_Z_METRICS, calculate_statistical_signals

logger = logging.getLogger(__name__)

# Core (spot / perp) metrics that carry a z-score, in display order. Options z-scores
# (tools.signals.OPTIONS_Z_METRICS) are rendered in their own "Options (Deribit)" block.
Z_SCORED_METRICS = ("spot_volume", "perp_volume", "perp_oi", "total_liquidations")

# Display order + unit for the options block. Units: vol points = annualised % IV.
OPTIONS_BLOCK = (
    ("dvol_close", "DVOL (30d implied vol index)", "vol_pts"),
    ("atm_iv_30d", "ATM IV 30d", "vol_pts"),
    ("skew_25d_30d", "25-delta skew 30d", "skew"),
    ("pcr_oi", "Put/call ratio (open interest)", "ratio"),
    ("pcr_volume_24h", "Put/call ratio (24h volume)", "ratio"),
    ("options_notional_volume", "Options notional volume", "usd"),
    ("options_block_notional_volume", "Block-trade notional volume", "usd"),
)

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def _load_prompt(filename: str, fallback: str) -> str:
    path = os.path.join(_PROMPTS_DIR, filename)
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Prompt {path} not found; using built-in fallback")
        return fallback


def _load_analysis_prompt() -> str:
    """Multi-token analysis prompt (whole-universe write-up)."""
    return _load_prompt(
        "statistical_analysis_prompt.md",
        "You are a crypto trading analyst. Analyze the z-score data and identify anomalies.\n\n"
        "For each token with significant moves (|z-score| >= 1.0):\n"
        "- Highlight outliers (|z-score| >= 2.5) with specific attention\n"
        "- Explain what the anomaly could mean\n"
        "- Offer suggestions on potential price implications\n"
        "- Rate your confidence level\n\n"
        "Skip tokens where all metrics have |z-score| < 1.0.\n\n"
        "Output format: Bulleted summary by token.",
    )


def _load_per_token_prompt() -> str:
    """Single-token analysis prompt (used for each TokenEvent)."""
    return _load_prompt(
        "per_token_analysis_prompt.md",
        "You are a crypto trading analyst. Analyze the z-score data for the "
        "single token provided and report key signals, a potential driver, a "
        "price implication, and a confidence level. Do not include a token header.",
    )


def _message_text(response: Any) -> str:
    """Extract plain text from a chat model response (str or content blocks)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class SignalsWorkflow:
    """
    Fetch market data, calculate z-scores, and use the LLM to generate
    trading insights from the statistical anomalies.
    """

    def __init__(self, llm=None):
        self._llm = llm
        logger.info("SignalsWorkflow initialized")

    def _get_llm(self):
        """Lazy-load the Vertex-hosted Claude model."""
        if self._llm is None:
            from utils.llm import get_llm
            self._llm = get_llm()
        return self._llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        tokens: Optional[List[str]] = None,
        lookback_days: int = 45,
        use_test_universe: bool = True,
        verbose: bool = False,
        show_stats: bool = False,
        analyze_all: bool = False,
        include_options: bool = True,
    ) -> Dict[str, Any]:
        """
        Run statistical signal analysis (blocking).

        Drains analyze_stream() and assembles a result dict:
            analysis: markdown, one "## TOKEN" section per analysed token
                      (significant tokens only unless analyze_all=True)
            stats:    analysis_date, thresholds, summary, signals (all tokens)
        """
        token_events: List[TokenEvent] = []
        final_event: Optional[FinalEvent] = None

        for event in self.analyze_stream(
            tokens=tokens,
            lookback_days=lookback_days,
            use_test_universe=use_test_universe,
            verbose=verbose,
            analyze_all=analyze_all,
            include_options=include_options,
        ):
            if isinstance(event, TokenEvent):
                token_events.append(event)
            elif isinstance(event, FinalEvent):
                final_event = event

        if final_event is None:
            return {"analysis": "", "stats": {}}

        if show_stats:
            self._print_detailed_stats(final_event.all_signals)

        stats = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "lookback_window": Z_WINDOW,
            "outlier_threshold": OUTLIER_THRESHOLD,
            "significant_threshold": INSIGNIFICANT_THRESHOLD,
            "summary": final_event.summary,
            "signals": final_event.all_signals,
        }

        return {"analysis": self.combine_token_events(token_events), "stats": stats}

    def analyze_stream(
        self,
        tokens: Optional[List[str]] = None,
        lookback_days: int = 45,
        use_test_universe: bool = True,
        verbose: bool = False,
        analyze_all: bool = False,
        include_options: bool = True,
    ) -> Iterator[StreamEvent]:
        """
        Streaming variant of analyze(). Yields a TokenEvent for each significant
        token as its LLM analysis finishes, then a single FinalEvent.

        With analyze_all=True every fetched token gets an LLM write-up, even when
        all its metrics are within |z| < INSIGNIFICANT_THRESHOLD (useful when the
        caller asked about specific tokens). include_options=False skips the
        Amberdata options provider (no options columns / z-scores / LLM block).

        Fetch and z-score phases happen up-front (synchronous); per-token LLM
        calls happen inside the loop so each TokenEvent arrives as soon as ready.
        """
        if tokens is None:
            tokens = TEST_TOKEN_UNIVERSE if use_test_universe else FULL_TOKEN_UNIVERSE
        tokens = [t.lower() for t in tokens]

        # Phase 1: fetch data (raises on failure - caller catches)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        if verbose:
            print(f"Fetching {len(tokens)} tokens, {lookback_days} days ({start_date:%Y-%m-%d} -> {end_date:%Y-%m-%d})...")
        token_data = fetch_token_metrics(tokens, start_date, end_date, include_options=include_options)

        if not token_data:
            yield FinalEvent(
                all_signals={},
                summary={"tokens_analyzed": 0, "tokens_with_outliers": [], "tokens_with_significant_moves": []},
            )
            return

        # Phase 2: z-scores
        signals = calculate_statistical_signals(token_data, window=Z_WINDOW)
        # Record whether the options feed was consulted at all, so the formatters can
        # tell "options disabled for this run" apart from "token has no listed options".
        for s in signals.values():
            s["options_enabled"] = bool(include_options)
        tokens_with_outliers = [t for t, s in signals.items() if s.get("has_outliers")]
        tokens_with_moves = [t for t, s in signals.items() if s.get("has_significant_moves")]
        if verbose:
            print(f"Signals computed for {len(signals)} tokens; "
                  f"{len(tokens_with_moves)} significant, {len(tokens_with_outliers)} with outliers.")

        # Phase 3: per-token LLM analysis
        to_analyze = list(signals) if analyze_all else tokens_with_moves
        for token in to_analyze:
            token_signals = signals[token]
            analysis_text = self._generate_token_analysis(token, token_signals, verbose=verbose)
            yield TokenEvent(token=token, signals=token_signals, analysis_text=analysis_text)

        # Phase 4: FinalEvent
        yield FinalEvent(
            all_signals=signals,
            summary={
                "tokens_analyzed": len(signals),
                "tokens_with_outliers": tokens_with_outliers,
                "tokens_with_significant_moves": tokens_with_moves,
            },
        )

    @staticmethod
    def combine_token_events(token_events: List[TokenEvent]) -> str:
        """Join per-token analyses into one markdown document."""
        if not token_events:
            return (
                "No significant statistical anomalies detected across the token universe. "
                f"All metrics are within normal ranges (|z-score| < {INSIGNIFICANT_THRESHOLD})."
            )
        return "\n".join(f"## {te.token.upper()}\n\n{te.analysis_text}\n" for te in token_events)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _generate_analysis(self, data: dict, verbose: bool = False) -> str:
        """
        Whole-universe LLM analysis from a signals payload
        ({analysis_date, outlier_threshold, tokens_with_signals}).
        Falls back to a formatted non-LLM summary on failure.
        """
        try:
            llm = self._get_llm()
            user_message = self._format_data_for_llm(data)
            if verbose:
                print(f"   Sending {len(user_message)} chars to LLM...")
            response = llm.invoke([
                SystemMessage(content=_load_analysis_prompt()),
                HumanMessage(content=user_message),
            ])
            return _message_text(response)
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return self._format_fallback_analysis(data)

    def _generate_token_analysis(self, token: str, token_signals: dict, verbose: bool = False) -> str:
        """
        Per-token LLM analysis.

        Returns the LLM text on success, or a string beginning with "<error: ..."
        if the call fails. Never raises - per-token failures must not abort the stream.
        """
        try:
            llm = self._get_llm()
            user_message = self._format_data_for_llm({
                "analysis_date": token_signals.get("latest_date", ""),
                "outlier_threshold": OUTLIER_THRESHOLD,
                "tokens_with_signals": {token: token_signals},
            })
            if verbose:
                print(f"   [{token.upper()}] sending {len(user_message)} chars to LLM...")
            response = llm.invoke([
                SystemMessage(content=_load_per_token_prompt()),
                HumanMessage(content=user_message),
            ])
            return _message_text(response)
        except Exception as e:
            logger.error(f"Error generating per-token analysis for {token}: {e}")
            return f"<error: {type(e).__name__}: {e}>"

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_data_for_llm(self, data: dict) -> str:
        """Format statistical data for LLM consumption."""
        lines = [
            f"# Statistical Signals Report - {data.get('analysis_date', 'Today')}",
            f"\nOutlier threshold: |z-score| >= {data.get('outlier_threshold', OUTLIER_THRESHOLD)}",
            "\n## Token Z-Scores\n",
        ]

        for token, signals in data.get("tokens_with_signals", {}).items():
            lines.append(f"### {token.upper()}")
            lines.append(f"Date: {signals.get('latest_date', 'N/A')}")
            lines.append(f"Has outliers: {signals.get('has_outliers', False)}")
            lines.append("")

            metrics = signals.get("metrics", {})
            for metric in Z_SCORED_METRICS + ("funding_rate", "price"):
                values = metrics.get(metric)
                if not isinstance(values, dict):
                    lines.append(f"- {metric}: not available from the data provider for this token")
                    continue
                z = values.get("z_score")
                if z is not None:
                    flag = " **OUTLIER**" if values.get("is_outlier") else ""
                    lines.append(f"- {metric}: z={z:.2f}{flag} (value: {self._format_value(values.get('value'))})")
                elif metric in Z_SCORED_METRICS:
                    shown = self._format_value(values.get("value"))
                    lines.append(f"- {metric}: no z-score (series constant or too short); latest value {shown}")
                elif metric == "price":
                    price, pct = values.get("value"), values.get("pct_change_1d")
                    if price is not None and pct is not None:
                        lines.append(f"- price: ${price:.2f} ({pct:+.2f}% 1d change)")
                    elif price is not None:
                        lines.append(f"- price: ${price:.2f}")
                elif metric == "funding_rate":
                    annual, h8 = values.get("value_annual_pct"), values.get("value_8h_pct")
                    if annual is not None and h8 is not None:
                        lines.append(f"- funding_rate: {annual:.2f}% annualized ({h8:.4f}% per 8h)")
                    elif annual is not None:
                        lines.append(f"- funding_rate: {annual:.2f}% annualized")

            lines.append("")
            lines.extend(self._format_options_block(signals))
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def _format_options_block(cls, signals: dict) -> List[str]:
        """'Options (Deribit)' lines for one token's signals dict.

        Units: DVOL / IV in vol points (annualised %); skew in vol points, positive =
        puts richer than calls; notional in USD; PCR dimensionless. Metrics without
        data are stated as 'not available' so the model never sees NaN.
        """
        lines = ["**Options (Deribit)**"]
        if signals.get("options_enabled", True) is False:
            lines.append("- options: not fetched (options feed disabled for this run)")
            return lines
        if not signals.get("options_listed", False):
            lines.append("- options: not listed on Deribit (no options metrics for this token)")
            return lines

        metrics = signals.get("metrics", {})
        for metric, label, unit in OPTIONS_BLOCK:
            values = metrics.get(metric)
            if not isinstance(values, dict) or values.get("value") is None:
                lines.append(f"- {metric}: not available")
                continue
            value = values["value"]
            if metric in OPTIONS_Z_METRICS:
                z = values.get("z_score")
                shown = cls._format_options_value(value, unit)
                if z is None:
                    lines.append(f"- {metric} ({label}): {shown}; no z-score (series constant or too short)")
                else:
                    flag = " **OUTLIER**" if values.get("is_outlier") else ""
                    lines.append(f"- {metric} ({label}): z={z:.2f}{flag} (value: {shown})")
            else:  # level metrics: skew_25d_30d, pcr_volume_24h
                change = values.get(f"change_{LEVEL_CHANGE_DAYS}d")
                shown = cls._format_options_value(value, unit)
                chg = (f", change vs {LEVEL_CHANGE_DAYS}d ago: {change:+.2f}" if change is not None
                       else f", no {LEVEL_CHANGE_DAYS}d-ago observation")
                hint = " (positive = puts richer than calls)" if unit == "skew" else ""
                lines.append(f"- {metric} ({label}): {shown}{hint}{chg}; level only, no z-score")
        return lines

    @classmethod
    def _format_options_value(cls, value, unit: str) -> str:
        if unit == "usd":
            # _format_value only adds "$" from $1K upwards; keep the unit explicit for
            # small / zero notional (e.g. no block trades on a USDC-settled alt).
            return cls._format_value(value) if abs(value) >= 1_000 else f"${value:,.2f}"
        if unit == "vol_pts":
            return f"{value:.1f} vol pts"
        if unit == "skew":
            return f"{value:+.2f} vol pts"
        if unit == "ratio":
            return f"{value:.2f}"
        return cls._format_value(value)

    @staticmethod
    def _format_value(val) -> str:
        """Format a numeric value for display."""
        if val is None:
            return "N/A"
        if isinstance(val, (int, float)):
            if abs(val) >= 1_000_000_000:
                return f"${val / 1e9:.2f}B"
            if abs(val) >= 1_000_000:
                return f"${val / 1e6:.2f}M"
            if abs(val) >= 1_000:
                return f"${val / 1e3:.2f}K"
            return f"{val:.2f}"
        return str(val)

    def _format_fallback_analysis(self, data: dict) -> str:
        """Formatted analysis without an LLM (used when the LLM call fails)."""
        lines = [f"# Statistical Signals Report - {data.get('analysis_date', 'Today')}", ""]

        for token, signals in data.get("tokens_with_signals", {}).items():
            if not signals.get("has_significant_moves"):
                continue
            lines.append(f"## {token.upper()}")

            outliers, significant = [], []
            for metric, values in signals.get("metrics", {}).items():
                if not isinstance(values, dict):
                    continue
                z = values.get("z_score")
                if z is None:
                    continue
                if abs(z) >= OUTLIER_THRESHOLD:
                    outliers.append(f"**{metric}**: z={z:.2f} ({'above' if z > 0 else 'below'} normal)")
                elif abs(z) >= INSIGNIFICANT_THRESHOLD:
                    significant.append(f"{metric}: z={z:.2f} ({'elevated' if z > 0 else 'depressed'})")

            if outliers:
                lines.append("**OUTLIERS:**")
                lines.extend(f"  - {o}" for o in outliers)
            if significant:
                lines.append("Significant moves:")
                lines.extend(f"  - {s}" for s in significant)
            lines.append("")

        return "\n".join(lines)

    def _print_detailed_stats(self, signals: Dict[str, Any]) -> None:
        """Print per-token z-score statistics for verification of the LLM's reasoning."""
        print(f"\n{'=' * 80}")
        print("DETAILED STATISTICS")
        print(f"{'=' * 80}\n")

        for token, data in sorted(signals.items()):
            has_outliers = data.get("has_outliers", False)
            has_moves = data.get("has_significant_moves", False)

            status = " | ".join(s for s, on in (("OUTLIER", has_outliers), ("SIGNIFICANT", has_moves)) if on)
            print(f"-- {token.upper()} [{status or 'normal'}]")
            print(f"   Date: {data.get('latest_date', 'N/A')}")

            metrics = data.get("metrics", {})
            for metric_name in Z_SCORED_METRICS:
                m = metrics.get(metric_name, {})
                z = m.get("z_score")
                if z is None:
                    print(f"   {metric_name:20s}: n/a")
                    continue
                if m.get("is_outlier"):
                    flag = " OUTLIER"
                elif m.get("is_significant"):
                    flag = " up" if z > 0 else " down"
                else:
                    flag = ""
                print(f"   {metric_name:20s}: z={z:+6.2f} value={self._format_value(m.get('value')):>12s}{flag}")

            price = metrics.get("price", {})
            if price.get("value") is not None:
                pct = price.get("pct_change_1d")
                suffix = f" ({pct:+.2f}% 1d change)" if pct is not None else ""
                print(f"   {'price':20s}: ${price['value']:.2f}{suffix}")

            fr = metrics.get("funding_rate", {})
            if fr.get("value_annual_pct") is not None:
                h8 = fr.get("value_8h_pct")
                suffix = f" ({h8:.4f}% per 8h)" if h8 is not None else ""
                print(f"   {'funding_rate':20s}: {fr['value_annual_pct']:.2f}% annualized{suffix}")

            if data.get("options_enabled", True) is False:
                print(f"   {'options':30s}: not fetched (--no-options)")
            elif data.get("options_listed"):
                for metric_name, _label, unit in OPTIONS_BLOCK:
                    m = metrics.get(metric_name, {})
                    if m.get("value") is None:
                        print(f"   {metric_name:30s}: n/a")
                        continue
                    shown = self._format_options_value(m["value"], unit)
                    if metric_name in OPTIONS_Z_METRICS:
                        z = m.get("z_score")
                        if z is None:
                            print(f"   {metric_name:30s}: value={shown} (no z)")
                            continue
                        flag = " OUTLIER" if m.get("is_outlier") else (" up" if m.get("is_significant") and z > 0 else " down" if m.get("is_significant") else "")
                        print(f"   {metric_name:30s}: z={z:+6.2f} value={shown}{flag}")
                    else:
                        chg = m.get(f"change_{LEVEL_CHANGE_DAYS}d")
                        suffix = f" ({LEVEL_CHANGE_DAYS}d chg {chg:+.2f})" if chg is not None else ""
                        print(f"   {metric_name:30s}: {shown}{suffix}")
            else:
                print(f"   {'options':30s}: not listed on Deribit")

            print(f"   {'-' * 50}\n")

        print(f"{'=' * 80}\n")


def run_signals_analysis(
    tokens: Optional[List[str]] = None,
    lookback_days: int = 45,
    use_test_universe: bool = True,
    verbose: bool = False,
    show_stats: bool = False,
    include_options: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper: build a SignalsWorkflow and run analyze().

    Returns:
        Dict with 'analysis' (markdown str) and 'stats' (dict).
    """
    return SignalsWorkflow().analyze(
        tokens=tokens,
        lookback_days=lookback_days,
        use_test_universe=use_test_universe,
        verbose=verbose,
        show_stats=show_stats,
        include_options=include_options,
    )
