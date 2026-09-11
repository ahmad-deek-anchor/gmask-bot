#!/usr/bin/env python
"""Run the z-score signals analysis and print the LLM write-up as markdown.

Examples
--------
    python run_signals.py                       # test universe, 45 days
    python run_signals.py --tokens btc eth sol  # specific tokens (each gets an LLM write-up)
    python run_signals.py --all                 # write up every token in the universe
    python run_signals.py --full-universe --days 60 --verbose
    python run_signals.py --json out/signals.json
    python run_signals.py --post-slack          # also post the write-up to Slack (bot token + channel id, else webhook)
    python run_signals.py --post-slack --slack-channel C0123456789   # override the channel
    python run_signals.py --no-options          # skip the Amberdata options metrics (faster, fewer API calls)

Data: Coin Metrics (spot) + Amberdata (perp derivatives + Deribit options). LLM: Claude on Vertex AI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from agents.events import FinalEvent, TokenEvent
from tools.metrics import FULL_TOKEN_UNIVERSE, INSIGNIFICANT_THRESHOLD, OUTLIER_THRESHOLD, TEST_TOKEN_UNIVERSE, Z_WINDOW
from workflows.signals_workflow import SignalsWorkflow


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical z-score signals with Claude analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    universe = parser.add_mutually_exclusive_group()
    universe.add_argument("--tokens", nargs="+", metavar="TOKEN",
                          help=f"Token symbols to analyze (default: test universe {TEST_TOKEN_UNIVERSE})")
    universe.add_argument("--full-universe", action="store_true",
                          help=f"Analyze the full {len(FULL_TOKEN_UNIVERSE)}-token universe")
    parser.add_argument("--days", type=int, default=45, help="Days of history to fetch (default: 45)")
    parser.add_argument("--all", dest="analyze_all", action="store_true",
                        help="LLM write-up for every token, not only those with |z| >= 1.0 "
                             "(implied by --tokens; use --significant-only to opt out)")
    parser.add_argument("--significant-only", action="store_true",
                        help="With --tokens: only write up tokens with a significant move")
    parser.add_argument("--verbose", "-v", action="store_true", help="Progress output + per-token z-score table")
    parser.add_argument("--post-slack", action="store_true",
                        help="Post the analysis to Slack: bot token + SLACK_CHANNEL_ID if both resolve, "
                             "else env SLACK_WEBHOOK_URL, else skip with a warning")
    parser.add_argument("--slack-channel", metavar="CHANNEL_ID",
                        help="Channel id for --post-slack (overrides SLACK_CHANNEL_ID / its secret)")
    parser.add_argument("--json", metavar="PATH", help="Write signals + analysis to this JSON file")
    parser.add_argument("--no-options", dest="no_options", action="store_true",
                        help="Skip the Amberdata options metrics (DVOL, IV, skew, put/call, options volume)")
    return parser.parse_args(argv)


SLACK_SKIP_WARNING = ("Slack post skipped: set SLACK_CHANNEL_ID (secret trading_signals_slack_channel_id) "
                      "or SLACK_WEBHOOK_URL")


def _slack_header(summary: dict) -> str:
    outliers = ", ".join(t.upper() for t in summary.get("tokens_with_outliers", [])) or "none"
    significant = ", ".join(t.upper() for t in summary.get("tokens_with_significant_moves", [])) or "none"
    return (f"*Statistical Signals - {datetime.now():%Y-%m-%d}*\n"
            f"Tokens analyzed: {summary.get('tokens_analyzed', 0)} • "
            f"Outliers (|z| >= {OUTLIER_THRESHOLD}): {outliers} • "
            f"Significant (|z| >= {INSIGNIFICANT_THRESHOLD}): {significant}\n\n")


def post_to_slack(report_md: str, channel_override: str | None = None, cfg=None) -> int:
    """Post the markdown report to Slack. Returns a process exit code.

    Precedence: bot token + channel id (flag, env SLACK_CHANNEL_ID, or its secret) ->
    env SLACK_WEBHOOK_URL -> skip with a WARNING and exit 0. Secrets are resolved
    lazily and only here; the webhook secret is never consulted.
    """
    from notifiers import slack as notifier
    if cfg is None:
        from utils.config import Config
        cfg = Config()
    logger = logging.getLogger(__name__)
    text = notifier.to_mrkdwn(report_md)

    channel = channel_override or cfg.SLACK_CHANNEL_ID
    token = cfg.SLACK_BOT_TOKEN if channel else None
    if channel and token:
        ok = notifier.post_via_bot(text, channel_id=channel, token=token)
        how = f"channel {channel} via bot"
    else:
        if channel and not token:
            logger.warning("SLACK_CHANNEL_ID set but no bot token resolved; trying the webhook")
        webhook = cfg.SLACK_WEBHOOK_URL
        if not webhook:
            logger.warning(SLACK_SKIP_WARNING)
            print(f"\n{SLACK_SKIP_WARNING}.")
            return 0
        ok = notifier.post_message(text, webhook_url=webhook)
        how = "incoming webhook"
    print(f"\nPosted to Slack ({how})." if ok else f"\nSlack post failed ({how}; see log).")
    return 0 if ok else 2


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "urllib3", "google", "cm_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.tokens:
        tokens, use_test = [t.lower() for t in args.tokens], False
        analyze_all = args.analyze_all or not args.significant_only
    elif args.full_universe:
        tokens, use_test = None, False
        analyze_all = args.analyze_all
    else:
        tokens, use_test = None, True
        analyze_all = args.analyze_all

    workflow = SignalsWorkflow()
    token_events: list[TokenEvent] = []
    final: FinalEvent | None = None

    print(f"# Statistical Signals - {datetime.now():%Y-%m-%d %H:%M}\n")

    try:
        for event in workflow.analyze_stream(
            tokens=tokens, lookback_days=args.days, use_test_universe=use_test,
            verbose=args.verbose, analyze_all=analyze_all, include_options=not args.no_options,
        ):
            if isinstance(event, TokenEvent):
                token_events.append(event)
                print(f"## {event.token.upper()}\n\n{event.analysis_text}\n", flush=True)
            elif isinstance(event, FinalEvent):
                final = event
    except Exception as e:
        logging.getLogger(__name__).error(f"Analysis failed: {type(e).__name__}: {e}")
        return 1

    if final is None:
        print("No result produced.")
        return 1

    summary = final.summary
    print("---")
    print(f"**Tokens analyzed:** {summary.get('tokens_analyzed', 0)}  ")
    print(f"**Outliers (|z| >= {OUTLIER_THRESHOLD}):** {', '.join(summary.get('tokens_with_outliers', [])) or 'none'}  ")
    print(f"**Significant (|z| >= {INSIGNIFICANT_THRESHOLD}):** {', '.join(summary.get('tokens_with_significant_moves', [])) or 'none'}")
    if not token_events:
        print("\n" + SignalsWorkflow.combine_token_events([]))

    if args.verbose:
        workflow._print_detailed_stats(final.all_signals)

    analysis_md = SignalsWorkflow.combine_token_events(token_events)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "lookback_days": args.days,
            "include_options": not args.no_options,
            "lookback_window": Z_WINDOW,
            "outlier_threshold": OUTLIER_THRESHOLD,
            "significant_threshold": INSIGNIFICANT_THRESHOLD,
            "summary": summary,
            "signals": final.all_signals,
            "analysis": analysis_md,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nSignals written to {path}")

    if args.post_slack:
        header = _slack_header(summary)
        return post_to_slack(header + analysis_md, channel_override=args.slack_channel)

    return 0


if __name__ == "__main__":
    sys.exit(main())
