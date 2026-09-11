"""Signal generation.

Consumes token metric DataFrames from tools/metrics.py and turns them into
z-score anomaly signals (outlier / significant / normal per metric).

Z-scored metrics
    spot_volume, perp_volume            30d rolling median, weekend/weekday split
    perp_oi, total_liquidations         30d rolling median, plain window
    OPTIONS_Z_METRICS (dvol_close, atm_iv_30d, pcr_oi, options_notional_volume,
    options_block_notional_volume)      30d rolling median, plain window; only for
                                        tokens whose frame carries options columns
Level-only metrics
    funding_rate                        annualised % (NO z-score)
    OPTIONS_LEVEL_METRICS (skew_25d_30d, pcr_volume_24h)
                                        latest level + change vs 7 days earlier
"""

import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from tools.metrics import (
    FULL_TOKEN_UNIVERSE,
    FUNDING_PERIODS_PER_YEAR,
    INSIGNIFICANT_THRESHOLD,
    OPTIONS_COLUMNS,
    OUTLIER_THRESHOLD,
    TEST_TOKEN_UNIVERSE,
    calculate_zscore_with_weekend_separation,
    fetch_token_metrics,
)

logger = logging.getLogger(__name__)

# Core z-scored metrics
VOLUME_Z_METRICS = ("spot_volume", "perp_volume")          # weekend/weekday split
OTHER_Z_METRICS = ("perp_oi", "total_liquidations")         # plain window

# Options metrics (Deribit via Amberdata). Present only for tokens with listed options.
OPTIONS_Z_METRICS = (
    "dvol_close", "atm_iv_30d", "pcr_oi", "options_notional_volume", "options_block_notional_volume",
)
OPTIONS_LEVEL_METRICS = ("skew_25d_30d", "pcr_volume_24h")  # level + 7d change, no z-score
LEVEL_CHANGE_DAYS = 7

# Every metric that carries a z-score, in display order
Z_SCORED_METRICS = VOLUME_Z_METRICS + OTHER_Z_METRICS + OPTIONS_Z_METRICS

# Minimum non-NaN observations before a metric is z-scored
MIN_Z_OBSERVATIONS = 10


def _z_entry(df: pd.DataFrame, metric: str, window: int, separate_weekends: bool) -> Optional[Dict[str, Any]]:
    """{value, z_score, is_outlier, is_significant} for the last valid observation of
    `metric`, or None when the column is missing / has <= MIN_Z_OBSERVATIONS points."""
    if metric not in df.columns or df[metric].notna().sum() <= MIN_Z_OBSERVATIONS:
        return None
    z_scores = calculate_zscore_with_weekend_separation(
        df[metric], df["time"], window=window, separate_weekends=separate_weekends
    )
    valid_mask = df[metric].notna() & z_scores.notna()
    if valid_mask.sum() > 0:
        last_valid_idx = valid_mask[valid_mask].index[-1]
        latest_z = z_scores.loc[last_valid_idx]
        latest_value = df[metric].loc[last_valid_idx]
    else:
        # No z anywhere (constant series -> zero std, e.g. $0 block notional every day on a
        # USDC-settled alt). Keep the latest observed value so it is reported as a level
        # rather than "not available"; z_score stays None.
        latest_z = np.nan
        observed = df[metric].dropna()
        latest_value = observed.iloc[-1] if len(observed) else np.nan

    is_outlier = bool(abs(latest_z) >= OUTLIER_THRESHOLD) if not pd.isna(latest_z) else False
    is_significant = bool(abs(latest_z) >= INSIGNIFICANT_THRESHOLD) if not pd.isna(latest_z) else False
    return {
        "value": float(latest_value) if not pd.isna(latest_value) else None,
        "z_score": float(latest_z) if not pd.isna(latest_z) else None,
        "is_outlier": is_outlier,
        "is_significant": is_significant,
    }


def _level_entry(df: pd.DataFrame, metric: str, change_days: int = LEVEL_CHANGE_DAYS) -> Optional[Dict[str, Any]]:
    """{value, change_7d} for a level-only metric: last non-NaN value and its change vs
    the last observation at least `change_days` earlier (None if no such observation)."""
    if metric not in df.columns or df[metric].notna().sum() == 0:
        return None
    d = df[["time", metric]].dropna().sort_values("time")
    last = d.iloc[-1]
    ref_time = pd.Timestamp(last["time"]) - timedelta(days=change_days)
    prior = d[d["time"] <= ref_time]
    change = float(last[metric] - prior.iloc[-1][metric]) if len(prior) else None
    return {"value": float(last[metric]), f"change_{change_days}d": change}


def calculate_statistical_signals(
    token_data: Dict[str, pd.DataFrame],
    window: int = 30
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate z-scores for all metrics across all tokens.

    Args:
        token_data: Dict mapping token -> DataFrame with metrics
        window: Rolling window for calculations

    Returns:
        Dict token -> {latest_date, metrics, has_outliers, has_significant_moves,
        options_listed}. `metrics[m]` is {value, z_score, is_outlier, is_significant}
        for every z-scored metric with enough history (incl. OPTIONS_Z_METRICS),
        {value, change_7d} for OPTIONS_LEVEL_METRICS, {value_annual_pct, value_8h_pct}
        for funding_rate and {value, pct_change_1d} for price. `options_listed` is
        True when the frame carries options columns (token has listed options);
        has_outliers / has_significant_moves cover the options z-scores too.
    """
    results = {}

    for token, df in token_data.items():
        if df.empty:
            continue

        options_listed = any(c in df.columns for c in OPTIONS_COLUMNS)
        token_signals = {
            "latest_date": df["time"].iloc[-1].strftime("%Y-%m-%d"),
            "metrics": {},
            "has_outliers": False,
            "has_significant_moves": False,
            "options_listed": options_listed,
        }

        z_plan = [(m, True) for m in VOLUME_Z_METRICS] + [(m, False) for m in OTHER_Z_METRICS]
        if options_listed:
            z_plan += [(m, False) for m in OPTIONS_Z_METRICS]

        for metric, separate_weekends in z_plan:
            entry = _z_entry(df, metric, window, separate_weekends)
            if entry is None:
                continue
            token_signals["metrics"][metric] = entry
            if entry["is_outlier"]:
                token_signals["has_outliers"] = True
            if entry["is_significant"]:
                token_signals["has_significant_moves"] = True

        if options_listed:
            for metric in OPTIONS_LEVEL_METRICS:
                entry = _level_entry(df, metric)
                if entry is not None:
                    token_signals["metrics"][metric] = entry

        # funding_rate is already an annualised percentage (providers/base.py schema,
        # e.g. 10.95 == 10.95 % p.a. == 0.01 % per 8h). Use the last non-NaN observation.
        if "funding_rate" in df.columns and df["funding_rate"].notna().sum() > 5:
            fr = df["funding_rate"].dropna()
            latest_funding = float(fr.iloc[-1])
            token_signals["metrics"]["funding_rate"] = {
                "value_annual_pct": latest_funding,
                "value_8h_pct": latest_funding / FUNDING_PERIODS_PER_YEAR,
            }

        if "price_pct_change" in df.columns:
            latest_pct = df["price_pct_change"].iloc[-1]
            latest_price = df["price"].iloc[-1]
            token_signals["metrics"]["price"] = {
                "value": float(latest_price) if not pd.isna(latest_price) else None,
                "pct_change_1d": float(latest_pct) if not pd.isna(latest_pct) else None,
            }

        results[token] = token_signals

    return results


@tool("get_statistical_signals_tool")
def get_statistical_signals_tool(
    tokens: Optional[List[str]] = None,
    lookback_days: int = 45,
    use_test_universe: bool = True
) -> str:
    """
    Calculate statistical z-score signals for crypto tokens.

    Analyzes spot volume, perp volume, perp OI, and liquidations using
    30-day rolling medians to identify anomalies. Separates weekends from
    weekdays for volume metrics to avoid false positives. For tokens with
    listed options (Deribit: btc, eth, sol, hype) the options metrics are z-scored
    too (dvol_close, atm_iv_30d, pcr_oi, options_notional_volume,
    options_block_notional_volume; IV in vol points, notional in USD) and
    skew_25d_30d / pcr_volume_24h are reported as levels with a 7-day change.

    Args:
        tokens: List of token symbols (default: test universe)
        lookback_days: Days of historical data to fetch (default: 45)
        use_test_universe: If True and no tokens provided, use test universe

    Returns:
        JSON with z-scores and analysis for each token
    """
    try:
        if tokens is None:
            tokens = TEST_TOKEN_UNIVERSE if use_test_universe else FULL_TOKEN_UNIVERSE

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        token_data = fetch_token_metrics(tokens, start_date, end_date)

        if not token_data:
            return json.dumps({"error": "No data available for any tokens"})

        signals = calculate_statistical_signals(token_data, window=30)

        tokens_with_outliers = [t for t, s in signals.items() if s.get("has_outliers")]
        tokens_with_moves = [t for t, s in signals.items() if s.get("has_significant_moves")]

        result = {
            "analysis_date": end_date.strftime("%Y-%m-%d"),
            "lookback_window": 30,
            "outlier_threshold": OUTLIER_THRESHOLD,
            "significant_threshold": INSIGNIFICANT_THRESHOLD,
            "summary": {
                "tokens_analyzed": len(signals),
                "tokens_with_outliers": tokens_with_outliers,
                "tokens_with_significant_moves": tokens_with_moves,
            },
            "signals": signals
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error calculating statistical signals: {e}")
        traceback.print_exc()
        return json.dumps({"error": str(e)})


@tool("get_multi_day_signals_tool")
def get_multi_day_signals_tool(
    tokens: Optional[List[str]] = None,
    days_to_analyze: int = 7,
    use_test_universe: bool = True
) -> str:
    """
    Get statistical signals for multiple days to identify trends.

    Args:
        tokens: List of token symbols
        days_to_analyze: Number of recent days to analyze
        use_test_universe: If True and no tokens provided, use test universe

    Returns:
        JSON with daily z-scores for trend analysis
    """
    try:
        if tokens is None:
            tokens = TEST_TOKEN_UNIVERSE if use_test_universe else FULL_TOKEN_UNIVERSE

        end_date = datetime.now()
        start_date = end_date - timedelta(days=45)

        token_data = fetch_token_metrics(tokens, start_date, end_date)

        if not token_data:
            return json.dumps({"error": "No data available"})

        results = {}

        for token, df in token_data.items():
            if df.empty or len(df) < 10:
                continue

            daily_signals = []
            metrics_to_track = [m for m in Z_SCORED_METRICS if m in df.columns]

            for i in range(-days_to_analyze, 0):
                if abs(i) >= len(df):
                    continue

                row_idx = i
                date = df["time"].iloc[row_idx]

                day_data = {"date": date.strftime("%Y-%m-%d")}

                for metric in metrics_to_track:
                    if metric in df.columns:
                        historical = df.iloc[:row_idx + 1] if row_idx < -1 else df
                        if len(historical) > 10:
                            is_volume = metric in VOLUME_Z_METRICS
                            z_scores = calculate_zscore_with_weekend_separation(
                                historical[metric],
                                historical["time"],
                                window=30,
                                separate_weekends=is_volume
                            )
                            day_data[f"{metric}_z"] = float(z_scores.iloc[-1]) if not pd.isna(z_scores.iloc[-1]) else None

                daily_signals.append(day_data)

            results[token] = daily_signals

        return json.dumps({
            "analysis_date": end_date.strftime("%Y-%m-%d"),
            "days_analyzed": days_to_analyze,
            "daily_signals": results
        }, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error in multi-day analysis: {e}")
        return json.dumps({"error": str(e)})
