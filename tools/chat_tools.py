"""@tool functions for the terminal chat agent (chat.py).

Thin, LLM-friendly wrappers over the programmatic entry points in
tools.metrics / tools.signals / workflows.signals_workflow. Every tool returns a
compact string (markdown table or small JSON) - never a raw DataFrame - and
validates token symbols against ``tools.metrics.FULL_TOKEN_UNIVERSE``.

Data units (as delivered by the providers, see providers/base.py):
    price               USD (Coin Metrics spot)
    spot_volume         USD / day (Coin Metrics)
    perp_volume         USD / day (Amberdata, summed over major exchanges)
    perp_oi             USD (Amberdata)
    funding_rate        annualised %, USD-margined perps (Amberdata)
    total_liquidations  USD / day (Amberdata)

Options (Amberdata options analytics, Deribit; only tokens with listed options,
see ``options_provider.supported_tokens()`` - btc, eth, sol, hype in the universe as of
Sept 2026; DVOL exists only for btc and eth):
    dvol_close, atm_iv_*            vol points (annualised % implied vol)
    skew_25d_30d                    vol points, put IV - call IV (positive = puts richer)
    pcr_oi, pcr_volume_24h          put/call ratio (dimensionless)
    options_notional_volume,
    options_block_notional_volume   USD / day
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from langchain_core.tools import tool

# Access fetch functions through the module so tests can monkeypatch
# tools.metrics.fetch_token_metrics / fetch_price_history.
from tools import metrics as _metrics
from tools.metrics import (
    FULL_TOKEN_UNIVERSE,
    resolve_token,
    INSIGNIFICANT_THRESHOLD,
    OPTIONS_DAILY_METRICS,
    OPTIONS_EXCHANGE,
    OUTLIER_THRESHOLD,
    TEST_TOKEN_UNIVERSE,
    Z_WINDOW,
    calculate_zscore_with_weekend_separation,
    summarise_gamma_exposure,
)
from tools.signals import LEVEL_CHANGE_DAYS, OPTIONS_LEVEL_METRICS, OPTIONS_Z_METRICS

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_CALL = 10
MAX_DAYS = 120
MIN_DAYS = 7

# Columns shown by get_token_metrics (options columns only when present in the frame)
METRIC_COLUMNS = ["price", "spot_volume", "perp_volume", "perp_oi", "funding_rate", "total_liquidations"] \
    + list(OPTIONS_DAILY_METRICS)
# Rows in get_zscore_signals: core metrics always; options z-metrics when the token has them
Z_METRICS = ["spot_volume", "perp_volume", "perp_oi", "total_liquidations", "funding_rate"]

VOL_PT_METRICS = ("dvol_close", "atm_iv_7d", "atm_iv_30d", "atm_iv_60d", "atm_iv_90d", "atm_iv_180d")
SKEW_METRICS = ("skew_25d_30d", "skew_10d_30d")
RATIO_METRICS = ("pcr_oi", "pcr_volume_24h", "ts_richness")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clamp_days(days, default: int) -> int:
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = default
    return min(max(days, MIN_DAYS), MAX_DAYS)


def _normalise_tokens(tokens) -> List[str]:
    """Accept a list, a comma/space separated string, or a single symbol."""
    if tokens is None:
        return []
    if isinstance(tokens, str):
        tokens = tokens.replace(",", " ").split()
    out: List[str] = []
    for t in tokens:
        t = str(t).strip().lower()
        if t and t not in out:
            out.append(t)
    return out


def validate_tokens(tokens) -> tuple[List[str], List[str], List[str]]:
    """Split requested tokens into (valid, unknown, dropped_over_cap).

    A token is valid when it is in the curated universe or when the dynamic Coin Metrics
    universe knows the asset (``tools.metrics.resolve_token``), so any of the top ~800
    assets by market cap works, not only the curated ~28.
    """
    requested = _normalise_tokens(tokens)
    valid = [t for t in requested if resolve_token(t)]
    unknown = [t for t in requested if t not in valid]
    dropped = valid[MAX_TOKENS_PER_CALL:]
    return valid[:MAX_TOKENS_PER_CALL], unknown, dropped


def _unknown_token_message(token: str) -> str:
    return (f"Unknown token '{token}': Coin Metrics has no asset by that symbol. Check the ticker, or call "
            f"list_top_assets for the top assets by market cap. Curated universe: {', '.join(FULL_TOKEN_UNIVERSE)}.")


def _notes(unknown: List[str], dropped: List[str]) -> List[str]:
    notes = []
    if unknown:
        notes.append(
            f"Unknown token(s) ignored: {', '.join(t.upper() for t in unknown)} "
            f"(no Coin Metrics asset by that symbol; call list_top_assets or list_token_universe)."
        )
    if dropped:
        notes.append(
            f"Token cap is {MAX_TOKENS_PER_CALL} per call; dropped: "
            f"{', '.join(t.upper() for t in dropped)}. Ask again for those separately."
        )
    return notes


def _date_range(days: int) -> tuple[datetime, datetime]:
    end = datetime.now()
    return end - timedelta(days=days), end


def _range_str(start: datetime, end: datetime) -> str:
    return f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"


def _fmt_usd(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    v = float(v)
    a = abs(v)
    if a >= 1e9:
        return f"${v / 1e9:.2f}B"
    if a >= 1e6:
        return f"${v / 1e6:.2f}M"
    if a >= 1e3:
        return f"${v / 1e3:.1f}K"
    return f"${v:.4f}" if a < 10 else f"${v:.2f}"


def _fmt_price(v) -> str:
    """Full-precision USD price (no K/M abbreviation)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    v = float(v)
    if abs(v) >= 1000:
        return f"${v:,.0f}"
    if abs(v) >= 1:
        return f"${v:,.2f}"
    return f"${v:.4f}"


def _fmt_pct(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{float(v):.{digits}f}%"


def _fmt_z(z) -> str:
    if z is None or (isinstance(z, float) and pd.isna(z)):
        return "n/a"
    return f"{float(z):+.2f}"


def _flag(z) -> str:
    if z is None or (isinstance(z, float) and pd.isna(z)):
        return "no data"
    a = abs(float(z))
    if a >= OUTLIER_THRESHOLD:
        return "OUTLIER"
    if a >= INSIGNIFICANT_THRESHOLD:
        return "significant"
    return "normal"


def _fmt_vol_pts(v, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{float(v):+.2f} vol pts" if signed else f"{float(v):.1f} vol pts"


def _fmt_ratio(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{float(v):.2f}"


def _fmt_num(v, digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{float(v):,.{digits}f}"


def _fmt_metric(metric: str, v) -> str:
    if metric == "funding_rate":
        return _fmt_pct(v)
    if metric == "price":
        return _fmt_price(v)
    if metric in VOL_PT_METRICS:
        return _fmt_vol_pts(v)
    if metric in SKEW_METRICS:
        return _fmt_vol_pts(v, signed=True)
    if metric in RATIO_METRICS:
        return _fmt_ratio(v)
    return _fmt_usd(v)


def _options_provider():
    """Options provider or None (lazy; tests monkeypatch providers.factory.get_options_provider)."""
    return _metrics.get_options_provider()


def _options_supported(provider) -> List[str]:
    try:
        return sorted(str(t).lower() for t in (provider.supported_tokens() or []))
    except Exception as e:
        logger.warning("options supported_tokens() failed: %s", e)
        return []


def _options_gate(token: str):
    """Validate `token` for an options tool.

    Returns (tok, provider, None) when the token has listed options, else
    (tok_or_None, provider_or_None, message) where `message` is the reply to return.
    """
    valid, _, _ = validate_tokens([token])
    if not valid:
        return None, None, _unknown_token_message(token)
    tok = valid[0]
    try:
        provider = _options_provider()
    except Exception as e:
        logger.error("options provider failed: %s", e)
        return tok, None, f"Options data unavailable: {type(e).__name__}: {e}"
    if provider is None:
        return tok, None, ("Options data is not configured (no Amberdata options provider / API key). "
                           "Spot and perp tools still work.")
    supported = _options_supported(provider)
    if tok not in supported:
        return tok, provider, (
            f"{tok.upper()} has no listed options on {OPTIONS_EXCHANGE}; options metrics are not "
            f"available for it. Tokens with options: {', '.join(t.upper() for t in supported) or 'none'}."
        )
    return tok, provider, None


def _exchange_name(provider) -> str:
    return str(getattr(provider, "exchange", OPTIONS_EXCHANGE))


def _date(ts) -> str:
    return ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]


def _last_valid(series: pd.Series):
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def _fetch_metrics(tokens: List[str], days: int):
    start, end = _date_range(days)
    return _metrics.fetch_token_metrics(tokens, start, end), start, end


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

@tool("list_token_universe")
def list_token_universe() -> str:
    """List the curated token universe (what the daily reports and snapshots cover) and explain that any Coin Metrics asset can be queried on demand.

    Call this when the user asks which tokens are covered. Symbols are lowercase tickers
    (e.g. 'btc', 'eth', 'sol'). For "what are the top 100 tokens" use list_top_assets.
    """
    return json.dumps({
        "full_universe": FULL_TOKEN_UNIVERSE,
        "count": len(FULL_TOKEN_UNIVERSE),
        "test_universe": TEST_TOKEN_UNIVERSE,
        "max_tokens_per_call": MAX_TOKENS_PER_CALL,
        "on_demand": "Any asset Coin Metrics tracks (~800 with market cap data) works with every price, "
                     "candle, metrics and z-score tool; just pass its ticker. The curated list is what the "
                     "daily snapshot and the full report cover. Use list_top_assets(n) to see the top assets "
                     "by market cap or spot volume.",
    })


@tool("list_top_assets")
def list_top_assets(n: int = 100, by: str = "market_cap", include_stables_and_wrapped: bool = False) -> str:
    """Top-N crypto assets by estimated market cap (default) or 24h trusted spot volume, from Coin Metrics.

    Use for "top 100 tokens", "largest assets", "what are the biggest coins", or to find the
    ticker for an asset before calling a price / metrics tool. Stablecoins, gold tokens and
    wrapped or staked duplicates (USDT, WBTC, stETH ...) are excluded unless asked for.

    Args:
        n: How many assets (default 100, max 300).
        by: 'market_cap' (default) or 'spot_volume'.
        include_stables_and_wrapped: Include stablecoins, gold tokens and wrapped/staked duplicates (default False).

    Returns:
        Markdown table: rank, ticker, market cap, 24h trusted spot volume, and whether the asset is
        in the curated universe (covered by the daily reports). Data is daily, as of the date shown.
    """
    from providers.factory import get_universe

    n = max(1, min(int(n), 300))
    by = "spot_volume" if str(by).lower().startswith("vol") else "market_cap"
    universe = get_universe()
    if universe is None:
        return "The dynamic universe is not available in this environment; curated universe: " + ", ".join(FULL_TOKEN_UNIVERSE)
    try:
        df = universe.top_assets(n=n, by=by, exclude_stables_and_wrapped=not include_stables_and_wrapped,
                                 classifier=_sector_classifier())
    except Exception as e:  # noqa: BLE001
        logger.error("list_top_assets failed: %s", e)
        return f"Error building the asset ranking: {type(e).__name__}: {e}"
    if df.empty:
        return "No ranking data returned by Coin Metrics."
    as_of = df["as_of"].iloc[0]
    label = "estimated market cap" if by == "market_cap" else "24h trusted spot volume"
    with_sectors = "sector" in df.columns
    lines = [f"### Top {len(df)} assets by {label} (Coin Metrics, as of {as_of})",
             "| # | ticker | market cap | 24h spot volume | curated |" + (" sector | sub-sector |" if with_sectors else ""),
             "|---|---|---|---|---|" + ("---|---|" if with_sectors else "")]
    for _, r in df.iterrows():
        line = (f"| {int(r['rank'])} | {r['symbol']} | {_fmt_usd(r['market_cap'])} | {_fmt_usd(r['spot_volume'])} | "
                f"{'yes' if r['symbol'] in FULL_TOKEN_UNIVERSE else ''} |")
        if with_sectors:
            line += f" {_join_list(r.get('sector'))} | {_join_list(r.get('sub_sector'))} |"
        lines.append(line)
    lines.append("_Tickers here work with every price, candle, metrics and z-score tool. 'Curated' marks the tokens the "
                 "daily snapshot and full report cover. Stablecoins and wrapped/staked duplicates excluded unless requested._"
                 + (" _Sector / sub-sector are Messari's taxonomy (blank = not classified)._" if with_sectors else ""))
    return "\n".join(lines)


def _join_list(v) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v if x)
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _sector_classifier():
    """MessariProvider.classify as a ``(symbols, cm_ids) -> frame`` callable, or None without Messari."""
    try:
        from providers.factory import get_messari_provider
        prov = get_messari_provider()
    except Exception as e:  # noqa: BLE001
        logger.debug("Messari classifier unavailable: %s", e)
        return None
    if prov is None:
        return None
    return lambda symbols, cm_ids: prov.classify(symbols, cm_ids=cm_ids)


@tool("get_token_metrics")
def get_token_metrics(token: str, days: int = 45) -> str:
    """Daily market metrics table for one token: price, spot volume, perp volume, perp OI, funding rate, liquidations, plus options metrics (DVOL, ATM IV, skew, put/call ratio, options volume) for tokens with listed options.

    Use this for questions about a token's recent raw data (e.g. "what has SOL
    funding been doing this month", "show me ETH open interest", "how has BTC
    implied vol moved"). Returns the latest values plus a markdown table with one
    row per day.

    Args:
        token: Token symbol, e.g. 'btc'. Must be in the supported universe.
        days: Days of history (default 45, min 7, max 120).

    Returns:
        Markdown: date range, latest values, daily table. Units: USD for
        price/volume/OI/liquidations/options notional; funding_rate is annualised %;
        dvol_close / atm_iv_30d in vol points (annualised % IV); skew_25d_30d in vol
        points (positive = puts richer); pcr_* dimensionless. Options columns only
        appear for tokens with listed options (Deribit).
    """
    days = _clamp_days(days, 45)
    valid, unknown, _ = validate_tokens([token])
    if not valid:
        return _unknown_token_message(token)
    tok = valid[0]

    try:
        data, start, end = _fetch_metrics([tok], days)
    except Exception as e:  # provider/network failure
        logger.error("get_token_metrics(%s) failed: %s", tok, e)
        return f"Error fetching metrics for {tok.upper()}: {type(e).__name__}: {e}"

    df = data.get(tok)
    if df is None or df.empty:
        return f"No data returned for {tok.upper()} over {_range_str(start, end)}."

    cols = [c for c in METRIC_COLUMNS if c in df.columns]
    lines = [
        f"### {tok.upper()} daily metrics ({_range_str(start, end)}, {len(df)} rows)",
        f"Latest row: {_date(df['time'].iloc[-1])}",
        "",
        "**Latest values**",
    ]
    for c in cols:
        lines.append(f"- {c}: {_fmt_metric(c, _last_valid(df[c]))}")
    pct = _last_valid(df["price_pct_change"]) if "price_pct_change" in df.columns else None
    if pct is not None:
        lines.append(f"- price_pct_change_1d: {pct:+.2f}%")
    if not any(c in df.columns for c in OPTIONS_DAILY_METRICS):
        lines.append(f"- options: not listed on {OPTIONS_EXCHANGE} (no options metrics for this token)")
    lines += ["", "| date | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for _, row in df.iterrows():
        lines.append(
            f"| {_date(row['time'])} | "
            + " | ".join(_fmt_metric(c, row[c]) for c in cols)
            + " |"
        )
    return "\n".join(lines)


@tool("get_zscore_signals")
def get_zscore_signals(tokens: List[str], days: int = 45) -> str:
    """Latest z-score anomaly signals for up to 10 tokens across spot volume, perp volume, perp OI, liquidations, funding rate and (for tokens with listed options) DVOL, ATM IV 30d, put/call OI ratio, options notional and block notional volume.

    This is the primary tool for "anomalies", "outliers", "unusual activity",
    "what stands out", "is implied vol unusually high" questions. Each metric's
    latest value is compared with its trailing 30-day rolling median; volume
    metrics compare weekdays with weekdays and weekends with weekends. |z| >= 2.5
    is an OUTLIER, |z| >= 1.0 is significant, |z| < 1.0 is normal. Options rows
    (dvol_close, atm_iv_30d, pcr_oi, options_notional_volume,
    options_block_notional_volume) only appear for tokens with Deribit options;
    25-delta skew and 24h put/call volume ratio are reported as levels with a
    7-day change in the context section (no z-score).

    Args:
        tokens: List of symbols, e.g. ['btc', 'eth', 'sol']. Max 10 per call.
        days: Days of history to fetch for the rolling window (default 45, min 7, max 120).

    Returns:
        Markdown table: token | metric | latest value | z-score | flag, plus a
        summary of tokens with outliers and price / funding context.
    """
    days = _clamp_days(days, 45)
    valid, unknown, dropped = validate_tokens(tokens)
    notes = _notes(unknown, dropped)
    if not valid:
        return "No valid tokens requested. " + " ".join(notes) if notes else \
            "No tokens given. Pass a list such as ['btc', 'eth']."

    try:
        from tools.signals import calculate_statistical_signals
        data, start, end = _fetch_metrics(valid, days)
        signals = calculate_statistical_signals(data, window=Z_WINDOW)
    except Exception as e:
        logger.error("get_zscore_signals(%s) failed: %s", valid, e)
        return f"Error computing z-scores for {', '.join(valid)}: {type(e).__name__}: {e}"

    rows: List[str] = []
    outlier_tokens: List[str] = []
    significant_tokens: List[str] = []
    context: List[str] = []
    missing = [t for t in valid if t not in data]

    for tok in valid:
        df = data.get(tok)
        if df is None or df.empty:
            continue
        sig = signals.get(tok, {})
        metrics = sig.get("metrics", {})
        latest_date = sig.get("latest_date") or _date(df["time"].iloc[-1])
        tok_out = False
        tok_sig = False

        token_metrics = Z_METRICS + [m for m in OPTIONS_Z_METRICS if m in df.columns]
        for m in token_metrics:
            if m == "funding_rate":
                # Funding is not z-scored by calculate_statistical_signals; do it here
                # against the plain rolling window (no weekend split). Values are
                # already annualised % from the provider.
                if m in df.columns and df[m].notna().sum() > 10:
                    z = calculate_zscore_with_weekend_separation(
                        df[m], df["time"], window=Z_WINDOW, separate_weekends=False
                    )
                    valid_mask = df[m].notna() & z.notna()
                    if valid_mask.any():
                        idx = valid_mask[valid_mask].index[-1]
                        zval, val = float(z.loc[idx]), float(df[m].loc[idx])
                    else:
                        zval, val = None, _last_valid(df[m])
                else:
                    zval, val = None, _last_valid(df[m]) if m in df.columns else None
            else:
                entry = metrics.get(m, {})
                zval, val = entry.get("z_score"), entry.get("value")

            flag = _flag(zval)
            if flag == "OUTLIER":
                tok_out = True
            if flag in ("OUTLIER", "significant"):
                tok_sig = True
            rows.append(f"| {tok.upper()} | {m} | {_fmt_metric(m, val)} | {_fmt_z(zval)} | {flag} |")

        if tok_out:
            outlier_tokens.append(tok.upper())
        if tok_sig:
            significant_tokens.append(tok.upper())

        price = _last_valid(df["price"]) if "price" in df.columns else None
        pct = _last_valid(df["price_pct_change"]) if "price_pct_change" in df.columns else None
        ctx = (f"- {tok.upper()} ({latest_date}): price {_fmt_price(price)}"
               + (f" ({pct:+.2f}% 1d)" if pct is not None else ""))
        if sig.get("options_listed"):
            parts = []
            for m in OPTIONS_LEVEL_METRICS:
                entry = metrics.get(m) or {}
                if entry.get("value") is None:
                    parts.append(f"{m} n/a")
                    continue
                chg = entry.get(f"change_{LEVEL_CHANGE_DAYS}d")
                shown = _fmt_metric(m, entry["value"])
                if chg is not None:
                    shown += f" ({LEVEL_CHANGE_DAYS}d chg {chg:+.2f})"
                parts.append(f"{m} {shown}")
            ctx += "; options levels (no z): " + ", ".join(parts)
        else:
            ctx += f"; options: not listed on {OPTIONS_EXCHANGE}"
        context.append(ctx)

    lines = [
        f"### Z-score signals ({_range_str(start, end)}; window {Z_WINDOW}d rolling median; "
        f"outlier |z|>={OUTLIER_THRESHOLD}, significant |z|>={INSIGNIFICANT_THRESHOLD})",
        "",
        "| token | metric | latest value | z-score | flag |",
        "|---|---|---|---|---|",
        *rows,
        "",
        f"**Outliers:** {', '.join(outlier_tokens) or 'none'}  ",
        f"**Significant (incl. outliers):** {', '.join(significant_tokens) or 'none'}",
        "",
        "**Price context**",
        *context,
    ]
    if missing:
        notes.append(f"No data returned for: {', '.join(t.upper() for t in missing)}.")
    if notes:
        lines += ["", "**Notes**"] + [f"- {n}" for n in notes]
    return "\n".join(lines)


@tool("get_price_history")
def get_price_history(token: str, days: int = 30) -> str:
    """Daily spot price history (Coin Metrics) for one token with a summary.

    Use for "price over the last N days", "how has X performed", high/low, or
    percentage change questions.

    Args:
        token: Token symbol, e.g. 'btc'.
        days: Days of history (default 30, min 7, max 120).

    Returns:
        Markdown: date range, summary (latest, high, low, period change %) and a
        daily table of price and 1-day % change.
    """
    days = _clamp_days(days, 30)
    valid, _, _ = validate_tokens([token])
    if not valid:
        return _unknown_token_message(token)
    tok = valid[0]

    try:
        df = _metrics.fetch_price_history(tok, days=days)
    except Exception as e:
        logger.error("get_price_history(%s) failed: %s", tok, e)
        return f"Error fetching price history for {tok.upper()}: {type(e).__name__}: {e}"

    start, end = _date_range(days)
    if df is None or df.empty:
        return f"No price data for {tok.upper()} over {_range_str(start, end)}."

    df = df.copy()
    df["chg"] = df["price"].pct_change() * 100
    prices = df["price"].dropna()
    first, last = float(prices.iloc[0]), float(prices.iloc[-1])
    change = (last / first - 1) * 100 if first else None

    lines = [
        f"### {tok.upper()} spot price, {_date(df['time'].iloc[0])} to {_date(df['time'].iloc[-1])} ({len(df)} days)",
        f"- latest: {_fmt_price(last)} ({_date(df['time'].iloc[-1])})",
        f"- high: {_fmt_price(prices.max())}  low: {_fmt_price(prices.min())}",
        f"- period change: {change:+.2f}%" if change is not None else "- period change: n/a",
        "",
        "| date | price | 1d % |",
        "|---|---|---|",
    ]
    for _, row in df.iterrows():
        chg = row["chg"]
        lines.append(
            f"| {_date(row['time'])} | {_fmt_price(row['price'])} | "
            f"{'n/a' if pd.isna(chg) else f'{chg:+.2f}%'} |"
        )
    return "\n".join(lines)


@tool("run_full_signals_analysis")
def run_full_signals_analysis(tokens: Optional[List[str]] = None, days: int = 45) -> str:
    """Run the full signals pipeline and return Claude's written per-token analysis.

    Slow (one extra LLM call per token with a significant move). Use only when
    the user explicitly wants the desk's full written report / narrative for a
    set of tokens; for quick questions prefer get_zscore_signals.

    Args:
        tokens: Symbols to analyse (max 10). Defaults to the test universe
            (btc, eth, sol, sui, hype, uni, jto) when omitted.
        days: Days of history to fetch (default 45).

    Returns:
        Markdown report: one section per significant token, then a summary of
        tokens analysed / with outliers / with significant moves.
    """
    days = _clamp_days(days, 45)
    if tokens:
        valid, unknown, dropped = validate_tokens(tokens)
        notes = _notes(unknown, dropped)
        if not valid:
            return "No valid tokens requested. " + " ".join(notes)
    else:
        valid, notes = list(TEST_TOKEN_UNIVERSE), []

    try:
        from workflows.signals_workflow import run_signals_analysis
        result = run_signals_analysis(tokens=valid, lookback_days=days)
    except Exception as e:
        logger.error("run_full_signals_analysis(%s) failed: %s", valid, e)
        return f"Error running signals analysis: {type(e).__name__}: {e}"

    start, end = _date_range(days)
    summary = (result.get("stats") or {}).get("summary", {})
    lines = [
        f"### Signals analysis for {', '.join(t.upper() for t in valid)} ({_range_str(start, end)})",
        "",
        result.get("analysis", "").strip() or "(no analysis text)",
        "",
        "---",
        f"Tokens analysed: {summary.get('tokens_analyzed', len(valid))}  ",
        f"Outliers: {', '.join(t.upper() for t in summary.get('tokens_with_outliers', [])) or 'none'}  ",
        f"Significant: {', '.join(t.upper() for t in summary.get('tokens_with_significant_moves', [])) or 'none'}",
    ]
    if notes:
        lines += ["", "**Notes**"] + [f"- {n}" for n in notes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# options tools (Amberdata options analytics, Deribit)
# ---------------------------------------------------------------------------

def _term_structure_lines(rows) -> List[str]:
    if not rows:
        return ["- term structure: not available"]
    lines = ["| days to expiry | ATM IV | fwd ATM IV |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {_fmt_num(r.get('days_to_expiration'))} | {_fmt_vol_pts(r.get('atm_iv'))} | "
                     f"{_fmt_vol_pts(r.get('fwd_atm_iv'))} |")
    ivs = [r["atm_iv"] for r in rows if r.get("atm_iv") is not None]
    if len(ivs) >= 2:
        shape = "contango (longer-dated IV above short-dated)" if ivs[-1] > ivs[0] else \
            "inverted / backwardation (short-dated IV above longer-dated)" if ivs[-1] < ivs[0] else "flat"
        lines.append(f"Shape: {shape}; front {_fmt_vol_pts(ivs[0])} vs back {_fmt_vol_pts(ivs[-1])}.")
    return lines


def _skew_lines(rows) -> List[str]:
    if not rows:
        return ["- skew by tenor: not available"]
    lines = ["| days to expiry | ATM IV | 25d put IV | 25d call IV | 25d skew | 10d skew |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {_fmt_num(r.get('days_to_expiration'))} | {_fmt_vol_pts(r.get('atm_iv'))} | "
            f"{_fmt_vol_pts(r.get('iv_put_25d'))} | {_fmt_vol_pts(r.get('iv_call_25d'))} | "
            f"{_fmt_vol_pts(r.get('skew_25d'), signed=True)} | {_fmt_vol_pts(r.get('skew_10d'), signed=True)} |"
        )
    lines.append("Skew = put IV - call IV in vol points; positive = puts richer (downside protection bid).")
    return lines


def _gamma_lines(gamma) -> List[str]:
    if not gamma:
        return ["- gamma exposure: not available"]
    lines = []
    head = []
    if gamma.get("snapshot_time"):
        head.append(f"snapshot {gamma['snapshot_time']}")
    if gamma.get("index_price") is not None:
        head.append(f"index {_fmt_price(gamma['index_price'])}")
    head.append(f"total net dealer gamma {_fmt_num(gamma.get('total_net_gamma'), 2)}")
    head.append(f"{gamma.get('n_strikes', 0)} strikes")
    lines.append("- " + "; ".join(head))
    fp = gamma.get("flip_point")
    lines.append(f"- gamma flip (net dealer gamma changes sign): {_fmt_price(fp) if fp is not None else 'not derivable (no sign change)'}")
    strikes = gamma.get("top_strikes") or []
    if strikes:
        lines += ["", "| strike | net dealer gamma | total dealer gamma |", "|---|---|---|"]
        for r in strikes:
            lines.append(f"| {_fmt_price(r['strike'])} | {_fmt_num(r['net_dealer_gamma'], 2)} | "
                         f"{_fmt_num(r.get('total_dealer_gamma'), 2)} |")
    lines.append("Positive net dealer gamma: dealers dampen moves near that strike; negative: dealers hedge with the move (amplifies).")
    return lines


def _block_lines(rows) -> List[str]:
    if not rows:
        return ["- block trades: none reported in the window"]
    lines = ["| expiry | strike | P/C | contracts | premium (USD) |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.get('expiry') or 'n/a'} | {_fmt_price(r.get('strike'))} | {r.get('put_call') or 'n/a'} | "
                     f"{_fmt_num(r.get('contract_volume'), 1)} | {_fmt_usd(r.get('premium_volume'))} |")
    return lines


@tool("get_options_snapshot")
def get_options_snapshot(token: str) -> str:
    """Current options picture for one token (Deribit via Amberdata): ATM IV term structure, skew by tenor, latest DVOL, put/call ratios, dealer gamma exposure by strike and the biggest block trades of the last 7 days.

    Use for "options snapshot", "how is vol priced", "term structure and skew",
    "what are the big options trades", "where is the gamma" questions. Only tokens
    with listed options (btc, eth, sol, hype on Deribit) have data; the tool says so
    otherwise. For a z-score view of IV vs its history use get_zscore_signals.

    Args:
        token: Token symbol, e.g. 'btc'.

    Returns:
        Compact markdown. Units: IV / DVOL in vol points (annualised %), skew in
        vol points (put IV - call IV, positive = puts richer), premium in USD.
    """
    tok, provider, msg = _options_gate(token)
    if msg:
        return msg
    try:
        snap = _metrics.fetch_options_snapshot(tok, provider=provider)
    except Exception as e:
        logger.error("get_options_snapshot(%s) failed: %s", tok, e)
        return f"Error fetching options snapshot for {tok.upper()}: {type(e).__name__}: {e}"

    ex = snap.get("exchange") or _exchange_name(provider)
    lines = [f"### {tok.upper()} options snapshot ({ex}, {datetime.now():%Y-%m-%d})"]
    if snap.get("currency"):
        lines.append(f"Underlying: {snap['currency']}")
    lines += ["", "**DVOL / put-call**"]
    dvol = snap.get("dvol")
    if dvol:
        chg = dvol.get("change_7d")
        lines.append(f"- DVOL close: {_fmt_vol_pts(dvol['value'])} ({dvol['time']})"
                     + (f", {chg:+.1f} vol pts vs 7d ago" if chg is not None else ""))
    else:
        lines.append("- DVOL: not available")
    pcr = snap.get("pcr")
    if pcr:
        chg = pcr.get("pcr_oi_change_7d")
        lines.append(f"- put/call OI ratio: {_fmt_ratio(pcr.get('pcr_oi'))}"
                     + (f" ({chg:+.2f} vs 7d ago)" if chg is not None else "")
                     + f"; put/call 24h volume ratio: {_fmt_ratio(pcr.get('pcr_volume_24h'))} ({pcr['time']})")
    else:
        lines.append("- put/call ratio: not available")
    lines += ["", "**ATM IV term structure**", *_term_structure_lines(snap.get("term_structure"))]
    lines += ["", "**Skew by tenor (delta surface)**", *_skew_lines(snap.get("delta_surface"))]
    lines += ["", "**Dealer gamma exposure**", *_gamma_lines(snap.get("gamma"))]
    lines += ["", "**Top block trades (last 7 days, by premium)**", *_block_lines(snap.get("block_trades"))]
    if snap.get("errors"):
        lines += ["", "**Notes**"] + [f"- {e}" for e in snap["errors"]]
    return "\n".join(lines)


@tool("get_vol_term_structure")
def get_vol_term_structure(token: str, exchange: str = "deribit") -> str:
    """ATM implied-vol term structure for one token: IV and forward IV per expiry bucket (snapshot) plus the latest constant-maturity ATM IVs (7d/30d/60d/90d/180d) and term-structure richness.

    Use for "term structure", "is the curve inverted", "front vs back-end vol",
    "30d vs 90d IV" questions. Only tokens with listed options (btc, eth, sol, hype).

    Args:
        token: Token symbol, e.g. 'eth'.
        exchange: Options exchange (default 'deribit'; the only one configured).

    Returns:
        Markdown tables. IV in vol points (annualised %).
    """
    tok, provider, msg = _options_gate(token)
    if msg:
        return msg
    notes: List[str] = []
    configured = _exchange_name(provider)
    if exchange and exchange.lower() != configured.lower():
        notes.append(f"Requested exchange '{exchange}' is not configured; showing {configured}.")

    try:
        ts = provider.get_term_structure(tok)
    except Exception as e:
        logger.error("get_vol_term_structure(%s) failed: %s", tok, e)
        return f"Error fetching term structure for {tok.upper()}: {type(e).__name__}: {e}"
    rows = _metrics._records(ts, ["days_to_expiration", "atm_iv", "fwd_atm_iv"])

    hist_rows = None
    try:
        end = datetime.now()
        hist = provider.get_term_structure_history(tok, end - timedelta(days=10), end)
        if hist is not None and not hist.empty:
            hist_rows = _metrics._records(
                hist.sort_values("time").tail(1),
                ["time", "atm_iv_7d", "atm_iv_30d", "atm_iv_60d", "atm_iv_90d", "atm_iv_180d", "ts_richness"],
            )
    except Exception as e:
        notes.append(f"constant-maturity history unavailable: {type(e).__name__}: {e}")

    lines = [f"### {tok.upper()} ATM IV term structure ({configured}, {datetime.now():%Y-%m-%d})", "",
             "**By expiry (snapshot)**", *_term_structure_lines(rows)]
    lines += ["", "**Constant-maturity ATM IV (latest daily)**"]
    if hist_rows:
        r = hist_rows[0]
        lines.append(f"- as of {r.get('time')}: 7d {_fmt_vol_pts(r.get('atm_iv_7d'))}, 30d {_fmt_vol_pts(r.get('atm_iv_30d'))}, "
                     f"60d {_fmt_vol_pts(r.get('atm_iv_60d'))}, 90d {_fmt_vol_pts(r.get('atm_iv_90d'))}, "
                     f"180d {_fmt_vol_pts(r.get('atm_iv_180d'))}; richness {_fmt_ratio(r.get('ts_richness'))}")
    else:
        lines.append("- not available")
    if notes:
        lines += ["", "**Notes**"] + [f"- {n}" for n in notes]
    return "\n".join(lines)


@tool("get_options_flow")
def get_options_flow(token: str, days: int = 7) -> str:
    """Daily options flow for one token over the last N days: contract, notional, premium and block-trade notional volume plus put/call ratios (open interest and 24h volume).

    Use for "options volume", "block flow", "is options activity picking up",
    "put/call ratio trend" questions. Only tokens with listed options (btc, eth, sol, hype).

    Args:
        token: Token symbol, e.g. 'btc'.
        days: Days of history (default 7, min 1, max 45). Each day is one upstream call.

    Returns:
        Markdown table (one row per day) with period totals. Notional / premium in USD.
    """
    tok, provider, msg = _options_gate(token)
    if msg:
        return msg
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 7
    days = min(max(days, 1), 45)
    end = datetime.now()
    start = end - timedelta(days=days)

    try:
        vol = provider.get_options_volume(tok, start, end)
        pcr = provider.get_put_call_ratio(tok, start, end)
    except Exception as e:
        logger.error("get_options_flow(%s) failed: %s", tok, e)
        return f"Error fetching options flow for {tok.upper()}: {type(e).__name__}: {e}"

    frames = [f for f in (vol, pcr) if f is not None and not f.empty and "time" in f.columns]
    if not frames:
        return f"No options flow data for {tok.upper()} over {_range_str(start, end)}."
    df = frames[0].copy()
    df["time"] = pd.to_datetime(df["time"])
    for other in frames[1:]:
        other = other.copy()
        other["time"] = pd.to_datetime(other["time"])
        df = df.merge(other, on="time", how="outer")
    df = df.sort_values("time").reset_index(drop=True)

    cols = [c for c in ("options_contract_volume", "options_notional_volume", "options_premium_volume",
                        "options_block_notional_volume", "pcr_oi", "pcr_volume_24h") if c in df.columns]
    labels = {
        "options_contract_volume": "contracts", "options_notional_volume": "notional",
        "options_premium_volume": "premium", "options_block_notional_volume": "block notional",
        "pcr_oi": "P/C OI", "pcr_volume_24h": "P/C vol 24h",
    }

    def _cell(c, v):
        if c == "options_contract_volume":
            return _fmt_num(v, 1)
        return _fmt_metric(c, v)

    lines = [f"### {tok.upper()} options flow ({_exchange_name(provider)}, {_range_str(start, end)}, {len(df)} days)", ""]
    lines += ["| date | " + " | ".join(labels[c] for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for _, row in df.iterrows():
        lines.append(f"| {_date(row['time'])} | " + " | ".join(_cell(c, row[c]) for c in cols) + " |")
    lines.append("")
    totals = []
    for c in ("options_notional_volume", "options_premium_volume", "options_block_notional_volume"):
        if c in df.columns and df[c].notna().any():
            totals.append(f"{labels[c]} {_fmt_usd(df[c].sum())}")
    if totals:
        lines.append("**Period totals:** " + ", ".join(totals))
    if "options_block_notional_volume" in df.columns and "options_notional_volume" in df.columns:
        bn, tn = df["options_block_notional_volume"].sum(), df["options_notional_volume"].sum()
        if tn and not pd.isna(tn) and tn > 0:
            lines.append(f"**Block share of notional:** {bn / tn * 100:.1f}%")
    for c in ("pcr_oi", "pcr_volume_24h"):
        if c in df.columns and df[c].notna().any():
            lines.append(f"**Latest {labels[c]}:** {_fmt_ratio(_last_valid(df[c]))}")
    lines.append("Units: notional / premium / block notional in USD per day; P/C ratios dimensionless (>1 = more puts).")
    return "\n".join(lines)


@tool("get_gamma_exposure")
def get_gamma_exposure(token: str) -> str:
    """Dealer gamma exposure by strike for one token (latest Deribit snapshot): the largest positive / negative net-gamma strikes, the gamma flip level and the index price.

    Use for "where is the gamma", "gamma flip", "pinning strikes", "dealer
    positioning" questions. Only tokens with listed options (btc, eth, sol, hype).

    Args:
        token: Token symbol, e.g. 'btc'.

    Returns:
        Markdown: snapshot time, index price, total net dealer gamma, flip point,
        table of the top strikes by |net dealer gamma|.
    """
    tok, provider, msg = _options_gate(token)
    if msg:
        return msg
    try:
        gex = provider.get_gamma_exposure(tok)
    except Exception as e:
        logger.error("get_gamma_exposure(%s) failed: %s", tok, e)
        return f"Error fetching gamma exposure for {tok.upper()}: {type(e).__name__}: {e}"
    gamma = summarise_gamma_exposure(gex, top_n=5)
    lines = [f"### {tok.upper()} dealer gamma exposure ({_exchange_name(provider)})", ""]
    if gamma is None:
        lines.append("- gamma exposure: not available (no snapshot returned)")
        return "\n".join(lines)
    lines += _gamma_lines(gamma)
    return "\n".join(lines)


def get_chat_tools() -> list:
    """Tools exposed to the chat agent.

    The five core chat tools, ``get_multi_day_signals_tool`` from tools.signals
    (daily z-score trend) and the four options tools. The other two signal tools
    duplicate get_zscore_signals / get_price_history and are omitted.
    """
    from tools.signals import get_multi_day_signals_tool

    return [
        list_token_universe,
        list_top_assets,
        get_token_metrics,
        get_zscore_signals,
        get_price_history,
        run_full_signals_analysis,
        get_multi_day_signals_tool,
        get_options_snapshot,
        get_vol_term_structure,
        get_options_flow,
        get_gamma_exposure,
    ]


__all__ = [
    "MAX_TOKENS_PER_CALL",
    "get_chat_tools",
    "list_top_assets",
    "get_gamma_exposure",
    "get_options_flow",
    "get_options_snapshot",
    "get_price_history",
    "get_token_metrics",
    "get_vol_term_structure",
    "get_zscore_signals",
    "list_token_universe",
    "run_full_signals_analysis",
    "validate_tokens",
]
