"""Token-metric fetch + z-score math.

Pulls raw daily data (price, spot volume, perp volume, perp OI, liquidations,
funding rate) through `providers.factory.get_provider()`, optionally merges the
daily options metrics (DVOL, ATM IV term structure, skew, put/call ratio,
options volume) from `providers.factory.get_options_provider()` for tokens with
listed options, and turns series into z-scores. Signal generation lives in
tools/signals.py.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from providers.factory import get_provider

logger = logging.getLogger(__name__)

# Suppress Coin Metrics client ERROR logs for expected "not supported" API responses
logging.getLogger("cm_client").setLevel(logging.CRITICAL)

# Token universes
FULL_TOKEN_UNIVERSE = [
    "btc", "eth", "sol", "near", "pol", "avax", "sui", "atom", "arb", "apt",
    "dot", "op", "hype", "sky", "aave", "syrup", "jto", "uni", "pendle",
    "morpho", "ldo", "fluid", "ena", "aero", "pump", "spx", "jup", "ray",
]

TEST_TOKEN_UNIVERSE = ["btc", "eth", "sol", "sui", "hype", "uni", "jto"]

# Z-score thresholds and rolling window (days)
OUTLIER_THRESHOLD = 2.5
INSIGNIFICANT_THRESHOLD = 1.0
Z_WINDOW = 30

# Minimum observations before a z-score is emitted
MIN_HISTORY = 10

# funding_rate is an annualised % of 8-hour periods (see providers/base.py)
FUNDING_PERIODS_PER_YEAR = 3 * 365

# Columns fetched per token (metric -> provider method, columns it yields)
METRIC_SOURCES = (
    ("perp_volume", "get_perp_volume", ["perp_volume"]),
    ("perp_oi", "get_perp_oi", ["perp_oi"]),
    ("funding_rate", "get_funding_rate", ["funding_rate"]),
    ("spot_volume", "get_spot_ohlcv", ["spot_volume"]),
    ("liquidations", "get_liquidations", ["long_liquidations", "short_liquidations", "total_liquidations"]),
)

# Daily options metrics (Amberdata options provider, Deribit). Only merged for tokens
# in `options_provider.supported_tokens()`; other tokens have none of these columns.
# Units: DVOL / IV in vol points (annualised %), skew in vol points (put IV - call IV,
# positive = puts richer), notional / premium in USD, PCR dimensionless.
OPTIONS_METRIC_SOURCES = (
    ("dvol", "get_dvol", ["dvol_open", "dvol_high", "dvol_low", "dvol_close"]),
    ("term_structure", "get_term_structure_history",
     ["atm_iv_7d", "atm_iv_30d", "atm_iv_60d", "atm_iv_90d", "atm_iv_180d", "ts_richness"]),
    ("skew", "get_skew_history", ["skew_25d_30d", "skew_10d_30d"]),
    ("put_call_ratio", "get_put_call_ratio", ["pcr_oi", "pcr_volume_24h"]),
    ("options_volume", "get_options_volume",
     ["options_contract_volume", "options_notional_volume", "options_premium_volume",
      "options_block_notional_volume"]),
)

# Every options column fetch_token_metrics may add
OPTIONS_COLUMNS = tuple(c for _, _, cols in OPTIONS_METRIC_SOURCES for c in cols)

# The options metrics surfaced to signals / LLM / chat (subset of OPTIONS_COLUMNS)
OPTIONS_DAILY_METRICS = (
    "dvol_close", "atm_iv_30d", "skew_25d_30d", "pcr_oi", "pcr_volume_24h",
    "options_notional_volume", "options_block_notional_volume",
)

OPTIONS_EXCHANGE = "deribit"


def get_options_provider():
    """Return the configured options provider, or None.

    Imported lazily so tools.metrics imports cleanly when providers/amberdata_options.py
    or `providers.factory.get_options_provider` does not exist (yet). Tests inject a fake
    by monkeypatching `providers.factory.get_options_provider`.
    """
    try:
        from providers import factory
    except ImportError:
        return None
    getter = getattr(factory, "get_options_provider", None)
    if getter is None:
        logger.info("providers.factory.get_options_provider not available; options metrics disabled")
        return None
    try:
        return getter()
    except ImportError as e:
        logger.info(f"Options provider unavailable ({e}); options metrics disabled")
        return None
    except Exception as e:  # missing key / probe failure must not sink the run
        logger.warning(f"Options provider could not be built ({type(e).__name__}: {e}); options metrics disabled")
        return None


def _options_supported_tokens(provider) -> set:
    if provider is None:
        return set()
    try:
        return {str(t).lower() for t in (provider.supported_tokens() or [])}
    except Exception as e:
        logger.warning(f"options provider supported_tokens() failed ({type(e).__name__}: {e}); options disabled")
        return set()


def calculate_zscore_with_weekend_separation(
    series: pd.Series,
    dates: pd.Series,
    window: int = 30,
    separate_weekends: bool = True,
) -> pd.Series:
    """
    Calculate z-scores against a rolling median, optionally with weekend/weekday separation.

    Args:
        series: Data series to calculate z-scores for
        dates: Corresponding dates (datetime-like Series aligned with `series`)
        window: Rolling window for median / std calculation
        separate_weekends: If True, compare each day only against prior days of the
            same type (weekend vs weekday), falling back to the plain window when
            fewer than 5 same-type observations exist.

    Returns:
        Series of z-scores aligned with `series` (NaN for the first MIN_HISTORY rows).
    """
    if not separate_weekends:
        rolling_median = series.rolling(window=window, min_periods=MIN_HISTORY).median()
        rolling_std = series.rolling(window=window, min_periods=MIN_HISTORY).std()
        return (series - rolling_median) / rolling_std.replace(0, np.nan)

    is_weekend = pd.to_datetime(dates).dt.dayofweek >= 5

    z_scores = pd.Series(index=series.index, dtype=float)

    for idx in range(len(series)):
        if idx < MIN_HISTORY:
            z_scores.iloc[idx] = np.nan
            continue

        current_is_weekend = is_weekend.iloc[idx]

        historical_mask = is_weekend.iloc[:idx] == current_is_weekend
        historical_values = series.iloc[:idx][historical_mask.values].tail(window)

        if len(historical_values) < 5:
            historical_values = series.iloc[max(0, idx - window):idx]

        if len(historical_values) > 0:
            median = historical_values.median()
            std = historical_values.std()
            if std > 0:
                z_scores.iloc[idx] = (series.iloc[idx] - median) / std
            else:
                z_scores.iloc[idx] = 0
        else:
            z_scores.iloc[idx] = np.nan

    return z_scores


def fetch_token_metrics(
    tokens: List[str],
    start_date: datetime,
    end_date: datetime,
    include_options: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all required daily metrics for `tokens` via the configured provider.

    Args:
        include_options: when True (default) and an options provider is configured,
            tokens in `options_provider.supported_tokens()` also get the daily options
            columns (OPTIONS_COLUMNS: dvol_*, atm_iv_*, skew_*, pcr_*, options_*_volume).
            Tokens without listed options do not get these columns at all. False skips
            the options provider entirely (it is never even constructed).

    Returns:
        Dict mapping token -> DataFrame with float64 columns
        time, price, price_pct_change, perp_volume, perp_oi, funding_rate,
        spot_volume, long_liquidations, short_liquidations, total_liquidations
        (+ OPTIONS_COLUMNS for tokens with listed options).
        Tokens with no price data are skipped. Metrics the provider has no data
        for are all-NaN columns (logged at INFO). Rows for the current, still
        incomplete UTC day are dropped so partial-day totals never enter z-scores.
    """
    provider = get_provider()
    results: Dict[str, pd.DataFrame] = {}

    options_provider = get_options_provider() if include_options else None
    options_tokens = _options_supported_tokens(options_provider)

    for token in tokens:
        token = token.lower()
        try:
            # --- price (required) ---
            price_df = provider.get_spot_price(token, start_date, end_date)
            if price_df is None or price_df.empty:
                logger.warning(f"No price data for {token} (spot provider); skipping token")
                continue

            df = price_df[["time", "price"]].copy()
            df["time"] = pd.to_datetime(df["time"])
            df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")

            missing: List[str] = []
            for name, method, columns in METRIC_SOURCES:
                try:
                    other = getattr(provider, method)(token, start_date, end_date)
                except Exception as e:  # one failing metric must not sink the token
                    logger.warning(f"{token}: {method} failed ({type(e).__name__}: {e}); {name} unavailable")
                    other = None
                df, ok = _merge_metric(df, other, columns)
                if not ok:
                    missing.append(name)

            if token in options_tokens:
                df, missing_opts = _merge_options_metrics(df, options_provider, token, start_date, end_date)
                missing.extend(f"options:{m}" for m in missing_opts)
            elif options_provider is not None:
                logger.info(f"{token}: no listed options on {OPTIONS_EXCHANGE}; options metrics skipped")

            df = _drop_partial_day(df)
            df = df.sort_values("time").reset_index(drop=True)
            df["price_pct_change"] = df["price"].pct_change() * 100
            if missing:
                logger.info(f"{token}: no data for {', '.join(missing)}; continuing with available metrics")
            results[token] = df

        except Exception as e:
            logger.error(f"Error fetching data for {token}: {e}")
            continue

    return results


def _merge_options_metrics(
    df: pd.DataFrame, provider, token: str, start_date: datetime, end_date: datetime
) -> tuple[pd.DataFrame, List[str]]:
    """Left-join every OPTIONS_METRIC_SOURCES frame onto `df`.

    A source that returns None / raises / lacks columns yields all-NaN columns, so a
    listed token always carries the full options column set. Returns (frame, names
    of sources that supplied nothing).
    """
    missing: List[str] = []
    for name, method, columns in OPTIONS_METRIC_SOURCES:
        try:
            fn = getattr(provider, method, None)
            other = fn(token, start_date, end_date) if fn is not None else None
        except Exception as e:
            logger.warning(f"{token}: options {method} failed ({type(e).__name__}: {e}); {name} unavailable")
            other = None
        present = [c for c in columns if other is not None and not other.empty and c in other.columns]
        if present:
            df, _ = _merge_metric(df, other, present)
        df = df.copy()
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        if not present:
            missing.append(name)
    return df, missing


def _drop_partial_day(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows on/after today's UTC date (the current day is incomplete)."""
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    return df[df["time"] < today]


def _merge_metric(df: pd.DataFrame, other, columns: List[str]) -> tuple[pd.DataFrame, bool]:
    """Left-join float64 `columns` from `other` onto `df` by day.

    Returns (frame, True) when `other` supplied the columns, else (frame with
    all-NaN columns, False).
    """
    if other is not None and not other.empty and all(c in other.columns for c in columns):
        other = other[["time"] + columns].copy()
        # Providers return tz-naive UTC days, but normalise defensively so a tz-aware
        # or intraday timestamp from one side still joins with the other.
        other["time"] = _to_utc_day(other["time"])
        for c in columns:
            other[c] = pd.to_numeric(other[c], errors="coerce").astype("float64")
        other = other.drop_duplicates("time", keep="last")
        df = df.copy()
        df["time"] = _to_utc_day(df["time"])
        return df.merge(other, on="time", how="left"), True
    df = df.copy()
    for c in columns:
        df[c] = np.nan
    return df, False


def _to_utc_day(values: pd.Series) -> pd.Series:
    """Coerce a time column to tz-naive UTC midnight (datetime64[ns])."""
    ts = pd.to_datetime(values, utc=True)
    return ts.dt.tz_convert(None).dt.normalize().astype("datetime64[ns]")


# ---------------------------------------------------------------------------
# Options snapshot (term structure, skew, gamma, block trades)
# ---------------------------------------------------------------------------

GAMMA_TOP_N = 3


def summarise_gamma_exposure(gex: Optional[pd.DataFrame], top_n: int = GAMMA_TOP_N) -> Optional[Dict[str, Any]]:
    """Condense a per-strike gamma-exposure frame (strike, net_dealer_gamma,
    total_dealer_gamma, index_price; attrs["snapshot_time"]) into
    {snapshot_time, index_price, total_net_gamma, top_strikes, flip_point, n_strikes}.

    top_strikes: the 2*top_n strikes with the largest |net_dealer_gamma| (sorted by
    strike). flip_point: strike where net dealer gamma changes sign (linear
    interpolation between adjacent strikes, the crossing closest to the index price);
    None if the sign never changes.
    """
    if gex is None or gex.empty or "strike" not in gex.columns or "net_dealer_gamma" not in gex.columns:
        return None
    g = gex.copy()
    g["strike"] = pd.to_numeric(g["strike"], errors="coerce")
    g["net_dealer_gamma"] = pd.to_numeric(g["net_dealer_gamma"], errors="coerce")
    g = g.dropna(subset=["strike", "net_dealer_gamma"])
    if g.empty:
        return None
    aggs = {"net_dealer_gamma": ("net_dealer_gamma", "sum")}
    if "total_dealer_gamma" in g.columns:
        g["total_dealer_gamma"] = pd.to_numeric(g["total_dealer_gamma"], errors="coerce")
        aggs["total_dealer_gamma"] = ("total_dealer_gamma", "sum")
    g = g.groupby("strike", as_index=False).agg(**aggs).sort_values("strike").reset_index(drop=True)

    index_price = None
    if "index_price" in gex.columns:
        ip = pd.to_numeric(gex["index_price"], errors="coerce").dropna()
        index_price = float(ip.iloc[-1]) if len(ip) else None

    top = g.reindex(g["net_dealer_gamma"].abs().sort_values(ascending=False).index).head(2 * top_n)
    top_strikes = []
    for _, r in top.sort_values("strike").iterrows():
        tot = r["total_dealer_gamma"] if "total_dealer_gamma" in top.columns else np.nan
        top_strikes.append({
            "strike": float(r["strike"]),
            "net_dealer_gamma": float(r["net_dealer_gamma"]),
            "total_dealer_gamma": float(tot) if pd.notna(tot) else None,
        })

    flip_point = None
    strikes = g["strike"].to_numpy(dtype=float)
    net = g["net_dealer_gamma"].to_numpy(dtype=float)
    candidates = []
    for i in range(len(g) - 1):
        a, b = net[i], net[i + 1]
        if a == 0:
            candidates.append(float(strikes[i]))
        elif (a < 0 < b) or (a > 0 > b):
            frac = a / (a - b)
            candidates.append(float(strikes[i] + frac * (strikes[i + 1] - strikes[i])))
    if candidates:
        ref = index_price if index_price is not None else float(np.median(strikes))
        flip_point = min(candidates, key=lambda x: abs(x - ref))

    snapshot_time = gex.attrs.get("snapshot_time") if hasattr(gex, "attrs") else None
    return {
        "snapshot_time": str(snapshot_time) if snapshot_time is not None else None,
        "index_price": index_price,
        "total_net_gamma": float(g["net_dealer_gamma"].sum()),
        "top_strikes": top_strikes,
        "flip_point": flip_point,
        "n_strikes": int(len(g)),
    }


def _records(df: Optional[pd.DataFrame], columns: Optional[List[str]] = None,
             limit: Optional[int] = None) -> Optional[List[dict]]:
    """DataFrame -> list of plain-python dicts (NaN -> None, timestamps -> 'YYYY-MM-DD')."""
    if df is None or df.empty:
        return None
    if columns:
        cols = [c for c in columns if c in df.columns]
        if not cols:
            return None
        df = df[cols]
    if limit is not None:
        df = df.head(limit)
    out = []
    for _, row in df.iterrows():
        rec = {}
        for k, v in row.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                rec[k] = v.strftime("%Y-%m-%d")
            elif v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
                rec[k] = None
            elif isinstance(v, np.integer):
                rec[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                rec[k] = float(v)
            else:
                rec[k] = v
        out.append(rec)
    return out


def _latest_row(df: Optional[pd.DataFrame], value_col: str, lookback_days: int = 7) -> Optional[Dict[str, Any]]:
    """Latest non-NaN value of `value_col` and its change vs the value `lookback_days` earlier."""
    if df is None or df.empty or value_col not in df.columns or "time" not in df.columns:
        return None
    d = df[["time", value_col]].copy()
    d["time"] = pd.to_datetime(d["time"])
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna().sort_values("time")
    if d.empty:
        return None
    last = d.iloc[-1]
    ref_time = pd.Timestamp(last["time"]) - timedelta(days=lookback_days)
    prior = d[d["time"] <= ref_time]
    change = float(last[value_col] - prior.iloc[-1][value_col]) if len(prior) else None
    return {
        "time": pd.Timestamp(last["time"]).strftime("%Y-%m-%d"),
        "value": float(last[value_col]),
        f"change_{lookback_days}d": change,
    }


def fetch_options_snapshot(token: str, provider=None, block_top_n: int = 10) -> Dict[str, Any]:
    """Current options picture for one token from the options provider.

    Returns a dict (never raises) with keys:
        token, exchange, listed (bool), currency,
        term_structure  list[{days_to_expiration, atm_iv, fwd_atm_iv}]              | None
        delta_surface   list[{days_to_expiration, atm_iv, iv_call_25d, iv_put_25d,
                              iv_call_10d, iv_put_10d, skew_25d, skew_10d}]         | None
        dvol            {time, value, change_7d}                                    | None
        pcr             {time, pcr_oi, pcr_oi_change_7d, pcr_volume_24h}            | None
        gamma           summarise_gamma_exposure(...)                               | None
        block_trades    list[{expiry, strike, put_call, contract_volume, premium_volume}] | None
        errors          list[str]
    `listed` is False (all sections None) when there is no provider or the token has
    no options on the exchange. Units: IV / DVOL in vol points (annualised %), skew in
    vol points (put IV - call IV, positive = puts richer), premium / notional in USD.
    """
    token = token.lower()
    provider = provider if provider is not None else get_options_provider()
    snap: Dict[str, Any] = {
        "token": token,
        "exchange": getattr(provider, "exchange", OPTIONS_EXCHANGE) if provider is not None else OPTIONS_EXCHANGE,
        "listed": False, "currency": None, "term_structure": None, "delta_surface": None,
        "dvol": None, "pcr": None, "gamma": None, "block_trades": None, "errors": [],
    }
    if provider is None:
        snap["errors"].append("no options provider configured")
        return snap
    if token not in _options_supported_tokens(provider):
        return snap
    snap["listed"] = True
    try:
        snap["currency"] = provider.currency_for(token)
    except Exception:
        snap["currency"] = None

    end = datetime.now()
    start = end - timedelta(days=14)

    def _call(name, method, *args, **kwargs):
        fn = getattr(provider, method, None)
        if fn is None:
            snap["errors"].append(f"{name}: provider has no {method}")
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"{token}: options {method} failed ({type(e).__name__}: {e})")
            snap["errors"].append(f"{name}: {type(e).__name__}: {e}")
            return None

    ts = _call("term_structure", "get_term_structure", token)
    snap["term_structure"] = _records(ts, ["days_to_expiration", "atm_iv", "fwd_atm_iv"])

    ds = _call("delta_surface", "get_delta_surface", token)
    snap["delta_surface"] = _records(ds, [
        "days_to_expiration", "atm_iv", "iv_call_25d", "iv_put_25d", "iv_call_10d", "iv_put_10d",
        "skew_25d", "skew_10d",
    ])

    dvol = _call("dvol", "get_dvol", token, start, end)
    snap["dvol"] = _latest_row(dvol, "dvol_close")

    pcr = _call("put_call_ratio", "get_put_call_ratio", token, start, end)
    oi, vol = _latest_row(pcr, "pcr_oi"), _latest_row(pcr, "pcr_volume_24h")
    if oi or vol:
        snap["pcr"] = {
            "time": (oi or vol)["time"],
            "pcr_oi": oi["value"] if oi else None,
            "pcr_oi_change_7d": oi["change_7d"] if oi else None,
            "pcr_volume_24h": vol["value"] if vol else None,
        }

    gex = _call("gamma_exposure", "get_gamma_exposure", token)
    snap["gamma"] = summarise_gamma_exposure(gex)

    blocks = _call("block_trades", "get_block_trades", token, end - timedelta(days=7), end, top_n=block_top_n)
    snap["block_trades"] = _records(
        blocks, ["expiry", "strike", "put_call", "contract_volume", "premium_volume"], limit=block_top_n
    )
    return snap


def fetch_price_history(token: str, days: int = 30) -> pd.DataFrame | None:
    """Daily spot price history for one token via the provider (time, price)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = get_provider().get_spot_price(token.lower(), start_date, end_date)
    if df is None or df.empty:
        return None
    df = df[["time", "price"]].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.sort_values("time").reset_index(drop=True)


@tool("get_token_price_history_tool")
def get_token_price_history_tool(token: str, days: int = 30) -> str:
    """
    Fetch historical daily price data for a specific token.

    Use this tool when you need the actual price history for a token,
    including daily prices and daily percentage changes.

    Args:
        token: Token symbol (e.g., 'btc', 'eth', 'sol', 'jto')
        days: Number of days of history to fetch (default: 30, min 7, max 90)

    Returns:
        JSON with a summary (current, high, low, change %) and the last 14 daily rows.
    """
    try:
        days = min(max(int(days), 7), 90)
        price_df = fetch_price_history(token, days=days)

        if price_df is None or price_df.empty:
            return json.dumps({"error": f"No price data available for {token}", "token": token.upper()})

        price_df["daily_change_pct"] = price_df["price"].pct_change() * 100

        records = []
        for _, row in price_df.iterrows():
            t = row["time"]
            records.append({
                "date": t.strftime("%Y-%m-%d") if hasattr(t, "strftime") else str(t)[:10],
                "price": round(float(row["price"]), 4) if pd.notna(row["price"]) else None,
                "daily_change_pct": round(float(row["daily_change_pct"]), 2) if pd.notna(row["daily_change_pct"]) else None,
            })

        prices = [r["price"] for r in records if r["price"] is not None]
        current_price = prices[-1] if prices else None
        first_price = prices[0] if prices else None
        change_pct = ((current_price / first_price) - 1) * 100 if current_price and first_price else None

        result = {
            "token": token.upper(),
            "days": days,
            "summary": {
                "current_price": current_price,
                "high": max(prices) if prices else None,
                "low": min(prices) if prices else None,
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
            },
            "history": records[-14:],
        }
        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error fetching price history for {token}: {e}")
        return json.dumps({"error": str(e), "token": token.upper()})
