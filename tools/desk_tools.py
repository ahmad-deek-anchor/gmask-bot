"""@tool functions over the Global Markets desk data in BigQuery (read-only).

Scope: the A1 / OTC derivatives desk book as risk-managed in **Haruko** (options,
futures/perps and spot balances across Deribit, Binance, OKX, Bybit, Kraken and OTC
counterparties), the OTC derivatives trade blotter, live Talos orders and the
desk's internal price feed.

**Scope (providers/desk_scope.py).** The derivatives desk is exactly three Haruko
portfolios - ``Derivs Risk`` (entity 20, A1 Ltd), ``ADSD`` (86, Anchorage Digital Swap
Dealer) and ``AD Hedge Co`` (87) - and every query here filters to them
(``entity_id IN (20, 86, 87)`` / ``strategy_name IN (...)``), then appends a scope
footer. Other Haruko accounts are excluded by default (Ahmad Deek, 2026-09-18). The
BigQuery export holds entities 20 and 86 only; 87 has no rows yet. Custody / HOLD /
CRMS / lending data live in other datasets the agent cannot read.

Every tool returns compact markdown (never a raw dump), states the as-of
timestamp of the snapshot it used and surfaces Haruko's ``data_quality_flag`` /
``valid_pricer_pct`` where the table carries them. USD is formatted with ``$``
and thousands separators.

Tables used (dataset ``brokerage_a1`` unless noted; verified 2026-09-10):

* ``fct_otc_haruko_position_pnl_history`` (+ ``fct_otcderivatives_trades``) - the
  position-level snapshot history behind ``get_derivs_pnl_eod``: Carson Levy's EOW
  method run live via ``sql/haruko_eod_pnl.sql`` / ``providers/haruko_eod.py``
  (3pm America/Chicago EOD cut, LTD differences; ~21 GB per run, cached 15 min).
  This is the **authoritative** source for monthly / MTD / YTD derivatives PnL.
* ``fct_otc_haruko_pnl_portfolio`` - portfolio snapshot every ~5 min, one row per
  entity: position/venue/asset counts, ``total_abs_size_usd`` (gross notional),
  ``total_equity_usd``, ``total_portfolio_pnl`` (day), WTD/MTD/QTD/YTD/LTD PnL,
  funding PnL, fees, ``total_delta_usd`` / ``total_gamma_usd`` /
  ``total_gamma_percent_usd`` / ``total_vega`` / ``total_theta``, risk-level labels,
  large-change flags, ``valid_pricer_pct`` and ``data_quality_flag``.
* ``fct_otc_haruko_pnl_portfolio_history_eod`` / ``fct_otc_haruko_greeks_history_eod``
  - one EOD row per entity per ``as_of_date`` (~23:55 UTC).
* ``fct_otc_haruko_pnl_position_history_eod`` - EOD position-level rows partitioned
  by ``as_of_date`` (``symbol``, ``underlying_asset``, ``venue``, ``counterparty``,
  ``strategy_name``, ``instrument_type`` SPOT/OPTIONS/FUTURES, ``size_usd`` (gross),
  ``delta_usd`` (signed), greeks, ``total_pnl`` (day), ``year_to_date_pnl``,
  ``funding_pnl``, ``avg_px``/``mark_px``, ``maturity``, ``pricer_valid``). Used for
  by-venue / by-strategy / by-underlying breakdowns and for perps, because the
  convenience views (``fct_otc_haruko_pnl_by_venue*``, ``fct_otc_haruko_futures_positions``,
  ``fct_perps_positions`` - stale since 2026-07-10) scan far more than the 2 GB cap.
* ``fct_otc_haruko_position_summary`` - live reconciled position per symbol
  (``position_type`` spot/futures/options, ``total_exchange_position``,
  ``total_position_delta_value``).
* ``fct_otcderivatives_trades`` + ``dim_otcderivatives_{products,assets,accounts,entities}``
  - OTC options blotter (status, direction, base/quote, quantity, strike, premium,
  expiration, counterparty).
* ``fct_a1_talos_open_orders_live`` - live Talos orders.
* ``pricing.fct_current_asset_prices_live`` (live tick per canonical symbol) and
  ``pricing.intraday_price`` (minute bars, USD quote, partitioned) for internal prices.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from langchain_core.tools import tool

from providers.bigquery import SQLGuardError, format_bytes
from providers.desk_scope import (ENTITY_NAMES, ENTITY_PORTFOLIO, entity_ids, entity_name, entity_sql, otc_entity_sql, portfolios,
                                  scope_label, scope_note, strategy_sql)
from providers.haruko_eod import METHOD_LINE, HarukoEodError, HarukoEodPnl

logger = logging.getLogger(__name__)

# dataset.table names (qualified with the data project by DeskBigQuery.table())
PORTFOLIO = "brokerage_a1.fct_otc_haruko_pnl_portfolio"
PORTFOLIO_EOD = "brokerage_a1.fct_otc_haruko_pnl_portfolio_history_eod"
GREEKS_EOD = "brokerage_a1.fct_otc_haruko_greeks_history_eod"
POSITIONS_EOD = "brokerage_a1.fct_otc_haruko_pnl_position_history_eod"
POSITION_SUMMARY = "brokerage_a1.fct_otc_haruko_position_summary"
OTC_TRADES = "brokerage_a1.fct_otcderivatives_trades"
OTC_PRODUCTS = "brokerage_a1.dim_otcderivatives_products"
OTC_ASSETS = "brokerage_a1.dim_otcderivatives_assets"
OTC_ACCOUNTS = "brokerage_a1.dim_otcderivatives_accounts"
OTC_ENTITIES = "brokerage_a1.dim_otcderivatives_entities"
OPEN_ORDERS = "brokerage_a1.fct_a1_talos_open_orders_live"
LIVE_PRICES = "pricing.fct_current_asset_prices_live"
INTRADAY_PRICES = "pricing.intraday_price"

QUERY_DISPLAY_ROWS = 40          # rows shown by query_desk_data
MAX_HISTORY_DAYS_PORTFOLIO = 200 // max(1, len(entity_ids()))   # entities x days must stay < the 200-row cap (66 for 3)
MAX_HISTORY_DAYS_GROUPED = 28    # (6 groups + other) x 28 = 196 rows < 200 cap
MAX_TOP_N = 50
EOD_LOOKBACK_DAYS = 7            # how far back to look for the latest EOD snapshot

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9_]{1,24}$")
_OPTION_TAIL_RE = re.compile(r"[-_]\d{1,2}[A-Z]{3}\d{2,4}.*$")     # BTC-25SEP26-80000-C, BTCUSD-25SEP26-...
_SETTLE_SUFFIXES = ("USDT", "USDC", "USD", "PERP", "PERPETUAL", "SWAP")


# ---------------------------------------------------------------------------
# access + error handling
# ---------------------------------------------------------------------------

def _get_bq():
    """DeskBigQuery or None (tests monkeypatch this)."""
    from providers.factory import get_desk_bigquery
    return get_desk_bigquery()


UNAVAILABLE = (
    "Desk data (BigQuery) is not available: google-cloud-bigquery or Application Default "
    "Credentials are missing. Market-data tools still work."
)

_haruko_eod: Dict[int, HarukoEodPnl] = {}   # one HarukoEodPnl (with its 15-min cache) per DeskBigQuery


def _get_haruko_eod(bq) -> HarukoEodPnl:
    """HarukoEodPnl bound to ``bq``; memoised so the TTL cache survives across tool calls."""
    inst = _haruko_eod.get(id(bq))
    if inst is None or inst.bq is not bq:
        inst = HarukoEodPnl(bq)
        _haruko_eod.clear()
        _haruko_eod[id(bq)] = inst
    return inst


def _guarded(name: str, body: Callable, scoped: bool = False) -> str:
    """Run ``body(bq)``; ``scoped`` tools (everything reading Haruko / OTC tables) get the portfolio-scope footer."""
    bq = _get_bq()
    if bq is None:
        return UNAVAILABLE
    try:
        out = body(bq)
        return f"{out}\n{scope_note()}" if scoped else out
    except SQLGuardError as e:
        return f"Rejected by the read-only guard: {e}"
    except Exception as e:  # noqa: BLE001 - surface to the model, never crash the agent
        logger.error("%s failed: %s", name, e, exc_info=logger.isEnabledFor(logging.DEBUG))
        return f"Error querying desk data in {name}: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _isna(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _usd(v, decimals: Optional[int] = None) -> str:
    """$1,234,567 (no decimals at >= $1,000; 2 decimals below; 4 below $1)."""
    if _isna(v):
        return "n/a"
    v = float(v)
    if v == 0:
        return "$0"
    if decimals is None:
        a = abs(v)
        decimals = 0 if a >= 1000 else (2 if a >= 1 else 4)
    s = f"{abs(v):,.{decimals}f}"
    return f"-${s}" if v < 0 else f"${s}"


def _num(v, decimals: int = 2) -> str:
    if _isna(v):
        return "n/a"
    return f"{float(v):,.{decimals}f}"


def _qty(v) -> str:
    """Coin quantity: thousands separators, up to 4 decimals, trailing zeros trimmed."""
    if _isna(v):
        return "n/a"
    v = float(v)
    s = f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _pct(v, decimals: int = 1) -> str:
    if _isna(v):
        return "n/a"
    return f"{float(v):.{decimals}f}%"


def _ts(v) -> str:
    """'YYYY-MM-DD HH:MM UTC' for timestamps, 'YYYY-MM-DD' for dates."""
    if _isna(v):
        return "n/a"
    if isinstance(v, pd.Timestamp):
        v = v.to_pydatetime()
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M UTC")
    s = str(v)
    return s[:16] + " UTC" if len(s) > 10 and s[10] in " T" else s[:10]


def _date(v) -> str:
    if _isna(v):
        return "n/a"
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    return str(v)[:10]


def _entity(eid) -> str:
    return entity_name(eid)


def _portfolio(eid) -> str:
    """Short portfolio name for table cells ('Derivs Risk', 'ADSD', 'AD Hedge Co')."""
    try:
        return ENTITY_PORTFOLIO.get(int(eid), f"entity {int(eid)}")
    except (TypeError, ValueError):
        return f"entity {eid}"


def _sign_side(v) -> str:
    if _isna(v) or float(v) == 0:
        return "flat"
    return "long" if float(v) > 0 else "short"


def _dq_line(flag, pct) -> str:
    flag_s = "n/a" if _isna(flag) else str(flag)
    line = f"Data quality: {flag_s}; valid pricers {_pct(pct)}"
    if flag_s.lower() not in ("normal", "n/a"):
        line += (" - CAVEAT: Haruko flags this snapshot; greeks/PnL on positions with an invalid "
                 "pricer may be stale or missing, treat the totals as indicative.")
    return line


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _cell(v) -> str:
    """Generic cell formatter for ad-hoc query output."""
    if _isna(v):
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    if isinstance(v, float):
        if math.isinf(v):
            return str(v)
        return f"{v:,.4f}" if abs(v) < 1 else f"{v:,.2f}"
    if isinstance(v, (pd.Timestamp, datetime)):
        return _ts(v)
    s = str(v).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= 80 else s[:77] + "..."


def _bytes_note(df: pd.DataFrame) -> str:
    return f"Bytes scanned: {format_bytes(df.attrs.get('bytes_processed'))}."


def _clamp(v, default: int, lo: int, hi: int) -> int:
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = default
    return min(max(v, lo), hi)


def _col(df: pd.DataFrame, name: str, default=None):
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


# ---------------------------------------------------------------------------
# symbol mapping (desk symbols -> Amberdata / Coin Metrics token universe)
# ---------------------------------------------------------------------------

def desk_symbol_to_token(symbol: Optional[str]) -> Optional[str]:
    """Map a desk / venue symbol to the lowercase token used by the market-data tools.

    Heuristic (documented for the prompt): uppercase; drop an option / dated-future tail
    (``-25SEP26-80000-C``, ``-25SEP26``); keep the part before the first ``-``, ``/`` or
    ``_`` (``BTC-USDT-SWAP`` -> ``BTC``, ``USDG_SOLANA`` -> ``USDG``, ``XAUT/USDT`` -> ``XAUT``);
    strip a trailing settlement suffix (``BTCUSDT`` -> ``BTC``, ``ETHUSD`` -> ``ETH``,
    ``BTCPERP`` -> ``BTC``) as long as at least three characters remain (so ``PYUSD``
    stays ``pyusd``). Returns None for empty input.
    Whether the result is in the supported universe is up to the caller
    (``tools.metrics.FULL_TOKEN_UNIVERSE``).
    """
    if not symbol:
        return None
    s = str(symbol).strip().upper()
    s = _OPTION_TAIL_RE.sub("", s)
    s = re.split(r"[-/_ ]", s)[0]
    for suf in _SETTLE_SUFFIXES:
        if s.endswith(suf) and len(s) - len(suf) >= 3:  # BTCUSDT -> BTC, but PYUSD stays PYUSD
            s = s[: -len(suf)]
            break
    return s.lower() or None


def _in_universe(token: Optional[str]) -> bool:
    if not token:
        return False
    try:
        from tools.metrics import FULL_TOKEN_UNIVERSE
        return token in FULL_TOKEN_UNIVERSE   # curated universe flag (display only, no network)
    except Exception:  # noqa: BLE001
        return False


def _token_hint(tokens: Sequence[str]) -> str:
    seen: List[str] = []
    for t in tokens:
        if t and t not in seen:
            seen.append(t)
    if not seen:
        return ""
    covered = [t for t in seen if _in_universe(t)]
    not_covered = [t for t in seen if t not in covered]
    parts = []
    if covered:
        parts.append("in the market-data universe (cross-check with get_zscore_signals / get_token_metrics): "
                     + ", ".join(covered))
    if not_covered:
        parts.append("not covered by the market-data tools: " + ", ".join(not_covered))
    return "Token mapping - " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# shared SQL fragments
# ---------------------------------------------------------------------------

def _latest_eod_cte(bq, lookback_days: int = EOD_LOOKBACK_DAYS) -> str:
    """CTEs selecting, per entity, the latest EOD snapshot (date + timestamp) of the position table."""
    t = bq.table(POSITIONS_EOD)
    return f"""
latest AS (
  SELECT entity_id, MAX(as_of_date) AS as_of_date
  FROM {t}
  WHERE as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_days} DAY)
    AND {entity_sql()}
  GROUP BY entity_id
),
snap AS (
  SELECT p.entity_id, p.as_of_date, MAX(p.as_of_timestamp) AS as_of_timestamp
  FROM {t} p
  JOIN latest l ON l.entity_id = p.entity_id AND l.as_of_date = p.as_of_date
  WHERE p.as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_days} DAY)
    AND {entity_sql('p.entity_id')}
  GROUP BY 1, 2
)"""


def _eod_join(alias: str = "p") -> str:
    return (f"JOIN snap s ON s.entity_id = {alias}.entity_id AND s.as_of_date = {alias}.as_of_date "
            f"AND s.as_of_timestamp = {alias}.as_of_timestamp")


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

_SNAPSHOT_COLS = [
    "position_timestamp", "entity_id", "position_count", "venue_count", "strategy_count", "asset_count",
    "symbol_count", "total_abs_size_usd", "total_size_usd", "total_equity_usd", "total_portfolio_pnl",
    "total_open_pnl", "total_realised_pnl", "total_week_to_date_pnl", "total_month_to_date_pnl",
    "total_quarter_to_date_pnl", "total_year_to_date_pnl", "total_life_to_date_pnl", "total_funding_pnl",
    "total_life_to_date_funding_pnl", "total_fees", "total_life_to_date_fees", "total_delta_usd",
    "total_delta_adjusted_usd", "total_abs_delta_usd", "total_gamma_usd", "total_gamma_percent_usd",
    "total_vega", "total_theta", "total_theta_bs", "total_vanna", "total_charm", "total_volga",
    "portfolio_delta_risk_level", "portfolio_gamma_risk_level", "portfolio_vega_risk_level",
    "large_delta_change_flag", "delta_change_severity", "delta_adjusted_usd_change",
    "large_ytd_pnl_change_flag", "ytd_pnl_change_severity", "ytd_pnl_change",
    "large_theta_bs_hourly_change_flag", "large_gamma_percent_usd_hourly_change_flag",
    "valid_pricer_count", "invalid_pricer_count", "valid_pricer_pct", "data_quality_flag",
    "hours_since_latest_calculation", "total_week_to_date_volume_usd", "total_month_to_date_volume_usd",
]


@tool("get_desk_risk_snapshot")
def get_desk_risk_snapshot() -> str:
    """Latest Haruko portfolio risk snapshot for the OTC/derivatives desk: positions, gross/net notional, equity, PnL (day, WTD, MTD, QTD, YTD, LTD), funding PnL, fees, USD greeks (delta, gamma, vega, theta), risk-level labels, large-change flags and data quality.

    Use for "what is our delta / gamma / YTD PnL", "how big is the book", "desk risk
    snapshot", "is the data quality ok". One section per portfolio in scope (Derivs Risk,
    ADSD, AD Hedge Co - see providers/desk_scope.py) plus a combined line. Snapshots land every ~5 minutes; the as-of time is
    stated. Covers the Haruko-risk-managed A1/OTC book only (options, futures/perps,
    spot balances) - not custody/HOLD. Greeks are Haruko USD-normalised totals:
    delta_usd = USD-equivalent delta; gamma_percent_usd = delta USD change per 1% spot
    move; vega = USD per vol point; theta = USD per day (as reported by Haruko).
    """
    def body(bq) -> str:
        sql = f"""
SELECT {", ".join(_SNAPSHOT_COLS)}
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY position_timestamp DESC) AS rn
  FROM {bq.table(PORTFOLIO)}
  WHERE position_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 DAY)
    AND {entity_sql()}
)
WHERE rn = 1
ORDER BY entity_id
LIMIT 10"""
        df = bq.query(sql)
        if df.empty:
            return "No portfolio snapshot in the last 3 days in fct_otc_haruko_pnl_portfolio."
        lines = ["### Desk risk snapshot (Haruko, fct_otc_haruko_pnl_portfolio)"]
        for _, r in df.iterrows():
            lines += ["", f"**{_entity(r['entity_id'])}** - as of {_ts(r['position_timestamp'])}"
                          + (f" (calc {int(r['hours_since_latest_calculation'])}h old)"
                             if not _isna(r.get("hours_since_latest_calculation")) and int(r["hours_since_latest_calculation"]) > 0 else "")]
            lines.append(f"- book: {int(r['position_count']):,} positions, {int(r['venue_count'])} venues, "
                         f"{int(r['asset_count'])} assets, {int(r['symbol_count']):,} symbols, "
                         f"{int(r['strategy_count']) if not _isna(r.get('strategy_count')) else 'n/a'} strategies")
            lines.append(f"- notional: gross {_usd(r['total_abs_size_usd'])}, net {_usd(r['total_size_usd'])}; "
                         f"equity {_usd(r['total_equity_usd'])}")
            lines.append(f"- PnL: day {_usd(r['total_portfolio_pnl'])} (open {_usd(r['total_open_pnl'])}, realised "
                         f"{_usd(r['total_realised_pnl'])}); WTD {_usd(r['total_week_to_date_pnl'])}; MTD "
                         f"{_usd(r['total_month_to_date_pnl'])}; QTD {_usd(r['total_quarter_to_date_pnl'])}; "
                         f"YTD {_usd(r['total_year_to_date_pnl'])}; LTD {_usd(r['total_life_to_date_pnl'])}")
            lines.append(f"- funding PnL: day {_usd(r['total_funding_pnl'])}, LTD {_usd(r['total_life_to_date_funding_pnl'])}; "
                         f"fees: day {_usd(r['total_fees'])}, LTD {_usd(r['total_life_to_date_fees'])}")
            lines.append(f"- greeks (USD): delta {_usd(r['total_delta_usd'])} (adjusted {_usd(r['total_delta_adjusted_usd'])}, "
                         f"gross {_usd(r['total_abs_delta_usd'])}); gamma {_usd(r['total_gamma_usd'])} "
                         f"(per 1% move {_usd(r['total_gamma_percent_usd'])}); vega {_usd(r['total_vega'])}/vol pt; "
                         f"theta {_usd(r['total_theta'])}/day (BS {_usd(r['total_theta_bs'])})"
                         + (f"; vanna {_num(r['total_vanna'], 0)}, charm {_num(r['total_charm'], 0)}, volga {_num(r['total_volga'], 0)}"
                            if not _isna(r.get("total_vanna")) else ""))
            lines.append(f"- risk levels: delta '{r['portfolio_delta_risk_level']}', gamma '{r['portfolio_gamma_risk_level']}', "
                         f"vega '{r['portfolio_vega_risk_level']}'")
            flags = []
            if bool(r.get("large_delta_change_flag")):
                flags.append(f"LARGE DELTA CHANGE ({r.get('delta_change_severity')}, {_usd(r.get('delta_adjusted_usd_change'))} vs previous snapshot)")
            if bool(r.get("large_ytd_pnl_change_flag")):
                flags.append(f"LARGE YTD PnL CHANGE ({r.get('ytd_pnl_change_severity')}, {_usd(r.get('ytd_pnl_change'))})")
            if bool(r.get("large_theta_bs_hourly_change_flag")):
                flags.append("LARGE THETA CHANGE (hourly)")
            if bool(r.get("large_gamma_percent_usd_hourly_change_flag")):
                flags.append("LARGE GAMMA CHANGE (hourly)")
            lines.append("- large-change flags: " + ("; ".join(flags) if flags else
                         f"none (delta change severity '{r.get('delta_change_severity')}', YTD PnL change severity '{r.get('ytd_pnl_change_severity')}')"))
            if not _isna(r.get("total_week_to_date_volume_usd")):
                lines.append(f"- volume: WTD {_usd(r['total_week_to_date_volume_usd'])}, MTD {_usd(r['total_month_to_date_volume_usd'])}")
            lines.append("- " + _dq_line(r["data_quality_flag"], r["valid_pricer_pct"])
                         + (f" ({int(r['valid_pricer_count'])} valid / {int(r['invalid_pricer_count'])} invalid)"
                            if not _isna(r.get("valid_pricer_count")) else ""))
        if len(df) > 1:
            s = df.sum(numeric_only=True)
            lines += ["", f"**Combined ({len(df)} entities)**: gross notional {_usd(s['total_abs_size_usd'])}; equity "
                          f"{_usd(s['total_equity_usd'])}; day PnL {_usd(s['total_portfolio_pnl'])}; YTD "
                          f"{_usd(s['total_year_to_date_pnl'])}; LTD {_usd(s['total_life_to_date_pnl'])}; delta "
                          f"{_usd(s['total_delta_usd'])}; gamma {_usd(s['total_gamma_usd'])}; vega {_usd(s['total_vega'])}; "
                          f"theta {_usd(s['total_theta'])}"]
        lines += ["", _bytes_note(df)]
        return "\n".join(lines)

    return _guarded("get_desk_risk_snapshot", body, scoped=True)


@tool("get_desk_pnl_history")
def get_desk_pnl_history(days: int = 30, by: str = "portfolio") -> str:
    """Daily end-of-day PnL history for the desk over the last N days, at portfolio level or broken down by strategy or venue.

    Use for "how has PnL trended", "PnL by venue over the last week", "which strategy
    made / lost money this month". ``by='portfolio'`` reads the EOD portfolio table (one
    row per entity per day: day PnL, WTD/MTD/YTD/LTD, equity, gross notional, data
    quality; max 90 days). ``by='strategy'`` / ``by='venue'`` aggregate the EOD position
    table (day PnL and latest gross notional / YTD per group, top 6 groups + 'other';
    max 28 days). EOD snapshots are taken ~23:55 UTC; day PnL is Haruko
    ``total_pnl`` (open + realised for that day). Covers the Haruko A1/OTC book only.

    Args:
        days: Days of history (default 30; clamped to 7..90 for portfolio, 7..28 for strategy/venue).
        by: 'portfolio' (default), 'strategy' or 'venue'.
    """
    mode = (by or "portfolio").strip().lower()
    if mode not in ("portfolio", "strategy", "venue"):
        return "Unknown breakdown; use by='portfolio', 'strategy' or 'venue'."

    def body(bq) -> str:
        if mode == "portfolio":
            n = _clamp(days, 30, 7, MAX_HISTORY_DAYS_PORTFOLIO)
            sql = f"""
SELECT as_of_date, as_of_timestamp, entity_id, total_portfolio_pnl, total_week_to_date_pnl, total_month_to_date_pnl,
       total_year_to_date_pnl, total_life_to_date_pnl, total_equity_usd, total_abs_size_usd, total_funding_pnl,
       total_fees, data_quality_flag, valid_pricer_pct
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY as_of_date, entity_id ORDER BY as_of_timestamp DESC) AS rn
  FROM {bq.table(PORTFOLIO_EOD)}
  WHERE as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n} DAY)
    AND {entity_sql()}
)
WHERE rn = 1
ORDER BY as_of_date, entity_id
LIMIT 200"""
            df = bq.query(sql)
            if df.empty:
                return f"No EOD portfolio rows in the last {n} days (fct_otc_haruko_pnl_portfolio_history_eod)."
            lines = [f"### Desk PnL history, portfolio level (EOD, last {n} days, "
                     f"{_date(df['as_of_date'].min())} to {_date(df['as_of_date'].max())})",
                     "Note: for monthly / MTD / YTD derivatives PnL use `get_derivs_pnl_eod` (Carson Levy's EOW "
                     "method, 3pm CT cut, LTD differences) - it is authoritative and matches the EOW report; "
                     "this table is Haruko's ~23:55 UTC portfolio snapshot for intraday / risk context."]
            for eid, g in df.groupby("entity_id", sort=True):
                g = g.sort_values("as_of_date")
                first, last = g.iloc[0], g.iloc[-1]
                lines += ["", f"**{_entity(eid)}** ({len(g)} days; latest EOD {_ts(last['as_of_timestamp'])})",
                          f"- period: sum of day PnL {_usd(g['total_portfolio_pnl'].sum())}; LTD PnL change "
                          f"{_usd(last['total_life_to_date_pnl'] - first['total_life_to_date_pnl'])} "
                          f"({_usd(first['total_life_to_date_pnl'])} -> {_usd(last['total_life_to_date_pnl'])}); "
                          f"latest YTD {_usd(last['total_year_to_date_pnl'])}, MTD {_usd(last['total_month_to_date_pnl'])}",
                          f"- best day {_usd(g['total_portfolio_pnl'].max())} ({_date(g.loc[g['total_portfolio_pnl'].idxmax(), 'as_of_date'])}), "
                          f"worst day {_usd(g['total_portfolio_pnl'].min())} ({_date(g.loc[g['total_portfolio_pnl'].idxmin(), 'as_of_date'])})",
                          "- " + _dq_line(last["data_quality_flag"], last["valid_pricer_pct"]), ""]
                rows = [[_date(r["as_of_date"]), _usd(r["total_portfolio_pnl"]), _usd(r["total_month_to_date_pnl"]),
                         _usd(r["total_year_to_date_pnl"]), _usd(r["total_life_to_date_pnl"]), _usd(r["total_equity_usd"]),
                         _usd(r["total_abs_size_usd"]), _usd(r["total_funding_pnl"]), str(r["data_quality_flag"])]
                        for _, r in g.iterrows()]
                lines += _md_table(["date", "day PnL", "MTD", "YTD", "LTD", "equity", "gross notional", "funding", "DQ flag"], rows)
            lines += ["", _bytes_note(df)]
            return "\n".join(lines)

        n = _clamp(days, 28, 7, MAX_HISTORY_DAYS_GROUPED)
        dim = "COALESCE(p.strategy_name, 'unassigned')" if mode == "strategy" else "COALESCE(p.venue, 'unknown')"
        t = bq.table(POSITIONS_EOD)
        sql = f"""
WITH snap AS (
  SELECT as_of_date, entity_id, MAX(as_of_timestamp) AS as_of_timestamp
  FROM {t}
  WHERE as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n} DAY)
    AND {entity_sql()}
  GROUP BY 1, 2
),
base AS (
  SELECT p.as_of_date, {dim} AS grp, p.total_pnl, p.size_usd, p.year_to_date_pnl, p.pricer_valid
  FROM {t} p
  {_eod_join('p')}
  WHERE p.as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n} DAY)
    AND {strategy_sql('p.strategy_name')}
),
top AS (
  SELECT grp FROM base
  WHERE as_of_date = (SELECT MAX(as_of_date) FROM base)
  GROUP BY grp ORDER BY SUM(ABS(size_usd)) DESC LIMIT 6
)
SELECT b.as_of_date, IF(b.grp IN (SELECT grp FROM top), b.grp, 'other') AS grp,
       SUM(b.total_pnl) AS day_pnl, SUM(b.size_usd) AS gross_usd, SUM(b.year_to_date_pnl) AS ytd_pnl, COUNT(*) AS n_positions,
       COUNTIF(b.pricer_valid) AS n_valid
FROM base b
GROUP BY 1, 2
ORDER BY 1, 2
LIMIT 200"""
        df = bq.query(sql)
        if df.empty:
            return f"No EOD position rows in the last {n} days (fct_otc_haruko_pnl_position_history_eod)."
        pivot = df.pivot_table(index="as_of_date", columns="grp", values="day_pnl", aggfunc="sum").fillna(0.0)
        latest_date = df["as_of_date"].max()
        latest = df[df["as_of_date"] == latest_date].set_index("grp")
        order = list(latest["gross_usd"].abs().sort_values(ascending=False).index)
        cols = [c for c in order if c in pivot.columns] + [c for c in pivot.columns if c not in order]
        pivot = pivot[cols]
        lines = [f"### Desk day PnL by {mode} (EOD, last {n} days, {_date(pivot.index.min())} to {_date(pivot.index.max())}, "
                 f"{scope_label()} combined)", "",
                 f"**Latest EOD ({_date(latest_date)}) by {mode}**", ""]
        has_dq = "n_valid" in latest.columns

        def _valid_share(g):
            if not has_dq or not latest.loc[g, "n_positions"]:
                return "n/a"
            return _pct(100.0 * float(latest.loc[g, "n_valid"]) / float(latest.loc[g, "n_positions"]), 0)

        rows = [[g, _usd(latest.loc[g, 'gross_usd']), _usd(latest.loc[g, 'day_pnl']), _usd(latest.loc[g, 'ytd_pnl']),
                 f"{int(latest.loc[g, 'n_positions']):,}", _valid_share(g), _usd(pivot[g].sum())] for g in cols if g in latest.index]
        lines += _md_table([mode, "gross notional", "day PnL", "YTD PnL", "positions", "valid pricers", f"sum day PnL ({n}d)"], rows)
        if has_dq:
            tot_n, tot_v = float(latest["n_positions"].sum()), float(latest["n_valid"].sum())
            share = 100.0 * tot_v / tot_n if tot_n else None
            lines += ["", "Data quality: " + (f"valid pricers {_pct(share, 1)} of positions at the latest EOD"
                                              + (" - CAVEAT: a low share means greeks/PnL on invalid-pricer positions may be stale or missing."
                                                 if share is not None and share < 90 else "") if share is not None else "n/a")]
        lines += ["", f"**Day PnL by date** (columns: {', '.join(cols)})", ""]
        rows = [[_date(d)] + [_usd(v) for v in pivot.loc[d, cols]] + [_usd(pivot.loc[d, cols].sum())] for d in pivot.index]
        lines += _md_table(["date"] + cols + ["total"], rows)
        lines += ["", f"Period total day PnL: {_usd(pivot.values.sum())}. 'other' = groups outside the top 6 by gross notional. "
                      "Day PnL = Haruko total_pnl (open + realised) per EOD snapshot.", _bytes_note(df)]
        return "\n".join(lines)

    return _guarded("get_desk_pnl_history", body, scoped=True)


@tool("get_derivs_pnl_eod")
def get_derivs_pnl_eod(period: str = "mtd", include_daily: bool = False) -> str:
    """Derivatives desk PnL for a period using Carson Levy's end-of-week (EOW) report method, run live in BigQuery - the authoritative Haruko PnL for monthly, MTD, YTD, weekly and custom ranges.

    Method: uniform 3pm America/Chicago end-of-day cut on
    ``fct_otc_haruko_position_pnl_history`` (one full-book snapshot per day, positions
    de-duplicated), PnL = difference in summed life-to-date PnL between EOD rows.
    Monthly figures therefore never use Haruko's month_to_date column (it resets
    mid-month; August 2026 shows -$18,923 there vs the true +$2,174,523). Total of the
    Derivs Risk + ADSD + AD Hedge Co portfolios combined (positions filtered on
    strategy_name; no per-portfolio split in this method). The query scans ~21 GB (~$0.13) and takes ~45 s cold; the
    result is cached for 15 min so follow-up periods are instant.

    Args:
        period: 'mtd' (default), 'wtd', 'ytd' (reports both Haruko SUM(year_to_date_pnl) and the
            LTD change since the first snapshot), 'last_week' (last complete Mon-Sun week),
            'last_month', 'month:YYYY-MM' (e.g. 'month:2026-08'), or 'range:YYYY-MM-DD..YYYY-MM-DD'.
        include_daily: also list the daily EOD PnL rows of the period (max 31 rows, most recent).
    """
    def body(bq) -> str:
        h = _get_haruko_eod(bq)
        try:
            df = h.frame()
        except HarukoEodError as e:
            return f"Could not run the EOW PnL query: {e}"
        if df.empty:
            return "The EOW PnL query returned no EOD snapshots (fct_otc_haruko_position_pnl_history)."
        try:
            res = h.period(period, df)
        except HarukoEodError as e:
            return f"Bad period {period!r}: {e}"
        latest = h.latest(df)
        first_date = df["eod_date"].iloc[0]

        lines = [f"### Derivatives desk PnL (Haruko, EOW method) - {res.label}",
                 f"Method: {METHOD_LINE}.", ""]
        if res.kind == "ytd":
            lines += [f"**YTD {res.end.year} PnL (Haruko SUM(year_to_date_pnl) at {_date(res.end)} EOD): {_usd(res.ytd_sum)}**",
                      f"**YTD as LTD change since first available snapshot ({_date(first_date)} -> {_date(res.end)}): "
                      f"{_usd(res.ytd_ltd_change)}**"]
        else:
            base_txt = (f"LTD {_usd(res.baseline_ltd)} at the first available snapshot {_date(res.baseline_date)}"
                        if res.from_first_row else f"LTD {_usd(res.baseline_ltd)} at {_date(res.baseline_date)} EOD")
            lines += [f"**{res.label} PnL: {_usd(res.pnl)}** ({base_txt} -> {_usd(res.end_ltd)} at {_date(res.end)} EOD; "
                      f"{res.n_days} EOD days)"]
        if len(res.daily):
            d = res.daily
            best, worst = d.loc[d["daily_pnl"].idxmax()], d.loc[d["daily_pnl"].idxmin()]
            lines.append(f"- best day {_usd(best['daily_pnl'])} ({_date(best['eod_date'])}), worst day "
                         f"{_usd(worst['daily_pnl'])} ({_date(worst['eod_date'])}); OTC trades in period: "
                         f"{int(d['n_trades'].fillna(0).sum()):,}, notional {_usd(d['notional_quote'].fillna(0).sum())}")
        lines.append(f"- Haruko's own columns at {_date(res.end)} EOD, for reference only (they reset mid-period, do not "
                     f"quote them as period PnL): MTD {_usd(res.haruko_mtd)}, WTD {_usd(res.haruko_wtd)}")
        for note in res.notes:
            lines.append(f"- {note}")

        if res.months is not None and len(res.months) > 1:
            lines += ["", "**Monthly PnL (LTD difference at month-end EOD)**", ""]
            rows = []
            for _, m in res.months.iterrows():
                tag = " (MTD, partial)" if m["partial"] else (" (from first snapshot)" if m["from_first_row"] else "")
                rows.append([m["month"] + tag, _usd(m["pnl"]), f"{_date(m['baseline'])} -> {_date(m['end'])}",
                             str(int(m["n_days"])), _usd(m["haruko_mtd"])])
            lines += _md_table(["month", "PnL", "LTD from -> to", "EOD days", "Haruko MTD col (ref)"], rows)

        if include_daily and len(res.daily):
            d = res.daily.tail(31)
            lines += ["", f"**Daily EOD PnL** ({'last 31 of ' if len(res.daily) > 31 else ''}{len(res.daily)} days)", ""]
            rows = [[_date(r["eod_date"]), _usd(r["daily_pnl"]), _usd(r["ltd_pnl"]), f"{int(r['n_positions']):,}",
                     f"{int(r['skew_secs']):+d}s" if not _isna(r["skew_secs"]) else "n/a",
                     f"{int(r['n_trades']):,}" if not _isna(r["n_trades"]) else "0"] for _, r in d.iterrows()]
            lines += _md_table(["EOD date", "day PnL", "LTD PnL", "positions", "cut skew", "trades"], rows)

        lines += ["", f"Latest EOD snapshot: {_date(latest['eod_date'])} (cut skew {int(latest['skew_secs']):+d}s, "
                      f"{int(latest['n_positions']):,} positions)."]
        if latest.get("stale"):
            lines.append(f"WARNING: the latest EOD snapshot is {latest['business_days_behind']} business days old "
                         f"(today {h.today().isoformat()} America/Chicago) - the series may be missing recent days; "
                         "check the Haruko snapshot job before quoting the figure as current.")
        lines += [f"Data coverage: EOD snapshots from {_date(first_date)} (first row) to {_date(latest['eod_date'])}; "
                  f"periods starting earlier are measured from {_date(first_date)}.",
                  f"Caveats: {scope_label()} portfolios combined (no per-portfolio split in this method); Haruko MTD/WTD "
                  "columns shown for reference only; PnL is Haruko mark-to-market life-to-date differences, not the "
                  "spot desk's booked PnL.",
                  f"{_bytes_note(df)} BigQuery cache hit: {_yes_no(df.attrs.get('cache_hit'))}; "
                  + (f"served from the in-process 15-min cache (age {int(df.attrs.get('cache_age_s', 0))}s)."
                     if df.attrs.get("from_cache") else "fresh run (result cached in-process for 15 min).")]
        return "\n".join(lines)

    return _guarded("get_derivs_pnl_eod", body, scoped=True)


def _yes_no(v) -> str:
    return "n/a" if v is None else ("yes" if v else "no")


@tool("get_desk_greeks_history")
def get_desk_greeks_history(days: int = 30) -> str:
    """Daily end-of-day USD greeks (delta, delta-adjusted, gamma, gamma per 1% move, vega, theta) for the desk over the last N days, per entity, with the risk-level labels and data-quality flag.

    Use for "how has our delta / gamma changed this month", "is vega building", "greeks
    trend". Source: fct_otc_haruko_greeks_history_eod (~23:55 UTC snapshots). Units as
    in get_desk_risk_snapshot (Haruko USD-normalised). Covers the Haruko A1/OTC book only.

    Args:
        days: Days of history (default 30, clamped to 7..90).
    """
    def body(bq) -> str:
        n = _clamp(days, 30, 7, MAX_HISTORY_DAYS_PORTFOLIO)
        sql = f"""
SELECT as_of_date, as_of_timestamp, entity_id, position_count, total_abs_size_usd, total_delta_usd, total_delta_adjusted_usd,
       total_abs_delta_usd, total_gamma_usd, total_gamma_percent_usd, total_vega, total_theta, total_theta_bs,
       portfolio_delta_risk_level, portfolio_gamma_risk_level, portfolio_vega_risk_level, data_quality_flag, valid_pricer_pct
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY as_of_date, entity_id ORDER BY as_of_timestamp DESC) AS rn
  FROM {bq.table(GREEKS_EOD)}
  WHERE as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n} DAY)
    AND {entity_sql()}
)
WHERE rn = 1
ORDER BY as_of_date, entity_id
LIMIT 200"""
        df = bq.query(sql)
        if df.empty:
            return f"No EOD greeks rows in the last {n} days (fct_otc_haruko_greeks_history_eod)."
        lines = [f"### Desk greeks history (EOD, last {n} days, {_date(df['as_of_date'].min())} to {_date(df['as_of_date'].max())})"]
        for eid, g in df.groupby("entity_id", sort=True):
            g = g.sort_values("as_of_date")
            first, last = g.iloc[0], g.iloc[-1]

            def chg(col):
                return _usd(last[col] - first[col])

            lines += ["", f"**{_entity(eid)}** ({len(g)} days; latest EOD {_ts(last['as_of_timestamp'])}; "
                          f"{int(last['position_count']):,} positions, gross notional {_usd(last['total_abs_size_usd'])})",
                      f"- change over period: delta {chg('total_delta_usd')}, gamma {chg('total_gamma_usd')}, vega {chg('total_vega')}, "
                      f"theta {chg('total_theta')}",
                      f"- latest risk levels: delta '{last['portfolio_delta_risk_level']}', gamma '{last['portfolio_gamma_risk_level']}', "
                      f"vega '{last['portfolio_vega_risk_level']}'",
                      "- " + _dq_line(last["data_quality_flag"], last["valid_pricer_pct"]), ""]
            rows = [[_date(r["as_of_date"]), _usd(r["total_delta_usd"]), _usd(r["total_delta_adjusted_usd"]), _usd(r["total_gamma_usd"]),
                     _usd(r["total_gamma_percent_usd"]), _usd(r["total_vega"]), _usd(r["total_theta"]), str(r["data_quality_flag"])]
                    for _, r in g.iterrows()]
            lines += _md_table(["date", "delta USD", "delta adj USD", "gamma USD", "gamma/1% USD", "vega", "theta", "DQ flag"], rows)
        lines += ["", "Units: Haruko USD-normalised greeks (vega per vol point, theta per day).", _bytes_note(df)]
        return "\n".join(lines)

    return _guarded("get_desk_greeks_history", body, scoped=True)


def _is_perp(symbol, maturity) -> bool:
    s = str(symbol or "").upper()
    if "PERP" in s or "SWAP" in s:
        return True
    if not _isna(maturity):
        try:
            year = pd.Timestamp(maturity).year
        except Exception:  # noqa: BLE001
            year = None
        if year is not None and year > 1971:
            return False
    return not bool(re.search(r"\d{1,2}[A-Z]{3}\d{2}", s))


@tool("get_perp_positions")
def get_perp_positions(top_n: int = 15) -> str:
    """The desk's perpetual-swap and dated-futures positions (latest EOD Haruko snapshot): symbol, venue, side, size, notional USD, entry vs mark, open / day PnL, funding PnL, delta, plus the live reconciled exchange position where available.

    Use for "what perps are we running", "largest perp positions", "are we paying or
    receiving funding", and to cross-check funding / OI signals from Amberdata for the
    same tokens (the output names the mapped tokens). Source: FUTURES rows of
    fct_otc_haruko_pnl_position_history_eod (EOD, ~23:55 UTC) joined with the live
    fct_otc_haruko_position_summary (futures). Note: fct_perps_positions (raw venue
    feed) is stale since 2026-07-10 and is not used. Sorted by |notional|. Covers the
    Haruko A1/OTC book only.

    Args:
        top_n: Positions to show (default 15, max 50).
    """
    def body(bq) -> str:
        n = _clamp(top_n, 15, 1, MAX_TOP_N)
        t = bq.table(POSITIONS_EOD)
        sql = f"""
WITH {_latest_eod_cte(bq)}
SELECT p.as_of_timestamp, p.entity_id, p.symbol, p.underlying_asset, p.venue, p.position, p.size_coin, p.size_usd,
       p.avg_px, p.mark_px, p.open_pnl, p.realised_pnl, p.total_pnl, p.funding_pnl, p.life_to_date_funding_pnl,
       p.year_to_date_pnl, p.delta_usd, p.maturity, p.pricer_valid, p.data_quality_flag
FROM {t} p
{_eod_join('p')}
WHERE p.as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {EOD_LOOKBACK_DAYS} DAY)
  AND {strategy_sql('p.strategy_name')}
  AND p.instrument_type = 'FUTURES'
  AND (p.position != 0 OR p.size_usd != 0)
ORDER BY ABS(p.size_usd) DESC
LIMIT {n}"""
        df = bq.query(sql)
        if df.empty:
            return "No open futures/perp positions in the latest EOD Haruko snapshot."
        live = pd.DataFrame()
        live_note = ""
        try:
            live = bq.query(f"""
SELECT snapshot_timestamp, symbol, total_exchange_position, total_computed_position, reconciliation_status, venues
FROM {bq.table(POSITION_SUMMARY)}
WHERE position_type = 'futures'
LIMIT 200""")
        except Exception as e:  # noqa: BLE001 - live overlay is optional
            live_note = f"(live position_summary unavailable: {type(e).__name__})"
        live_by_symbol: Dict[str, pd.Series] = {}
        if not live.empty:
            counts = live["symbol"].value_counts()
            for _, r in live.iterrows():
                if counts.get(r["symbol"], 0) == 1:
                    live_by_symbol[str(r["symbol"])] = r

        as_of = df["as_of_timestamp"].max()
        gross = float(df["size_usd"].abs().sum())
        net_delta = float(df["delta_usd"].sum())
        perps = df[[_is_perp(s, m) for s, m in zip(df["symbol"], df["maturity"])]]
        lines = [f"### Desk perp / futures positions (Haruko EOD {_ts(as_of)}; top {len(df)} by notional)",
                 f"- {len(df)} positions shown ({len(perps)} perpetual, {len(df) - len(perps)} dated futures); gross notional "
                 f"{_usd(gross)}; net delta {_usd(net_delta)}; day PnL {_usd(df['total_pnl'].sum())}; day funding "
                 f"{_usd(df['funding_pnl'].sum())}; LTD funding {_usd(df['life_to_date_funding_pnl'].sum())}",
                 f"- invalid pricers on {int((~df['pricer_valid'].astype(bool)).sum())} of {len(df)} rows"
                 + (f"; live exchange positions as of {_ts(live['snapshot_timestamp'].max())}" if not live.empty else " " + live_note), ""]
        rows = []
        tokens = []
        for _, r in df.iterrows():
            side = _sign_side(r["position"] if not _isna(r["position"]) and float(r["position"]) != 0 else r["delta_usd"])
            kind = "perp" if _is_perp(r["symbol"], r["maturity"]) else f"fut {_date(r['maturity'])}"
            lv = live_by_symbol.get(str(r["symbol"]))
            live_qty = _qty(lv["total_exchange_position"]) if lv is not None else "-"
            tokens.append(desk_symbol_to_token(r["underlying_asset"] or r["symbol"]))
            rows.append([str(r["symbol"]), str(r["venue"]), _portfolio(r["entity_id"]), kind, side, _qty(r["size_coin"]),
                         _usd(r["size_usd"]), _num(r["avg_px"], 2), _num(r["mark_px"], 2), _usd(r["open_pnl"]),
                         _usd(r["total_pnl"]), _usd(r["funding_pnl"]), _usd(r["life_to_date_funding_pnl"]),
                         _usd(r["delta_usd"]), live_qty, "ok" if bool(r["pricer_valid"]) else "INVALID"])
        lines += _md_table(["symbol", "venue", "entity", "type", "side", "qty (coin)", "notional", "avg px", "mark", "open PnL",
                            "day PnL", "funding (day)", "funding LTD", "delta USD", "live qty", "pricer"], rows)
        lines += ["", "qty (coin) is Haruko size_coin (unsigned; side from the position sign). Live qty = reconciled exchange "
                      "position from fct_otc_haruko_position_summary, summed per symbol across venues (so BTCUSDT on Binance "
                      "and Bybit show the same figure; Deribit inverse contracts are in USD); '-' when absent. "
                      "Funding PnL positive = received.",
                  _token_hint(tokens), _bytes_note(df)]
        return "\n".join(lines)

    return _guarded("get_perp_positions", body, scoped=True)


@tool("get_desk_positions_by_symbol")
def get_desk_positions_by_symbol(symbol: Optional[str] = None, top_n: int = 20) -> str:
    """Desk exposure and PnL by underlying asset from the latest EOD Haruko snapshot: gross notional, net delta USD, gamma, vega, theta, day and YTD PnL, position counts by instrument type (spot / options / futures) and pricer validity.

    Use to answer "what is our BTC exposure", "which tokens are we long/short", "how much
    HYPE risk do we have", and to cross an Amberdata / Coin Metrics signal with the
    desk's book in the same token. With ``symbol`` (token or desk symbol such as 'btc',
    'BTC-PERP', 'ETHUSDT' - mapped to the underlying) the breakdown is by instrument
    type and venue plus the live spot position; without it, the top N underlyings by
    gross notional. Source: fct_otc_haruko_pnl_position_history_eod (EOD ~23:55 UTC),
    Derivs Risk / ADSD / AD Hedge Co combined. Covers the Haruko A1/OTC book only, not custody.

    Args:
        symbol: Optional token / desk symbol to focus on (e.g. 'btc', 'BTC-PERPETUAL').
        top_n: Underlyings to show when no symbol is given (default 20, max 50).
    """
    def body(bq) -> str:
        n = _clamp(top_n, 20, 1, MAX_TOP_N)
        t = bq.table(POSITIONS_EOD)
        token = desk_symbol_to_token(symbol) if symbol else None
        if symbol and not token:
            return f"Could not interpret symbol '{symbol}'."
        if token and not _SYMBOL_RE.match(token):
            return f"Symbol '{symbol}' contains unsupported characters."
        where_sym = f"AND UPPER(p.underlying_asset) = '{token.upper()}'" if token else ""
        group_cols = "p.instrument_type, p.venue" if token else "p.underlying_asset, p.instrument_type"
        sql = f"""
WITH {_latest_eod_cte(bq)}
SELECT {group_cols}, COUNT(*) AS n_positions, SUM(p.size_usd) AS gross_usd, SUM(p.delta_usd) AS delta_usd,
       SUM(p.gamma_usd) AS gamma_usd, SUM(p.vega) AS vega, SUM(p.theta) AS theta, SUM(p.total_pnl) AS day_pnl,
       SUM(p.year_to_date_pnl) AS ytd_pnl, COUNTIF(p.pricer_valid) AS n_valid, MAX(p.as_of_timestamp) AS as_of
FROM {t} p
{_eod_join('p')}
WHERE p.as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {EOD_LOOKBACK_DAYS} DAY)
  AND {strategy_sql('p.strategy_name')}
  {where_sym}
GROUP BY {group_cols}
ORDER BY ABS(gross_usd) DESC
LIMIT 200"""
        df = bq.query(sql)
        if df.empty:
            return (f"No positions with underlying {token.upper()} in the latest EOD Haruko snapshot." if token
                    else "No positions in the latest EOD Haruko snapshot.")
        as_of = df["as_of"].max()

        if token:
            lines = [f"### Desk exposure in {token.upper()} (Haruko EOD {_ts(as_of)}, {scope_label()})"]
            tot = df.sum(numeric_only=True)
            lines.append(f"- total: {int(tot['n_positions']):,} positions, gross notional {_usd(tot['gross_usd'])}, net delta "
                         f"{_usd(tot['delta_usd'])} ({_sign_side(tot['delta_usd'])}), gamma {_usd(tot['gamma_usd'])}, vega "
                         f"{_usd(tot['vega'])}, theta {_usd(tot['theta'])}; day PnL {_usd(tot['day_pnl'])}, YTD {_usd(tot['ytd_pnl'])}; "
                         f"valid pricers {_pct(100.0 * tot['n_valid'] / tot['n_positions'] if tot['n_positions'] else None)}")
            live_line = ""
            try:
                live = bq.query(f"""
SELECT snapshot_timestamp, symbol, position_type, total_exchange_position, total_computed_position, total_position_delta_value,
       reconciliation_status, venues
FROM {bq.table(POSITION_SUMMARY)}
WHERE UPPER(symbol) = '{token.upper()}'
LIMIT 20""")
                if not live.empty:
                    parts = [f"{r['position_type']} {_qty(r['total_computed_position'])} {token.upper()} "
                             f"(delta value {_usd(r['total_position_delta_value'])}, {r['reconciliation_status']}, venues: {r['venues']})"
                             for _, r in live.iterrows()]
                    live_line = f"- live reconciled position ({_ts(live['snapshot_timestamp'].max())}): " + "; ".join(parts)
            except Exception as e:  # noqa: BLE001
                live_line = f"- live position_summary unavailable: {type(e).__name__}"
            if live_line:
                lines.append(live_line)
            lines += [""]
            rows = [[str(r["instrument_type"]), str(r["venue"]), f"{int(r['n_positions']):,}", _usd(r["gross_usd"]), _usd(r["delta_usd"]),
                     _usd(r["gamma_usd"]), _usd(r["vega"]), _usd(r["theta"]), _usd(r["day_pnl"]), _usd(r["ytd_pnl"]),
                     f"{int(r['n_valid'])}/{int(r['n_positions'])}"] for _, r in df.iterrows()]
            lines += _md_table(["instrument", "venue", "positions", "gross notional", "net delta USD", "gamma USD", "vega", "theta",
                                "day PnL", "YTD PnL", "valid pricers"], rows)
            lines += ["", _token_hint([token]), _bytes_note(df)]
            return "\n".join(lines)

        agg = df.groupby("underlying_asset").agg(
            n_positions=("n_positions", "sum"), gross_usd=("gross_usd", "sum"), delta_usd=("delta_usd", "sum"),
            gamma_usd=("gamma_usd", "sum"), vega=("vega", "sum"), theta=("theta", "sum"), day_pnl=("day_pnl", "sum"),
            ytd_pnl=("ytd_pnl", "sum"), n_valid=("n_valid", "sum"))
        agg = agg.reindex(agg["gross_usd"].abs().sort_values(ascending=False).index).head(n)
        mix = df.pivot_table(index="underlying_asset", columns="instrument_type", values="n_positions", aggfunc="sum").fillna(0)
        lines = [f"### Desk exposure by underlying (Haruko EOD {_ts(as_of)}, {scope_label()}; top {len(agg)} of "
                 f"{df['underlying_asset'].nunique()} underlyings by gross notional)", ""]
        rows = []
        tokens = []
        for u, r in agg.iterrows():
            tok = desk_symbol_to_token(u)
            tokens.append(tok)
            m = mix.loc[u] if u in mix.index else {}
            mix_s = "/".join(f"{int(m.get(k, 0))}" for k in ("SPOT", "OPTIONS", "FUTURES"))
            rows.append([str(u), tok or "-", _usd(r["gross_usd"]), _usd(r["delta_usd"]), _usd(r["gamma_usd"]), _usd(r["vega"]),
                         _usd(r["theta"]), _usd(r["day_pnl"]), _usd(r["ytd_pnl"]), mix_s,
                         _pct(100.0 * r["n_valid"] / r["n_positions"] if r["n_positions"] else None, 0)])
        lines += _md_table(["underlying", "token", "gross notional", "net delta USD", "gamma USD", "vega", "theta", "day PnL",
                            "YTD PnL", "spot/opt/fut", "valid pricers"], rows)
        tot = df.sum(numeric_only=True)
        lines += ["", f"All underlyings: gross notional {_usd(tot['gross_usd'])}, net delta {_usd(tot['delta_usd'])}, day PnL "
                      f"{_usd(tot['day_pnl'])}, YTD {_usd(tot['ytd_pnl'])}. Stablecoin/fiat rows (USD, USDC, USDT...) are balances, not risk.",
                  _token_hint([t for t in tokens if t and t not in ("usd", "usdc", "usdt", "usdg", "pyusd", "usdtb")]), _bytes_note(df)]
        return "\n".join(lines)

    return _guarded("get_desk_positions_by_symbol", body, scoped=True)


@tool("get_otc_derivatives_trades")
def get_otc_derivatives_trades(days: int = 30, top_n: int = 20) -> str:
    """Recent OTC derivatives (bilateral options) trades from the desk blotter: execution time, status, direction, call/put, underlying, quantity, notional (quote), strike, premium, expiry, counterparty and entity, plus a summary by status and product.

    Use for "what OTC options have we traded recently", "open OTC trades with counterparty
    X", "OTC notional this month". Source: fct_otcderivatives_trades joined to
    dim_otcderivatives_products / assets / accounts / entities. Direction is the desk's
    side (BUY = desk bought the option). Statuses: ACTIVE, CLOSED_OUT, MATURED, CANCELED,
    VOIDED. Notional is quantity_total_quote (quote currency, USD). Newest first.

    Args:
        days: Look-back window in days by exec_time (default 30, max 365).
        top_n: Trades to list (default 20, max 50).
    """
    def body(bq) -> str:
        d = _clamp(days, 30, 1, 365)
        n = _clamp(top_n, 20, 1, MAX_TOP_N)
        sql = f"""
SELECT t.exec_time, t.status, t.direction, p.product_type, p.option_type, ba.symbol AS base_symbol, qa.symbol AS quote_symbol,
       SAFE_CAST(t.quantity_total_base AS FLOAT64) AS qty_base, SAFE_CAST(t.quantity_total_quote AS FLOAT64) AS notional_quote,
       SAFE_CAST(t.strike AS FLOAT64) AS strike, SAFE_CAST(t.premium_per_base_unit AS FLOAT64) AS premium_per_unit,
       t.premium_currency_type, t.settlement_type, t.expiration, a.counterparty_name, e.entity_name, t.package_indicator, t.venue
FROM {bq.table(OTC_TRADES)} t
LEFT JOIN {bq.table(OTC_PRODUCTS)} p ON p.id = t.product_id
LEFT JOIN {bq.table(OTC_ASSETS)} ba ON ba.id = p.base_id
LEFT JOIN {bq.table(OTC_ASSETS)} qa ON qa.id = p.quote_id
LEFT JOIN {bq.table(OTC_ACCOUNTS)} a ON a.id = t.account_id
LEFT JOIN {bq.table(OTC_ENTITIES)} e ON e.id = t.entity_id
WHERE t.exec_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {d} DAY)
  AND {otc_entity_sql('t.entity_id')}
ORDER BY t.exec_time DESC
LIMIT {n}"""
        df = bq.query(sql)
        summary = bq.query(f"""
SELECT t.status, p.option_type, ba.symbol AS base_symbol, COUNT(*) AS n_trades,
       SUM(SAFE_CAST(t.quantity_total_quote AS FLOAT64)) AS notional_quote, MAX(t.exec_time) AS last_exec
FROM {bq.table(OTC_TRADES)} t
LEFT JOIN {bq.table(OTC_PRODUCTS)} p ON p.id = t.product_id
LEFT JOIN {bq.table(OTC_ASSETS)} ba ON ba.id = p.base_id
WHERE t.exec_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {d} DAY)
  AND {otc_entity_sql('t.entity_id')}
GROUP BY 1, 2, 3
ORDER BY n_trades DESC
LIMIT 100""")
        if df.empty:
            return f"No OTC derivatives trades executed in the last {d} days (fct_otcderivatives_trades)."
        lines = [f"### OTC derivatives trades, last {d} days (as of {_ts(datetime.now(timezone.utc))}; blotter latest exec {_ts(df['exec_time'].max())})"]
        if not summary.empty:
            tot_n = int(summary["n_trades"].sum())
            tot_notional = float(summary["notional_quote"].fillna(0).sum())
            by_status = summary.groupby("status").agg(n=("n_trades", "sum"), notional=("notional_quote", "sum"))
            lines.append(f"- {tot_n:,} trades, notional {_usd(tot_notional)} in the window; by status: "
                         + "; ".join(f"{s} {int(r['n'])} ({_usd(r['notional'])})" for s, r in by_status.iterrows()))
            by_prod = summary.groupby(["base_symbol", "option_type"]).agg(n=("n_trades", "sum"), notional=("notional_quote", "sum"))
            by_prod = by_prod.sort_values("notional", ascending=False).head(8)
            lines.append("- by underlying / type: " + "; ".join(
                f"{b} {o or '?'} {int(r['n'])} ({_usd(r['notional'])})" for (b, o), r in by_prod.iterrows()))
        lines += ["", f"**Latest {len(df)} trades**", ""]
        rows = []
        for _, r in df.iterrows():
            prem = f"{_num(r['premium_per_unit'], 4)} {r['premium_currency_type'] or ''}".strip()
            rows.append([_ts(r["exec_time"]), str(r["status"]), str(r["direction"]), f"{r['base_symbol']}/{r['quote_symbol']} {r['option_type'] or r['product_type'] or ''}".strip(),
                         _qty(r["qty_base"]), _usd(r["notional_quote"]), _num(r["strike"], 0), prem, _date(r["expiration"]),
                         str(r["counterparty_name"] or "n/a"), str(r["entity_name"] or "n/a")])
        lines += _md_table(["exec time", "status", "side", "product", "qty (base)", "notional", "strike", "premium/unit", "expiry",
                            "counterparty", "entity"], rows)
        lines += ["", "Premium per unit is quoted in the premium currency type (QUOTE = USD per base unit, BASE = in the "
                      "underlying coin). Side is the desk's side.", _bytes_note(df)]
        return "\n".join(lines)

    return _guarded("get_otc_derivatives_trades", body, scoped=True)


@tool("get_open_orders")
def get_open_orders() -> str:
    """Live open orders on Talos for the A1 desk: symbol, side, amount and currency, limit price, filled / remaining, strategy, status, counterparty, end time.

    Use for "what orders are working", "do we have resting bids in XRP". Source:
    fct_a1_talos_open_orders_live (refreshed continuously; the loaded_at time is
    reported as the as-of). Amount is in the stated currency (base coin or USD).
    """
    def body(bq) -> str:
        sql = f"""
SELECT loaded_at, counterparty, order_id, symbol, side, strategy, amount, currency, limit_price, cum_qty, leaves_qty, amount_left,
       avg_px_all_in, status, hours_left, start_time, end_time, is_multileg, last_market
FROM {bq.table(OPEN_ORDERS)}
ORDER BY symbol, side, limit_price
LIMIT 200"""
        df = bq.query(sql)
        if df.empty:
            return "No open Talos orders right now (fct_a1_talos_open_orders_live is empty)."
        as_of = df["loaded_at"].max()
        lines = [f"### Open Talos orders (as of {_ts(as_of)}; {len(df)} orders)"]
        by_sym = df.groupby(["symbol", "side"]).size()
        lines.append("- by symbol/side: " + "; ".join(f"{s} {sd.lower()} x{c}" for (s, sd), c in by_sym.items()))
        lines.append("")
        rows = []
        for _, r in df.iterrows():
            filled = _qty(r["cum_qty"]) if not _isna(r["cum_qty"]) else "0"
            rows.append([str(r["symbol"]), str(r["side"]), f"{_qty(r['amount'])} {r['currency']}", _num(r["limit_price"], 4 if not _isna(r['limit_price']) and abs(float(r['limit_price'])) < 10 else 2),
                         filled, _qty(r["leaves_qty"]) if not _isna(r["leaves_qty"]) else _qty(r["amount_left"]),
                         str(r["strategy"] or "-"), str(r["status"]), str(r["counterparty"] or "-"),
                         _num(r["hours_left"], 1) if not _isna(r["hours_left"]) else "-", str(r["end_time"] or "-")])
        lines += _md_table(["symbol", "side", "amount", "limit", "filled", "remaining", "strategy", "status", "counterparty",
                            "hours left", "end time"], rows)
        lines += ["", _bytes_note(df)]
        return "\n".join(lines)

    return _guarded("get_open_orders", body)


@tool("get_internal_price")
def get_internal_price(asset: str, hours: int = 24) -> str:
    """The desk's internal price for an asset: latest live tick (pricing.fct_current_asset_prices_live, with source and timestamp) plus the intraday path over the last N hours from pricing.intraday_price (minute bars, USD quote) - open, high, low, last, change %, and an hourly table.

    Use for "what is our internal BTC price", "how did HYPE move over the last 6 hours
    on our feed", or to compare the internal mark with Coin Metrics / Amberdata prices.
    Intraday bars may lag ~1 hour. Coin Metrics hourly reference rates also exist in
    pricing.fct_coinmetrics_hourly_prices (lowercase asset, ~2-day lag, expensive scan)
    - reach them via query_desk_data with a `date` filter if needed.

    Args:
        asset: Symbol such as 'BTC', 'eth', 'HYPE' (letters/digits/underscore only).
        hours: Look-back window for the intraday path (default 24, 1..168).
    """
    def body(bq) -> str:
        sym = (asset or "").strip().upper()
        if not _SYMBOL_RE.match(sym):
            return f"Invalid asset symbol '{asset}'. Use letters/digits only, e.g. 'BTC'."
        h = _clamp(hours, 24, 1, 168)
        live = bq.query(f"""
SELECT canonical_symbol, price, provider_timestamp, source
FROM {bq.table(LIVE_PRICES)}
WHERE UPPER(canonical_symbol) = '{sym}'
ORDER BY provider_timestamp DESC
LIMIT 5""")
        intraday = bq.query(f"""
SELECT TIMESTAMP_TRUNC(price_timestamp_minute, HOUR) AS hour,
       ARRAY_AGG(SAFE_CAST(exchange_price AS FLOAT64) ORDER BY price_timestamp_minute DESC LIMIT 1)[OFFSET(0)] AS price,
       MAX(SAFE_CAST(exchange_price AS FLOAT64)) AS high, MIN(SAFE_CAST(exchange_price AS FLOAT64)) AS low,
       MAX(price_timestamp_minute) AS last_minute, COUNT(*) AS n_bars
FROM {bq.table(INTRADAY_PRICES)}
WHERE base_symbol = '{sym}' AND quote_symbol = 'USD'
  AND price_timestamp_minute >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {h} HOUR)
GROUP BY 1
ORDER BY 1
LIMIT 200""")
        if live.empty and intraday.empty:
            return f"No internal price for {sym}: not in fct_current_asset_prices_live and no USD intraday bars in the last {h} hours."
        lines = [f"### Internal price: {sym}"]
        if not live.empty:
            r = live.iloc[0]
            lines.append(f"- live: {_usd(r['price'], 2 if float(r['price']) >= 1 else 6)} ({r['source']}, {_ts(r['provider_timestamp'])})")
            if len(live) > 1:
                lines.append("- other sources: " + "; ".join(f"{x['source']} {_usd(x['price'], 2 if float(x['price']) >= 1 else 6)} ({_ts(x['provider_timestamp'])})"
                                                             for _, x in live.iloc[1:].iterrows()))
        else:
            lines.append("- live tick: not available in fct_current_asset_prices_live")
        if not intraday.empty:
            first, last = intraday.iloc[0], intraday.iloc[-1]
            chg = (float(last["price"]) / float(first["price"]) - 1) * 100 if float(first["price"]) else None
            dec = 2 if float(last["price"]) >= 1 else 6
            lines.append(f"- intraday (USD bars, last {h}h, {int(intraday['n_bars'].sum()):,} minutes, {_ts(first['hour'])} to "
                         f"{_ts(last['last_minute'])}): first {_usd(first['price'], dec)}, last {_usd(last['price'], dec)}, "
                         + (f"change {chg:+.2f}%, " if chg is not None else "") + f"high {_usd(intraday['high'].max(), dec)}, low {_usd(intraday['low'].min(), dec)}")
            step = max(1, math.ceil(len(intraday) / 24))
            shown = intraday.iloc[::-1].iloc[::step].iloc[::-1]
            lines += ["", f"| hour (UTC) | last | high | low |", "|---|---|---|---|"]
            lines += [f"| {_ts(r['hour'])[:16]} | {_usd(r['price'], dec)} | {_usd(r['high'], dec)} | {_usd(r['low'], dec)} |" for _, r in shown.iterrows()]
            if step > 1:
                lines.append(f"(every {step}th hour shown)")
        else:
            lines.append(f"- intraday: no USD minute bars for {sym} in the last {h} hours")
        tok = desk_symbol_to_token(sym)
        lines += ["", _token_hint([tok] if tok else []), _bytes_note(intraday if not intraday.empty else live)]
        return "\n".join(lines)

    return _guarded("get_internal_price", body)


@tool("list_desk_tables")
def list_desk_tables(keyword: str = "") -> str:
    """List the BigQuery tables/views the agent can read in the desk datasets (brokerage_a1, pricing), optionally filtered by a keyword matched against table names, descriptions and column names.

    Use to discover where something lives ("which tables track PnL by venue",
    "is there a funding table") before describe_desk_table / query_desk_data. Returns
    table, type, row count (tables only; views show n/a), partitioning and matching
    columns. Metadata comes from a 24 h cache. Other datasets (custody/HOLD, CRMS,
    lending, marketdata, reporting...) are not accessible.

    Args:
        keyword: Optional case-insensitive filter, e.g. 'venue', 'funding', 'perps'.
    """
    def body(bq) -> str:
        rows = bq.list_tables(keyword=keyword or None)
        if not rows:
            return (f"No table in {', '.join(bq.allowed_datasets)} matches '{keyword}'." if keyword
                    else f"No tables found in {', '.join(bq.allowed_datasets)}.")
        shown = rows[:60]
        lines = [f"### Desk tables in {bq.data_project} ({len(rows)} match" + (f" '{keyword}'" if keyword else "") + f"; showing {len(shown)})", ""]
        out = []
        for r in shown:
            desc = (r.get("description") or "").replace("\n", " ")
            desc = desc if len(desc) <= 70 else desc[:67] + "..."
            is_view = str(r.get("type") or "").upper().endswith("VIEW")
            out.append([r["table"], str(r.get("type") or ""),
                        f"{r['num_rows']:,}" if r.get("num_rows") is not None and not is_view else "n/a",
                        r.get("partitioning") or "-", str(r.get("n_columns")), ", ".join(r.get("matched_columns") or []) or "-", desc])
        lines += _md_table(["table", "type", "rows", "partitioning", "cols", "matching columns", "description"], out)
        if len(rows) > len(shown):
            lines.append(f"...{len(rows) - len(shown)} more; narrow the keyword.")
        lines += ["", "Use describe_desk_table('dataset.table') for columns and query_desk_data(sql) to read. Views report n/a rows; "
                      "some Haruko convenience views scan >2 GB and will be refused by the cost cap - prefer the partitioned "
                      "*_history_eod tables with an as_of_date filter."]
        return "\n".join(lines)

    return _guarded("list_desk_tables", body)


@tool("describe_desk_table")
def describe_desk_table(table: str) -> str:
    """Schema of one desk table: columns with types and descriptions, table type, row count, size, last modified, partitioning and clustering.

    Use before writing SQL with query_desk_data. ``table`` is 'dataset.table', e.g.
    'brokerage_a1.fct_otc_haruko_pnl_by_venue_history' or 'pricing.intraday_price'.
    Only brokerage_a1 and pricing are accessible.

    Args:
        table: 'dataset.table' (project prefix optional).
    """
    def body(bq) -> str:
        info = bq.describe_table(table)
        lines = [f"### {bq.data_project}.{info['table']} ({info.get('type') or 'TABLE'})"]
        if info.get("description"):
            lines.append(info["description"].strip())
        meta = []
        if info.get("num_rows") is not None:
            meta.append(f"rows {info['num_rows']:,}")
        if info.get("num_bytes"):
            meta.append(f"size {format_bytes(info['num_bytes'])}")
        if info.get("modified"):
            meta.append(f"modified {str(info['modified'])[:16].replace('T', ' ')} UTC")
        meta.append(f"partitioning {info.get('partitioning') or 'none'}")
        if info.get("clustering"):
            meta.append("clustered by " + ", ".join(info["clustering"]))
        lines.append("- " + "; ".join(meta))
        cols = info.get("columns") or []
        lines += ["", f"**Columns ({len(cols)})**", ""]
        shown = cols[:150]
        lines += _md_table(["column", "type", "description"],
                           [[c["name"], c["type"] + ("" if (c.get("mode") or "NULLABLE") == "NULLABLE" else f" {c['mode']}"),
                             (c.get("description") or "").replace("\n", " ")[:90]] for c in shown])
        if len(cols) > len(shown):
            lines.append(f"...{len(cols) - len(shown)} more columns.")
        if info.get("partitioning"):
            lines += ["", f"Filter on the partition column ({info['partitioning']}) to keep scans under the cost cap."]
        return "\n".join(lines)

    return _guarded("describe_desk_table", body)


@tool("query_desk_data")
def query_desk_data(sql: str) -> str:
    """Run a read-only BigQuery SQL query against the desk datasets and return the result as a markdown table (max 40 rows shown, 200 returned) with the bytes scanned.

    Escape hatch for questions the purpose-built desk tools do not cover. Rules
    enforced by a guard before anything runs: single SELECT/WITH statement only (no
    DML/DDL/scripts); every table must be `anc-global-markets.brokerage_a1.*` or
    `anc-global-markets.pricing.*` (fully qualify names in backticks); a LIMIT of at
    most 200 is applied; the job is capped at 2 GB scanned - filter partitioned tables
    on their partition column (as_of_date, position_timestamp, price_timestamp_minute).
    Use list_desk_tables / describe_desk_table first to get exact column names.
    Timestamps are UTC. Never use this to read anything outside the two datasets - it
    will be refused. Desk scope: any query on a Haruko / OTC table (`fct_otc_haruko_*`,
    `fct_otcderivatives_*`) must filter to the desk's portfolios - `entity_id IN (20, 86, 87)`
    or `strategy_name IN ('Derivs Risk', 'ADSD', 'AD Hedge Co')` - the result warns when it
    does not.

    Args:
        sql: Standard (GoogleSQL) SELECT statement.
    """
    def body(bq) -> str:
        safe = bq.validate(sql)
        df = bq.query(safe)
        if df.empty:
            return f"Query returned no rows. {_bytes_note(df)}"
        shown = df.head(QUERY_DISPLAY_ROWS)
        headers = [str(c) for c in shown.columns]
        rows = [[_cell(v) for v in rec] for rec in shown.itertuples(index=False, name=None)]
        lines = [f"### Query result: {len(df)} row(s)" + (f", showing first {len(shown)}" if len(df) > len(shown) else "")
                 + (f" (row cap {bq.max_rows} reached - add filters/aggregation for more)" if len(df) >= bq.max_rows else ""), ""]
        lines += _md_table(headers, rows)
        lines += ["", _bytes_note(df)]
        if _unscoped_desk_sql(safe):
            lines.append(f"WARNING: this query reads a Haruko / OTC table without a portfolio filter. Desk answers must be "
                         f"limited to {scope_label()}: add `{entity_sql()}` or `{strategy_sql()}` and re-run.")
        return "\n".join(lines)

    return _guarded("query_desk_data", body)


_SCOPED_TABLE_RE = re.compile(r"fct_otc_haruko|fct_otcderivatives|fct_perps", re.IGNORECASE)
_SCOPE_FILTER_RE = re.compile(r"\b(entity_id|strategy_name)\b", re.IGNORECASE)


def _unscoped_desk_sql(sql: str) -> bool:
    """True when a free-form query touches a Haruko / OTC table and never mentions entity_id / strategy_name."""
    return bool(_SCOPED_TABLE_RE.search(sql)) and not _SCOPE_FILTER_RE.search(sql)


def get_desk_tools() -> list:
    """All desk-data tools, in the order they should be registered."""
    return [
        get_desk_risk_snapshot,
        get_desk_pnl_history,
        get_derivs_pnl_eod,
        get_desk_greeks_history,
        get_perp_positions,
        get_desk_positions_by_symbol,
        get_otc_derivatives_trades,
        get_open_orders,
        get_internal_price,
        list_desk_tables,
        describe_desk_table,
        query_desk_data,
    ]


DESK_TOOL_NAMES = [t.name for t in get_desk_tools()]

__all__ = [
    "DESK_TOOL_NAMES",
    "ENTITY_NAMES",
    "desk_symbol_to_token",
    "describe_desk_table",
    "get_derivs_pnl_eod",
    "get_desk_greeks_history",
    "get_desk_pnl_history",
    "get_desk_positions_by_symbol",
    "get_desk_risk_snapshot",
    "get_desk_tools",
    "get_internal_price",
    "get_open_orders",
    "get_otc_derivatives_trades",
    "get_perp_positions",
    "list_desk_tables",
    "query_desk_data",
]
