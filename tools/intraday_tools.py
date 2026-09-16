"""Live and intraday price tools (Coin Metrics market-level endpoints).

Three @tool functions the chat agent and the Slack bot can call for "what is X trading at
right now", "how did it move in the last hour", and "what is going through the tape":

    get_live_price(token)                              last trade + top of book + 1h/24h change
    get_intraday_candles(token, frequency, lookback)   OHLCV bars, 1m .. 4h, with a summary
    get_recent_trades(token, minutes, min_trade_usd)   tape summary and the largest prints

Everything is per market (Coinbase USD first, then the fallbacks the provider knows), not an
index: the key has no access to Coin Metrics reference rates. Latency measured 2026-09-16:
1-minute candles ~1 minute behind, trades and quotes live. Every answer names the market and
the timestamp so the reader knows what they are looking at.

Provider access goes through ``providers.factory.get_provider()``; the composite exposes
the Coin Metrics provider as ``.spot``. Tests monkeypatch ``get_provider``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from providers import factory as _factory
from providers.coinmetrics import INTRADAY_FREQUENCIES, MAX_TAPE_MINUTES
from tools.chat_tools import _fmt_price, _fmt_usd, validate_tokens
from tools.metrics import FULL_TOKEN_UNIVERSE

logger = logging.getLogger(__name__)

MAX_TABLE_ROWS = 48          # bars printed in full; longer windows print the summary + the last rows
MAX_PRINTS = 10              # largest trades listed by get_recent_trades
DEFAULT_MIN_TRADE_USD = 50_000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _spot_provider():
    """The Coin Metrics provider (composite .spot), or the provider itself when not composed."""
    provider = _factory.get_provider()
    return getattr(provider, "spot", provider)


def _token(token: str) -> tuple[Optional[str], Optional[str]]:
    valid, _, _ = validate_tokens([token])
    if not valid:
        return None, f"Unknown token '{token}'. Supported symbols: {', '.join(FULL_TOKEN_UNIVERSE)}."
    return valid[0], None


def _market_label(market: str) -> str:
    """'coinbase-btc-usd-spot' -> 'Coinbase BTC-USD spot'."""
    parts = market.split("-")
    if len(parts) >= 4:
        return f"{parts[0].capitalize()} {parts[1].upper()}-{parts[2].upper()} {parts[3]}"
    return market


def _ts(dt) -> str:
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _age(dt) -> str:
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    if secs < 90:
        return f"{secs:.0f}s ago"
    if secs < 5400:
        return f"{secs / 60:.0f} min ago"
    return f"{secs / 3600:.1f} h ago"


def _pct(new, old) -> str:
    try:
        if old is None or pd.isna(old) or float(old) == 0 or new is None or pd.isna(new):
            return "n/a"
        return f"{(float(new) / float(old) - 1) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _close_at_or_before(df: pd.DataFrame, when: datetime) -> Optional[float]:
    """Close of the last bar that ended at or before ``when`` (None if no such bar)."""
    if df is None or df.empty:
        return None
    when = pd.Timestamp(when).tz_convert("UTC") if pd.Timestamp(when).tzinfo else pd.Timestamp(when, tz="UTC")
    prior = df[df["time"] <= when]
    if prior.empty:
        return None
    v = prior["close"].iloc[-1]
    return None if pd.isna(v) else float(v)


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

@tool("get_live_price")
def get_live_price(token: str) -> str:
    """Current price of one token right now: last trade, best bid/ask and spread on its primary spot market (Coinbase USD where listed), plus the change over the last hour and 24 hours and the 24h high/low/volume.

    Use for "what is BTC trading at", "where is ETH now", "current price", "spot right now".
    For history use get_intraday_candles (intraday) or get_price_history (daily).

    Args:
        token: Token symbol, e.g. 'btc'.

    Returns:
        Markdown lines naming the market and the timestamp of each figure. Live, not an index:
        last trade and top of book are real time; candles lag the bar close by about a minute.
    """
    tok, err = _token(token)
    if err:
        return err
    try:
        prov = _spot_provider()
        trade = prov.get_latest_trade(tok)
        quote = prov.get_latest_quote(tok)
        hourly = prov.get_intraday_candles(tok, frequency="1h", lookback_minutes=26 * 60)
        minute = prov.get_intraday_candles(tok, frequency="1m", lookback_minutes=5)
    except Exception as e:  # noqa: BLE001
        logger.error("get_live_price(%s) failed: %s", tok, e)
        return f"Error fetching live price for {tok.upper()}: {type(e).__name__}: {e}"

    if trade is None and quote is None and (minute is None or minute.empty):
        return f"No live market data for {tok.upper()} on any of its spot markets right now."

    market = (trade or quote or {}).get("market") or (minute.attrs.get("market") if minute is not None else None)
    lines = [f"### {tok.upper()} live ({_market_label(market)})"]

    if trade:
        lines.append(f"- last trade: {_fmt_price(trade['price'])} for {trade['amount']:.6g} {tok.upper()} "
                     f"({_fmt_usd(trade['usd'])}, {trade['side'] or 'n/a'}) at {_ts(trade['time'])} ({_age(trade['time'])})")
    if quote:
        spread = f"{quote['spread_bp']:.2f} bp" if quote.get("spread_bp") is not None else "n/a"
        lines.append(f"- bid / ask: {_fmt_price(quote['bid'])} / {_fmt_price(quote['ask'])} (mid {_fmt_price(quote['mid'])}, "
                     f"spread {spread}) at {_ts(quote['time'])} ({_age(quote['time'])})")
    if minute is not None and not minute.empty:
        last = minute.iloc[-1]
        lines.append(f"- last 1m bar ({_ts(last['time'])}): close {_fmt_price(last['close'])}, "
                     f"high {_fmt_price(last['high'])}, low {_fmt_price(last['low'])}, volume {_fmt_usd(last['usd_volume'])}")

    ref = (trade or {}).get("price") or (quote or {}).get("mid") or (float(minute.iloc[-1]["close"]) if minute is not None and not minute.empty else None)
    if hourly is not None and not hourly.empty and ref is not None:
        now = datetime.now(timezone.utc)
        c1h = _close_at_or_before(hourly, now - timedelta(hours=1))
        c24 = _close_at_or_before(hourly, now - timedelta(hours=24))
        last24 = hourly[hourly["time"] >= pd.Timestamp(now - timedelta(hours=24))]
        lines.append(f"- change: 1h {_pct(ref, c1h)}, 24h {_pct(ref, c24)}")
        if not last24.empty:
            lines.append(f"- 24h range: high {_fmt_price(last24['high'].max())}, low {_fmt_price(last24['low'].min())}, "
                         f"volume {_fmt_usd(last24['usd_volume'].sum())} on this market")
    lines.append("_Source: Coin Metrics market data for the named market (not an index). Trades and quotes are live; "
                 "candles close ~1 minute behind._")
    return "\n".join(lines)


@tool("get_intraday_candles")
def get_intraday_candles(token: str, frequency: str = "5m", lookback_minutes: int = 180) -> str:
    """Intraday OHLCV bars for one token from its primary spot market, with a summary (open, high, low, last, change, volume, VWAP).

    Use for "how has BTC moved in the last hour / today", "intraday chart", "5-minute candles",
    "what did it do overnight". Frequencies: 1m, 5m, 10m, 15m, 30m, 1h, 4h.

    Args:
        token: Token symbol, e.g. 'eth'.
        frequency: Bar size; one of 1m, 5m, 10m, 15m, 30m, 1h, 4h (default 5m).
        lookback_minutes: Window to fetch, in minutes (default 180; max one week). Windows longer
            than 48 bars print the summary plus the most recent 48 bars.

    Returns:
        Markdown: market, window, summary and a bar table (time, open, high, low, close, USD volume).
    """
    tok, err = _token(token)
    if err:
        return err
    if frequency not in INTRADAY_FREQUENCIES:
        return f"Unsupported frequency '{frequency}'. Use one of: {', '.join(INTRADAY_FREQUENCIES)}."
    try:
        df = _spot_provider().get_intraday_candles(tok, frequency=frequency, lookback_minutes=lookback_minutes)
    except Exception as e:  # noqa: BLE001
        logger.error("get_intraday_candles(%s, %s) failed: %s", tok, frequency, e)
        return f"Error fetching {frequency} candles for {tok.upper()}: {type(e).__name__}: {e}"
    if df is None or df.empty:
        return f"No {frequency} candles for {tok.upper()} in the last {lookback_minutes} minutes on any of its spot markets."

    market = df.attrs.get("market", "?")
    o = float(df["open"].iloc[0]); last = float(df["close"].iloc[-1])
    hi, lo = float(df["high"].max()), float(df["low"].min())
    usd = df["usd_volume"].fillna(0.0)
    vwap = float((df["vwap"].fillna(df["close"]) * usd).sum() / usd.sum()) if usd.sum() > 0 else None
    lines = [
        f"### {tok.upper()} {frequency} candles, {_ts(df['time'].iloc[0])} to {_ts(df['time'].iloc[-1])} "
        f"({len(df)} bars, {_market_label(market)})",
        f"- open {_fmt_price(o)}  last {_fmt_price(last)}  change {_pct(last, o)}",
        f"- high {_fmt_price(hi)}  low {_fmt_price(lo)}  range {(hi - lo) / lo * 100 if lo else 0:.2f}%",
        f"- volume {_fmt_usd(usd.sum())}" + (f"  vwap {_fmt_price(vwap)}" if vwap else "") +
        (f"  trades {int(df['trades'].fillna(0).sum()):,}" if df["trades"].notna().any() else ""),
    ]
    shown = df if len(df) <= MAX_TABLE_ROWS else df.iloc[-MAX_TABLE_ROWS:]
    if len(shown) < len(df):
        lines.append(f"- table shows the last {len(shown)} bars; the summary covers all {len(df)}")
    lines += ["", "| time (UTC) | open | high | low | close | usd volume |", "|---|---|---|---|---|---|"]
    for _, r in shown.iterrows():
        lines.append(f"| {pd.Timestamp(r['time']).strftime('%m-%d %H:%M')} | {_fmt_price(r['open'])} | {_fmt_price(r['high'])} | "
                     f"{_fmt_price(r['low'])} | {_fmt_price(r['close'])} | {_fmt_usd(r['usd_volume'])} |")
    lines.append(f"_Source: Coin Metrics market candles, {_market_label(market)}; bars close ~1 minute behind real time._")
    return "\n".join(lines)


@tool("get_recent_trades")
def get_recent_trades(token: str, minutes: int = 5, min_trade_usd: float = DEFAULT_MIN_TRADE_USD) -> str:
    """The tape for one token over the last few minutes on its primary spot market: trade count, buy vs sell notional (taker side), VWAP, and the largest individual prints.

    Use for "what is going through right now", "any big prints", "is the tape buy- or
    sell-heavy". Max window 60 minutes.

    Args:
        token: Token symbol, e.g. 'sol'.
        minutes: Window in minutes (default 5, max 60).
        min_trade_usd: Only prints at or above this USD size are listed individually (default 50,000).

    Returns:
        Markdown: window, summary lines, and up to 10 largest prints (time, side, size, price, USD).
    """
    tok, err = _token(token)
    if err:
        return err
    minutes = max(1, min(int(minutes), MAX_TAPE_MINUTES))
    try:
        df = _spot_provider().get_recent_trades(tok, minutes=minutes)
    except Exception as e:  # noqa: BLE001
        logger.error("get_recent_trades(%s) failed: %s", tok, e)
        return f"Error fetching trades for {tok.upper()}: {type(e).__name__}: {e}"
    if df is None or df.empty:
        return f"No trades for {tok.upper()} in the last {minutes} minutes on any of its spot markets."

    market = df.attrs.get("market", "?")
    total = float(df["usd"].sum())
    side = df["side"].str.lower()
    buys, sells = float(df.loc[side == "buy", "usd"].sum()), float(df.loc[side == "sell", "usd"].sum())
    vwap = float((df["price"] * df["amount"]).sum() / df["amount"].sum()) if df["amount"].sum() > 0 else None
    first, last = float(df["price"].iloc[0]), float(df["price"].iloc[-1])
    lines = [
        f"### {tok.upper()} tape, last {minutes} min ({_market_label(market)}), "
        f"{_ts(df['time'].iloc[0])} to {_ts(df['time'].iloc[-1])}",
        f"- {len(df):,} trades, {_fmt_usd(total)} notional" + (f", vwap {_fmt_price(vwap)}" if vwap else ""),
        f"- taker buys {_fmt_usd(buys)} ({buys / total * 100 if total else 0:.0f}%)  taker sells {_fmt_usd(sells)} "
        f"({sells / total * 100 if total else 0:.0f}%)",
        f"- first {_fmt_price(first)}  last {_fmt_price(last)}  change {_pct(last, first)}",
    ]
    big = df[df["usd"] >= float(min_trade_usd)].sort_values("usd", ascending=False).head(MAX_PRINTS)
    if big.empty:
        lines.append(f"- no single print at or above {_fmt_usd(min_trade_usd)}; largest was {_fmt_usd(df['usd'].max())}")
    else:
        lines += [f"- largest prints (>= {_fmt_usd(min_trade_usd)}):", "", "| time (UTC) | side | size | price | usd |", "|---|---|---|---|---|"]
        for _, r in big.iterrows():
            lines.append(f"| {pd.Timestamp(r['time']).strftime('%H:%M:%S')} | {r['side']} | {r['amount']:.6g} | "
                         f"{_fmt_price(r['price'])} | {_fmt_usd(r['usd'])} |")
    lines.append(f"_Source: Coin Metrics market trades, {_market_label(market)}; live. Side is the taker (aggressor) side._")
    return "\n".join(lines)


def get_intraday_tools() -> list:
    """Live / intraday price tools, in registration order."""
    return [get_live_price, get_intraday_candles, get_recent_trades]


__all__ = ["get_live_price", "get_intraday_candles", "get_recent_trades", "get_intraday_tools"]
