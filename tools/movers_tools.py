"""Market-wide movers screen over the top-N assets (Coin Metrics hourly candles).

    get_top_movers(n=100, min_cap_usd=1e9)   levels, T-24h and T-7d moves, volume ratio, leaders / laggards

One call replaces a hundred price lookups: for every asset in the ranking it pulls eight days of
hourly bars from the primary spot market (a thread pool, ~30-60 s cold, memoised 10 minutes) and
reports the last close, the change versus 24 hours and 7 days earlier, and today's USD volume
against the trailing seven-day average. Built for the daily commentary playbook; also answers
"what moved today", "biggest gainers above $1B", "who is trading on unusual volume".
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from providers import factory as _factory
from tools.chat_tools import _fmt_price, _fmt_usd
from tools.desk_tools import _md_table
from tools.metrics import FULL_TOKEN_UNIVERSE

logger = logging.getLogger(__name__)

MAX_N = 150
LOOKBACK_HOURS = 8 * 24 + 2
TTL_S = 10 * 60
WORKERS = 8
MAJORS = ("btc", "eth", "sol", "hype")

_memo: dict = {}
_memo_lock = threading.Lock()


def _spot_provider():
    provider = _factory.get_provider()
    return getattr(provider, "spot", provider)


def _pct(new, old) -> Optional[float]:
    try:
        if old is None or new is None or pd.isna(old) or pd.isna(new) or float(old) == 0:
            return None
        return (float(new) / float(old) - 1) * 100
    except (TypeError, ValueError):
        return None


def _close_at_or_before(df: pd.DataFrame, when: pd.Timestamp, slack: pd.Timedelta = pd.Timedelta(hours=3)) -> Optional[float]:
    """Close of the last bar at or before ``when``; the first bar of the window if it starts up to ``slack`` later."""
    prior = df[df["time"] <= when]
    if prior.empty:
        first = df.iloc[0]
        if first["time"] - when <= slack and not pd.isna(first["close"]):
            return float(first["close"])
        return None
    v = prior["close"].iloc[-1]
    return None if pd.isna(v) else float(v)


def _one(spot, symbol: str, now: pd.Timestamp) -> dict:
    rec = {"symbol": symbol, "last": None, "chg_24h": None, "chg_7d": None, "vol_24h": None, "vol_ratio": None,
           "market": None, "as_of": None, "error": None}
    try:
        df = spot.get_intraday_candles(symbol, frequency="1h", lookback_minutes=LOOKBACK_HOURS * 60)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}"
        return rec
    if df is None or df.empty:
        rec["error"] = "no candles"
        return rec
    last_t = df["time"].iloc[-1]
    last = float(df["close"].iloc[-1])
    rec.update({"last": last, "market": df.attrs.get("market"), "as_of": last_t,
                "chg_24h": _pct(last, _close_at_or_before(df, last_t - pd.Timedelta(hours=24))),
                "chg_7d": _pct(last, _close_at_or_before(df, last_t - pd.Timedelta(days=7)))})
    usd = df["usd_volume"].fillna(0.0)
    last24 = usd[df["time"] > last_t - pd.Timedelta(hours=24)].sum()
    prior = usd[(df["time"] <= last_t - pd.Timedelta(hours=24)) & (df["time"] > last_t - pd.Timedelta(days=8))]
    days = max(1.0, (prior.shape[0] / 24.0)) if prior.shape[0] else 0.0
    rec["vol_24h"] = float(last24)
    rec["vol_ratio"] = (float(last24) / (float(prior.sum()) / days)) if days and prior.sum() > 0 else None
    return rec


def movers_frame(n: int = 100, now: Optional[datetime] = None) -> pd.DataFrame:
    """rank, symbol, market_cap, sector, last, chg_24h, chg_7d, vol_24h, vol_ratio, market, as_of, error (memoised)."""
    n = max(10, min(int(n), MAX_N))
    key = ("movers", n)
    t_now = time.time()
    with _memo_lock:
        hit = _memo.get(key)
        if hit is not None and hit[0] > t_now:
            return hit[1].copy()
    universe = _factory.get_universe()
    if universe is None:
        raise RuntimeError("the dynamic universe is not available in this environment")
    try:
        from tools.chat_tools import _sector_classifier
        top = universe.top_assets(n=n, classifier=_sector_classifier())
    except Exception:  # noqa: BLE001
        top = universe.top_assets(n=n)
    top = top.rename(columns={"as_of": "rank_as_of"})     # the per-token rows carry their own as_of
    spot = _spot_provider()
    stamp = pd.Timestamp(now or datetime.now(timezone.utc))
    with ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(lambda s: _one(spot, s, stamp), top["symbol"].tolist()))
    df = top.merge(pd.DataFrame(rows), on="symbol", how="left")
    if "sector" not in df.columns:
        df["sector"] = None
    df["sector1"] = df["sector"].map(lambda v: v[0] if isinstance(v, list) and v else "")
    with _memo_lock:
        _memo[key] = (t_now + TTL_S, df)
    return df.copy()


def _p(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{float(v):+.1f}%"


def _x(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{float(v):.1f}x"


def _row(r) -> list:
    return [str(int(r["rank"])), r["symbol"].upper(), _fmt_price(r["last"]), _p(r["chg_24h"]), _p(r["chg_7d"]),
            _fmt_usd(r["vol_24h"]), _x(r["vol_ratio"]), r.get("sector1") or ""]


@tool("get_top_movers")
def get_top_movers(n: int = 100, min_cap_usd: float = 1e9, leaders: int = 8) -> str:
    """Market-wide movers screen over the top-N assets by market cap: last price on the primary spot market, change versus 24 hours ago (T-24h) and 7 days ago (T-7d), 24h USD volume and its ratio to the trailing 7-day average, plus Messari sector. Returns the majors (BTC, ETH, SOL, HYPE), the leaders and laggards above min_cap_usd, and the smaller names moving on unusual volume.

    Use for "what moved today", "biggest gainers / losers", "who is trading on unusual volume",
    and to build the daily commentary in one call instead of one lookup per token. Takes 30-60
    seconds cold (one candle request per asset), then is cached for 10 minutes.

    Args:
        n: Assets to screen (default 100, max 150).
        min_cap_usd: Market-cap floor for the leaders / laggards table (default $1B).
        leaders: How many leaders and how many laggards to list (default 8).

    Returns:
        Markdown: as-of time, majors table, leaders and laggards above the floor, small caps on volume, breadth.
    """
    try:
        df = movers_frame(n)
    except Exception as e:  # noqa: BLE001
        logger.error("get_top_movers failed: %s", e)
        return f"Error building the movers screen: {type(e).__name__}: {e}"
    ok = df[df["last"].notna()]
    if ok.empty:
        return "No price data came back for the ranking (Coin Metrics candles unavailable)."
    as_of = ok["as_of"].max()
    heads = ["#", "Ticker", "Last", "T-24h", "T-7d", "24h vol", "Vol vs 7d avg", "Sector"]
    out = [f"### Movers across the top {len(df)} assets, as of {pd.Timestamp(as_of).strftime('%Y-%m-%d %H:%M UTC')} "
           f"(hourly closes, primary spot market per token; {len(ok)} of {len(df)} priced)"]
    up, down = int((ok["chg_24h"] > 0).sum()), int((ok["chg_24h"] < 0).sum())
    med24 = ok["chg_24h"].median() if ok["chg_24h"].notna().any() else None
    med7 = ok["chg_7d"].median() if ok["chg_7d"].notna().any() else None
    out.append(f"- breadth T-24h: {up} up / {down} down; median T-24h {_p(med24)}, median T-7d {_p(med7)}")
    majors = ok[ok["symbol"].isin(MAJORS)].set_index("symbol").reindex([m for m in MAJORS if m in ok["symbol"].values]).reset_index()
    if len(majors):
        out.append("")
        out.append("**Majors**")
        out += _md_table(heads, [_row(r) for _, r in majors.iterrows()])
    big = ok[(ok["market_cap"] >= float(min_cap_usd)) & ok["chg_24h"].notna()]
    k = max(1, min(int(leaders), 20))
    if len(big):
        out.append("")
        out.append(f"**Leaders above {_fmt_usd(min_cap_usd)} market cap (T-24h)**")
        out += _md_table(heads, [_row(r) for _, r in big.sort_values("chg_24h", ascending=False).head(k).iterrows()])
        out.append("")
        out.append(f"**Laggards above {_fmt_usd(min_cap_usd)} market cap (T-24h)**")
        out += _md_table(heads, [_row(r) for _, r in big.sort_values("chg_24h").head(k).iterrows()])
    small = ok[(ok["market_cap"] < float(min_cap_usd)) & ok["vol_ratio"].notna()]
    if len(small):
        hot = small[small["vol_ratio"] >= 1.5].sort_values("chg_24h", ascending=False, key=lambda s: s.abs()).head(6)
        if len(hot):
            out.append("")
            out.append(f"**Smaller names (below {_fmt_usd(min_cap_usd)}) moving on unusual volume (>= 1.5x their 7-day average)**")
            out += _md_table(heads, [_row(r) for _, r in hot.iterrows()])
    missing = df[df["last"].isna()]["symbol"].tolist()
    if missing:
        out.append(f"- no candles for: {', '.join(m.upper() for m in missing[:15])}{' ...' if len(missing) > 15 else ''}")
    out.append("_Source: Coin Metrics hourly candles on each token's primary spot market (Coinbase USD where listed); T-24h and "
               "T-7d compare the latest hourly close with the close 24 hours / 7 days earlier; volume ratio = last 24h USD volume "
               "over the trailing 7-day daily average on that market. Sector per Messari. Curated tokens: "
               f"{', '.join(FULL_TOKEN_UNIVERSE[:6])} ..._")
    return "\n".join(out)


def get_movers_tools() -> list:
    return [get_top_movers]


MOVERS_TOOL_NAMES = [t.name for t in get_movers_tools()]

__all__ = ["MOVERS_TOOL_NAMES", "get_movers_tools", "get_top_movers", "movers_frame"]
