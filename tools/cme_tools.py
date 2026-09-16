"""@tool functions for CME crypto futures and BTC ETF on-chain flows (Coin Metrics).

    get_cme_curve(token, include_micro)             active outrights: close, basis vs spot, volume, OI
    get_cme_open_interest(token, days, contract)    daily OI / volume history, CME share of all venues
    get_btc_etf_onchain_flows(days, hourly)         Coin Metrics on-chain ETF in/out flows and ETF supply

CME publishes open interest once a day (21:00 UTC); candles are daily closes, not settlement
prices; there is no CME mark or index on our key, so basis is computed against the last trade
on the token's primary spot market (Coinbase USD for the majors) and every answer says so.
Provider access goes through ``providers.factory.get_provider()`` (composite ``.spot``).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from providers import factory as _factory
from providers.coinmetrics import CME_BASES
from tools.chat_tools import _fmt_price as _price_fmt, _fmt_usd as _usd_fmt
from tools.desk_tools import _md_table
from tools.intraday_tools import _market_label, _ts

logger = logging.getLogger(__name__)

MAX_HISTORY_ROWS = 45


def _spot_provider():
    provider = _factory.get_provider()
    return getattr(provider, "spot", provider)


def _base(token: str) -> str:
    return (token or "").strip().lower()


def _fmt_usd(v) -> str:
    """NA-safe USD (pandas NA / NaN / None -> n/a, exact zero -> $0)."""
    if v is None or pd.isna(v):
        return "n/a"
    return "$0" if float(v) == 0 else _usd_fmt(float(v))


def _fmt_price(v) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return _price_fmt(float(v))


def _pct(v, digits: int = 2) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{float(v):+.{digits}f}%"


def _n(v, digits: int = 0) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{float(v):,.{digits}f}"


def _day(t) -> str:
    return "n/a" if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d")


@tool("get_cme_curve")
def get_cme_curve(token: str, include_micro: bool = False) -> str:
    """CME futures curve for BTC, ETH, SOL or XRP: every active outright contract with its last daily close, basis versus spot (simple and annualised), the day's USD volume and CME open interest (contracts, USD, coins).

    Use for "CME basis", "where is the CME curve", "front-month premium", "how much open
    interest is on CME", "contango / backwardation on CME". Standard contracts by default;
    include_micro adds the micro contracts (MBT, MET, MSL, MXP).

    Args:
        token: 'btc', 'eth', 'sol' or 'xrp'.
        include_micro: Also list micro contracts (default False).

    Returns:
        Markdown table plus the spot reference used (market, price, time), the candle and OI
        as-of dates and the basis convention (close / spot - 1, annualised ACT/365).
    """
    b = _base(token)
    if b not in CME_BASES:
        return f"CME lists crypto futures we can read for {', '.join(x.upper() for x in CME_BASES)}; '{token}' is not one of them."
    try:
        df = _spot_provider().cme_curve(b, include_micro=include_micro)
    except Exception as e:  # noqa: BLE001
        logger.error("get_cme_curve(%s) failed: %s", b, e)
        return f"Error fetching the CME curve for {b.upper()}: {type(e).__name__}: {e}"
    if df is None or df.empty:
        return f"No CME futures data returned for {b.upper()} (no active contracts with candles or open interest)."
    spot = df.attrs.get("spot")
    lines = [f"### CME {b.upper()} futures curve ({len(df)} contract{'s' if len(df) != 1 else ''}"
             + ("" if include_micro else ", standard size only") + ")"]
    if spot:
        lines.append(f"- spot reference: {_fmt_price(spot)} last trade on {_market_label(df.attrs.get('spot_market') or '')} "
                     f"at {_ts(df.attrs['spot_time'])}")
    else:
        lines.append("- spot reference unavailable: basis not computed")
    lines.append(f"- futures closes as of {_day(df.attrs.get('as_of'))} (daily candle close); open interest as of "
                 f"{_day(df.attrs.get('oi_as_of'))} (CME publishes OI once a day, 21:00 UTC)")
    rows = []
    for _, r in df.iterrows():
        size = f"{r['contract_size']:g} {b.upper()}" if pd.notna(r["contract_size"]) else "?"
        rows.append([f"{r['symbol']} ({r['label']})", size, f"{r['days_to_expiry']:.0f}", _fmt_price(r["close"]),
                     _pct(r["basis_pct"]), _pct(r["basis_ann_pct"], 1), _fmt_usd(r["usd_volume"]),
                     _n(r["oi_contracts"]), _fmt_usd(r["oi_usd"]), _n(r["oi_base"])])
    lines.append("")
    lines += _md_table(["Contract", "Size", "Days", "Close", "Basis", "Ann. basis", "Volume", "OI (contracts)",
                        "OI (USD)", f"OI ({b.upper()})"], rows)
    std = df[df["is_standard"] & df["oi_usd"].notna()]
    if len(std):
        tot_usd = float(std["oi_usd"].sum()); tot_base = float(std["oi_base"].fillna(0).sum())
        front = std.iloc[0]
        lines.append(f"- total standard-contract OI {_fmt_usd(tot_usd)} ({_n(tot_base)} {b.upper()}); front month "
                     f"{front['symbol']} carries {front['oi_usd'] / tot_usd * 100 if tot_usd else 0:.0f}% of it")
    ann = df[df["is_standard"] & df["basis_ann_pct"].notna()]
    if len(ann) >= 2:
        shape = "upward-sloping (contango)" if ann["close"].iloc[-1] > ann["close"].iloc[0] else "downward-sloping (backwardation)"
        lines.append(f"- curve {shape}: {ann['symbol'].iloc[0]} {_pct(ann['basis_ann_pct'].iloc[0], 1)} to "
                     f"{ann['symbol'].iloc[-1]} {_pct(ann['basis_ann_pct'].iloc[-1], 1)} annualised")
    lines.append("_Source: Coin Metrics CME market data. Basis = daily close / spot last trade - 1 (not a synchronous "
                 "mark); annualised = basis x 365 / days to expiry (simple, ACT/365). Weekly Bitcoin Friday futures excluded._")
    return "\n".join(lines)


@tool("get_cme_open_interest")
def get_cme_open_interest(token: str, days: int = 30, contract: str = "") -> str:
    """Daily CME open interest and volume history for BTC, ETH, SOL or XRP: all active outrights summed (standard + micro + weekly) or one contract, with the all-venue futures open interest from Coin Metrics so CME's share can be quoted.

    Use for "how has CME open interest changed", "CME OI trend", "is CME positioning growing",
    "CME share of BTC futures OI", or a single contract's OI / volume / close history.

    Args:
        token: 'btc', 'eth', 'sol' or 'xrp'.
        days: History window in days (default 30, max 365).
        contract: Optional single contract symbol, e.g. 'BTCZ6' (default: all active outrights summed).

    Returns:
        Markdown summary (latest OI, change over the window, CME share) and a daily table
        (date, OI contracts, OI USD, volume USD, close for single contracts).
    """
    b = _base(token)
    if b not in CME_BASES:
        return f"CME lists crypto futures we can read for {', '.join(x.upper() for x in CME_BASES)}; '{token}' is not one of them."
    days = max(2, min(int(days), 365))
    try:
        df = _spot_provider().cme_history(b, contract=contract or None, days=days)
    except Exception as e:  # noqa: BLE001
        logger.error("get_cme_open_interest(%s) failed: %s", b, e)
        return f"Error fetching CME history for {b.upper()}: {type(e).__name__}: {e}"
    if df is None or df.empty:
        what = f"contract {contract.upper()}" if contract else f"{b.upper()} futures"
        return f"No CME open interest / volume history returned for {what} over the last {days} days."
    single = df.attrs.get("contract")
    label = f"{contract.upper()}" if single else f"all active {b.upper()} outrights ({len(df.attrs.get('contracts', []))} contracts, standard + micro + weekly)"
    oi = df.dropna(subset=["oi_usd"])
    lines = [f"### CME {b.upper()} open interest, last {days} days - {label}"]
    if len(oi):
        last, first = oi.iloc[-1], oi.iloc[0]
        chg = (last["oi_usd"] / first["oi_usd"] - 1) * 100 if first["oi_usd"] else None
        lines.append(f"- latest OI {_fmt_usd(last['oi_usd'])} ({_n(last['oi_contracts'])} contracts) as of {_day(last['time'])}; "
                     f"{_pct(chg)} vs {_day(first['time'])} ({_fmt_usd(first['oi_usd'])})")
        lines.append(f"- window high {_fmt_usd(oi['oi_usd'].max())} on {_day(oi.loc[oi['oi_usd'].idxmax(), 'time'])}, "
                     f"low {_fmt_usd(oi['oi_usd'].min())} on {_day(oi.loc[oi['oi_usd'].idxmin(), 'time'])}")
        share = pd.to_numeric(oi["cme_share_oi_pct"], errors="coerce").dropna()
        if not single and len(share):
            lines.append(f"- CME share of all-venue {b.upper()} futures OI: {share.iloc[-1]:.1f}% latest "
                         f"(window average {share.mean():.1f}%; all venues {_fmt_usd(oi['all_venue_oi_usd'].iloc[-1])})")
    vol = df.dropna(subset=["usd_volume"])
    if len(vol):
        lines.append(f"- volume: {_fmt_usd(vol['usd_volume'].iloc[-1])} on {_day(vol['time'].iloc[-1])}; "
                     f"daily average {_fmt_usd(vol['usd_volume'].mean())} over {len(vol)} day{'s' if len(vol) != 1 else ''}")
    shown = df if len(df) <= MAX_HISTORY_ROWS else df.iloc[-MAX_HISTORY_ROWS:]
    heads = ["Date", "OI (contracts)", "OI (USD)", "Volume (USD)"] + (["Close"] if single else ["All-venue OI", "CME share"])
    rows = []
    for _, r in shown.iterrows():
        row = [_day(r["time"]), _n(r["oi_contracts"]), _fmt_usd(r["oi_usd"]), _fmt_usd(r["usd_volume"])]
        if single:
            row.append(_fmt_price(r.get("close")))
        else:
            row += [_fmt_usd(r.get("all_venue_oi_usd")), "n/a" if pd.isna(r.get("cme_share_oi_pct")) else f"{float(r['cme_share_oi_pct']):.1f}%"]
        rows.append(row)
    lines.append("")
    lines += _md_table(heads, rows)
    if len(shown) < len(df):
        lines.append(f"(table shows the last {len(shown)} of {len(df)} days)")
    lines.append("_Source: Coin Metrics CME market data (OI published once a day at 21:00 UTC; volume from daily candles) "
                 "and Coin Metrics reported all-venue futures OI / volume. CME rows without OI are days CME did not publish (weekends)._")
    return "\n".join(lines)


@tool("get_btc_etf_onchain_flows")
def get_btc_etf_onchain_flows(days: int = 30, hourly: bool = False) -> str:
    """Bitcoin ETF flows and holdings inferred on-chain by Coin Metrics: daily (or hourly) USD inflow, outflow and net flow into ETF-labelled addresses, plus total ETF-held BTC and its USD value.

    Use for "BTC ETF flows", "are ETFs buying", "how much BTC do the ETFs hold", especially
    intraday (hourly) before issuers publish. BTC only; for issuer-reported flows across
    assets use the Messari ETF tools when available.

    Args:
        days: Window in days (default 30, max 730; hourly windows are capped at 7 days).
        hourly: Hourly flows instead of daily (default False).

    Returns:
        Markdown summary (latest net flow, window totals, holdings) and a table of the last rows.
    """
    freq = "1h" if hourly else "1d"
    days = max(1, min(int(days), 7 if hourly else 730))
    try:
        df = _spot_provider().etf_onchain_flows("btc", days=days, frequency=freq)
    except Exception as e:  # noqa: BLE001
        logger.error("get_btc_etf_onchain_flows failed: %s", e)
        return f"Error fetching BTC ETF on-chain flows: {type(e).__name__}: {e}"
    if df is None or df.empty:
        return f"No BTC ETF on-chain flow data returned for the last {days} days."
    last = df.iloc[-1]
    net = df["net_flow_usd"]
    lines = [f"### BTC ETF on-chain flows, last {days} day{'s' if days != 1 else ''} ({'hourly' if hourly else 'daily'}, Coin Metrics)"]
    lines.append(f"- latest {'hour' if hourly else 'day'} ({_ts(last['time']) if hourly else _day(last['time'])}): "
                 f"net {_fmt_usd(last['net_flow_usd'])} (in {_fmt_usd(last['flow_in_usd'])}, out {_fmt_usd(last['flow_out_usd'])})")
    pos, neg = int((net > 0).sum()), int((net < 0).sum())
    lines.append(f"- window: net {_fmt_usd(net.sum())} over {len(df)} {'hours' if hourly else 'days'} "
                 f"({pos} net-inflow, {neg} net-outflow); largest inflow {_fmt_usd(net.max())} on "
                 f"{_day(df.loc[net.idxmax(), 'time'])}, largest outflow {_fmt_usd(net.min())} on {_day(df.loc[net.idxmin(), 'time'])}")
    sup = df.dropna(subset=["supply_btc"])
    if len(sup):
        s_last, s_first = sup.iloc[-1], sup.iloc[0]
        lines.append(f"- ETF-held supply {_n(s_last['supply_btc'])} BTC ({_fmt_usd(s_last['supply_usd'])}) as of {_day(s_last['time'])}; "
                     f"{_n(s_last['supply_btc'] - s_first['supply_btc'])} BTC vs {_day(s_first['time'])}")
    shown = df if len(df) <= MAX_HISTORY_ROWS else df.iloc[-MAX_HISTORY_ROWS:]
    rows = [[_ts(r["time"]) if hourly else _day(r["time"]), _fmt_usd(r["flow_in_usd"]), _fmt_usd(r["flow_out_usd"]),
             _fmt_usd(r["net_flow_usd"]), _n(r["supply_btc"]) if pd.notna(r["supply_btc"]) else ""] for _, r in shown.iterrows()]
    lines.append("")
    lines += _md_table(["Time" if hourly else "Date", "Inflow", "Outflow", "Net", "ETF supply (BTC)"], rows)
    if len(shown) < len(df):
        lines.append(f"(table shows the last {len(shown)} of {len(df)} rows)")
    lines.append("_Source: Coin Metrics FlowInEtfUSD / FlowOutEtfUSD / SplyEtfNtv, inferred from ETF-labelled on-chain "
                 "addresses (BTC only): they lag issuer reports by about a day and will not match Blockworks / Farside "
                 "creation-redemption figures exactly. Latest day may be incomplete until the next daily close._")
    return "\n".join(lines)


def get_cme_tools() -> list:
    """CME futures + BTC ETF on-chain tools, in registration order."""
    return [get_cme_curve, get_cme_open_interest, get_btc_etf_onchain_flows]


CME_TOOL_NAMES = [t.name for t in get_cme_tools()]

__all__ = ["CME_TOOL_NAMES", "get_btc_etf_onchain_flows", "get_cme_curve", "get_cme_open_interest", "get_cme_tools"]
