"""@tool functions over Messari (providers/messari.py): news, sector classification, crypto ETF
flows (Blockworks data), Intel events and social signals.

    get_crypto_news(tokens, hours, limit)   headlines from Messari's curated feed, per asset or market-wide
    classify_tokens(tokens)                 Messari sector / sub-sector / tags for tickers
    get_sector_members(sector, n)           ranked assets in a sector, sub-sector or tag ("DePIN", "Layer-1")
    list_crypto_sectors()                   the taxonomy itself
    get_etf_overview(asset)                 latest ETF AUM / flows / product counts per underlying + top issuers
    get_etf_flows(asset, days)              daily ETF flow history (issuer-reported), BTC also with on-chain flows
    get_intel_events(tokens, days)          Messari Intel events: upgrades, governance, listings, legal, hacks
    get_social_signals(tokens)              mindshare, sentiment and post counts with Messari's insight text

Tickers here are resolved against Messari's own asset list (47k assets), not the Coin Metrics
universe, so a token can be classified even when we cannot price it. When the Coin Metrics
universe is available its asset id ('tao_bittensor') is passed along to break ticker
collisions. Every answer names Messari as the source and prints UTC times. Nothing writes.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import pandas as pd
from langchain_core.tools import tool

from providers.messari import NEWS_MAX_ASSETS
from tools.chat_tools import _fmt_usd as _usd_fmt
from tools.desk_tools import _md_table

logger = logging.getLogger(__name__)

SOURCE = "Messari"
MAX_NEWS = 40
MAX_MEMBERS = 100
UNAVAILABLE = ("Messari is not available: no API key (env MESSARI_API_KEY or secret messari_api_key2 in "
               "anchorage-trading-solutions). News, sector classification and ETF-flow tools are off; "
               "market-data and desk tools still work.")


# ---------------------------------------------------------------------------
# access + error handling
# ---------------------------------------------------------------------------

def _get_messari():
    """MessariProvider or None (tests monkeypatch this)."""
    from providers.factory import get_messari_provider
    return get_messari_provider()


def _guarded(name: str, body: Callable) -> str:
    prov = _get_messari()
    if prov is None:
        return UNAVAILABLE
    try:
        return body(prov)
    except Exception as e:  # noqa: BLE001 - surface to the model, never crash the agent
        logger.error("%s failed: %s", name, e, exc_info=logger.isEnabledFor(logging.DEBUG))
        return f"Error calling Messari in {name}: {type(e).__name__}: {e}"


def _cm_ids(tokens: List[str]) -> dict:
    """symbol -> Coin Metrics asset id when the dynamic universe is available (collision breaker)."""
    try:
        from providers.factory import get_universe
        universe = get_universe()
    except Exception:  # noqa: BLE001
        return {}
    if universe is None:
        return {}
    out = {}
    for t in tokens:
        try:
            cm = universe.resolve(t)
        except Exception:  # noqa: BLE001
            cm = None
        if cm:
            out[t] = cm
    return out


def _tokens(tokens) -> List[str]:
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


def _join(vals, empty: str = "-") -> str:
    vals = [str(v) for v in (vals or []) if v not in (None, "")]
    return ", ".join(vals) if vals else empty


MAX_TAGS = 6
MAX_EVENTS = 30
MAX_FLOW_ROWS = 45
ETF_ASSET_ALIASES = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana", "xrp": "ripple", "multi": "multi-asset",
                     "basket": "multi-asset", "hype": "hyperliquid", "link": "chainlink", "zec": "zcash", "ada": "cardano",
                     "avax": "avalanche", "doge": "dogecoin", "dot": "polkadot", "ltc": "litecoin", "xlm": "stellar",
                     "hbar": "hedera-hashgraph", "uni": "uniswap", "trx": "tron", "tao": "bittensor", "pol": "polygon"}
MAX_ETF_ASSET_ROWS = 15


def _usd(v) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return "$0" if float(v) == 0 else _usd_fmt(float(v))


def _n(v, digits: int = 0) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{float(v):,.{digits}f}"


def _pct(v, digits: int = 1, signed: bool = True) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"{float(v):{'+' if signed else ''}.{digits}f}%"


def _day(t) -> str:
    return "n/a" if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d")


def _clip(text: str, n: int = 220) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _tagged(assets) -> str:
    """Assets Messari tagged on an item: tickers first, capped (a generic markets piece can carry 150 tags)."""
    vals = [str(a) for a in (assets or []) if a]
    vals.sort(key=lambda a: (not a.isupper(), a))
    shown = ", ".join(vals[:MAX_TAGS])
    return shown + (f" +{len(vals) - MAX_TAGS} more" if len(vals) > MAX_TAGS else "")


def _ts(t) -> str:
    try:
        return pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
    except Exception:  # noqa: BLE001
        return str(t)


def _sentiment(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    v = float(v)
    word = "positive" if v > 0.15 else "negative" if v < -0.15 else "neutral"
    return f" - Messari sentiment {v:+.1f} ({word})"


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

@tool("get_crypto_news")
def get_crypto_news(tokens: List[str] = [], hours: int = 24, limit: int = 15, include_blogs: bool = False) -> str:
    """Latest crypto news headlines from Messari's curated feed: market-wide, or for up to 5 tokens.

    Use for "what's the news on X", "why is X moving", "anything on the ETF / regulation
    today", "headlines from the last 6 hours". Each item carries its publish time (UTC),
    source, link, the assets Messari tagged and, where scored, Messari's sentiment.

    Args:
        tokens: Up to 5 tickers (e.g. ['btc', 'eth']); empty = whole market.
        hours: Look-back window in hours (default 24, max 336).
        limit: Max items (default 15, max 40).
        include_blogs: Also include project blogs and forums, not only news outlets (default False).

    Returns:
        Markdown list, newest first, plus the window and a note when Messari's asset filter was
        slow and the feed had to be filtered client-side (may miss items).
    """
    def body(prov) -> str:
        toks = _tokens(tokens)[:NEWS_MAX_ASSETS]
        n = max(1, min(int(limit), MAX_NEWS))
        types = ("News", "Blog", "Forum") if include_blogs else ("News",)
        df = prov.news(assets=toks or None, hours=hours, limit=n, source_types=types, cm_ids=_cm_ids(toks))
        label = f"{', '.join(t.upper() for t in toks)}" if toks else "crypto market"
        if df is None:
            return f"Messari's news feed did not respond for {label} (timeout or API error). Try again in a minute."
        since = _ts(df.attrs.get("window_start"))
        unknown = df.attrs.get("unknown") or []
        out = [f"### News: {label}, since {since} ({SOURCE} news feed, {len(df)} item{'s' if len(df) != 1 else ''})"]
        if unknown:
            out.append(f"Messari does not know ticker(s) {', '.join(u.upper() for u in unknown)}; matched by title text only.")
        if df.empty:
            out.append(f"No {'news / blog / forum' if include_blogs else 'news'} items in that window.")
        for _, r in df.iterrows():
            tags = f" [{_tagged(r['assets'])}]" if r["assets"] else ""
            out.append(f"- {_ts(r['time'])} - **{r['title']}** ({r['source']}){tags}{_sentiment(r['sentiment'])}"
                       + (f" - {r['url']}" if r["url"] else ""))
        notes = [f"_Source: {SOURCE} news feed ({'news, blogs and forums' if include_blogs else 'news outlets only'})."]
        if df.attrs.get("fallback"):
            notes.append("Messari's per-asset filter timed out, so the last "
                         f"{'/'.join(s for s in df.attrs.get('source_types', []))} items were filtered here by asset tag "
                         "and by ticker / project name in the title - an item that mentions the asset only in its body may be missing.")
        notes.append("Sentiment is Messari's model score, descriptive only._")
        out.append(" ".join(notes))
        return "\n".join(out)
    return _guarded("get_crypto_news", body)


@tool("classify_tokens")
def classify_tokens(tokens: List[str]) -> str:
    """Messari's sector, sub-sector and tags for one or more tickers (e.g. 'tao' -> AI / Decentralized AI; 'uni' -> DeFi / Decentralized Exchange), with Messari rank and project name.

    Use for "what sector is X", "is X a DePIN token", "classify these tokens", or to group a
    list of tokens by sector before comparing them. Works for any asset Messari lists (47k),
    not only those we can price.

    Args:
        tokens: Tickers, e.g. ['tao', 'render', 'hnt'] (max 25).

    Returns:
        Markdown table: ticker, Messari project, rank, sector(s), sub-sector(s), tags; unknown tickers listed.
    """
    def body(prov) -> str:
        toks = _tokens(tokens)[:25]
        if not toks:
            return "Give at least one ticker to classify."
        df = prov.classify(toks, cm_ids=_cm_ids(toks))
        known = df[df["slug"].notna()]
        unknown = df[df["slug"].isna()]["symbol"].tolist()
        out = [f"### Sector classification ({SOURCE} taxonomy)"]
        if len(known):
            rows = [[r["symbol"].upper(), f"{r['name']} ({r['slug']})", str(int(r["rank"])) if r["rank"] else "unranked",
                     _join(r["sector"]), _join(r["sub_sector"]), _join(r["tags"])] for _, r in known.iterrows()]
            out += _md_table(["Ticker", "Messari project", "Rank", "Sector", "Sub-sector", "Tags"], rows)
        if unknown:
            out.append(f"Not in Messari's asset list: {', '.join(u.upper() for u in unknown)}.")
        out.append(f"_Source: {SOURCE} sectorV2 / subSectorV2 taxonomy (Messari's own classification, not GICS); "
                   "rank = Messari market-cap rank. Use get_sector_members to list a sector's constituents._")
        return "\n".join(out)
    return _guarded("classify_tokens", body)


@tool("get_sector_members")
def get_sector_members(sector: str, n: int = 25) -> str:
    """The top-N assets (by Messari rank) in a Messari sector, sub-sector or tag: e.g. 'DePIN', 'AI', 'Meme', 'Layer-1', 'Layer-2', 'Decentralized Exchange', 'Lending', 'Real World Assets', 'Stablecoins', 'Privacy', 'Proof-of-Work'.

    Use for "which tokens are DePIN", "list the L2s", "top 10 AI tokens", "what's in the RWA
    sector". Call list_crypto_sectors to see the available names. Pair with
    get_zscore_signals / get_price_history on the returned tickers for a sector view.

    Args:
        sector: Sector, sub-sector or tag name (case-insensitive; 'depin' works).
        n: How many assets (default 25, max 100).

    Returns:
        Markdown table: rank, ticker, project, sub-sectors, tags, plus the total number of matches.
    """
    def body(prov) -> str:
        name = (sector or "").strip()
        if not name:
            return "Give a sector, sub-sector or tag name (see list_crypto_sectors)."
        df = prov.sector_members(name, n=max(1, min(int(n), MAX_MEMBERS)))
        kind = df.attrs.get("kind")
        if df.empty or kind is None:
            return (f"No Messari sector, sub-sector or tag matches '{name}'. Call list_crypto_sectors for the "
                    "available names.")
        level = {"sector": "sector", "sub_sector": "sub-sector", "tags": "tag"}[kind]
        total = df.attrs.get("total", len(df))
        out = [f"### {name}: {len(df)} of {total} ranked assets ({SOURCE} {level})"]
        if kind == "sector":
            heads, cols = ("Sub-sector", "Tags"), ("sub_sector", "tags")
        elif kind == "sub_sector":
            heads, cols = ("Sector", "Tags"), ("sector", "tags")
        else:
            heads, cols = ("Sector", "Sub-sector"), ("sector", "sub_sector")
        rows = [[str(int(r["rank"])), r["symbol"], f"{r['name']} ({r['slug']})", _join(r[cols[0]]), _join(r[cols[1]])]
                for _, r in df.iterrows()]
        out += _md_table(["Rank", "Ticker", "Project", *heads], rows)
        out.append(f"_Source: {SOURCE} taxonomy; rank = Messari market-cap rank (unranked assets excluded). "
                   "Tickers may need list_top_assets / get_live_price to confirm we can price them._")
        return "\n".join(out)
    return _guarded("get_sector_members", body)


@tool("list_crypto_sectors")
def list_crypto_sectors() -> str:
    """Messari's crypto sector taxonomy: every sector with its sub-sectors, how many ranked assets it holds and its three largest tickers.

    Use for "what sectors are there", "how is the market classified", or to find the exact
    name before calling get_sector_members.

    Returns:
        Markdown table: sector, ranked assets, top tickers, sub-sectors.
    """
    def body(prov) -> str:
        df = prov.sectors()
        if df.empty:
            return "Messari returned no sector data."
        out = [f"### Crypto sectors ({SOURCE} taxonomy, {int(df['assets'].sum())} sector memberships over ranked assets)"]
        rows = [[r["sector"], str(int(r["assets"])), _join(r["top"]), _join(r["sub_sectors"][:12]) +
                 (f" (+{len(r['sub_sectors']) - 12} more)" if len(r["sub_sectors"]) > 12 else "")] for _, r in df.iterrows()]
        out += _md_table(["Sector", "Assets", "Top", "Sub-sectors"], rows)
        out.append(f"_Source: {SOURCE} sectorV2 / subSectorV2. An asset can sit in more than one sector "
                   "(HYPE is DeFi and Networks). Use get_sector_members(name) for the constituents._")
        return "\n".join(out)
    return _guarded("list_crypto_sectors", body)


# ---------------------------------------------------------------------------
# ETFs (Blockworks data via Messari)
# ---------------------------------------------------------------------------

def _etf_asset(asset: str) -> str:
    a = (asset or "bitcoin").strip().lower()
    return ETF_ASSET_ALIASES.get(a, a)


def _onchain_net_flows(days: int) -> Optional[pd.DataFrame]:
    """Coin Metrics on-chain BTC ETF net flows keyed by day, or None when the spot provider lacks them."""
    try:
        from providers.factory import get_provider
        spot = getattr(get_provider(), "spot", None) or get_provider()
        df = spot.etf_onchain_flows("btc", days=days + 2)
    except Exception as e:  # noqa: BLE001
        logger.debug("on-chain ETF flows unavailable: %s", e)
        return None
    if df is None or df.empty:
        return None
    out = df[["time", "net_flow_usd"]].copy()
    out["day"] = out["time"].dt.floor("D")
    return out[["day", "net_flow_usd"]]


@tool("get_etf_overview")
def get_etf_overview(asset: str = "bitcoin") -> str:
    """Latest crypto ETF picture from Messari (Blockworks Research data): per underlying asset the spot and futures ETF AUM, latest daily net flow, product counts, the US / Europe / APAC split, plus the largest issuers by AUM with their 30-day flows.

    Use for "how big are the BTC ETFs", "ETF AUM", "how many ETH ETF products", "which issuer
    is gathering assets", "regional ETF split". For the day-by-day flow history use
    get_etf_flows. Assets: bitcoin, ethereum, solana, xrp, multi-asset (tickers accepted).

    Args:
        asset: Underlying to highlight (default 'bitcoin'); every asset with ETFs is listed anyway.

    Returns:
        Markdown: per-asset table, the highlighted asset's regional breakdown, top issuers, as-of date.
    """
    def body(prov) -> str:
        target = _etf_asset(asset)
        assets = prov.etf_assets()
        if assets is None:
            return "Messari's ETF asset table did not respond (timeout or API error). Try again in a minute."
        if assets.empty:
            return "Messari returned no ETF asset rows."
        as_of = assets["as_of"].max()
        out = [f"### Crypto ETFs by underlying asset ({SOURCE} / Blockworks Research, as of {_day(as_of)})"]
        ranked = assets.sort_values("spot_aum_usd", ascending=False, na_position="last")
        shown = ranked.head(MAX_ETF_ASSET_ROWS)
        if target not in shown["id"].str.lower().tolist():
            shown = pd.concat([shown, ranked[ranked["id"].str.lower() == target]])
        rows = []
        for _, r in shown.iterrows():
            rows.append([r["id"], _usd(r.get("spot_aum_usd")), _usd(r.get("spot_flow_usd")), _n(r.get("spot_products")),
                         _usd(r.get("futures_aum_usd")), _usd(r.get("futures_flow_usd")), _usd(r.get("total_volume_usd"))])
        out += _md_table(["Asset", "Spot AUM", "Spot flow (latest day)", "Spot products", "Futures AUM", "Futures flow", "Volume"], rows)
        rest = ranked.iloc[len(shown):] if len(ranked) > len(shown) else ranked.iloc[0:0]
        if len(rest):
            out.append(f"... {len(rest)} more underlyings with ETPs (combined spot AUM {_usd(rest['spot_aum_usd'].sum())}: "
                       f"{', '.join(rest['id'].head(12).tolist())}{' ...' if len(rest) > 12 else ''}). Total spot ETF AUM across all "
                       f"underlyings {_usd(ranked['spot_aum_usd'].sum())}.")
        hit = assets[assets["id"].str.lower() == target]
        if len(hit):
            r = hit.iloc[0]
            out.append("")
            out.append(f"**{r['id']} spot ETFs by region** (AUM / latest flow / products): "
                       f"US {_usd(r.get('us_spot_aum_usd'))} / {_usd(r.get('us_spot_flow_usd'))} / {_n(r.get('us_spot_products'))}; "
                       f"Europe {_usd(r.get('europe_spot_aum_usd'))} / {_usd(r.get('europe_spot_flow_usd'))} / {_n(r.get('europe_spot_products'))}; "
                       f"APAC {_usd(r.get('apac_spot_aum_usd'))} / {_usd(r.get('apac_spot_flow_usd'))} / {_n(r.get('apac_spot_products'))}. "
                       f"Delta-one {_usd(r.get('deltaone_aum_usd'))} vs leveraged {_usd(r.get('leveraged_aum_usd'))}.")
        else:
            out.append(f"(no ETF row for '{target}'; assets with ETFs are listed above)")
        issuers = prov.etf_providers(n=10)
        if issuers is not None and len(issuers):
            out.append("")
            out.append(f"**Largest issuers** (all crypto ETPs, as of {_day(issuers['as_of'].max())}):")
            out += _md_table(["Issuer", "AUM", "Latest flow", "30d flow", "AUM 30d chg", "Products", "BTC share", "Fee (wtd)"],
                             [[r["name"], _usd(r.get("aum_usd")), _usd(r.get("flow_usd")), _usd(r.get("rolling30d_flow_usd")),
                               _pct(r.get("aum30d_change_perc") * 100 if pd.notna(r.get("aum30d_change_perc")) else None),
                               _n(r.get("num_products")),
                               _pct(r.get("bitcoin_aum_perc") * 100 if pd.notna(r.get("bitcoin_aum_perc")) else None, signed=False),
                               _pct(r.get("weighted_aum_expense_ratio") * 100 if pd.notna(r.get("weighted_aum_expense_ratio")) else None, 2, signed=False)]
                              for _, r in issuers.iterrows()])
        out.append(f"_Source: {SOURCE} ETF datasets (Blockworks Research, daily, issuer-reported). A flow shown as n/a is "
                   "not yet published for the latest day - it is not zero. Issuer flow of $0 on the latest row is Blockworks' "
                   "placeholder before publication._")
        return "\n".join(out)
    return _guarded("get_etf_overview", body)


@tool("get_etf_flows")
def get_etf_flows(asset: str = "bitcoin", days: int = 30) -> str:
    """Daily crypto ETF flow history for one underlying (bitcoin, ethereum, solana, xrp, multi-asset) from Messari / Blockworks: spot and futures net flows, spot AUM, US / Europe / APAC spot flows and product counts, with window totals. For bitcoin the Coin Metrics on-chain net flow is shown alongside for comparison.

    Use for "ETF flows this week", "how much went into the ETH ETFs", "biggest outflow day",
    "cumulative ETF flows since ...".

    Args:
        asset: Underlying (default 'bitcoin'; tickers accepted).
        days: Window in days (default 30, max 365).

    Returns:
        Markdown summary (latest, totals, best / worst day) and a daily table.
    """
    def body(prov) -> str:
        target = _etf_asset(asset)
        n = max(1, min(int(days), 365))
        df = prov.etf_asset_timeseries(target, days=n)
        if df is None:
            return f"Messari's ETF time series did not respond for {target} (timeout or API error). Try again in a minute."
        if df.empty:
            return f"Messari has no ETF time series for '{target}'. Assets with ETFs: bitcoin, ethereum, solana, xrp, multi-asset."
        onchain = _onchain_net_flows(n) if target == "bitcoin" else None
        if onchain is not None:
            df = df.assign(day=df["time"].dt.floor("D")).merge(onchain, on="day", how="left").drop(columns="day")
        flows = df.dropna(subset=["spot_flow_usd"]) if "spot_flow_usd" in df.columns else df.iloc[0:0]
        out = [f"### {target} ETF flows, last {n} days ({SOURCE} / Blockworks Research)"]
        if len(flows):
            last = flows.iloc[-1]
            out.append(f"- latest published day {_day(last['time'])}: spot net flow {_usd(last['spot_flow_usd'])}"
                       + (f", futures {_usd(last.get('futures_flow_usd'))}" if pd.notna(last.get("futures_flow_usd")) else "")
                       + f"; spot AUM {_usd(last.get('spot_aum_usd'))} across {_n(last.get('spot_products'))} products")
            pos, neg = int((flows["spot_flow_usd"] > 0).sum()), int((flows["spot_flow_usd"] < 0).sum())
            out.append(f"- window: spot net {_usd(flows['spot_flow_usd'].sum())} over {len(flows)} published days "
                       f"({pos} inflow, {neg} outflow); best {_usd(flows['spot_flow_usd'].max())} on "
                       f"{_day(flows.loc[flows['spot_flow_usd'].idxmax(), 'time'])}, worst {_usd(flows['spot_flow_usd'].min())} on "
                       f"{_day(flows.loc[flows['spot_flow_usd'].idxmin(), 'time'])}")
            if "us_spot_flow_usd" in flows.columns:
                out.append(f"- by region (window): US {_usd(flows['us_spot_flow_usd'].sum())}, Europe "
                           f"{_usd(flows.get('europe_spot_flow_usd', pd.Series(dtype=float)).sum())}, APAC "
                           f"{_usd(flows.get('apac_spot_flow_usd', pd.Series(dtype=float)).sum())}")
        unpublished = df[df["spot_flow_usd"].isna()] if "spot_flow_usd" in df.columns else df.iloc[0:0]
        if len(unpublished):
            out.append(f"- {_day(unpublished['time'].iloc[-1])}: flows not yet published (AUM {_usd(unpublished.iloc[-1].get('spot_aum_usd'))})")
        shown = df if len(df) <= MAX_FLOW_ROWS else df.iloc[-MAX_FLOW_ROWS:]
        heads = ["Date", "Spot flow", "Futures flow", "Spot AUM", "US spot flow", "Europe", "APAC"] + (["On-chain net (CM)"] if onchain is not None else [])
        rows = []
        for _, r in shown.iterrows():
            row = [_day(r["time"]), _usd(r.get("spot_flow_usd")), _usd(r.get("futures_flow_usd")), _usd(r.get("spot_aum_usd")),
                   _usd(r.get("us_spot_flow_usd")), _usd(r.get("europe_spot_flow_usd")), _usd(r.get("apac_spot_flow_usd"))]
            if onchain is not None:
                row.append(_usd(r.get("net_flow_usd")))
            rows.append(row)
        out.append("")
        out += _md_table(heads, rows)
        if len(shown) < len(df):
            out.append(f"(table shows the last {len(shown)} of {len(df)} days)")
        note = (f"_Source: {SOURCE} ETF asset time series (Blockworks Research, issuer-reported creations / redemptions, "
                "daily; n/a = not yet published, not zero).")
        if onchain is not None:
            note += (" The on-chain column is Coin Metrics FlowInEtfUSD - FlowOutEtfUSD inferred from ETF-labelled "
                     "bitcoin addresses: a different method that lags by about a day, so the two will not match exactly.")
        out.append(note + "_")
        return "\n".join(out)
    return _guarded("get_etf_flows", body)


# ---------------------------------------------------------------------------
# Intel events + social signals
# ---------------------------------------------------------------------------

@tool("get_intel_events")
def get_intel_events(tokens: List[str], days: int = 30, importance: str = "", limit: int = 20) -> str:
    """Messari Intel events for up to 5 tokens: protocol upgrades, governance proposals, token unlock decisions, exchange listings, legal / regulatory actions, hacks and recoveries, each with date, importance (High / Medium / Low), status (Proposed, Discussed, Planned, In-Progress, Completed, Rejected), category and a link.

    Use for "what's coming up for SOL", "any governance votes on AAVE", "recent upgrades or
    incidents for X", "why did X move" when the news feed is thin. Dates are event dates;
    Planned / Proposed items may lie in the future.

    Args:
        tokens: Up to 5 tickers.
        days: How far back the event date may be (default 30, max 3650).
        importance: Optional filter: 'high', 'medium' or 'low' (comma-separated for several).
        limit: Max events (default 20, max 30).

    Returns:
        Markdown list, newest first: date, importance, status, name, category, assets, one-line detail, link.
    """
    def body(prov) -> str:
        toks = _tokens(tokens)[:NEWS_MAX_ASSETS]
        if not toks:
            return "Give at least one ticker."
        imp = [x.strip() for x in str(importance or "").split(",") if x.strip()] or None
        n = max(1, min(int(limit), MAX_EVENTS))
        df = prov.intel_events(toks, days=days, importance=imp, limit=n, cm_ids=_cm_ids(toks))
        label = ", ".join(t.upper() for t in toks)
        if df is None:
            return f"Messari Intel did not respond for {label} (timeout or API error). Try again in a minute."
        unknown = df.attrs.get("unknown") or []
        out = [f"### Intel events: {label}, since {_day(df.attrs.get('since'))} ({SOURCE} Intel, {len(df)} event{'s' if len(df) != 1 else ''})"]
        if unknown:
            out.append(f"Messari does not know ticker(s) {', '.join(u.upper() for u in unknown)}.")
        if df.empty:
            out.append("No Intel events in that window" + (f" at importance {', '.join(imp)}" if imp else "") + ".")
        for _, r in df.head(n).iterrows():
            out.append(f"- {_day(r['date'])} [{r['importance']} / {r['status']}] **{r['name']}** - {r['category']}"
                       + (f" / {r['subcategory']}" if r["subcategory"] else "") + f" - {_join(r['assets'])}"
                       + (f" (also {_join(r['secondary'])})" if r["secondary"] else "")
                       + (f": {_clip(r['details'])}" if r["details"] else "") + (f" - {r['link']}" if r["link"] else ""))
        out.append(f"_Source: {SOURCE} Intel (analyst-curated event database); importance and status are Messari's labels. "
                   "Event dates can be in the future for planned items._")
        return "\n".join(out)
    return _guarded("get_intel_events", body)


@tool("get_social_signals")
def get_social_signals(tokens: List[str]) -> str:
    """Messari Signals for up to 10 tokens: mindshare (the token's share of tracked crypto-influencer attention, %) over 24h / 7d / 30d with its change, social sentiment (-1..1), post and author counts, and Messari's generated one-paragraph read of why attention is moving.

    Use for "is X getting attention", "what is crypto Twitter saying about X", "mindshare",
    "social sentiment". Descriptive only: attention is not a trade signal.

    Args:
        tokens: Up to 10 tickers.

    Returns:
        Markdown table plus one insight line per token, with the as-of caveat.
    """
    def body(prov) -> str:
        toks = _tokens(tokens)[:10]
        if not toks:
            return "Give at least one ticker."
        df = prov.signals(toks, cm_ids=_cm_ids(toks))
        if df is None:
            return f"Messari Signals did not respond for {', '.join(t.upper() for t in toks)} (timeout or API error)."
        unknown = df.attrs.get("unknown") or []
        out = [f"### Social signals ({SOURCE} Signals: mindshare = share of tracked influencer attention)"]
        if unknown:
            out.append(f"Messari does not know ticker(s) {', '.join(u.upper() for u in unknown)}.")
        if df.empty:
            out.append("No signal data for those tokens.")
            return "\n".join(out)
        rows = [[r["symbol"], _pct(r["mindshare_24h"], 2, signed=False), _pct(r["mindshare_24h_chg"]),
                 _pct(r["mindshare_7d"], 2, signed=False), _pct(r["mindshare_7d_chg"]),
                 _pct(r["mindshare_30d"], 2, signed=False),
                 "n/a" if pd.isna(r["sentiment_24h"]) else f"{float(r['sentiment_24h']):+.2f}",
                 "n/a" if pd.isna(r["sentiment_7d"]) else f"{float(r['sentiment_7d']):+.2f}",
                 _n(r["posts_24h"]), _n(r["authors_24h"])] for _, r in df.iterrows()]
        out += _md_table(["Ticker", "Mindshare 24h", "chg", "7d", "chg", "30d", "Sentiment 24h", "7d mean", "Posts 24h", "Authors 24h"], rows)
        for _, r in df.iterrows():
            if r["insight_24h"]:
                out.append(f"- **{r['symbol']}** (Messari 24h read): {_clip(r['insight_24h'], 400)}")
        out.append(f"_Source: {SOURCE} Signals. Mindshare change is relative (% of the previous period's share); sentiment "
                   "is a model score in [-1, 1]; the insight text is Messari's generated summary of the drivers, quoted as "
                   "vendor commentary, not our view._")
        return "\n".join(out)
    return _guarded("get_social_signals", body)


def get_messari_tools() -> list:
    """All Messari tools, in registration order."""
    return [get_crypto_news, classify_tokens, get_sector_members, list_crypto_sectors,
            get_etf_overview, get_etf_flows, get_intel_events, get_social_signals]


MESSARI_TOOL_NAMES = [t.name for t in get_messari_tools()]

__all__ = ["MESSARI_TOOL_NAMES", "SOURCE", "UNAVAILABLE", "classify_tokens", "get_crypto_news", "get_etf_flows",
           "get_etf_overview", "get_intel_events", "get_messari_tools", "get_sector_members", "get_social_signals",
           "list_crypto_sectors"]
