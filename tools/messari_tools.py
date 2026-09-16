"""@tool functions over Messari (providers/messari.py): news, sector classification, and -
in later commits - crypto ETF flows, Intel events and social signals.

    get_crypto_news(tokens, hours, limit)   headlines from Messari's curated feed, per asset or market-wide
    classify_tokens(tokens)                 Messari sector / sub-sector / tags for tickers
    get_sector_members(sector, n)           ranked assets in a sector, sub-sector or tag ("DePIN", "Layer-1")
    list_crypto_sectors()                   the taxonomy itself

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


def get_messari_tools() -> list:
    """All Messari tools, in registration order."""
    return [get_crypto_news, classify_tokens, get_sector_members, list_crypto_sectors]


MESSARI_TOOL_NAMES = [t.name for t in get_messari_tools()]

__all__ = ["MESSARI_TOOL_NAMES", "SOURCE", "UNAVAILABLE", "classify_tokens", "get_crypto_news",
           "get_messari_tools", "get_sector_members", "list_crypto_sectors"]
