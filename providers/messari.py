"""Messari provider: news, asset classification, crypto ETF data, Intel events and social signals.

Key: Secret Manager ``messari_api_key2`` (Enterprise; the older ``messari_api_key`` is dead).
Header ``x-messari-api-key``; every response is ``{data, error, metadata}`` and some list
endpoints nest a second ``{data, metadata}`` inside ``data``. Rate limit 200-600 req/min.
Checked 2026-09-16 on our key:

    /news/v1/news/feed                       ~420 'News'-type items a day, 100 per page.
                                             Filtering by ``assetIds`` times out (524) on
                                             Messari's side almost every time, so ``news()``
                                             fails fast on the filtered call and falls back to
                                             the unfiltered feed filtered client-side.
    /metrics/v2/assets                       47,806 assets, 500 per page, sorted by rank
                                             (rank 0 = unranked); ``search=`` finds by symbol /
                                             name / slug. Carries the taxonomy: ``sectorV2``
                                             (13 sectors such as DeFi, Networks, AI, Meme,
                                             DePIN), ``subSectorV2`` (~100), ``tags``.
    /metrics/v2/protocols/etfs               issuer table (77 issuers) and per-asset table
                                             (bitcoin, ethereum, solana, xrp, multi-asset ...)
                                             with daily AUM / flow / product counts by type
                                             (spot, futures), region and strategy; flows for
                                             the latest day are null until published.
    /intel/v1/events                         Messari Intel events (governance, upgrades,
                                             listings ...) with importance and status.
    /signal/v1/assets                        mindshare (share of influencer attention) over
                                             24h / 7d / 30d with a generated insight text.
    /research                                403: not on our plan.

Everything here is memoised with a TTL (news 5 min, classification 24 h, ETF 1 h, Intel
30 min, signals 15 min); the ranked asset table is also cached on disk
(``MESSARI_CACHE_PATH``, default data/messari_assets_cache.json) like the Coin Metrics
universe. Nothing raises to the caller except programming errors; a vendor failure returns
None / an empty frame and is logged.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd
import requests

from providers._http import AmberdataHTTP

logger = logging.getLogger(__name__)

BASE_URL = "https://api.messari.io"
NEWS_FEED = "/news/v1/news/feed"
NEWS_SOURCES = "/news/v1/news/sources"
ASSETS = "/metrics/v2/assets"
ASSET_DETAILS = "/metrics/v2/assets/details"
ETF_PROVIDERS = "/metrics/v2/protocols/etfs"
ETF_ASSETS = "/metrics/v2/protocols/etfs/assets"
ETF_ASSET_TIMESERIES = "/metrics/v2/protocols/etfs/assets/{asset}/metrics/overview/time-series/1d"
ETF_PROVIDER_TIMESERIES = "/metrics/v2/protocols/etfs/{provider}/metrics/overview/time-series/1d"
INTEL_EVENTS = "/intel/v1/events"
SIGNALS_ASSETS = "/signal/v1/assets"

RETRY_STATUSES = {429, 500, 502, 503, 504, 524}
NEWS_PAGE = 100                  # the feed rejects limit > 100
NEWS_MAX_PAGES = 8               # fallback scan: up to 800 items (~2 days of 'News'-type items)
NEWS_FILTERED_TIMEOUT_S = 12     # asset-filtered call: fail fast, then fall back
NEWS_MAX_ASSETS = 5
NEWS_SOURCE_TYPES = ("News", "Blog", "Forum")
ASSET_PAGE = 500
ASSET_MAX_PAGES = 12             # ranked assets end around page 7-8 (~3,500); rank 0 = unranked
SEARCH_LIMIT = 25
TTL_NEWS_S = 5 * 60
TTL_ASSETS_S = 24 * 3600
TTL_SEARCH_S = 24 * 3600
TTL_ETF_S = 3600
TTL_INTEL_S = 30 * 60
TTL_SIGNALS_S = 15 * 60
DEFAULT_CACHE_PATH = Path(os.getenv("MESSARI_CACHE_PATH", "data/messari_assets_cache.json"))

# Symbols whose Messari slug is not the obvious one (verified 2026-09-16 via /metrics/v2/assets?search=).
SYMBOL_TO_SLUG = {
    "btc": "bitcoin", "eth": "ethereum", "sol": "solana", "xrp": "xrp", "bnb": "binance-coin",
    "hype": "hyperliquid", "tao": "bittensor-0", "sky": "sky-protocol", "pol": "polygon-ecosystem-token",
    "usdt": "tether", "usdc": "usd-coin",
}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class MessariHTTP(AmberdataHTTP):
    """Messari flavour of the shared HTTP client: own header, 524 counted as retryable,
    envelope unwrapping and page-number pagination."""

    def __init__(self, api_key: str, session: Optional[requests.Session] = None, timeout: int = 20,
                 max_retries: int = 2, backoff: float = 1.0):
        super().__init__(api_key, session=session, timeout=timeout, max_retries=max_retries, backoff=backoff,
                         header_name="x-messari-api-key", retry_statuses=RETRY_STATUSES, label="Messari")

    @property
    def slow(self) -> bool:
        """True when the last request died of a 524 / gateway timeout or a client-side timeout."""
        return self.last_status == 524 or (self.last_status is None and bool(self.last_error))

    def page(self, path: str, params: Optional[dict] = None, timeout: Optional[float] = None,
             retries: Optional[int] = None) -> tuple[Optional[object], dict]:
        """One request. Returns (data, metadata); data is None on any failure or API error."""
        env = self.request(BASE_URL + path, params, timeout=timeout, retries=retries)
        if not isinstance(env, dict):
            return None, {}
        if env.get("error"):
            logger.warning("Messari %s error: %s", path, env["error"])
            self.last_status, self.last_error = self.last_status or 200, str(env["error"])
            return None, {}
        data, meta = env.get("data"), env.get("metadata") or {}
        # some list endpoints (ETF tables) nest a second envelope inside data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            meta = data.get("metadata") or meta
            data = data["data"]
        return data, meta if isinstance(meta, dict) else {}

    def pages(self, path: str, params: Optional[dict] = None, max_pages: int = 10,
              page_key: str = "page", first_page: int = 1,
              stop: Optional[Callable[[list], bool]] = None) -> Optional[list]:
        """Follow ``metadata.page / totalPages``; ``stop(rows)`` can end the scan early.
        Returns the rows (possibly empty) or None when the very first page failed."""
        rows: list = []
        page = first_page
        for _ in range(max_pages):
            data, meta = self.page(path, dict(params or {}, **{page_key: page}))
            if data is None:
                return None if not rows else rows
            batch = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
            rows.extend(batch)
            if not batch or (stop is not None and stop(batch)):
                break
            total = meta.get("totalPages")
            try:
                if total is not None and page >= int(total):
                    break
            except (TypeError, ValueError):
                pass
            page += 1
        return rows


# ---------------------------------------------------------------------------
# TTL memoisation (instance-level, thread-safe, DataFrame copies)
# ---------------------------------------------------------------------------

def _key_part(value):
    if isinstance(value, dict):
        return tuple(sorted((str(k), _key_part(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_key_part(v) for v in value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M")
    if isinstance(value, str):
        return value.lower()
    return value


def _ttl_memoised(ttl_s: float):
    def deco(method: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            key = (method.__name__,) + tuple(_key_part(a) for a in args) \
                + tuple(sorted((k, _key_part(v)) for k, v in kwargs.items()))
            now = time.time()
            with self._memo_lock:
                hit = self._memo.get(key)
                if hit is not None and hit[0] > now:
                    val = hit[1]
                    return val.copy() if isinstance(val, pd.DataFrame) else val
            result = method(self, *args, **kwargs)
            if result is not None:   # never cache a failure
                with self._memo_lock:
                    self._memo[key] = (now + ttl_s, result)
            return result.copy() if isinstance(result, pd.DataFrame) else result
        wrapper.__name__ = method.__name__
        wrapper.__doc__ = method.__doc__
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [x for x in v if x not in (None, "")]
    return [v]


def _rank(v) -> Optional[int]:
    try:
        r = int(v)
    except (TypeError, ValueError):
        return None
    return r if r > 0 else None


def _asset_record(a: dict) -> dict:
    return {
        "slug": a.get("slug") or "",
        "symbol": (a.get("symbol") or "").upper(),
        "name": a.get("name") or "",
        "rank": _rank(a.get("rank")),
        "sector": _to_list(a.get("sectorV2")),
        "sub_sector": _to_list(a.get("subSectorV2")),
        "tags": _to_list(a.get("tags")),
        "category": a.get("category") or "",
        "has_news": bool(a.get("hasNews")),
        "has_intel": bool(a.get("hasIntel")),
    }


ASSET_COLUMNS = ["slug", "symbol", "name", "rank", "sector", "sub_sector", "tags", "category", "has_news", "has_intel"]


_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_ETF_RENAMES = (
    ("type_spot_unitedstates_", "us_spot_"), ("type_spot_europe_", "europe_spot_"), ("type_spot_apac_", "apac_spot_"),
    ("type_spot_", "spot_"), ("type_futures_", "futures_"), ("strategy_deltaone_", "deltaone_"),
    ("strategy_leveraged_", "leveraged_"), ("volume_unitedstates_usd", "us_volume_usd"),
    ("volume_europe_usd", "europe_volume_usd"), ("volume_apac_usd", "apac_volume_usd"), ("_product_count", "_products"),
)


def metric_name(raw: str) -> str:
    """Messari metric key (camelCase or kebab-case) -> snake_case with the ETF prefixes shortened:
    'typeSpotUnitedstatesFlowUsd' / 'type-spot-unitedstates-flow-usd' -> 'us_spot_flow_usd'."""
    s = _CAMEL_RE.sub("_", str(raw)).replace("-", "_").lower()
    for a, b in _ETF_RENAMES:
        s = s.replace(a, b)
    return s


def _epoch(v):
    """Messari 'asOf' / point timestamps are epoch seconds (sometimes ms); return a UTC Timestamp or NaT."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return pd.NaT
    return pd.to_datetime(v, unit="ms" if v > 1e11 else "s", utc=True)


def _qualifier(cm_asset_id: Optional[str]) -> Optional[str]:
    """'tao_bittensor' -> 'bittensor'; 'btc' -> None."""
    if cm_asset_id and "_" in cm_asset_id:
        q = cm_asset_id.split("_", 1)[1].strip().lower()
        return q or None
    return None


def _pick(candidates: list[dict], qualifier: Optional[str]) -> Optional[dict]:
    """Among rows sharing a symbol: the one whose slug carries the Coin Metrics qualifier,
    else the best (lowest positive) rank, else the first."""
    if not candidates:
        return None
    if qualifier:
        q = qualifier.replace("_", "-")
        hits = [c for c in candidates if q in (c.get("slug") or "") or q in (c.get("name") or "").lower()]
        if hits:
            candidates = hits
    ranked = [c for c in candidates if c.get("rank")]
    if ranked:
        return min(ranked, key=lambda c: c["rank"])
    return candidates[0]


def _title_pattern(symbol: str, name: str) -> re.Pattern:
    parts = [re.escape(symbol)] if symbol and len(symbol) >= 3 else []
    if name and len(name) >= 3:
        parts.append(re.escape(name))
    if not parts:
        parts = [re.escape(symbol or name or "\x00")]
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])", re.IGNORECASE)


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------

class MessariProvider:
    """News, taxonomy, ETF, Intel and signals from Messari (see module docstring)."""

    def __init__(self, api_key: str, session: Optional[requests.Session] = None, timeout: int = 20,
                 max_retries: int = 2, backoff: float = 1.0, cache_path: Optional[Path] = None,
                 ttl_s: int = TTL_ASSETS_S):
        self.http = MessariHTTP(api_key, session=session, timeout=timeout, max_retries=max_retries, backoff=backoff)
        self._memo: dict = {}
        self._memo_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self._ttl_s = ttl_s
        self._assets: Optional[pd.DataFrame] = None
        self._assets_built_at: float = 0.0
        self._resolved: dict[str, Optional[dict]] = {}     # "symbol|qualifier" -> asset record
        self._load_cache()

    # ------------------------------------------------------------------ cache

    def _load_cache(self) -> None:
        try:
            if not self._cache_path.exists():
                return
            raw = json.loads(self._cache_path.read_text())
            built = float(raw.get("built_at", 0))
            if time.time() - built > self._ttl_s:
                return
            self._assets = pd.DataFrame(raw["assets"], columns=ASSET_COLUMNS)
            self._assets_built_at = built
            self._resolved = dict(raw.get("resolved", {}))
            logger.info("Messari asset cache loaded: %d ranked assets, built %s", len(self._assets),
                        datetime.fromtimestamp(built, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        except Exception as e:  # noqa: BLE001
            logger.warning("Messari asset cache unreadable (%s); rebuilding on demand", e)
            self._assets = None

    def _save_cache(self) -> None:
        if self._assets is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"built_at": self._assets_built_at, "assets": self._assets.to_dict("records"),
                       "resolved": self._resolved}
            self._cache_path.write_text(json.dumps(payload, default=str))
        except Exception as e:  # noqa: BLE001
            logger.debug("Messari asset cache not written: %s", e)

    # ------------------------------------------------------------------ assets / taxonomy

    def _fresh(self) -> bool:
        return self._assets is not None and (time.time() - self._assets_built_at) < self._ttl_s

    def asset_table(self, force: bool = False) -> pd.DataFrame:
        """Every *ranked* Messari asset (rank > 0) with its taxonomy; columns ASSET_COLUMNS.

        Pages of 500 sorted by rank until a page is entirely unranked (about 7-8 requests,
        ~6 s); cached 24 h in memory and on disk. Raises RuntimeError when Messari returns
        nothing at all (callers turn that into a tool message)."""
        with self._lock:
            if self._fresh() and not force:
                return self._assets
            t0 = time.time()
            rows = self.http.pages(ASSETS, {"limit": ASSET_PAGE, "sort": "rank", "order": "asc"},
                                   max_pages=ASSET_MAX_PAGES,
                                   stop=lambda batch: all(not _rank(r.get("rank")) for r in batch))
            if not rows:
                raise RuntimeError(f"Messari returned no assets ({self.http.last_error or 'no response'})")
            recs = [_asset_record(r) for r in rows]
            recs = [r for r in recs if r["rank"] and r["slug"]]
            df = pd.DataFrame(recs, columns=ASSET_COLUMNS).drop_duplicates("slug").sort_values("rank").reset_index(drop=True)
            self._assets, self._assets_built_at = df, time.time()
            logger.info("Messari asset table built: %d ranked assets in %.1fs", len(df), time.time() - t0)
            self._save_cache()
            return df

    def warm(self) -> None:
        """Build the asset table if stale; safe from a background thread."""
        try:
            self.asset_table()
        except Exception as e:  # noqa: BLE001
            logger.warning("Messari warm-up failed: %s", e)

    @_ttl_memoised(TTL_SEARCH_S)
    def search_assets(self, text: str) -> Optional[pd.DataFrame]:
        """``/metrics/v2/assets?search=`` (name, symbol, slug or contract address); None on failure."""
        data, _ = self.http.page(ASSETS, {"search": text, "limit": SEARCH_LIMIT, "page": 1})
        if data is None:
            return None
        recs = [_asset_record(r) for r in data if isinstance(r, dict)]
        return pd.DataFrame(recs, columns=ASSET_COLUMNS)

    def asset_record(self, symbol: str, cm_asset_id: Optional[str] = None) -> Optional[dict]:
        """The Messari asset behind a ticker (dict with ASSET_COLUMNS keys) or None.

        Order: SYMBOL_TO_SLUG override -> exact symbol (else slug / name) match in the ranked
        table -> the same match over ``search`` results. Several assets share a ticker (SKY is Sky Protocol
        and Skycastle); the Coin Metrics id qualifier ('sky_sky', 'tao_bittensor') breaks the
        tie, else the best rank wins. Results are cached with the asset table."""
        s = (symbol or "").strip().lower()
        if not s:
            return None
        qual = _qualifier(cm_asset_id)
        key = f"{s}|{qual or ''}"
        if key in self._resolved:
            return self._resolved[key]
        rec: Optional[dict] = None
        try:
            table = None
            try:
                table = self.asset_table()
            except Exception as e:  # noqa: BLE001
                logger.debug("asset_table unavailable for resolution: %s", e)
            override = SYMBOL_TO_SLUG.get(s)
            if override and table is not None:
                hit = table[table["slug"] == override]
                if len(hit):
                    rec = hit.iloc[0].to_dict()
            if rec is None and table is not None:
                cands = table[table["symbol"] == s.upper()]
                if cands.empty:   # a slug or project name instead of a ticker ('helium', 'bittensor')
                    cands = table[(table["slug"] == s) | (table["name"].str.lower() == s)]
                rec = _pick([r for r in cands.to_dict("records")], qual)
            if rec is None:
                found = self.search_assets(override or s)
                if found is not None and len(found):
                    exact = found[(found["symbol"] == s.upper()) | (found["slug"] == s) | (found["name"].str.lower() == s)
                                  | (found["slug"] == override if override else False)]
                    rec = _pick(exact.to_dict("records"), qual)
        except Exception as e:  # noqa: BLE001
            logger.debug("Messari resolution failed for %s: %s", s, e)
        if rec is not None:
            rec = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in rec.items()}
            rec["rank"] = _rank(rec.get("rank"))
        self._resolved[key] = rec
        return rec

    def slug_for(self, symbol: str, cm_asset_id: Optional[str] = None) -> Optional[str]:
        rec = self.asset_record(symbol, cm_asset_id)
        return rec["slug"] if rec else None

    def classify(self, symbols: Sequence[str], cm_ids: Optional[dict] = None) -> pd.DataFrame:
        """One row per requested ticker: symbol, slug, name, rank, sector, sub_sector, tags
        (slug None when Messari does not know the ticker). ``cm_ids`` maps symbol -> Coin
        Metrics asset id for collision-safe resolution."""
        out = []
        for sym in symbols:
            s = str(sym).strip().lower()
            rec = self.asset_record(s, (cm_ids or {}).get(s))
            out.append({"symbol": s, "slug": rec["slug"] if rec else None, "name": rec["name"] if rec else None,
                        "rank": rec["rank"] if rec else None,
                        "sector": rec["sector"] if rec else [], "sub_sector": rec["sub_sector"] if rec else [],
                        "tags": rec["tags"] if rec else []})
        return pd.DataFrame(out, columns=["symbol", "slug", "name", "rank", "sector", "sub_sector", "tags"])

    def sector_members(self, sector: str, n: int = 25) -> pd.DataFrame:
        """Ranked assets whose sector, sub-sector or tag matches ``sector`` (case-insensitive,
        'depin' == 'DePIN'). ``df.attrs['kind']`` says which level matched."""
        table = self.asset_table()
        key = (sector or "").strip().lower().replace("_", " ").replace("-", " ")
        if not key:
            return pd.DataFrame(columns=ASSET_COLUMNS)

        def norm(vals) -> set:
            return {str(v).lower().replace("-", " ").replace("_", " ") for v in (vals or [])}

        kind = None
        for col in ("sector", "sub_sector", "tags"):
            mask = table[col].map(lambda v: key in norm(v))
            if mask.any():
                kind = col
                break
        if kind is None:
            # substring match as a last resort ("exchange" -> "Decentralized Exchange")
            for col in ("sector", "sub_sector", "tags"):
                mask = table[col].map(lambda v: any(key in x for x in norm(v)))
                if mask.any():
                    kind = col
                    break
        if kind is None:
            out = pd.DataFrame(columns=ASSET_COLUMNS)
            out.attrs["kind"] = None
            return out
        out = table[mask].sort_values("rank").head(max(1, int(n))).reset_index(drop=True)
        out.attrs["kind"] = kind
        out.attrs["total"] = int(mask.sum())
        return out

    def sectors(self) -> pd.DataFrame:
        """Messari's sector taxonomy over ranked assets: sector, assets (count), sub_sectors
        (list, most common first), top (three best-ranked tickers)."""
        table = self.asset_table()
        ex = table[["symbol", "rank", "sector", "sub_sector"]].explode("sector").dropna(subset=["sector"])
        rows = []
        for sec, grp in ex.groupby("sector"):
            subs = grp["sub_sector"].explode().dropna().value_counts()
            top = grp.sort_values("rank")["symbol"].head(3).tolist()
            rows.append({"sector": sec, "assets": int(len(grp)), "sub_sectors": subs.index.tolist(), "top": top})
        return pd.DataFrame(rows, columns=["sector", "assets", "sub_sectors", "top"]).sort_values(
            "assets", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ news

    def news_sources(self) -> Optional[pd.DataFrame]:
        data, _ = self.http.page(NEWS_SOURCES, {"limit": 100, "page": 1})
        if data is None:
            return None
        return pd.DataFrame([{"id": s.get("id"), "name": s.get("sourceName"), "type": s.get("sourceType")}
                             for s in data if isinstance(s, dict)])

    @staticmethod
    def _news_frame(rows: list, slugs: Optional[list[str]] = None) -> pd.DataFrame:
        recs = []
        for r in rows:
            assets = [a for a in (r.get("assets") or []) if isinstance(a, dict)]
            sents = [a for a in (r.get("sentiment") or []) if isinstance(a, dict)] if isinstance(r.get("sentiment"), list) else []
            sent_by_slug = {a.get("slug"): a.get("sentiment") for a in sents if a.get("slug")}
            mentioned = {a.get("slug") for a in assets if a.get("slug")} | set(sent_by_slug)
            sym = [a.get("symbol") or a.get("slug") for a in assets if (a.get("symbol") or a.get("slug"))]
            sentiment = None
            if slugs:
                vals = [sent_by_slug[s] for s in slugs if s in sent_by_slug and sent_by_slug[s] is not None]
                sentiment = vals[0] if vals else None
            elif len(sent_by_slug) == 1:
                sentiment = next(iter(sent_by_slug.values()))
            src = r.get("source") or {}
            ts = r.get("publishTime") or r.get("publishTimeMillis")
            recs.append({
                "time": pd.to_datetime(ts, utc=True, unit="ms" if isinstance(ts, (int, float)) else None),
                "title": r.get("title") or "", "source": src.get("sourceName") or "", "source_type": src.get("sourceType") or "",
                "url": r.get("url") or "", "assets": sym, "mentioned": sorted(mentioned), "sentiment": sentiment,
                "description": r.get("description") or "",
            })
        cols = ["time", "title", "source", "source_type", "url", "assets", "mentioned", "sentiment", "description"]
        df = pd.DataFrame(recs, columns=cols)
        if len(df):
            df = df.sort_values("time", ascending=False).drop_duplicates("url").reset_index(drop=True)
        return df

    @_ttl_memoised(TTL_NEWS_S)
    def news(self, assets: Optional[Sequence[str]] = None, hours: int = 24, limit: int = 20,
             source_types: Sequence[str] = ("News",), cm_ids: Optional[dict] = None) -> Optional[pd.DataFrame]:
        """Latest items from the Messari news feed, newest first.

        Columns: time (UTC), title, source, source_type, url, assets (symbols tagged by
        Messari), mentioned (slugs incl. sentiment tags), sentiment (-1..1 for the requested
        asset when Messari scored it, else None), description. ``df.attrs``: window_start,
        slugs, fallback (True when the asset filter timed out and the unfiltered feed was
        filtered client-side by asset tag or by ticker / name in the title), unknown (tickers
        Messari could not resolve). None when Messari returned nothing at all."""
        hours = max(1, min(int(hours), 24 * 14))
        limit = max(1, min(int(limit), 200))
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        types = [t for t in source_types if t in NEWS_SOURCE_TYPES] or ["News"]
        base = {"publishedAfter": _utc_iso(since), "sourceTypes": ",".join(types), "limit": NEWS_PAGE, "sort": 2}

        slugs, patterns, unknown = [], [], []
        for a in list(assets or [])[:NEWS_MAX_ASSETS]:
            rec = self.asset_record(a, (cm_ids or {}).get(str(a).lower()))
            if rec:
                slugs.append(rec["slug"])
                patterns.append(_title_pattern(rec["symbol"] or str(a).upper(), rec["name"]))
            else:
                unknown.append(str(a))
                patterns.append(_title_pattern(str(a).upper(), ""))

        fallback = False
        rows: Optional[list] = None
        if slugs:
            data, _ = self.http.page(NEWS_FEED, dict(base, assetIds=",".join(slugs), page=1),
                                     timeout=NEWS_FILTERED_TIMEOUT_S, retries=0)
            if data is not None:
                rows = [r for r in data if isinstance(r, dict)]
            elif not self.http.slow:
                return None
        if rows is None and (slugs or unknown):
            fallback = True
            scanned = self.http.pages(NEWS_FEED, base, max_pages=NEWS_MAX_PAGES)
            if scanned is None:
                return None
            want = set(slugs)
            rows = []
            for r in scanned:
                tagged = {a.get("slug") for a in (r.get("assets") or []) if isinstance(a, dict)}
                if isinstance(r.get("sentiment"), list):
                    tagged |= {a.get("slug") for a in r["sentiment"] if isinstance(a, dict)}
                title = r.get("title") or ""
                if (want & tagged) or any(p.search(title) for p in patterns):
                    rows.append(r)
        if rows is None:   # no asset filter: plain feed
            rows = self.http.pages(NEWS_FEED, base, max_pages=max(1, math.ceil(limit / NEWS_PAGE)))
            if rows is None:
                return None
        df = self._news_frame(rows, slugs or None).head(limit)
        df.attrs.update({"window_start": since, "slugs": slugs, "fallback": fallback, "unknown": unknown,
                         "source_types": types})
        return df


    # ------------------------------------------------------------------ ETFs (Blockworks via Messari)

    @staticmethod
    def _latest_table(data) -> pd.DataFrame:
        rows = []
        for r in data or []:
            if not isinstance(r, dict):
                continue
            rec = {"id": r.get("id") or r.get("slug") or "", "slug": r.get("slug") or "", "name": r.get("name") or "",
                   "as_of": _epoch(r.get("asOf"))}
            for k, v in (r.get("metrics") or {}).items():
                rec[metric_name(k)] = pd.to_numeric(v, errors="coerce") if v is not None else pd.NA
            rows.append(rec)
        return pd.DataFrame(rows)

    @_ttl_memoised(TTL_ETF_S)
    def etf_assets(self) -> Optional[pd.DataFrame]:
        """Latest crypto ETF figures per underlying asset (bitcoin, ethereum, solana, xrp, multi-asset ...).

        Columns: id, slug, name, as_of (UTC) and the Blockworks metrics in snake_case:
        spot_aum_usd, spot_flow_usd, spot_products, futures_aum_usd, futures_flow_usd,
        futures_products, us_spot_aum_usd / europe_spot_ / apac_spot_ (aum, flow, products),
        deltaone_* and leveraged_* (aum, flow, products), total_volume_usd, us_volume_usd ...
        A null flow means "not yet published for that day", never zero. None on failure."""
        data, _ = self.http.page(ETF_ASSETS)
        if data is None:
            return None
        return self._latest_table(data)

    @_ttl_memoised(TTL_ETF_S)
    def etf_providers(self, n: int = 25) -> Optional[pd.DataFrame]:
        """Latest figures per ETF issuer, largest AUM first: id (Blockworks slug), name, as_of,
        aum_usd, flow_usd, num_products, weighted_aum_expense_ratio, bitcoin_aum_perc,
        spot_aum_perc, rolling_30d_flow_usd, aum_30d_change_perc. None on failure."""
        data, _ = self.http.page(ETF_PROVIDERS, {"limit": max(1, min(int(n), 100)), "page": 1, "sort": "aum-usd", "order": "desc"})
        if data is None:
            return None
        df = self._latest_table(data)
        if "aum_usd" in df.columns:
            df = df.sort_values("aum_usd", ascending=False).reset_index(drop=True)
        return df

    def _timeseries(self, path: str, days: int) -> Optional[pd.DataFrame]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(1, min(int(days), 3650)))
        data, meta = self.http.page(path, {"start": _utc_iso(start), "end": _utc_iso(end)})
        if data is None:
            return None
        points = (data.get("points") if isinstance(data, dict) else None) or []
        schema = [metric_name(c.get("slug")) for c in (meta.get("pointSchemas") or []) if isinstance(c, dict)]
        if not schema and points:
            schema = ["time"] + [f"m{i}" for i in range(1, len(points[0]))]
        rows = [dict(zip(schema, pt)) for pt in points if isinstance(pt, (list, tuple))]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["time"] = df["time"].map(_epoch)
        for c in df.columns:
            if c != "time":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("time").reset_index(drop=True)

    @_ttl_memoised(TTL_ETF_S)
    def etf_asset_timeseries(self, asset: str = "bitcoin", days: int = 30) -> Optional[pd.DataFrame]:
        """Daily ETF history for one underlying (identifier 'bitcoin', 'ethereum', 'solana', 'xrp'):
        time plus the same snake_case metric columns as ``etf_assets``. Null flows on the latest
        day mean not yet published. None on failure, empty frame when Messari has no rows."""
        return self._timeseries(ETF_ASSET_TIMESERIES.format(asset=(asset or "bitcoin").strip().lower()), days)

    @_ttl_memoised(TTL_ETF_S)
    def etf_provider_timeseries(self, provider: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Daily history for one issuer (Blockworks slug such as 'blackrock', 'fidelity', 'grayscale')."""
        return self._timeseries(ETF_PROVIDER_TIMESERIES.format(provider=(provider or "").strip().lower()), days)

    # ------------------------------------------------------------------ Intel events

    @_ttl_memoised(TTL_INTEL_S)
    def intel_events(self, assets: Sequence[str], days: int = 30, importance: Optional[Sequence[str]] = None,
                     limit: int = 50, cm_ids: Optional[dict] = None) -> Optional[pd.DataFrame]:
        """Messari Intel events (upgrades, governance, listings, legal, hacks ...) that name any of
        ``assets`` as primary or secondary, newest first.

        Columns: date (UTC), name, importance (High / Medium / Low), status (Proposed, Discussed,
        Planned, In-Progress, Completed, Rejected), category, subcategory, assets (primary
        symbols), secondary (symbols), details (markdown), link (first resource), id.
        ``attrs``: slugs, unknown, since. None on failure."""
        slugs, unknown = [], []
        for a in list(assets or [])[:NEWS_MAX_ASSETS]:
            rec = self.asset_record(a, (cm_ids or {}).get(str(a).lower()))
            (slugs.append(rec["slug"]) if rec else unknown.append(str(a)))
        cols = ["date", "name", "importance", "status", "category", "subcategory", "assets", "secondary", "details", "link", "id"]
        since = datetime.now(timezone.utc) - timedelta(days=max(1, min(int(days), 3650)))
        if not slugs:
            df = pd.DataFrame(columns=cols)
            df.attrs.update({"slugs": [], "unknown": unknown, "since": since})
            return df
        params = {"primaryOrSecondaryAssets": ",".join(slugs), "startTime": _utc_iso(since),
                  "limit": max(1, min(int(limit), 100)), "page": 1}
        if importance:
            params["importance"] = ",".join(str(i).capitalize() for i in importance)
        data, _ = self.http.page(INTEL_EVENTS, params)
        if data is None:
            return None
        rows = []
        for e in data:
            if not isinstance(e, dict):
                continue
            res = [r for r in (e.get("resources") or []) if isinstance(r, dict) and r.get("link")]
            rows.append({
                "date": pd.to_datetime(e.get("eventDate") or e.get("submissionDate"), utc=True, errors="coerce"),
                "name": e.get("eventName") or "", "importance": e.get("importance") or "", "status": e.get("status") or "",
                "category": e.get("category") or "", "subcategory": e.get("subcategory") or "",
                "assets": [a.get("symbol") or a.get("slug") for a in (e.get("primaryAssets") or []) if isinstance(a, dict)],
                "secondary": [a.get("symbol") or a.get("slug") for a in (e.get("secondaryAssets") or []) if isinstance(a, dict)],
                "details": e.get("eventDetails") or "", "link": res[0]["link"] if res else "", "id": e.get("id") or "",
            })
        df = pd.DataFrame(rows, columns=cols)
        if len(df):
            df = df.sort_values("date", ascending=False).reset_index(drop=True)
        df.attrs.update({"slugs": slugs, "unknown": unknown, "since": since})
        return df

    # ------------------------------------------------------------------ social signals (mindshare)

    @_ttl_memoised(TTL_SIGNALS_S)
    def signals(self, assets: Sequence[str], cm_ids: Optional[dict] = None) -> Optional[pd.DataFrame]:
        """Messari Signals for up to 10 assets: mindshare (share of tracked influencer attention, %)
        over 24h / 7d / 30d with its change, the 24h sentiment close and 7d mean (-1..1), post and
        author counts, and Messari's generated 24h / 7d insight text.

        Columns: symbol, slug, name, mindshare_24h, mindshare_24h_chg, mindshare_7d, mindshare_7d_chg,
        mindshare_30d, mindshare_30d_chg, sentiment_24h, sentiment_7d, posts_24h, authors_24h,
        posts_7d, insight_24h, insight_7d. ``attrs``: unknown. None on failure."""
        slugs, unknown = [], []
        for a in list(assets or [])[:10]:
            rec = self.asset_record(a, (cm_ids or {}).get(str(a).lower()))
            (slugs.append(rec["slug"]) if rec else unknown.append(str(a)))
        cols = ["symbol", "slug", "name", "mindshare_24h", "mindshare_24h_chg", "mindshare_7d", "mindshare_7d_chg",
                "mindshare_30d", "mindshare_30d_chg", "sentiment_24h", "sentiment_7d", "posts_24h", "authors_24h",
                "posts_7d", "insight_24h", "insight_7d"]
        if not slugs:
            df = pd.DataFrame(columns=cols)
            df.attrs["unknown"] = unknown
            return df
        data, _ = self.http.page(SIGNALS_ASSETS, {"assetIds": ",".join(slugs), "limit": len(slugs), "page": 1})
        if data is None:
            return None

        def g(d, *path):
            for k in path:
                d = d.get(k) if isinstance(d, dict) else None
            return d

        rows = []
        for r in data:
            if not isinstance(r, dict):
                continue
            ms, se, pm = r.get("mindshare") or {}, r.get("sentiment") or {}, r.get("postMetrics") or {}
            rows.append({
                "symbol": (r.get("symbol") or "").upper(), "slug": r.get("slug") or "", "name": r.get("name") or "",
                "mindshare_24h": g(ms, "24h", "percentage"), "mindshare_24h_chg": g(ms, "24h", "percentageChange"),
                "mindshare_7d": g(ms, "7d", "percentage"), "mindshare_7d_chg": g(ms, "7d", "percentageChange"),
                "mindshare_30d": g(ms, "30d", "percentage"), "mindshare_30d_chg": g(ms, "30d", "percentageChange"),
                "sentiment_24h": g(se, "24h", "close"), "sentiment_7d": g(se, "7d", "mean"),
                "posts_24h": g(pm, "24h", "totalPosts"), "authors_24h": g(pm, "24h", "uniqueAuthorsMentioning"),
                "posts_7d": g(pm, "7d", "totalPosts"),
                "insight_24h": g(ms, "24h", "insight", "message") or "", "insight_7d": g(ms, "7d", "insight", "message") or "",
            })
        df = pd.DataFrame(rows, columns=cols)
        for c in cols[3:14]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # keep the caller's order
        order = {s: i for i, s in enumerate(slugs)}
        df = df.assign(_o=df["slug"].map(order)).sort_values("_o").drop(columns="_o").reset_index(drop=True)
        df.attrs["unknown"] = unknown
        return df


__all__ = [
    "ASSET_COLUMNS", "BASE_URL", "MessariHTTP", "MessariProvider", "NEWS_MAX_ASSETS", "NEWS_SOURCE_TYPES",
    "RETRY_STATUSES", "SYMBOL_TO_SLUG", "TTL_ASSETS_S", "TTL_ETF_S", "TTL_INTEL_S", "TTL_NEWS_S", "TTL_SIGNALS_S",
    "metric_name",
]
