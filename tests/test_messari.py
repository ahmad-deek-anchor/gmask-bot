"""providers.messari + tools/messari_tools.py against a fake requests session. No network.

Canned payloads are trimmed copies of real Messari responses (2026-09-16): the asset list
(sorted by rank, rank 0 = unranked), the news feed (assets tagged per item, sentiment as a
list of per-asset scores), and the 524 behaviour of the asset-filtered news call.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import requests

import providers.messari as msr
from providers.messari import ASSET_COLUMNS, MessariHTTP, MessariProvider
from tools import messari_tools as mt

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _asset(slug, symbol, name, rank, sectors, subs, tags=()):
    return {"id": slug + "-id", "slug": slug, "symbol": symbol, "name": name, "rank": rank,
            "category": "Cryptocurrency", "sector": "x", "sectorV2": list(sectors), "subSectorV2": list(subs),
            "tags": list(tags), "hasNews": True, "hasIntel": True}


RANKED = [
    _asset("bitcoin", "BTC", "Bitcoin", 1, ["Networks"], ["Layer-1"], ["Proof-of-Work"]),
    _asset("ethereum", "ETH", "Ethereum", 2, ["Networks"], ["Layer-1"], ["Proof-of-Stake"]),
    _asset("hyperliquid", "HYPE", "Hyperliquid", 10, ["DeFi", "Networks"], ["Derivatives", "Decentralized Exchange", "Layer-1"]),
    _asset("bittensor-0", "TAO", "Bittensor", 40, ["AI"], ["Decentralized AI"]),
    _asset("sky-protocol", "SKY", "Sky", 61, ["DeFi"], ["Lending", "Stablecoin Issuer"]),
    _asset("helium", "HNT", "Helium", 95, ["DePIN"], ["Wireless"]),
    _asset("render", "RENDER", "Render", 120, ["DePIN", "AI"], ["Compute"]),
    _asset("skycastle", "SKY", "Skycastle", 3787, ["Gaming"], ["Metaverse"]),
]
UNRANKED = [_asset("tao-private-network", "SN65", "TAO Private Network", 0, [], []),
            _asset("cowgorithm", "COW", "Cowgorithm", 0, [], [])]

SEARCH_RESULTS = {
    "pendle": [_asset("pendle", "PENDLE", "Pendle", 88, ["DeFi"], ["Yield"]), _asset("pendle-fan", "PENDLEF", "Pendle Fan", 0, [], [])],
    "wif": [_asset("dogwifhat", "WIF", "dogwifhat", 130, ["Meme"], ["Dog"])],
}


def _news_item(title, when, slugs=(), source="CoinDesk", stype="News", sentiment=None, url=None):
    return {
        "assets": [{"id": s + "-id", "name": s.title(), "slug": s, "symbol": s[:3].upper()} for s in slugs],
        "publishTimeMillis": int(when.timestamp() * 1000), "publishTime": _iso(when),
        "source": {"id": "src", "sourceName": source, "sourceType": stype},
        "title": title, "url": url or f"https://example.com/{re.sub('[^a-z0-9]+', '-', title.lower())}",
        "category": None, "subcategory": None, "description": None,
        "sentiment": sentiment,
    }


FEED = [
    _news_item("Bitcoin slides after Senate cloture vote fails", NOW - timedelta(hours=1), ["bitcoin"],
               sentiment=[{"id": "b", "name": "Bitcoin", "slug": "bitcoin", "symbol": "BTC", "sentiment": -0.6}]),
    _news_item("Ethereum ETF issuers file for staking", NOW - timedelta(hours=2), ["ethereum"]),
    _news_item("BTC options open interest hits record", NOW - timedelta(hours=3), []),           # title match only
    _news_item("Aleo mainnet update", NOW - timedelta(hours=4), ["aleo"], source="Aleo Blog", stype="Blog"),
    _news_item("Helium adds carrier partner", NOW - timedelta(hours=5), ["helium"]),
    _news_item("Bitcoin slides after Senate cloture vote fails", NOW - timedelta(hours=1), ["bitcoin"],
               url="https://example.com/bitcoin-slides-after-senate-cloture-vote-fails"),           # duplicate url
]


class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text if payload is None else json.dumps(payload)
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Serves the Messari endpoints used by the provider; records every request."""

    def __init__(self, filtered_news="524", fail_assets=False):
        self.headers = {}
        self.calls: list[tuple[str, dict]] = []
        self.filtered_news = filtered_news
        self.fail_assets = fail_assets

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        path = urlparse(url).path
        self.calls.append((path, params))
        if path == msr.ASSETS:
            if self.fail_assets:
                return FakeResponse(500, text="boom")
            if "search" in params:
                q = params["search"].lower()
                hits = SEARCH_RESULTS.get(q, [a for a in RANKED + UNRANKED if q in a["slug"] or q == a["symbol"].lower()])
                return FakeResponse(200, {"data": hits, "error": None,
                                          "metadata": {"pageSize": 25, "page": 1, "totalRows": len(hits), "totalPages": 1}})
            page = int(params.get("page", 1))
            pages = {1: RANKED, 2: UNRANKED, 3: UNRANKED}
            return FakeResponse(200, {"data": pages.get(page, []), "error": None,
                                      "metadata": {"pageSize": 500, "page": page, "totalRows": 1200, "totalPages": 3}})
        if path == msr.NEWS_FEED:
            if "assetIds" in params:
                if self.filtered_news == "524":
                    return FakeResponse(524, text="<html>cloudflare timeout</html>")
                if self.filtered_news == "timeout":
                    raise requests.Timeout("read timed out")
                if self.filtered_news == "403":
                    return FakeResponse(403, {"error": "forbidden"})
                want = set(params["assetIds"].split(","))
                rows = [r for r in FEED if want & {a["slug"] for a in r["assets"]}]
                return FakeResponse(200, {"data": rows, "error": None, "metadata": {"limit": 100, "page": 1, "totalRows": len(rows), "totalPages": 1}})
            types = set(params.get("sourceTypes", "News").split(","))
            rows = [r for r in FEED if r["source"]["sourceType"] in types]
            page = int(params.get("page", 1))
            return FakeResponse(200, {"data": rows if page == 1 else [], "error": None,
                                      "metadata": {"limit": 100, "page": page, "totalRows": len(rows), "totalPages": 1}})
        if path == msr.NEWS_SOURCES:
            return FakeResponse(200, {"data": [{"id": "1", "sourceName": "CoinDesk", "sourceType": "News"}], "error": None, "metadata": {}})
        return FakeResponse(404, {"error": "not found"})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(msr.time, "sleep", lambda s: None)
    import providers._http as http
    monkeypatch.setattr(http.time, "sleep", lambda s: None)


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def provider(session, tmp_path):
    return MessariProvider("k", session=session, cache_path=tmp_path / "messari.json")


@pytest.fixture
def tools(provider, monkeypatch):
    monkeypatch.setattr(mt, "_get_messari", lambda: provider)
    monkeypatch.setattr(mt, "_cm_ids", lambda toks: {})
    return provider


# --- HTTP ------------------------------------------------------------------

def test_http_header_and_envelope_unwrapping(session):
    http = MessariHTTP("secret", session=session)
    assert session.headers["x-messari-api-key"] == "secret" and "x-api-key" not in session.headers
    data, meta = http.page(msr.ASSETS, {"limit": 500, "page": 1})
    assert isinstance(data, list) and data[0]["slug"] == "bitcoin" and meta["totalPages"] == 3

    class Nested(FakeSession):
        def get(self, url, params=None, timeout=None):
            return FakeResponse(200, {"data": {"data": [{"id": "blackrock"}], "metadata": {"page": 1, "totalPages": 1}}, "error": None})
    data, meta = MessariHTTP("k", session=Nested()).page("/x")
    assert data == [{"id": "blackrock"}] and meta["totalPages"] == 1


def test_http_524_is_retried_then_reported_slow(session):
    http = MessariHTTP("k", session=session, max_retries=1)
    data, _ = http.page(msr.NEWS_FEED, {"assetIds": "bitcoin"})
    assert data is None and http.slow and http.last_status == 524
    assert sum(1 for p, _ in session.calls if p == msr.NEWS_FEED) == 2      # one retry
    data, _ = http.page(msr.NEWS_FEED, {"assetIds": "bitcoin"}, retries=0)
    assert data is None and sum(1 for p, _ in session.calls if p == msr.NEWS_FEED) == 3   # per-call override: no retry
    session.filtered_news = "403"
    data, _ = http.page(msr.NEWS_FEED, {"assetIds": "bitcoin"})
    assert data is None and not http.slow and http.last_status == 403


def test_http_pages_stops_on_total_pages_and_stop_callback(session):
    http = MessariHTTP("k", session=session)
    rows = http.pages(msr.ASSETS, {"limit": 500}, max_pages=10)
    assert len(rows) == len(RANKED) + 2 * len(UNRANKED)            # 3 pages, totalPages honoured
    rows = http.pages(msr.ASSETS, {"limit": 500}, max_pages=10, stop=lambda b: all(not r["rank"] for r in b))
    assert len(rows) == len(RANKED) + len(UNRANKED)                # stopped after the first all-unranked page


# --- asset table / resolution ------------------------------------------------

def test_asset_table_keeps_ranked_only_and_caches_to_disk(provider, session, tmp_path):
    df = provider.asset_table()
    assert list(df.columns) == ASSET_COLUMNS
    assert df["slug"].tolist()[:3] == ["bitcoin", "ethereum", "hyperliquid"] and "cowgorithm" not in df["slug"].tolist()
    n_calls = len(session.calls)
    provider.asset_table()
    assert len(session.calls) == n_calls                                   # memoised
    fresh = MessariProvider("k", session=FakeSession(), cache_path=tmp_path / "messari.json")
    assert fresh._assets is not None and len(fresh._assets) == len(df)    # served from the file
    assert fresh.asset_record("hype")["slug"] == "hyperliquid" and fresh.http.call_count == 0


def test_asset_table_failure_raises(tmp_path):
    prov = MessariProvider("k", session=FakeSession(fail_assets=True), cache_path=tmp_path / "m.json")
    with pytest.raises(RuntimeError):
        prov.asset_table()


def test_resolution_overrides_collisions_and_search(provider, session):
    assert provider.slug_for("BTC") == "bitcoin"                              # override
    assert provider.slug_for("tao") == "bittensor-0"                          # override (Messari's odd slug)
    assert provider.slug_for("sky") == "sky-protocol"                         # two SKYs: override / best rank
    assert provider.slug_for("hnt") == "helium"                               # exact symbol in the table
    assert provider.slug_for("pendle") == "pendle"                            # not in table -> search, ranked match
    assert ("/metrics/v2/assets", {"search": "pendle", "limit": 25, "page": 1}) in session.calls
    assert provider.slug_for("notacoin") is None and provider.slug_for("") is None
    n = len(session.calls)
    provider.slug_for("pendle"); provider.slug_for("notacoin")
    assert len(session.calls) == n                                            # resolutions cached


def test_qualifier_breaks_ticker_collisions(provider):
    del msr.SYMBOL_TO_SLUG["sky"]
    try:
        provider._resolved.clear()
        assert provider.slug_for("sky") == "sky-protocol"                     # best rank wins without a qualifier
        provider._resolved.clear()
        assert provider.slug_for("sky", cm_asset_id="sky_skycastle") == "skycastle"
    finally:
        msr.SYMBOL_TO_SLUG["sky"] = "sky-protocol"


def test_classify_sector_members_and_sectors(provider):
    df = provider.classify(["hype", "tao", "zzz"])
    assert df["slug"].tolist() == ["hyperliquid", "bittensor-0", None]
    assert df.iloc[0]["sector"] == ["DeFi", "Networks"] and df.iloc[2]["sector"] == []

    depin = provider.sector_members("depin", n=10)
    assert depin["symbol"].tolist() == ["HNT", "RENDER"] and depin.attrs["kind"] == "sector" and depin.attrs["total"] == 2
    l1 = provider.sector_members("Layer-1", n=2)
    assert l1["slug"].tolist() == ["bitcoin", "ethereum"] and l1.attrs["kind"] == "sub_sector"
    pow_ = provider.sector_members("proof-of-work")
    assert pow_["slug"].tolist() == ["bitcoin"] and pow_.attrs["kind"] == "tags"
    assert provider.sector_members("exchange")["slug"].tolist() == ["hyperliquid"]     # substring fallback
    none = provider.sector_members("underwater basket weaving")
    assert none.empty and none.attrs["kind"] is None

    sec = provider.sectors()
    assert sec.iloc[0]["sector"] == "Networks" and sec.iloc[0]["assets"] == 3 and sec.iloc[0]["top"] == ["BTC", "ETH", "HYPE"]
    depin_row = sec.set_index("sector").loc["DePIN"]
    assert depin_row["assets"] == 2 and depin_row["sub_sectors"] == ["Wireless", "Compute"]


# --- news ------------------------------------------------------------------

def test_news_market_wide_dedupes_and_sorts(provider):
    df = provider.news(hours=24, limit=10)
    assert list(df.columns)[:5] == ["time", "title", "source", "source_type", "url"]
    assert len(df) == 4                                                   # 5 News items minus the duplicate url
    assert df["time"].is_monotonic_decreasing and str(df["time"].dtype).startswith("datetime64[ns, UTC]")
    assert df.iloc[0]["sentiment"] == -0.6 and df.iloc[0]["assets"] == ["BIT"]
    assert df.attrs["fallback"] is False and df.attrs["slugs"] == []
    blogs = provider.news(hours=24, limit=10, source_types=("News", "Blog"))
    assert "Aleo mainnet update" in blogs["title"].tolist()


def test_news_asset_filter_falls_back_when_messari_times_out(provider, session):
    df = provider.news(assets=["btc"], hours=12, limit=10)
    assert df.attrs["fallback"] is True and df.attrs["slugs"] == ["bitcoin"]
    assert df["title"].tolist() == ["Bitcoin slides after Senate cloture vote fails", "BTC options open interest hits record"]
    assert df.iloc[0]["sentiment"] == -0.6 and pd.isna(df.iloc[1]["sentiment"])
    filtered = [p for p in session.calls if p[0] == msr.NEWS_FEED and "assetIds" in p[1]]
    assert len(filtered) == 1                                              # failed fast: no retry on the filtered call
    # an unknown ticker still gets a title-text match and is reported
    df2 = provider.news(assets=["helium", "xyzzy"], hours=12, limit=10)
    assert df2.attrs["unknown"] == ["xyzzy"] and df2["title"].tolist() == ["Helium adds carrier partner"]


def test_news_asset_filter_is_combined_with_the_headline_scan(tmp_path):
    prov = MessariProvider("k", session=FakeSession(filtered_news="ok"), cache_path=tmp_path / "m.json")
    df = prov.news(assets=["eth"], hours=24, limit=5)
    assert df.attrs["fallback"] is False and df.attrs["scanned"] is True
    assert df["title"].tolist() == ["Ethereum ETF issuers file for staking"]          # tagged item, not duplicated by the scan
    df = prov.news(assets=["btc"], hours=24, limit=5)
    # Messari's filter returns the tagged item; the scan adds the untagged headline mention
    assert df["title"].tolist() == ["Bitcoin slides after Senate cloture vote fails", "BTC options open interest hits record"]


def test_news_null_data_means_no_rows_not_failure(tmp_path):
    class Quiet(FakeSession):
        def get(self, url, params=None, timeout=None):
            if urlparse(url).path == msr.NEWS_FEED and "assetIds" in (params or {}):
                return FakeResponse(200, {"data": None, "error": None, "metadata": {"limit": 100, "page": 1, "totalRows": 0, "totalPages": 0}})
            return super().get(url, params=params, timeout=timeout)
    prov = MessariProvider("k", session=Quiet(), cache_path=tmp_path / "m.json")
    data, _ = prov.http.page(msr.NEWS_FEED, {"assetIds": "zcash"})
    assert data == [] and not prov.http.slow
    df = prov.news(assets=["hnt"], hours=24, limit=5)                                  # Messari: null; scan finds the headline
    assert df is not None and df.attrs["fallback"] is False and df["title"].tolist() == ["Helium adds carrier partner"]


def test_news_client_timeout_and_hard_errors_still_scan(tmp_path):
    prov = MessariProvider("k", session=FakeSession(filtered_news="timeout"), cache_path=tmp_path / "m.json")
    df = prov.news(assets=["btc"], hours=24, limit=5)
    assert df is not None and df.attrs["fallback"] is True and len(df) == 2
    prov = MessariProvider("k", session=FakeSession(filtered_news="403"), cache_path=tmp_path / "m2.json")
    df = prov.news(assets=["btc"], hours=24, limit=5)
    assert df is not None and df.attrs["fallback"] is True and len(df) == 2           # a 403 on the filter no longer kills the answer


def test_news_is_memoised_for_five_minutes(provider, session):
    provider.news(hours=24, limit=10)
    n = len(session.calls)
    provider.news(hours=24, limit=10)
    assert len(session.calls) == n


# --- tools -------------------------------------------------------------------

def test_tools_unavailable_without_provider(monkeypatch):
    monkeypatch.setattr(mt, "_get_messari", lambda: None)
    assert mt.get_crypto_news.invoke({}) == mt.UNAVAILABLE
    assert mt.classify_tokens.invoke({"tokens": ["btc"]}) == mt.UNAVAILABLE


def test_get_crypto_news_tool(tools):
    out = mt.get_crypto_news.invoke({"tokens": ["btc"], "hours": 12, "limit": 5})
    assert out.startswith("### News: BTC, since")
    assert "**Bitcoin slides after Senate cloture vote fails** (CoinDesk) [BIT] - Messari sentiment -0.6 (negative) - https://" in out
    assert "matched by Messari's asset tags" in out and "per-asset filter timed out" in out and "UTC" in out
    market = mt.get_crypto_news.invoke({"limit": 3})
    assert market.startswith("### News: crypto market") and market.count("\n- ") == 3
    assert "Aleo" not in market
    assert "Aleo" in mt.get_crypto_news.invoke({"limit": 10, "include_blogs": True})


def test_get_crypto_news_tool_reports_vendor_failure(tools, monkeypatch):
    monkeypatch.setattr(tools, "news", lambda **kw: None)
    assert "did not respond" in mt.get_crypto_news.invoke({"tokens": ["btc"]})


def test_classify_and_sector_tools(tools):
    out = mt.classify_tokens.invoke({"tokens": ["hype", "tao", "zzz"]})
    assert "| HYPE | Hyperliquid (hyperliquid) | 10 | DeFi, Networks | Derivatives, Decentralized Exchange, Layer-1 |" in out
    assert "Not in Messari's asset list: ZZZ." in out and "not GICS" in out
    assert mt.classify_tokens.invoke({"tokens": []}) == "Give at least one ticker to classify."

    out = mt.get_sector_members.invoke({"sector": "DePIN", "n": 5})
    assert out.startswith("### DePIN: 2 of 2 ranked assets (Messari sector)")
    assert "| 95 | HNT | Helium (helium) | Wireless | - |" in out
    assert "No Messari sector" in mt.get_sector_members.invoke({"sector": "nonsense"})

    out = mt.list_crypto_sectors.invoke({})
    assert "| Networks | 3 | BTC, ETH, HYPE | Layer-1, Derivatives, Decentralized Exchange |" in out
    assert "| DePIN | 2 | HNT, RENDER | Wireless, Compute |" in out


def test_tools_registered_in_chat_default_tools():
    from chat import default_tools
    names = [t.name for t in default_tools()]
    assert set(mt.MESSARI_TOOL_NAMES) <= set(names)
    assert names.index("get_crypto_news") > names.index("get_recent_trades")


# --- ETF / Intel / signals (Commit C) ------------------------------------------

ETF_ASSET_ROWS = [
    {"id": "bitcoin", "slug": "bitcoin", "name": "bitcoin", "asOf": 1789344000,
     "metrics": {"typeSpotAumUsd": 117933987263, "typeSpotFlowUsd": None, "typeSpotProductCount": 136,
                 "typeFuturesAumUsd": 3637095747, "typeFuturesFlowUsd": None, "typeFuturesProductCount": 6,
                 "typeSpotUnitedstatesAumUsd": 102458910211, "typeSpotUnitedstatesFlowUsd": None, "typeSpotUnitedstatesProductCount": 59,
                 "typeSpotEuropeAumUsd": 9916885521.0, "typeSpotEuropeFlowUsd": None, "typeSpotEuropeProductCount": 36,
                 "typeSpotApacAumUsd": 1094806417, "typeSpotApacFlowUsd": None, "typeSpotApacProductCount": 15,
                 "strategyDeltaoneAumUsd": 117217862805, "strategyLeveragedAumUsd": 1751943443, "totalVolumeUsd": None}},
    {"id": "ethereum", "slug": "ethereum", "name": "ethereum", "asOf": 1789344000,
     "metrics": {"typeSpotAumUsd": 20074096625, "typeSpotFlowUsd": 12345678.0, "typeSpotProductCount": 69,
                 "typeFuturesAumUsd": 1693531884, "typeFuturesFlowUsd": -1000.0, "typeFuturesProductCount": 6,
                 "typeSpotUnitedstatesAumUsd": 16521391841, "typeSpotUnitedstatesFlowUsd": 10000000.0, "typeSpotUnitedstatesProductCount": 22,
                 "totalVolumeUsd": 2.5e9}},
    {"id": "multi-asset", "slug": "", "name": "multi-asset", "asOf": 1789344000, "metrics": {"typeSpotAumUsd": 4952829876, "typeSpotFlowUsd": None}},
]
ETF_PROVIDER_ROWS = [
    {"id": "blackrock", "slug": "blackrock", "name": "BlackRock", "asOf": 1789344000,
     "metrics": {"aumUsd": 74152364308, "flowUsd": 0, "numProducts": 8, "weightedAumExpenseRatio": 0.00249, "bitcoinAumPerc": 0.8628,
                 "spotAumPerc": 0.9988, "rolling30dFlowUsd": 4752107990, "aum30dChangePerc": 0.3583}},
    {"id": "grayscale", "slug": "grayscale", "name": "Grayscale", "asOf": 1789344000,
     "metrics": {"aumUsd": 20953748520, "flowUsd": 0, "numProducts": 18, "weightedAumExpenseRatio": 0.011, "bitcoinAumPerc": 0.7215,
                 "spotAumPerc": 0.9989, "rolling30dFlowUsd": 223932364, "aum30dChangePerc": 0.3238}},
]
ETF_SCHEMA = [{"slug": s, "name": s, "is_timestamp": s == "time"} for s in
              ("time", "type-spot-aum-usd", "type-spot-flow-usd", "type-spot-product-count", "type-futures-aum-usd", "type-futures-flow-usd",
               "type-spot-unitedstates-flow-usd", "type-spot-europe-flow-usd", "type-spot-apac-flow-usd")]
ETF_POINTS = [
    [1788825600, 117462924236, 8451486, 136, 3669840760, 9044040, -3640200, 7596541, 1248721],       # 2026-09-08
    [1788912000, 117436226586.99998, -99021372, 136, 3666610638, -27346493, -109130127, 7127470, 262666],
    [1788998400, 115443362686, -280177060, 136, 3573067788, -31931044, -285891462, 5393463, 0],
    [1789344000, 117933987263, None, 136, 3637095747, None, None, None, None],                       # latest: unpublished
]
INTEL_ROWS = [
    {"id": "e1", "submissionDate": "2026-09-11T10:00:00Z", "eventDate": "2026-09-11T10:00:00Z", "eventName": "EURCV on Aave V4",
     "primaryAssets": [{"slug": "aave", "symbol": "AAVE", "name": "Aave"}], "secondaryAssets": [{"slug": "ethereum", "symbol": "ETH", "name": "Ethereum"}],
     "eventDetails": "Proposal to onboard **EURCV** [link](https://x.com/a).", "status": "Proposed", "category": "Protocol Management",
     "subcategory": "New Supported Asset or Market", "importance": "Low", "resources": [{"name": "post", "link": "https://x.com/a"}], "globalEvent": False},
    {"id": "e2", "submissionDate": "2026-09-01T10:00:00Z", "eventDate": "2026-09-01T10:00:00Z", "eventName": "SEC Transfer Agent Rule Modernization",
     "primaryAssets": [{"slug": "bitcoin", "symbol": "BTC", "name": "Bitcoin"}], "secondaryAssets": None,
     "eventDetails": "The SEC proposed ...", "status": "Discussed", "category": "Legal and Regulatory", "subcategory": "Regulation",
     "importance": "Medium", "resources": [], "globalEvent": True},
]
SIGNAL_ROWS = [
    {"id": "x", "name": "Bitcoin", "symbol": "BTC", "slug": "bitcoin",
     "mindshare": {"24h": {"percentage": 4.58, "percentageChange": -18.1, "insight": {"message": "Bitcoin's mindshare is declining as the Clarity Act vote fails."}},
                   "7d": {"percentage": 5.62, "percentageChange": -3.0, "insight": {"message": "7d read."}}, "30d": {"percentage": 6.93, "percentageChange": -7.1}},
     "sentiment": {"24h": {"open": 0.28, "high": 0.28, "low": 0.17, "close": 0.17, "mean": 0.22}, "7d": {"mean": 0.28}},
     "postMetrics": {"24h": {"totalPosts": 450, "uniqueAuthorsMentioning": 403}, "7d": {"totalPosts": 4218}}},
    {"id": "y", "name": "Ethereum", "symbol": "ETH", "slug": "ethereum",
     "mindshare": {"24h": {"percentage": 2.34, "percentageChange": 5.0}, "7d": {"percentage": 3.0, "percentageChange": 1.0}, "30d": {"percentage": 3.5}},
     "sentiment": {}, "postMetrics": {}},
]


class FakeSessionC(FakeSession):
    """Adds the ETF / Intel / signal endpoints."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.etf_fail = False

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        path = urlparse(url).path
        if path == msr.ETF_ASSETS:
            self.calls.append((path, params))
            if self.etf_fail:
                return FakeResponse(503, text="down")
            return FakeResponse(200, {"data": {"data": ETF_ASSET_ROWS}, "error": None})           # nested envelope
        if path == msr.ETF_PROVIDERS:
            self.calls.append((path, params))
            return FakeResponse(200, {"data": {"data": ETF_PROVIDER_ROWS, "metadata": {"page": 1, "totalPages": 26}}, "error": None})
        if path.startswith("/metrics/v2/protocols/etfs/assets/"):
            self.calls.append((path, params))
            if "start" not in params or "end" not in params:
                return FakeResponse(200, {"data": None, "error": "parse time range: missing time range end", "metadata": None})
            if "/assets/dogecoin/" in path:
                return FakeResponse(200, {"data": {"points": []}, "error": None, "metadata": {"pointSchemas": ETF_SCHEMA}})
            return FakeResponse(200, {"data": {"points": ETF_POINTS}, "error": None, "metadata": {"pointSchemas": ETF_SCHEMA}})
        if path == msr.INTEL_EVENTS:
            self.calls.append((path, params))
            want = set(params.get("primaryOrSecondaryAssets", "").split(","))
            rows = [r for r in INTEL_ROWS if want & ({a["slug"] for a in r["primaryAssets"]} | {a["slug"] for a in (r["secondaryAssets"] or [])})]
            if params.get("importance"):
                rows = [r for r in rows if r["importance"] in params["importance"].split(",")]
            return FakeResponse(200, {"data": rows, "error": None, "metadata": {"limit": 50, "page": 1, "totalRows": len(rows), "totalPages": 1}})
        if path == msr.SIGNALS_ASSETS:
            self.calls.append((path, params))
            want = params.get("assetIds", "").split(",")
            rows = [r for r in SIGNAL_ROWS if r["slug"] in want]
            return FakeResponse(200, {"data": rows, "error": None, "metadata": {"pageSize": len(rows), "page": 1, "totalRows": len(rows), "totalPages": 1}})
        return super().get(url, params=params, timeout=timeout)


@pytest.fixture
def session_c():
    return FakeSessionC()


@pytest.fixture
def provider_c(session_c, tmp_path):
    return MessariProvider("k", session=session_c, cache_path=tmp_path / "messari.json")


@pytest.fixture
def tools_c(provider_c, monkeypatch):
    monkeypatch.setattr(mt, "_get_messari", lambda: provider_c)
    monkeypatch.setattr(mt, "_cm_ids", lambda toks: {})
    monkeypatch.setattr(mt, "_onchain_net_flows", lambda days: None)
    return provider_c


def test_metric_name_normalisation():
    assert msr.metric_name("typeSpotUnitedstatesFlowUsd") == "us_spot_flow_usd"
    assert msr.metric_name("type-spot-flow-usd") == "spot_flow_usd"
    assert msr.metric_name("strategyDeltaoneProductCount") == "deltaone_products"
    assert msr.metric_name("volumeApacUsd") == "apac_volume_usd" and msr.metric_name("aumUsd") == "aum_usd"


def test_etf_assets_and_providers(provider_c, session_c):
    df = provider_c.etf_assets()
    assert df["id"].tolist() == ["bitcoin", "ethereum", "multi-asset"]
    btc = df.set_index("id").loc["bitcoin"]
    assert btc["spot_aum_usd"] == 117933987263 and pd.isna(btc["spot_flow_usd"])           # null flow stays NA, never 0
    assert btc["us_spot_products"] == 59 and btc["europe_spot_aum_usd"] == pytest.approx(9916885521.0)
    assert str(btc["as_of"])[:10] == "2026-09-14"
    eth = df.set_index("id").loc["ethereum"]
    assert eth["spot_flow_usd"] == 12345678.0 and eth["total_volume_usd"] == 2.5e9
    n = len(session_c.calls)
    provider_c.etf_assets()
    assert len(session_c.calls) == n                                                          # memoised (1 h)
    prov = provider_c.etf_providers(n=5)
    assert prov["id"].tolist() == ["blackrock", "grayscale"] and prov.iloc[0]["rolling30d_flow_usd"] == 4752107990
    session_c.etf_fail = True
    fresh = MessariProvider("k", session=session_c, cache_path=None)
    assert fresh.etf_assets() is None


def test_etf_asset_timeseries_parses_points(provider_c, session_c):
    df = provider_c.etf_asset_timeseries("bitcoin", days=10)
    assert list(df.columns) == ["time", "spot_aum_usd", "spot_flow_usd", "spot_products", "futures_aum_usd", "futures_flow_usd",
                                "us_spot_flow_usd", "europe_spot_flow_usd", "apac_spot_flow_usd"]
    assert len(df) == 4 and df["time"].is_monotonic_increasing and str(df["time"].iloc[0])[:10] == "2026-09-08"
    assert df.iloc[1]["spot_flow_usd"] == -99021372 and pd.isna(df.iloc[-1]["spot_flow_usd"])
    call = [p for p in session_c.calls if p[0].startswith("/metrics/v2/protocols/etfs/assets/bitcoin/")][0]
    assert "start" in call[1] and "end" in call[1]                                             # both required by Messari
    assert provider_c.etf_asset_timeseries("dogecoin", days=10).empty


def test_intel_events_and_signals(provider_c):
    ev = provider_c.intel_events(["eth", "btc"], days=30)
    assert ev["name"].tolist() == ["EURCV on Aave V4", "SEC Transfer Agent Rule Modernization"]      # newest first
    assert ev.iloc[0]["assets"] == ["AAVE"] and ev.iloc[0]["secondary"] == ["ETH"] and ev.iloc[0]["link"] == "https://x.com/a"
    assert ev.iloc[1]["secondary"] == [] and ev.iloc[1]["link"] == ""
    hi = provider_c.intel_events(["btc"], days=30, importance=["medium"])
    assert hi["name"].tolist() == ["SEC Transfer Agent Rule Modernization"]
    none = provider_c.intel_events(["zzz"], days=30)
    assert none.empty and none.attrs["unknown"] == ["zzz"]

    sg = provider_c.signals(["eth", "btc", "zzz"])
    assert sg["symbol"].tolist() == ["ETH", "BTC"] and sg.attrs["unknown"] == ["zzz"]          # caller's order kept
    btc = sg.set_index("symbol").loc["BTC"]
    assert btc["mindshare_24h"] == 4.58 and btc["sentiment_24h"] == 0.17 and btc["posts_24h"] == 450
    assert btc["insight_24h"].startswith("Bitcoin's mindshare is declining")
    assert pd.isna(sg.set_index("symbol").loc["ETH"]["sentiment_24h"])


def test_etf_tools(tools_c):
    out = mt.get_etf_overview.invoke({"asset": "btc"})
    assert out.startswith("### Crypto ETFs by underlying asset (Messari / Blockworks Research, as of 2026-09-14)")
    assert "| bitcoin | $117.93B | n/a | 136 | $3.64B | n/a | n/a |" in out
    assert "**bitcoin spot ETFs by region**" in out and "US $102.46B / n/a / 59" in out
    assert "| BlackRock | $74.15B | $0 | $4.75B | +35.8% | 8 | 86.3% | 0.25% |" in out
    assert "not zero" in out
    assert "(no ETF row for 'dogecoin'" in mt.get_etf_overview.invoke({"asset": "dogecoin"})

    out = mt.get_etf_flows.invoke({"asset": "bitcoin", "days": 10})
    assert out.startswith("### bitcoin ETF flows, last 10 days")
    assert "latest published day 2026-09-10: spot net flow $-280.18M, futures $-31.93M" in out
    assert "window: spot net $-370.75M over 3 published days (1 inflow, 2 outflow)" in out
    assert "2026-09-14: flows not yet published" in out
    assert "| 2026-09-14 | n/a | n/a | $117.93B | n/a | n/a | n/a |" in out and "On-chain" not in out
    assert "no ETF time series for 'dogecoin'" in mt.get_etf_flows.invoke({"asset": "dogecoin"})


def test_etf_flows_tool_merges_onchain_column(tools_c, monkeypatch):
    onchain = pd.DataFrame({"day": pd.to_datetime(["2026-09-08", "2026-09-09"], utc=True), "net_flow_usd": [5e6, -7e6]})
    monkeypatch.setattr(mt, "_onchain_net_flows", lambda days: onchain)
    out = mt.get_etf_flows.invoke({"asset": "bitcoin", "days": 10})
    assert "| On-chain net (CM) |" in out and "| 2026-09-09 | $-99.02M | $-27.35M | $117.44B | $-109.13M | $7.13M | $262.7K | $-7.00M |" in out
    assert "will not match exactly" in out


def test_intel_and_signal_tools(tools_c):
    out = mt.get_intel_events.invoke({"tokens": ["eth", "btc", "zzz"], "days": 30})
    assert out.startswith("### Intel events: ETH, BTC, ZZZ, since")
    assert "Messari does not know ticker(s) ZZZ." in out
    assert "- 2026-09-11 [Low / Proposed] **EURCV on Aave V4** - Protocol Management / New Supported Asset or Market - AAVE (also ETH): Proposal to onboard **EURCV** [link](https://x.com/a). - https://x.com/a" in out
    assert "No Intel events" in mt.get_intel_events.invoke({"tokens": ["eth"], "days": 30, "importance": "high"})
    assert mt.get_intel_events.invoke({"tokens": []}) == "Give at least one ticker."

    out = mt.get_social_signals.invoke({"tokens": ["btc", "eth"]})
    assert "| BTC | 4.58% | -18.1% | 5.62% | -3.0% | 6.93% | +0.17 | +0.28 | 450 | 403 |" in out
    assert "| ETH | 2.34% | +5.0% | 3.00% | +1.0% | 3.50% | n/a | n/a | n/a | n/a |" in out
    assert "**BTC** (Messari 24h read): Bitcoin's mindshare is declining" in out and "ETH** (Messari" not in out


def test_all_messari_tools_registered():
    from chat import default_tools
    names = [t.name for t in default_tools()]
    assert set(mt.MESSARI_TOOL_NAMES) <= set(names) and len(mt.MESSARI_TOOL_NAMES) == 8
