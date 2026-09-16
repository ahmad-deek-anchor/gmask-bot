"""TokenUniverse and dynamic token resolution against a fake Coin Metrics client. No network."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from providers.coinmetrics import CoinMetricsProvider, spot_markets_for
from providers.universe import STABLES_AND_WRAPPED, TokenUniverse, display_symbol

NOW = datetime.now(timezone.utc)
DAY = NOW.strftime("%Y-%m-%d")


class _Result:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


class FakeClient:
    """Reference data, a market-cap catalog, asset metrics and a spot-market catalog."""

    def __init__(self):
        self.calls = []
        self.caps = {"btc": 1.5e12, "eth": 3e11, "usdt": 1.8e11, "wbtc": 9e9, "tao_bittensor": 3e9,
                     "sky_sky": 1.2e9, "sky": 5e6, "ondo": 1e9, "zzz_dead": float("nan")}
        self.vols = {"btc": 2e10, "eth": 1e10, "usdt": 5e10, "wbtc": 1e8, "tao_bittensor": 2e8,
                     "sky_sky": 3e7, "sky": 1e5, "ondo": 4e7, "zzz_dead": float("nan")}

    def reference_data_assets(self, **kw):
        self.calls.append("reference_data_assets")
        return _Result(pd.DataFrame({"asset": list(self.caps) + ["fartcoin"], "full_name": "x"}))

    def catalog_asset_metrics_v2(self, metrics, **kw):
        self.calls.append("catalog_asset_metrics_v2")
        return _Result(pd.DataFrame({"asset": list(self.caps), "metric": metrics[0], "frequency": "1d"}))

    def get_asset_metrics(self, assets, metrics, **kw):
        self.calls.append(("get_asset_metrics", tuple(assets)))
        rows = []
        for a in assets:
            for d in (2, 1):
                rows.append({"asset": a, "time": NOW - timedelta(days=d),
                             "CapMrktEstUSD": self.caps[a] * (0.9 if d == 2 else 1.0),
                             "volume_trusted_spot_usd_1d": self.vols[a]})
        return _Result(pd.DataFrame(rows))

    def catalog_market_candles_v2(self, asset, market_type, **kw):
        self.calls.append(("catalog_market_candles_v2", asset))
        recent, stale = NOW - timedelta(days=1), NOW - timedelta(days=400)
        rows = [
            ("binance-tao_bittensor-usdt-spot", recent), ("kraken-tao_bittensor-usd-spot", recent),
            ("coinbase-tao_bittensor-usd-spot", recent), ("mexc-tao_bittensor-btc-spot", recent),
            ("bitfinex-tao_bittensor-usd-spot", stale),
        ] if asset == "tao_bittensor" else [("lbank-ondo-usdt-spot", recent)] if asset == "ondo" else []
        return _Result(pd.DataFrame({"market": [r[0] for r in rows], "frequency": "1d",
                                     "min_time": NOW - timedelta(days=700), "max_time": [r[1] for r in rows]}))

    def get_market_candles(self, markets, **kw):
        (m,) = markets
        self.calls.append(("candles", m))
        if "tao_bittensor" not in m:
            raise RuntimeError("400 market not supported")
        days = pd.date_range(end=NOW.date(), periods=5, freq="D", tz="UTC")
        return _Result(pd.DataFrame({"market": m, "time": days, "price_open": 300.0, "price_high": 310.0,
                                     "price_low": 290.0, "price_close": 305.0, "vwap": 300.0, "volume": 10.0,
                                     "candle_usd_volume": 3000.0, "candle_trades_count": 5}))


@pytest.fixture
def universe(tmp_path):
    return TokenUniverse(FakeClient(), cache_path=tmp_path / "universe_cache.json", ttl_s=3600)


def test_ranking_uses_latest_day_and_sorts_by_cap(universe):
    df = universe.ranking()
    assert df["asset"].tolist()[:3] == ["btc", "eth", "usdt"]
    assert df.set_index("asset").loc["btc", "market_cap"] == 1.5e12       # latest day, not the 0.9x older one
    assert "zzz_dead" not in df["asset"].tolist()                          # NaN cap dropped
    assert universe._client.calls.count("catalog_asset_metrics_v2") == 1


def test_top_assets_excludes_stables_and_wrapped_by_default(universe):
    top = universe.top_assets(n=6)
    assert top["asset"].tolist() == ["btc", "eth", "tao_bittensor", "sky_sky", "ondo"]   # 'sky' (Skycoin) collapsed into 'sky'
    assert top["symbol"].tolist() == ["btc", "eth", "tao", "sky", "ondo"]
    assert top["rank"].tolist() == [1, 2, 3, 4, 5]
    with_all = universe.top_assets(n=5, exclude_stables_and_wrapped=False)
    assert "usdt" in with_all["asset"].tolist()
    assert {"usdt", "wbtc", "steth", "bnb_bsc"} <= STABLES_AND_WRAPPED


def test_top_assets_by_volume(universe):
    top = universe.top_assets(n=3, by="spot_volume", exclude_stables_and_wrapped=False)
    assert top["asset"].tolist() == ["usdt", "btc", "eth"]
    with pytest.raises(ValueError):
        universe.top_assets(by="price")


def test_ranking_is_cached_in_memory_and_on_disk(universe, tmp_path):
    universe.ranking(); universe.ranking()
    assert universe._client.calls.count("catalog_asset_metrics_v2") == 1
    fresh = TokenUniverse(FakeClient(), cache_path=tmp_path / "universe_cache.json", ttl_s=3600)
    assert fresh._ranking is not None and len(fresh._ranking) == len(universe.ranking())
    assert "catalog_asset_metrics_v2" not in fresh._client.calls   # served from the file


def test_resolve_exact_qualified_and_unknown(universe):
    assert universe.resolve("btc") == "btc"
    assert universe.resolve("TAO") == "tao_bittensor"       # qualified id, case-insensitive
    assert universe.resolve("sky") == "sky_sky"             # largest market cap among sky / sky_sky
    assert universe.resolve("fartcoin") == "fartcoin"       # known asset with no market cap still resolves
    assert universe.resolve("notacoin") is None
    assert universe.resolve("") is None
    assert universe.is_known("ondo") and not universe.is_known("notacoin")


def test_spot_markets_prefers_known_exchanges_and_drops_stale(universe):
    markets = universe.spot_markets("tao_bittensor")
    assert markets == ["coinbase-tao_bittensor-usd-spot", "kraken-tao_bittensor-usd-spot", "binance-tao_bittensor-usdt-spot"]
    assert "bitfinex" not in " ".join(markets)                # stale market excluded
    assert universe.spot_markets("ondo") == ["lbank-ondo-usdt-spot"]   # fallback: any live market
    assert universe.spot_markets("nomarkets") == []
    universe.spot_markets("tao_bittensor")
    assert sum(1 for c in universe._client.calls if c == ("catalog_market_candles_v2", "tao_bittensor")) == 1


def test_display_symbol():
    assert display_symbol("tao_bittensor") == "tao" and display_symbol("btc") == "btc"


def test_provider_uses_curated_markets_for_curated_tokens_and_universe_for_others():
    client = FakeClient()
    prov = CoinMetricsProvider(client)
    assert prov._spot_markets("btc") == spot_markets_for("btc")           # curated: no catalog call
    assert ("catalog_market_candles_v2", "btc") not in client.calls
    assert prov._spot_markets("tao")[0] == "coinbase-tao_bittensor-usd-spot"
    assert prov._asset_id("tao") == "tao_bittensor" and prov._asset_id("sky") == "sky_sky"
    df = prov.get_spot_ohlcv("tao", NOW - timedelta(days=5), NOW)
    assert df is not None and float(df["close"].iloc[-1]) == 305.0


def test_validate_tokens_accepts_dynamic_symbols_when_a_universe_exists(monkeypatch):
    from providers import factory
    from tools import chat_tools
    prov = CoinMetricsProvider(FakeClient())
    monkeypatch.setattr(factory, "get_provider", lambda: prov)
    valid, unknown, _ = chat_tools.validate_tokens("btc tao notacoin")
    assert valid == ["btc", "tao"] and unknown == ["notacoin"]


def test_validate_tokens_falls_back_to_curated_without_a_universe(monkeypatch):
    from providers import factory
    from tools import chat_tools
    monkeypatch.setattr(factory, "get_provider", lambda: object())   # no .universe attribute
    valid, unknown, _ = chat_tools.validate_tokens("btc tao")
    assert valid == ["btc"] and unknown == ["tao"]


def test_list_top_assets_tool(monkeypatch):
    from providers import factory
    from tools import chat_tools
    prov = CoinMetricsProvider(FakeClient())
    monkeypatch.setattr(factory, "get_provider", lambda: prov)
    out = chat_tools.list_top_assets.invoke({"n": 3})
    assert "Top 3 assets by estimated market cap" in out
    assert "| 1 | btc |" in out and "| 3 | tao |" in out and "usdt" not in out
    out_all = chat_tools.list_top_assets.invoke({"n": 3, "include_stables_and_wrapped": True})
    assert "usdt" in out_all
