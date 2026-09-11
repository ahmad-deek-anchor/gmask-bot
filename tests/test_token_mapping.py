"""Guard: the token universes are well-formed and every token resolves to a
Coin Metrics asset id and at least one spot market."""

from providers.coinmetrics import ASSET_MAP, asset_id_for, spot_markets_for
from tools.metrics import FULL_TOKEN_UNIVERSE, TEST_TOKEN_UNIVERSE


def test_universes_are_lowercase_and_unique():
    for universe in (FULL_TOKEN_UNIVERSE, TEST_TOKEN_UNIVERSE):
        assert universe == [t.lower() for t in universe]
        assert len(set(universe)) == len(universe)


def test_test_universe_is_subset_of_full_universe():
    missing = [t for t in TEST_TOKEN_UNIVERSE if t not in FULL_TOKEN_UNIVERSE]
    assert missing == [], f"test-universe tokens not in full universe: {missing}"


def test_every_token_has_a_coinmetrics_asset_id_and_markets():
    for token in FULL_TOKEN_UNIVERSE:
        asset_id = asset_id_for(token)
        assert asset_id and asset_id == asset_id.lower()
        markets = spot_markets_for(token)
        assert markets, f"no spot markets for {token}"
        assert all(m.startswith(("coinbase-", "binance-", "kraken-", "bybit-", "okex-")) for m in markets)
        assert all(m.endswith("-spot") and f"-{asset_id}-" in m for m in markets)


def test_asset_map_overrides_apply():
    assert asset_id_for("pol") == "pol"  # Coin Metrics re-listed Polygon as "pol" (Sept 2026)
    assert asset_id_for("SKY") == "sky_sky"  # "sky" on Coin Metrics is Skycoin
    assert spot_markets_for("sky")[0] == "coinbase-sky_sky-usd-spot"
    assert asset_id_for("btc") == "btc"
    # every remapped token must still be in the universe, otherwise the override is dead
    assert all(t in FULL_TOKEN_UNIVERSE for t in ASSET_MAP)
