"""CompositeProvider routing and factory wiring, with fake providers. No network."""

from datetime import datetime

import pandas as pd
import pytest

from providers.base import MarketDataProvider
from providers.composite import CompositeProvider

START, END = datetime(2026, 8, 1), datetime(2026, 9, 10)

SPOT_METHODS = ["get_spot_ohlcv", "get_spot_price"]
DERIV_METHODS = ["get_funding_rate", "get_perp_oi", "get_perp_volume", "get_liquidations"]


class RecordingProvider(MarketDataProvider):
    """Records every call and returns a tagged one-row frame."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def _hit(self, method, token, start, end):
        self.calls.append((method, token, start, end))
        return pd.DataFrame({"time": [pd.Timestamp("2026-09-10")], "source": [self.name]})

    def get_spot_ohlcv(self, token, start_date, end_date):
        return self._hit("get_spot_ohlcv", token, start_date, end_date)

    def get_spot_price(self, token, start_date, end_date):
        return self._hit("get_spot_price", token, start_date, end_date)

    def get_funding_rate(self, token, start_date, end_date):
        return self._hit("get_funding_rate", token, start_date, end_date)

    def get_perp_oi(self, token, start_date, end_date):
        return self._hit("get_perp_oi", token, start_date, end_date)

    def get_perp_volume(self, token, start_date, end_date):
        return self._hit("get_perp_volume", token, start_date, end_date)

    def get_liquidations(self, token, start_date, end_date):
        return self._hit("get_liquidations", token, start_date, end_date)


@pytest.fixture
def composite():
    spot, deriv = RecordingProvider("spot"), RecordingProvider("deriv")
    return CompositeProvider(spot=spot, derivatives=deriv), spot, deriv


@pytest.mark.parametrize("method", SPOT_METHODS)
def test_spot_methods_route_to_spot_provider(composite, method):
    comp, spot, deriv = composite
    df = getattr(comp, method)("btc", START, END)
    assert df["source"].iloc[0] == "spot"
    assert spot.calls == [(method, "btc", START, END)]
    assert deriv.calls == []


@pytest.mark.parametrize("method", DERIV_METHODS)
def test_derivative_methods_route_to_derivatives_provider(composite, method):
    comp, spot, deriv = composite
    df = getattr(comp, method)("eth", START, END)
    assert df["source"].iloc[0] == "deriv"
    assert deriv.calls == [(method, "eth", START, END)]
    assert spot.calls == []


def test_composite_passes_through_none(composite):
    comp, spot, deriv = composite
    deriv.get_perp_oi = lambda token, s, e: None
    assert comp.get_perp_oi("btc", START, END) is None


def test_composite_is_a_market_data_provider(composite):
    comp, _, _ = composite
    assert isinstance(comp, MarketDataProvider)
    assert comp.spot is not comp.derivatives


def test_factory_builds_composite_without_cache(monkeypatch):
    """get_provider() wires CoinMetrics(spot) + Amberdata(derivs) into a CompositeProvider,
    memoises it, and reset_provider() drops it. All heavy constructors are faked."""
    import sys
    import types

    import providers.factory as factory
    from providers.coinmetrics import CoinMetricsProvider

    monkeypatch.setenv("COINMETRICS_API_KEY", "cm-key")
    monkeypatch.setenv("AMBERDATA_API_KEY", "ad-key")

    class FakeCMClient:
        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key

    class FakeAmberdata(RecordingProvider):
        def __init__(self, api_key, session=None, timeout=30):
            super().__init__("amberdata")
            self.api_key = api_key

    fake_cm_mod = types.ModuleType("coinmetrics.api_client")
    fake_cm_mod.CoinMetricsClient = FakeCMClient
    monkeypatch.setitem(sys.modules, "coinmetrics.api_client", fake_cm_mod)

    fake_ad_mod = types.ModuleType("providers.amberdata")
    fake_ad_mod.AmberdataProvider = FakeAmberdata
    monkeypatch.setitem(sys.modules, "providers.amberdata", fake_ad_mod)

    factory.reset_provider()
    try:
        p = factory.get_provider()
        assert isinstance(p, CompositeProvider)
        assert isinstance(p.spot, CoinMetricsProvider)
        assert p.spot._client.api_key == "cm-key"
        assert isinstance(p.derivatives, FakeAmberdata)
        assert p.derivatives.api_key == "ad-key"
        assert factory.get_provider() is p  # memoised
        assert "CachedProvider" not in type(p).__name__

        factory.reset_provider()
        assert factory.get_provider() is not p
    finally:
        factory.reset_provider()


def test_config_env_override_short_circuits_secret_manager(monkeypatch):
    import sys
    import types

    from utils.config import Config

    called = []
    fake_secrets = types.ModuleType("utils.secrets")
    fake_secrets.get_secret = lambda *a, **k: called.append((a, k)) or "from-secret"
    monkeypatch.setitem(sys.modules, "utils.secrets", fake_secrets)

    monkeypatch.setenv("COINMETRICS_API_KEY", "env-key")
    monkeypatch.delenv("AMBERDATA_API_KEY", raising=False)
    monkeypatch.setenv("VERTEX_MODEL", "claude-test")

    cfg = Config()
    assert cfg.COINMETRICS_API_KEY == "env-key"
    assert called == []
    assert cfg.AMBERDATA_API_KEY == "from-secret"
    assert called[0][0][0] == "amberdata_key"
    assert called[0][1]["env_var"] == "AMBERDATA_API_KEY"
    assert cfg.VERTEX_MODEL == "claude-test"
    assert cfg.VERTEX_PROJECT == "anchorage-ai-development"
    assert cfg.GCP_SECRETS_PROJECT == "anchorage-trading-solutions"
