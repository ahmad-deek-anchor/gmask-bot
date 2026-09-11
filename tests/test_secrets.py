"""Tests for utils.secrets (no network: Secret Manager client is faked)."""

import sys
import types

import pytest

from utils import secrets


class _FakePayload:
    def __init__(self, data: bytes):
        self.data = data


class _FakeResponse:
    def __init__(self, data: bytes):
        self.payload = _FakePayload(data)


class FakeSecretManagerClient:
    """Stand-in for google.cloud.secretmanager.SecretManagerServiceClient."""

    values: dict[str, str] = {}
    calls: list[str] = []
    instances = 0

    def __init__(self):
        type(self).instances += 1

    def access_secret_version(self, request):
        name = request["name"]
        type(self).calls.append(name)
        if name not in self.values:
            raise RuntimeError(f"404 Secret Version [{name}] not found")
        return _FakeResponse(self.values[name].encode("utf-8"))


@pytest.fixture
def fake_sm(monkeypatch):
    """Install a fake google.cloud.secretmanager module and reset caches."""
    FakeSecretManagerClient.values = {}
    FakeSecretManagerClient.calls = []
    FakeSecretManagerClient.instances = 0

    fake_mod = types.ModuleType("google.cloud.secretmanager")
    fake_mod.SecretManagerServiceClient = FakeSecretManagerClient

    google_pkg = sys.modules.get("google") or types.ModuleType("google")
    cloud_pkg = sys.modules.get("google.cloud") or types.ModuleType("google.cloud")
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud.secretmanager", fake_mod)
    monkeypatch.setattr(cloud_pkg, "secretmanager", fake_mod, raising=False)

    secrets.clear_cache()
    yield FakeSecretManagerClient
    secrets.clear_cache()


def test_env_override_wins_default_name(monkeypatch, fake_sm):
    fake_sm.values["projects/p/secrets/my_key/versions/latest"] = "from-sm"
    monkeypatch.setenv("MY_KEY", "from-env")
    assert secrets.get_secret("my_key", project="p") == "from-env"
    assert fake_sm.calls == []  # never touched Secret Manager


def test_env_override_custom_var(monkeypatch, fake_sm):
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_VAR", "custom-env")
    assert secrets.get_secret("my_key", project="p", env_var="CUSTOM_VAR") == "custom-env"
    assert fake_sm.calls == []


def test_empty_env_falls_through(monkeypatch, fake_sm):
    monkeypatch.setenv("MY_KEY", "")
    fake_sm.values["projects/p/secrets/my_key/versions/latest"] = "from-sm"
    assert secrets.get_secret("my_key", project="p") == "from-sm"


def test_secret_manager_path_and_resource_name(monkeypatch, fake_sm):
    monkeypatch.delenv("COINMETRICS_TRIAL_API", raising=False)
    resource = "projects/anchorage-trading-solutions/secrets/coinmetrics_trial_api/versions/latest"
    fake_sm.values[resource] = "sm-value"
    assert secrets.get_secret("coinmetrics_trial_api") == "sm-value"
    assert fake_sm.calls == [resource]


def test_failure_returns_none_and_warns(monkeypatch, fake_sm, caplog):
    monkeypatch.delenv("MISSING", raising=False)
    with caplog.at_level("WARNING", logger="utils.secrets"):
        assert secrets.get_secret("missing", project="p") is None
    assert any("missing" in rec.getMessage() for rec in caplog.records)


def test_client_constructor_failure_returns_none(monkeypatch, fake_sm):
    monkeypatch.delenv("X", raising=False)

    class Boom:
        def __init__(self):
            raise PermissionError("no ADC")

    sys.modules["google.cloud.secretmanager"].SecretManagerServiceClient = Boom
    assert secrets.get_secret("x", project="p") is None


def test_caching(monkeypatch, fake_sm):
    monkeypatch.delenv("CACHED", raising=False)
    resource = "projects/p/secrets/cached/versions/latest"
    fake_sm.values[resource] = "v1"
    assert secrets.get_secret("cached", project="p") == "v1"
    fake_sm.values[resource] = "v2"
    assert secrets.get_secret("cached", project="p") == "v1"  # cached
    assert len(fake_sm.calls) == 1
    secrets.clear_cache()
    assert secrets.get_secret("cached", project="p") == "v2"
    assert len(fake_sm.calls) == 2


def test_failure_is_cached_too(monkeypatch, fake_sm):
    monkeypatch.delenv("NOPE", raising=False)
    assert secrets.get_secret("nope", project="p") is None
    assert secrets.get_secret("nope", project="p") is None
    assert len(fake_sm.calls) == 1


def test_cache_is_per_project(monkeypatch, fake_sm):
    monkeypatch.delenv("K", raising=False)
    fake_sm.values["projects/a/secrets/k/versions/latest"] = "a-val"
    fake_sm.values["projects/b/secrets/k/versions/latest"] = "b-val"
    assert secrets.get_secret("k", project="a") == "a-val"
    assert secrets.get_secret("k", project="b") == "b-val"
