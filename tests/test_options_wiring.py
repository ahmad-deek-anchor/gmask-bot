"""Options wiring (tools.metrics / tools.signals / workflow / chat tools / CLI) with a fake
options provider injected via providers.factory.get_options_provider. No network, no Vertex.

The fake implements the exact interface of providers/amberdata_options.py from
OPTIONS_SPEC.md (supported_tokens, currency_for, get_dvol, get_term_structure_history,
get_skew_history, get_put_call_ratio, get_options_volume, get_term_structure,
get_delta_surface, get_gamma_exposure, get_block_trades).
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import providers.factory as factory
import tools.metrics as metrics_mod
from tools.chat_tools import (
    get_chat_tools,
    get_gamma_exposure,
    get_options_flow,
    get_options_snapshot,
    get_token_metrics,
    get_vol_term_structure,
    get_zscore_signals,
)
from tools.metrics import (
    OPTIONS_COLUMNS,
    OPTIONS_DAILY_METRICS,
    OUTLIER_THRESHOLD,
    fetch_options_snapshot,
    fetch_token_metrics,
    summarise_gamma_exposure,
)
from tools.signals import OPTIONS_LEVEL_METRICS, OPTIONS_Z_METRICS, calculate_statistical_signals
from workflows.signals_workflow import SignalsWorkflow

FETCH_END = datetime(2026, 8, 31)  # in the past: no partial "today" row to drop
DAYS = 45


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    import sys
    import types

    fake_llm = types.ModuleType("utils.llm")
    fake_llm.get_llm = lambda *a, **k: (_ for _ in ()).throw(AssertionError("get_llm() must not be called"))
    monkeypatch.setitem(sys.modules, "utils.llm", fake_llm)


# ----------------------------------------------------------------------------
# fakes
# ----------------------------------------------------------------------------

class FakeSpotPerpProvider:
    """Minimal CompositeProvider stand-in (spot + perp) with calm data."""

    def __init__(self, days=DAYS, end=FETCH_END):
        self.time = pd.date_range(end=end, periods=days, freq="D")
        self.n = days

    def get_spot_price(self, token, s, e):
        return pd.DataFrame({"time": self.time, "price": np.linspace(100, 110, self.n)})

    def get_spot_ohlcv(self, token, s, e):
        return pd.DataFrame({"time": self.time, "open": 1, "high": 1, "low": 1, "close": 1,
                             "spot_volume": np.full(self.n, 5.0)})

    def get_perp_volume(self, token, s, e):
        return pd.DataFrame({"time": self.time, "perp_volume": np.full(self.n, 7.0)})

    def get_perp_oi(self, token, s, e):
        return pd.DataFrame({"time": self.time, "perp_oi": np.full(self.n, 9.0)})

    def get_funding_rate(self, token, s, e):
        return pd.DataFrame({"time": self.time, "funding_rate": np.full(self.n, 8.0)})

    def get_liquidations(self, token, s, e):
        return pd.DataFrame({"time": self.time, "long_liquidations": 1.0, "short_liquidations": 2.0,
                             "total_liquidations": 3.0})


class FakeOptionsProvider:
    """AmberdataOptionsProvider stand-in: btc/eth listed, 45 days of synthetic data with a
    DVOL spike on the last day. `sol` is listed but has no DVOL (mirrors Deribit SOL_USDC)."""

    exchange = "deribit"

    def __init__(self, days=DAYS, end=FETCH_END, seed=0):
        self.calls = []
        self.time = pd.date_range(end=end, periods=days, freq="D")
        self.n = days
        self.rng = np.random.default_rng(seed)

    def _log(self, name, token):
        self.calls.append((name, token))

    def supported_tokens(self):
        self.calls.append(("supported_tokens", None))
        return ["btc", "eth", "sol"]

    def currency_for(self, token):
        return {"btc": "BTC", "eth": "ETH", "sol": "SOL_USDC"}.get(token.lower())

    def get_dvol(self, token, s, e):
        self._log("get_dvol", token)
        if token == "sol":
            return None
        close = 50 + self.rng.normal(0, 1.0, self.n)
        close[-1] = 80.0  # spike
        return pd.DataFrame({"time": self.time, "dvol_open": close - 0.5, "dvol_high": close + 1,
                             "dvol_low": close - 1, "dvol_close": close})

    def get_term_structure_history(self, token, s, e):
        self._log("get_term_structure_history", token)
        base = 55 + self.rng.normal(0, 0.5, self.n)
        return pd.DataFrame({"time": self.time, "atm_iv_7d": base - 3, "atm_iv_30d": base,
                             "atm_iv_60d": base + 1, "atm_iv_90d": base + 2, "atm_iv_180d": base + 3,
                             "ts_richness": np.full(self.n, 1.05)})

    def get_skew_history(self, token, s, e, tenor_days=30):
        self._log("get_skew_history", token)
        skew = np.linspace(1.0, 4.0, self.n)  # rising skew -> 7d change positive
        return pd.DataFrame({"time": self.time, "skew_25d_30d": skew, "skew_10d_30d": skew * 1.5})

    def get_put_call_ratio(self, token, s, e):
        self._log("get_put_call_ratio", token)
        return pd.DataFrame({"time": self.time, "pcr_oi": 0.6 + self.rng.normal(0, 0.01, self.n),
                             "pcr_volume_24h": np.full(self.n, 0.9)})

    def get_options_volume(self, token, s, e):
        self._log("get_options_volume", token)
        notional = 1.0e9 + self.rng.normal(0, 2e7, self.n)
        return pd.DataFrame({"time": self.time, "options_contract_volume": np.full(self.n, 12_000.0),
                             "options_notional_volume": notional, "options_premium_volume": notional * 0.02,
                             "options_block_notional_volume": notional * 0.3})

    # snapshots
    def get_term_structure(self, token, timestamp=None):
        self._log("get_term_structure", token)
        return pd.DataFrame({"days_to_expiration": [7, 30, 90], "atm_iv": [48.0, 52.0, 56.0],
                             "fwd_atm_iv": [48.0, 53.0, 58.0]})

    def get_delta_surface(self, token, timestamp=None):
        self._log("get_delta_surface", token)
        return pd.DataFrame({"days_to_expiration": [7, 30], "atm_iv": [48.0, 52.0],
                             "iv_call_10d": [50.0, 53.0], "iv_call_25d": [47.0, 51.0],
                             "iv_put_10d": [58.0, 60.0], "iv_put_25d": [51.0, 54.5],
                             "skew_25d": [4.0, 3.5], "skew_10d": [8.0, 7.0]})

    def get_gamma_exposure(self, token, date=None):
        self._log("get_gamma_exposure", token)
        df = pd.DataFrame({
            "strike": [90_000.0, 95_000.0, 100_000.0, 105_000.0, 110_000.0],
            "net_dealer_gamma": [-500.0, -200.0, 100.0, 800.0, 300.0],
            "total_dealer_gamma": [900.0, 600.0, 700.0, 1200.0, 500.0],
            "index_price": [101_000.0] * 5,
        })
        df.attrs["snapshot_time"] = "2026-08-31T00:00:00Z"
        return df

    def get_block_trades(self, token, s, e, top_n=20):
        self._log("get_block_trades", token)
        return pd.DataFrame({"expiry": ["2026-09-26", "2026-12-25"], "strike": [100_000.0, 120_000.0],
                             "put_call": ["P", "C"], "contract_volume": [250.0, 400.0],
                             "premium_volume": [2_500_000.0, 1_200_000.0]})


class NoDataOptionsProvider(FakeOptionsProvider):
    """Listed token but every endpoint empty -> 'not available' everywhere."""

    def get_dvol(self, *a, **k): return None
    def get_term_structure_history(self, *a, **k): return None
    def get_skew_history(self, *a, **k): return None
    def get_put_call_ratio(self, *a, **k): return pd.DataFrame()
    def get_options_volume(self, *a, **k): return None
    def get_term_structure(self, *a, **k): return None
    def get_delta_surface(self, *a, **k): return None
    def get_gamma_exposure(self, *a, **k): return None
    def get_block_trades(self, *a, **k): return None


@pytest.fixture
def fake_options(monkeypatch):
    provider = FakeOptionsProvider()
    factory_calls = []

    def _get_options_provider():
        factory_calls.append(1)
        return provider

    # raising=False: providers.factory may not expose get_options_provider yet
    monkeypatch.setattr(factory, "get_options_provider", _get_options_provider, raising=False)
    monkeypatch.setattr(metrics_mod, "get_provider", lambda: FakeSpotPerpProvider())
    provider.factory_calls = factory_calls
    return provider


def _fetch(tokens, include_options=True):
    return fetch_token_metrics(tokens, FETCH_END - timedelta(days=DAYS - 1), FETCH_END,
                               include_options=include_options)


# ----------------------------------------------------------------------------
# tools.metrics: merge
# ----------------------------------------------------------------------------

def test_options_columns_merge_for_listed_tokens_only(fake_options):
    out = _fetch(["btc", "uni"])
    btc, uni = out["btc"], out["uni"]

    assert set(OPTIONS_COLUMNS) <= set(btc.columns)
    assert len(btc) == DAYS
    assert btc["dvol_close"].iloc[-1] == pytest.approx(80.0)
    assert btc["dvol_close"].notna().all()
    assert btc["skew_25d_30d"].iloc[-1] == pytest.approx(4.0)
    assert (btc["options_block_notional_volume"] / btc["options_notional_volume"]).round(6).eq(0.3).all()
    for c in OPTIONS_COLUMNS:
        assert btc[c].dtype == np.float64
    # existing columns untouched
    assert (btc["spot_volume"] == 5.0).all() and (btc["perp_oi"] == 9.0).all()
    assert btc["time"].is_monotonic_increasing

    # uni is not listed: no options columns at all
    assert not any(c in uni.columns for c in OPTIONS_COLUMNS)
    assert {"price", "spot_volume", "perp_oi", "funding_rate", "total_liquidations"} <= set(uni.columns)

    # provider was only asked for the listed token
    assert ("get_dvol", "btc") in fake_options.calls
    assert not any(tok == "uni" for _, tok in fake_options.calls)
    assert fake_options.factory_calls == [1]  # built once per fetch


def test_listed_token_with_missing_series_gets_nan_columns(fake_options):
    """sol is listed but has no DVOL: column present, all NaN; other options data present."""
    sol = _fetch(["sol"])["sol"]
    assert "dvol_close" in sol.columns and sol["dvol_close"].isna().all()
    assert sol["atm_iv_30d"].notna().all()
    assert sol["pcr_oi"].notna().all()


def test_failing_options_endpoint_does_not_sink_token(fake_options, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(fake_options, "get_put_call_ratio", boom)
    monkeypatch.setattr(fake_options, "supported_tokens", lambda: ["btc"])
    btc = _fetch(["btc"])["btc"]
    assert btc["pcr_oi"].isna().all() and btc["pcr_volume_24h"].isna().all()
    assert btc["dvol_close"].notna().all()


def test_include_options_false_never_touches_options_provider(fake_options):
    out = _fetch(["btc"], include_options=False)
    assert not any(c in out["btc"].columns for c in OPTIONS_COLUMNS)
    assert fake_options.calls == []
    assert fake_options.factory_calls == []


def test_missing_factory_function_means_no_options(monkeypatch):
    monkeypatch.delattr(factory, "get_options_provider", raising=False)
    monkeypatch.setattr(metrics_mod, "get_provider", lambda: FakeSpotPerpProvider())
    assert metrics_mod.get_options_provider() is None
    out = _fetch(["btc"])
    assert "btc" in out and not any(c in out["btc"].columns for c in OPTIONS_COLUMNS)


def test_factory_returning_none_or_raising_is_graceful(monkeypatch):
    monkeypatch.setattr(metrics_mod, "get_provider", lambda: FakeSpotPerpProvider())
    monkeypatch.setattr(factory, "get_options_provider", lambda: None, raising=False)
    assert "dvol_close" not in _fetch(["btc"])["btc"].columns

    def boom():
        raise RuntimeError("no key")

    monkeypatch.setattr(factory, "get_options_provider", boom, raising=False)
    assert metrics_mod.get_options_provider() is None
    assert "dvol_close" not in _fetch(["btc"])["btc"].columns


# ----------------------------------------------------------------------------
# tools.signals: z-scores + levels
# ----------------------------------------------------------------------------

def test_options_zscores_and_levels(fake_options):
    data = _fetch(["btc", "uni"])
    signals = calculate_statistical_signals(data, window=30)
    btc, uni = signals["btc"], signals["uni"]

    assert btc["options_listed"] is True
    dvol = btc["metrics"]["dvol_close"]
    assert set(dvol) == {"value", "z_score", "is_outlier", "is_significant"}
    assert dvol["value"] == pytest.approx(80.0)
    assert dvol["z_score"] >= OUTLIER_THRESHOLD and dvol["is_outlier"] is True
    assert btc["has_outliers"] is True and btc["has_significant_moves"] is True
    for m in OPTIONS_Z_METRICS:
        assert m in btc["metrics"], m
        assert btc["metrics"][m]["z_score"] is not None
    # existing keys intact
    for m in ("spot_volume", "perp_volume", "perp_oi", "total_liquidations"):
        assert m in btc["metrics"]
    assert btc["metrics"]["funding_rate"]["value_annual_pct"] == pytest.approx(8.0)
    assert "price" in btc["metrics"]
    # level metrics: value + 7d change, no z
    skew = btc["metrics"]["skew_25d_30d"]
    assert set(skew) == {"value", "change_7d"}
    assert skew["value"] == pytest.approx(4.0)
    assert skew["change_7d"] == pytest.approx(7 * 3.0 / (DAYS - 1))
    assert btc["metrics"]["pcr_volume_24h"]["value"] == pytest.approx(0.9)
    assert btc["metrics"]["pcr_volume_24h"]["change_7d"] == pytest.approx(0.0)

    assert uni["options_listed"] is False
    assert not any(m in uni["metrics"] for m in OPTIONS_Z_METRICS + OPTIONS_LEVEL_METRICS)
    assert uni["has_outliers"] is False


def test_sol_without_dvol_has_other_options_metrics(fake_options):
    sig = calculate_statistical_signals(_fetch(["sol"]), window=30)["sol"]
    assert sig["options_listed"] is True
    assert "dvol_close" not in sig["metrics"]
    assert "atm_iv_30d" in sig["metrics"] and "pcr_oi" in sig["metrics"]


def test_multi_day_tool_tracks_options_z(fake_options, monkeypatch):
    import json

    import tools.signals as sig_mod
    from tools.signals import get_multi_day_signals_tool

    data = _fetch(["btc", "uni"])
    monkeypatch.setattr(sig_mod, "fetch_token_metrics", lambda *a, **k: data)
    out = json.loads(get_multi_day_signals_tool.invoke({"tokens": ["btc", "uni"], "days_to_analyze": 3}))
    btc_days, uni_days = out["daily_signals"]["btc"], out["daily_signals"]["uni"]
    assert len(btc_days) == 3
    assert "dvol_close_z" in btc_days[-1] and btc_days[-1]["dvol_close_z"] > OUTLIER_THRESHOLD
    assert "perp_oi_z" in btc_days[-1]
    assert "dvol_close_z" not in uni_days[-1]


# ----------------------------------------------------------------------------
# workflow: LLM prompt block
# ----------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)

        class R:
            content = "write-up"
        return R()


def _prompt_for(data, token):
    wf = SignalsWorkflow(llm=_FakeLLM())
    signals = calculate_statistical_signals(data, window=30)
    return wf._format_data_for_llm({"analysis_date": "2026-08-31", "outlier_threshold": OUTLIER_THRESHOLD,
                                    "tokens_with_signals": {token: signals[token]}})


def test_llm_prompt_options_block(fake_options):
    data = _fetch(["btc", "uni", "sol"])

    btc = _prompt_for(data, "btc")
    assert "**Options (Deribit)**" in btc
    dvol_line = next(l for l in btc.splitlines() if l.startswith("- dvol_close"))
    assert "**OUTLIER**" in dvol_line and "80.0 vol pts" in dvol_line and "z=" in dvol_line
    skew_line = next(l for l in btc.splitlines() if l.startswith("- skew_25d_30d"))
    assert "+4.00 vol pts" in skew_line and "puts richer" in skew_line and "z=" not in skew_line
    assert "change vs 7d ago: +0.48" in skew_line  # 7 * 3/44
    notional_line = next(l for l in btc.splitlines() if l.startswith("- options_notional_volume"))
    assert "(value: $" in notional_line and notional_line.rstrip(")").endswith(("M", "B"))  # USD formatting
    pcr_line = next(l for l in btc.splitlines() if l.startswith("- pcr_oi"))
    assert "z=" in pcr_line
    assert "nan" not in btc.lower()
    # the core block is unchanged (fake perp_oi is constant -> no z, but the line is there)
    assert "- perp_oi:" in btc and "- spot_volume:" in btc and "funding_rate: 8.00% annualized" in btc
    assert btc.index("**Options (Deribit)**") > btc.index("- funding_rate:")

    uni = _prompt_for(data, "uni")
    assert "options: not listed on Deribit" in uni
    assert "dvol_close" not in uni

    sol = _prompt_for(data, "sol")
    assert "- dvol_close: not available" in sol
    assert "- atm_iv_30d (ATM IV 30d): z=" in sol


def test_llm_prompt_all_options_missing_for_listed_token(monkeypatch):
    monkeypatch.setattr(factory, "get_options_provider", lambda: NoDataOptionsProvider(), raising=False)
    monkeypatch.setattr(metrics_mod, "get_provider", lambda: FakeSpotPerpProvider())
    data = _fetch(["btc"])
    prompt = _prompt_for(data, "btc")
    assert "not listed" not in prompt
    for m in OPTIONS_DAILY_METRICS:
        assert f"- {m}: not available" in prompt, m


def test_workflow_stream_passes_include_options(fake_options):
    events = list(SignalsWorkflow(llm=_FakeLLM()).analyze_stream(tokens=["btc"], lookback_days=DAYS,
                                                                  include_options=False))
    assert fake_options.calls == []
    final = events[-1]
    assert final.all_signals["btc"]["options_listed"] is False


def test_detailed_stats_prints_options(fake_options, capsys):
    wf = SignalsWorkflow(llm=_FakeLLM())
    signals = calculate_statistical_signals(_fetch(["btc", "uni"]), window=30)
    wf._print_detailed_stats(signals)
    out = capsys.readouterr().out
    assert "dvol_close" in out and "OUTLIER" in out
    assert "not listed on Deribit" in out


# ----------------------------------------------------------------------------
# run_signals CLI: --no-options
# ----------------------------------------------------------------------------

def test_cli_no_options_flag(monkeypatch, fake_options):
    import run_signals
    from agents.events import FinalEvent

    assert run_signals.parse_args(["--no-options"]).no_options is True
    assert run_signals.parse_args([]).no_options is False

    seen = {}

    class _WF(SignalsWorkflow):
        def __init__(self, *a, **k): ...

        def analyze_stream(self, **kw):
            seen.update(kw)
            yield FinalEvent(all_signals={}, summary={"tokens_analyzed": 0, "tokens_with_outliers": [],
                                                      "tokens_with_significant_moves": []})

    monkeypatch.setattr(run_signals, "SignalsWorkflow", _WF)
    assert run_signals.main(["--tokens", "btc", "--no-options"]) == 0
    assert seen["include_options"] is False
    assert run_signals.main(["--tokens", "btc"]) == 0
    assert seen["include_options"] is True
    assert fake_options.calls == []  # the stub workflow never fetched


# ----------------------------------------------------------------------------
# tools.metrics.fetch_options_snapshot + gamma summary
# ----------------------------------------------------------------------------

def test_summarise_gamma_exposure_top_strikes_and_flip():
    g = summarise_gamma_exposure(FakeOptionsProvider().get_gamma_exposure("btc"), top_n=2)
    assert g["index_price"] == 101_000.0
    assert g["snapshot_time"] == "2026-08-31T00:00:00Z"
    assert g["n_strikes"] == 5
    assert [s["strike"] for s in g["top_strikes"]] == [90_000.0, 95_000.0, 105_000.0, 110_000.0]
    # sign change between 95k (-200) and 100k (+100): 95k + 200/300 * 5k
    assert g["flip_point"] == pytest.approx(95_000 + 5_000 * 200 / 300)
    assert g["total_net_gamma"] == pytest.approx(500.0)
    assert summarise_gamma_exposure(None) is None
    assert summarise_gamma_exposure(pd.DataFrame()) is None
    same_sign = pd.DataFrame({"strike": [1.0, 2.0], "net_dealer_gamma": [1.0, 2.0]})
    assert summarise_gamma_exposure(same_sign)["flip_point"] is None


def test_fetch_options_snapshot_shape(fake_options):
    snap = fetch_options_snapshot("BTC")
    assert snap["token"] == "btc" and snap["listed"] is True and snap["currency"] == "BTC"
    assert snap["exchange"] == "deribit"
    assert [r["days_to_expiration"] for r in snap["term_structure"]] == [7, 30, 90]
    assert snap["delta_surface"][0]["skew_25d"] == 4.0
    assert snap["dvol"]["value"] == pytest.approx(80.0) and snap["dvol"]["change_7d"] is not None
    assert snap["pcr"]["pcr_oi"] is not None and snap["pcr"]["pcr_volume_24h"] == pytest.approx(0.9)
    assert snap["gamma"]["flip_point"] is not None
    assert len(snap["block_trades"]) == 2 and snap["block_trades"][0]["put_call"] == "P"
    assert snap["errors"] == []

    uni = fetch_options_snapshot("uni")
    assert uni["listed"] is False and uni["term_structure"] is None and uni["gamma"] is None


def test_fetch_options_snapshot_without_provider(monkeypatch):
    monkeypatch.setattr(factory, "get_options_provider", lambda: None, raising=False)
    snap = fetch_options_snapshot("btc")
    assert snap["listed"] is False and "no options provider configured" in snap["errors"]


def test_fetch_options_snapshot_endpoint_errors_are_collected(fake_options, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("HTTP 429")

    monkeypatch.setattr(fake_options, "get_gamma_exposure", boom)
    snap = fetch_options_snapshot("eth")
    assert snap["listed"] is True and snap["gamma"] is None
    assert any("gamma_exposure" in e and "HTTP 429" in e for e in snap["errors"])
    assert snap["term_structure"]  # other sections still filled


# ----------------------------------------------------------------------------
# chat tools
# ----------------------------------------------------------------------------

def test_chat_tools_include_options_tools():
    names = [t.name for t in get_chat_tools()]
    for n in ("get_options_snapshot", "get_vol_term_structure", "get_options_flow", "get_gamma_exposure"):
        assert n in names
    assert all(t.description for t in get_chat_tools())


def test_get_options_snapshot_tool(fake_options):
    out = get_options_snapshot.invoke({"token": "btc"})
    lines = out.splitlines()
    assert lines[0].startswith("### BTC options snapshot (deribit")
    assert "Underlying: BTC" in out
    assert "- DVOL close: 80.0 vol pts" in out
    assert "- put/call OI ratio: 0." in out and "put/call 24h volume ratio: 0.90" in out
    assert "**ATM IV term structure**" in out and "| 30 | 52.0 vol pts | 53.0 vol pts |" in out
    assert "contango" in out
    assert "**Skew by tenor (delta surface)**" in out and "| +4.00 vol pts | +8.00 vol pts |" in out
    assert "**Dealer gamma exposure**" in out and "gamma flip" in out and "$98,333" in out
    assert "**Top block trades" in out and "| 2026-09-26 | $100,000 | P | 250.0 | $2.50M |" in out
    assert "**Notes**" not in out


def test_get_options_snapshot_unsupported_token_and_no_provider(fake_options, monkeypatch):
    out = get_options_snapshot.invoke({"token": "uni"})
    assert "UNI has no listed options on deribit" in out and "BTC, ETH, SOL" in out
    assert not any(tok == "uni" for _, tok in fake_options.calls)

    assert "Unknown token 'doge'" in get_options_snapshot.invoke({"token": "doge"})

    monkeypatch.setattr(factory, "get_options_provider", lambda: None, raising=False)
    assert "not configured" in get_options_snapshot.invoke({"token": "btc"})


def test_get_options_snapshot_all_sections_missing(monkeypatch):
    monkeypatch.setattr(factory, "get_options_provider", lambda: NoDataOptionsProvider(), raising=False)
    out = get_options_snapshot.invoke({"token": "eth"})
    assert out.startswith("### ETH options snapshot")
    assert "- DVOL: not available" in out
    assert "- put/call ratio: not available" in out
    assert "- term structure: not available" in out
    assert "- skew by tenor: not available" in out
    assert "- gamma exposure: not available" in out
    assert "- block trades: none reported" in out
    assert "nan" not in out.lower()


def test_get_vol_term_structure_tool(fake_options):
    out = get_vol_term_structure.invoke({"token": "eth"})
    assert out.startswith("### ETH ATM IV term structure (deribit")
    assert "| days to expiry | ATM IV | fwd ATM IV |" in out
    assert "| 7 | 48.0 vol pts | 48.0 vol pts |" in out
    assert "Shape: contango" in out
    assert "**Constant-maturity ATM IV (latest daily)**" in out
    assert "- as of 2026-08-31: 7d " in out and "180d " in out and "richness 1.05" in out

    out = get_vol_term_structure.invoke({"token": "eth", "exchange": "okex"})
    assert "Requested exchange 'okex' is not configured; showing deribit" in out

    assert "no listed options" in get_vol_term_structure.invoke({"token": "jto"})


def test_get_vol_term_structure_no_data(monkeypatch):
    monkeypatch.setattr(factory, "get_options_provider", lambda: NoDataOptionsProvider(), raising=False)
    out = get_vol_term_structure.invoke({"token": "btc"})
    assert "- term structure: not available" in out and "- not available" in out


def test_get_options_flow_tool(fake_options):
    out = get_options_flow.invoke({"token": "btc", "days": 7})
    lines = out.splitlines()
    assert lines[0].startswith("### BTC options flow (deribit")
    assert f"{DAYS} days" in lines[0]  # fake ignores the range; row count reflects the frame
    assert "| date | contracts | notional | premium | block notional | P/C OI | P/C vol 24h |" in out
    rows = [l for l in lines if l.startswith("| 2026-")]
    assert len(rows) == DAYS
    assert "12,000.0" in rows[-1] and "$1." in rows[-1] and "0.90" in rows[-1]
    assert "**Period totals:** notional $" in out
    assert "**Block share of notional:** 30.0%" in out
    assert "**Latest P/C OI:** 0." in out

    assert "no listed options" in get_options_flow.invoke({"token": "uni"})
    # days is clamped to [1, 45]
    assert get_options_flow.invoke({"token": "btc", "days": 500}).startswith("### BTC options flow")


def test_get_options_flow_no_data(monkeypatch):
    monkeypatch.setattr(factory, "get_options_provider", lambda: NoDataOptionsProvider(), raising=False)
    out = get_options_flow.invoke({"token": "btc"})
    assert out.startswith("No options flow data for BTC")


def test_get_gamma_exposure_tool(fake_options):
    out = get_gamma_exposure.invoke({"token": "btc"})
    assert out.startswith("### BTC dealer gamma exposure (deribit)")
    assert "index $101,000" in out and "5 strikes" in out
    assert "gamma flip" in out and "$98,333" in out
    assert "| strike | net dealer gamma | total dealer gamma |" in out
    assert "| $105,000 | 800.00 | 1,200.00 |" in out
    assert "no listed options" in get_gamma_exposure.invoke({"token": "sui"})


def test_get_gamma_exposure_no_data(monkeypatch):
    monkeypatch.setattr(factory, "get_options_provider", lambda: NoDataOptionsProvider(), raising=False)
    assert "gamma exposure: not available" in get_gamma_exposure.invoke({"token": "btc"})


def test_options_tool_provider_error_is_reported(fake_options, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("amberdata 503")

    monkeypatch.setattr(fake_options, "get_gamma_exposure", boom)
    out = get_gamma_exposure.invoke({"token": "btc"})
    assert out.startswith("Error fetching gamma exposure for BTC") and "amberdata 503" in out


def test_get_zscore_signals_and_token_metrics_include_options(fake_options, monkeypatch):
    data = _fetch(["btc", "uni"])
    monkeypatch.setattr(metrics_mod, "fetch_token_metrics", lambda *a, **k: data)

    out = get_zscore_signals.invoke({"tokens": ["btc", "uni"], "days": 45})
    rows = [l for l in out.splitlines() if l.startswith("| BTC |") or l.startswith("| UNI |")]
    btc_metrics = [l.split("|")[2].strip() for l in rows if l.startswith("| BTC |")]
    uni_metrics = [l.split("|")[2].strip() for l in rows if l.startswith("| UNI |")]
    core = ["spot_volume", "perp_volume", "perp_oi", "total_liquidations", "funding_rate"]
    assert btc_metrics == core + list(OPTIONS_Z_METRICS)
    assert uni_metrics == core
    dvol_row = next(l for l in rows if l.startswith("| BTC | dvol_close"))
    assert "80.0 vol pts" in dvol_row and "OUTLIER" in dvol_row
    assert "**Outliers:** BTC" in out
    ctx_btc = next(l for l in out.splitlines() if l.startswith("- BTC ("))
    assert "options levels (no z): skew_25d_30d +4.00 vol pts (7d chg +0.48), pcr_volume_24h 0.90 (7d chg +0.00)" in ctx_btc
    ctx_uni = next(l for l in out.splitlines() if l.startswith("- UNI ("))
    assert "options: not listed on deribit" in ctx_uni

    out = get_token_metrics.invoke({"token": "btc", "days": 45})
    for c in OPTIONS_DAILY_METRICS:
        assert f"- {c}:" in out, c
    assert "- dvol_close: 80.0 vol pts" in out
    assert "- skew_25d_30d: +4.00 vol pts" in out
    assert "- pcr_oi: 0." in out
    header = next(l for l in out.splitlines() if l.startswith("| date |"))
    assert "dvol_close" in header and "options_notional_volume" in header

    out = get_token_metrics.invoke({"token": "uni", "days": 45})
    assert "- options: not listed on deribit" in out
    assert "dvol_close" not in out


# ----------------------------------------------------------------------------
# regressions from the live integration run (2026-09-10)
# ----------------------------------------------------------------------------

def test_no_options_run_is_labelled_not_fetched_not_unlisted(fake_options, capsys):
    """--no-options used to render BTC as 'options: not listed on Deribit' in both the LLM
    prompt and the detailed stats, telling the model a listed token had no options."""
    wf = SignalsWorkflow(llm=_FakeLLM())
    final = list(wf.analyze_stream(tokens=["btc"], lookback_days=DAYS, include_options=False))[-1]
    btc = final.all_signals["btc"]
    assert btc["options_enabled"] is False and btc["options_listed"] is False

    prompt = wf._format_data_for_llm({"analysis_date": "2026-08-31", "outlier_threshold": OUTLIER_THRESHOLD,
                                      "tokens_with_signals": {"btc": btc}})
    assert "- options: not fetched (options feed disabled for this run)" in prompt
    assert "not listed" not in prompt and "dvol_close" not in prompt

    wf._print_detailed_stats(final.all_signals)
    out = capsys.readouterr().out
    assert "not fetched (--no-options)" in out and "not listed" not in out

    # with the feed on, the flag is True and an unlisted token is still reported as unlisted
    final_on = list(wf.analyze_stream(tokens=["btc", "uni"], lookback_days=DAYS, include_options=True))[-1]
    assert final_on.all_signals["btc"]["options_enabled"] is True
    assert final_on.all_signals["uni"]["options_enabled"] is True
    uni_prompt = wf._format_data_for_llm({"analysis_date": "2026-08-31", "outlier_threshold": OUTLIER_THRESHOLD,
                                          "tokens_with_signals": {"uni": final_on.all_signals["uni"]}})
    assert "options: not listed on Deribit" in uni_prompt and "not fetched" not in uni_prompt

    # signals dicts without the flag (older callers / calculate_statistical_signals alone) still work
    legacy = dict(final_on.all_signals["uni"])
    legacy.pop("options_enabled")
    assert "options: not listed on Deribit" in "\n".join(SignalsWorkflow._format_options_block(legacy))


@pytest.mark.parametrize("value, expected", [
    (0.0, "$0.00"),            # SOL / HYPE block notional (no block trades on USDC-settled alts)
    (512.5, "$512.50"),
    (3_389.15, "$3.39K"),      # TRX daily notional
    (2.5e6, "$2.50M"),
    (1.588e9, "$1.59B"),
])
def test_options_usd_values_always_carry_dollar_sign(value, expected):
    assert SignalsWorkflow._format_options_value(value, "usd") == expected


def test_llm_prompt_zero_block_notional_shows_usd(monkeypatch):
    class ZeroBlocks(FakeOptionsProvider):
        def get_options_volume(self, token, s, e):
            df = super().get_options_volume(token, s, e)
            df["options_block_notional_volume"] = 0.0
            return df

    monkeypatch.setattr(factory, "get_options_provider", lambda: ZeroBlocks(), raising=False)
    monkeypatch.setattr(metrics_mod, "get_provider", lambda: FakeSpotPerpProvider())
    prompt = _prompt_for(_fetch(["sol"]), "sol")
    line = next(l for l in prompt.splitlines() if l.startswith("- options_block_notional_volume"))
    # constant series -> std 0 -> no z; the value must still read as USD, never bare "0.00"
    assert "$0.00" in line and " 0.00" not in line.replace("$0.00", "")
