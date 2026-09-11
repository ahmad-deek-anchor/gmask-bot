"""Unit tests for providers.amberdata_options — no network.

Canned payloads are trimmed copies of real Amberdata options responses
(Deribit, probed 2026-09-10). Timestamps are epoch ms on some endpoints and
ISO-8601 on others, exactly as the API returns them.
"""

from datetime import datetime

import pandas as pd
import pytest

import providers._http as http_mod
import providers.amberdata_options as opts
from providers.amberdata_options import AmberdataOptionsProvider, currencies_from_instruments

START = datetime(2026, 9, 7)
END = datetime(2026, 9, 10)
TODAY = pd.Timestamp("2026-09-10")   # "current UTC day" for the partial-day rule

D07 = 1788739200000  # 2026-09-07T00:00:00Z
D08 = 1788825600000
D09 = 1788912000000
D10 = 1788998400000  # today -> partial, must be dropped

INSTRUMENT_ROWS = [
    {"currency": "BTC", "exchange": "deribit", "expiration": 1789113600000, "instrument": "DERIBIT-BTC-11SEP26-60000.0-C", "putCall": "C", "strike": 60000},
    {"currency": "BTC_USDC", "exchange": "deribit", "expiration": 1789113600000, "instrument": "DERIBIT-BTC_USDC-11SEP26-60000.0-C", "putCall": "C", "strike": 60000},
    {"currency": "ETH", "exchange": "deribit", "expiration": 1789113600000, "instrument": "DERIBIT-ETH-11SEP26-1800.0-C", "putCall": "C", "strike": 1800},
    {"currency": "SOL_USDC", "exchange": "deribit", "expiration": 1789113600000, "instrument": "DERIBIT-SOL_USDC-11SEP26-100.0-C", "putCall": "C", "strike": 100},
    {"currency": "XRP_USDC", "exchange": "deribit", "expiration": 1789113600000, "instrument": "DERIBIT-XRP_USDC-11SEP26-0.9-C", "putCall": "C", "strike": 0.9},
]

DVOL_ROWS = [  # newest first, as the API returns them; ms timestamps in exchangeTimestamp
    {"close": 39.63, "currency": "BTC", "exchange": "deribit", "exchangeTimestamp": D10, "high": 41.26, "instrument": "DERIBIT_BTC_DVOL_INDEX", "low": 39.02, "open": 40.2},
    {"close": 40.2, "currency": "BTC", "exchange": "deribit", "exchangeTimestamp": D09, "high": 40.87, "instrument": "DERIBIT_BTC_DVOL_INDEX", "low": 39.75, "open": 39.92},
    {"close": 39.92, "currency": "BTC", "exchange": "deribit", "exchangeTimestamp": D08, "high": 40.32, "instrument": "DERIBIT_BTC_DVOL_INDEX", "low": 38.55, "open": 38.79},
    {"close": 38.79, "currency": "BTC", "exchange": "deribit", "exchangeTimestamp": D07, "high": 39.32, "instrument": "DERIBIT_BTC_DVOL_INDEX", "low": 38.13, "open": 39.32},
]

RICHNESS_ROWS = [  # ISO timestamps
    {"atm180days": 40.188, "atm30days": 38.528, "atm60days": 38.481, "atm7days": 39.989, "atm90days": 39.469, "counter": 10, "currency": "BTC", "exchange": "deribit", "ratio": 9.936, "richness": 0.9936, "timestamp": "2026-09-10T00:00:00.000Z"},
    {"atm180days": 40.368, "atm30days": 38.019, "atm60days": 38.595, "atm7days": 39.289, "atm90days": 39.775, "counter": 10, "currency": "BTC", "exchange": "deribit", "ratio": 9.807, "richness": 0.9807, "timestamp": "2026-09-09T00:00:00.000Z"},
    {"atm180days": None, "atm30days": 36.684, "atm60days": 37.804, "atm7days": 35.812, "atm90days": 39.058, "counter": 9, "currency": "BTC", "exchange": "deribit", "ratio": 9.45, "richness": 0.945, "timestamp": "2026-09-08T00:00:00.000Z"},
]

PCR_ROWS = [
    {"currency": "BTC", "exchange": "deribit", "putCallRatioOpenInterest": 0.55, "putCallRatioVolume24hr": 0.81, "timestamp": D10},
    {"currency": "BTC", "exchange": "deribit", "putCallRatioOpenInterest": 0.53, "putCallRatioVolume24hr": 0.36, "timestamp": D09},
    {"currency": "BTC", "exchange": "deribit", "putCallRatioOpenInterest": 0.54, "putCallRatioVolume24hr": 0.48, "timestamp": D08},
]

# volume-aggregates without timeInterval=day: per-minute rows; blocked fields null when no block
VOLAGG_ROWS = [
    {"contractVolumeBlocked": None, "contractVolumeOnScreen": 0.1, "currency": "BTC", "exchange": "deribit", "notionalVolumeBlocked": None, "notionalVolumeOnScreen": 7844.771, "premiumVolumeBlocked": None, "premiumVolumeOnScreen": 419.695, "timestamp": D08 + 86340000},
    {"contractVolumeBlocked": 100, "contractVolumeOnScreen": 26.4, "currency": "BTC", "exchange": "deribit", "notionalVolumeBlocked": 7852972, "notionalVolumeOnScreen": 2073193.864, "premiumVolumeBlocked": 28663.348, "premiumVolumeOnScreen": 5893.003, "timestamp": D08 + 75780000},
    # already-daily row for 09-09 (what timeInterval=day returns)
    {"contractVolumeBlocked": 9172.7, "contractVolumeOnScreen": 10975.2, "currency": "BTC", "exchange": "deribit", "notionalVolumeBlocked": 722709156.533, "notionalVolumeOnScreen": 865562624.93, "premiumVolumeBlocked": 9426126.819, "premiumVolumeOnScreen": 12577245.712, "timestamp": D09},
    # today -> dropped
    {"contractVolumeBlocked": None, "contractVolumeOnScreen": 7.3, "currency": "BTC", "exchange": "deribit", "notionalVolumeBlocked": None, "notionalVolumeOnScreen": 561480.729, "premiumVolumeBlocked": None, "premiumVolumeOnScreen": 6887.606, "timestamp": D10 + 3600000},
]


def _surface_row(ts, dte, atm, c10, c25, p10, p25, index_price=76930.9):
    return {"atm": atm, "currency": "BTC", "daysToExpiration": dte, "delta50": atm, "deltaCall05": c10 + 3, "deltaCall10": c10,
            "deltaCall25": c25, "deltaPut05": p10 + 3, "deltaPut10": p10, "deltaPut25": p25, "exchange": "deribit",
            "indexPrice": index_price, "multiplier": 1, "openInterest": 442432.8, "timestamp": ts, "underlyingPrice": index_price + 250}


SURFACE_SNAPSHOT_ROWS = [  # live snapshot (10 tenors, trimmed to 3)
    _surface_row("2026-09-10T16:09:00.000Z", 7, 37.96, 43.91, 39.95, 41.60, 38.40),
    _surface_row("2026-09-10T16:09:00.000Z", 30, 37.644, 42.031, 38.822, 43.353, 38.7115),
    _surface_row("2026-09-10T16:09:00.000Z", 180, 40.06, 44.27, 40.98, 46.33, 41.55),
]

SURFACE_HISTORY_ROWS = [  # timeInterval=day -> one 00:00Z surface per day
    _surface_row("2026-09-10T00:00:00.000Z", 30, 38.528, 43.151, 39.854, 43.506, 39.592),   # today -> dropped
    _surface_row("2026-09-10T00:00:00.000Z", 7, 39.99, 44.0, 41.0, 43.0, 40.0),
    _surface_row("2026-09-09T00:00:00.000Z", 30, 38.019, 43.044, 39.583, 42.650, 38.757),
    _surface_row("2026-09-09T00:00:00.000Z", 7, 39.29, 44.0, 41.0, 43.0, 42.0),
    _surface_row("2026-09-08T00:00:00.000Z", 30, 36.684, 41.0, 38.0, 44.0, 40.0),
]

TERM_ROWS = [
    {"atm": 35.51, "currency": "BTC", "daysToExpiration": 1, "exchange": "deribit", "fwdAtm": None, "timestamp": "2026-09-10T16:09:00.000Z"},
    {"atm": 39.51, "currency": "BTC", "daysToExpiration": 2, "exchange": "deribit", "fwdAtm": 43.13, "timestamp": "2026-09-10T16:09:00.000Z"},
    {"atm": 40.12, "currency": "BTC", "daysToExpiration": 180, "exchange": "deribit", "fwdAtm": 41.02, "timestamp": "2026-09-10T16:09:00.000Z"},
    {"currency": "BTC", "daysToExpiration": 90, "exchange": "deribit", "fwdAtm": None, "timestamp": "2026-09-10T16:09:00.000Z"},  # no atm -> dropped
]

SNAP_OLD, SNAP_NEW = 1788991200000, 1788994800000  # 22:00Z and 23:00Z on 2026-09-09


def _gex(snap, strike, exp, pc, net, total, gamma, idx):
    return {"currency": "BTC", "dealerNetInventory": net, "dealerTotalInventory": total, "exchange": "deribit",
            "expirationTimestamp": exp, "gammaLevel": gamma, "indexPrice": idx,
            "instrumentNormalized": f"DERIBIT-BTC-X-{strike}.0-{pc}", "putCall": pc, "snapshotTimestamp": snap, "strike": strike}


GEX_ROWS = [
    _gex(SNAP_NEW, 71000, 1789027200000, "P", -26.6, -27.0, 0.0, 78220.66),
    _gex(SNAP_NEW, 71000, 1793347200000, "P", -10.0, -20.0, -0.001, 78220.66),   # other expiry, same strike
    _gex(SNAP_NEW, 71000, 1793347200000, "C", 6.6, 7.0, 0.002, 78220.66),         # call, same strike
    _gex(SNAP_NEW, 98000, 1793347200000, "C", 265.2, 299.0, 0.00299, 78220.66),
    _gex(SNAP_OLD, 71000, 1789027200000, "P", -999.0, -999.0, 0.0, 78022.84),      # older snapshot -> ignored
]

BLOCK_ROWS = [
    {"contractVolume": 325, "currency": "BTC", "exchange": "deribit", "expirationTimestamp": 1790323200000, "premiumVolume": 706367.2367, "putCall": "C", "strike": 80000},
    {"contractVolume": 158, "currency": "BTC", "exchange": "deribit", "expirationTimestamp": 1793347200000, "premiumVolume": 642235.227, "putCall": "C", "strike": 80000},
    {"contractVolume": 200, "currency": "BTC", "exchange": "deribit", "expirationTimestamp": 1793347200000, "premiumVolume": 580774.888, "putCall": "P", "strike": 76000},
    {"contractVolume": 40, "currency": "BTC", "exchange": "deribit", "expirationTimestamp": 1788940800000, "premiumVolume": 1576.046, "putCall": "C", "strike": 79500},
]


# ----------------------------------------------------------------------
# fakes
# ----------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if not self.responses:
            raise AssertionError("unexpected extra request")
        return self.responses.pop(0)


def _ok(rows, nxt=None):
    return FakeResponse(200, {"status": 200, "title": "OK", "description": "Successful request",
                              "payload": {"data": rows, "metadata": {"next": nxt, "api-version": "2023-09-30"}}})


def _range_cap_400(start, end):
    msg = (f"the range specified [{start}T00:00:00Z - {end}T00:00:00Z] is over the maximum allowed (1d), "
           "please reduce the size of your query")
    return FakeResponse(400, {"status": 400, "title": "BAD REQUEST", "error": True, "message": msg}, text=msg)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(opts, "_utc_today", lambda: TODAY)


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(http_mod.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(opts.time, "sleep", lambda s: slept.append(s))
    return slept


def _provider(monkeypatch, table, **kwargs):
    """Provider whose _get returns table[path] (list, or None for HTTP errors) and records calls."""
    p = AmberdataOptionsProvider("k", session=FakeSession([]), **kwargs)
    calls = []

    def fake_get(path, params):
        calls.append((path, dict(params)))
        v = table.get(path, [])
        return None if v is None else list(v)

    monkeypatch.setattr(p, "_get", fake_get)
    table.setdefault("instruments/information", INSTRUMENT_ROWS)
    return p, calls


def _check_daily(df):
    assert list(df.columns)[0] == "time"
    assert str(df["time"].dtype) == "datetime64[ns]"
    assert df["time"].dt.tz is None
    assert (df["time"] == df["time"].dt.normalize()).all()
    assert df["time"].is_monotonic_increasing
    assert not df["time"].duplicated().any()
    assert (df["time"] < TODAY).all()
    for c in df.columns[1:]:
        assert str(df[c].dtype) == "float64", c


# ----------------------------------------------------------------------
# currency mapping / discovery
# ----------------------------------------------------------------------

def test_currencies_from_instruments_prefers_unsuffixed_then_usdc():
    m = currencies_from_instruments(INSTRUMENT_ROWS + [{"currency": "SOL_USDT"}, {"currency": None}, "junk"])
    assert m == {"btc": "BTC", "eth": "ETH", "sol": "SOL_USDC", "xrp": "XRP_USDC"}


def test_currency_for_and_supported_tokens_are_discovered_once(monkeypatch):
    p, calls = _provider(monkeypatch, {})
    assert p.currency_for("btc") == "BTC"
    assert p.currency_for("SOL") == "SOL_USDC"
    assert p.currency_for("eth") == "ETH"
    assert p.currency_for("uni") is None
    assert p.supported_tokens() == ["btc", "eth", "sol", "xrp"]
    assert [c[0] for c in calls] == ["instruments/information"]
    assert calls[0][1] == {"exchange": "deribit"}


def test_discovery_failure_falls_back_to_static_map(monkeypatch, caplog):
    p, calls = _provider(monkeypatch, {"instruments/information": None})
    with caplog.at_level("WARNING"):
        assert p.currency_for("sol") == "SOL_USDC"
    assert "static currency map" in caplog.text
    assert "hype" in p.supported_tokens()


def test_unknown_token_returns_none_without_data_calls(monkeypatch, caplog):
    p, calls = _provider(monkeypatch, {"volatility/index": DVOL_ROWS})
    with caplog.at_level("INFO"):
        assert p.get_dvol("uni", START, END) is None
    assert [c[0] for c in calls] == ["instruments/information"]
    assert "no options" in caplog.text


def test_exchange_is_normalised_and_session_gets_headers():
    s = FakeSession([])
    p = AmberdataOptionsProvider("secret-key", exchange="Deribit", session=s)
    assert p.exchange == "deribit"
    assert s.headers["x-api-key"] == "secret-key"
    assert "gzip" in s.headers["Accept-Encoding"]


# ----------------------------------------------------------------------
# daily series
# ----------------------------------------------------------------------

def test_dvol_ms_timestamps_partial_day_dropped(monkeypatch):
    p, calls = _provider(monkeypatch, {"volatility/index": DVOL_ROWS})
    df = p.get_dvol("btc", START, END)
    _check_daily(df)
    assert list(df.columns) == ["time", "dvol_open", "dvol_high", "dvol_low", "dvol_close"]
    assert df["time"].tolist() == [pd.Timestamp("2026-09-07"), pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-09")]
    assert df["dvol_close"].tolist() == [38.79, 39.92, 40.2]
    path, params = calls[-1]
    assert path == "volatility/index"
    assert params == {"exchange": "deribit", "currency": "BTC", "timeInterval": "day",
                      "startDate": "2026-09-07", "endDate": "2026-09-11"}   # endDate exclusive


def test_term_structure_history_iso_timestamps_and_nulls(monkeypatch):
    p, calls = _provider(monkeypatch, {"volatility/term-structures/richness": RICHNESS_ROWS})
    df = p.get_term_structure_history("btc", START, END)
    _check_daily(df)
    assert list(df.columns) == ["time", "atm_iv_7d", "atm_iv_30d", "atm_iv_60d", "atm_iv_90d", "atm_iv_180d", "ts_richness"]
    assert df["time"].tolist() == [pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-09")]
    assert df["atm_iv_30d"].tolist() == [36.684, 38.019]
    assert pd.isna(df["atm_iv_180d"].iloc[0]) and df["atm_iv_180d"].iloc[1] == 40.368
    assert "timeInterval" not in calls[-1][1]


def test_put_call_ratio_columns(monkeypatch):
    p, _ = _provider(monkeypatch, {"trades-flow/put-call-ratio": PCR_ROWS})
    df = p.get_put_call_ratio("eth", START, END)
    _check_daily(df)
    assert list(df.columns) == ["time", "pcr_oi", "pcr_volume_24h"]
    assert df["pcr_oi"].tolist() == [0.54, 0.53]
    assert df["pcr_volume_24h"].tolist() == [0.48, 0.36]


def test_options_volume_sums_on_screen_and_block_per_day(monkeypatch):
    p, _ = _provider(monkeypatch, {"trades-flow/volume-aggregates": VOLAGG_ROWS})
    df = p.get_options_volume("btc", START, END)
    _check_daily(df)
    assert list(df.columns) == ["time", "options_contract_volume", "options_notional_volume",
                                "options_premium_volume", "options_block_notional_volume"]
    assert df["time"].tolist() == [pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-09")]
    r08 = df.iloc[0]
    assert r08["options_contract_volume"] == pytest.approx(0.1 + 26.4 + 100)
    assert r08["options_notional_volume"] == pytest.approx(7844.771 + 2073193.864 + 7852972)
    assert r08["options_premium_volume"] == pytest.approx(419.695 + 5893.003 + 28663.348)
    assert r08["options_block_notional_volume"] == pytest.approx(7852972)
    r09 = df.iloc[1]
    assert r09["options_notional_volume"] == pytest.approx(722709156.533 + 865562624.93)
    assert r09["options_block_notional_volume"] == pytest.approx(722709156.533)


@pytest.mark.parametrize("method,path", [
    ("get_dvol", "volatility/index"),
    ("get_term_structure_history", "volatility/term-structures/richness"),
    ("get_put_call_ratio", "trades-flow/put-call-ratio"),
    ("get_options_volume", "trades-flow/volume-aggregates"),
    ("get_skew_history", "volatility/delta-surfaces/constant"),
    ("get_block_trades", "trades-flow/block-volumes"),
])
def test_empty_and_failed_daily_series_return_none(monkeypatch, caplog, method, path):
    p, _ = _provider(monkeypatch, {path: []})
    with caplog.at_level("INFO"):
        assert getattr(p, method)("btc", START, END) is None
    assert "no " in caplog.text
    p2, _ = _provider(monkeypatch, {path: None})
    assert getattr(p2, method)("btc", START, END) is None


# ----------------------------------------------------------------------
# chunking: day-by-day fallback on the 1-day range cap
# ----------------------------------------------------------------------

def test_range_cap_400_triggers_day_loop_and_tolerates_day_failures(no_sleep, caplog):
    s = FakeSession([
        _ok(INSTRUMENT_ROWS),
        _range_cap_400("2026-09-07", "2026-09-11"),          # whole range rejected
        _ok([DVOL_ROWS[3]]),                                   # 09-07
        FakeResponse(500, text="boom"), FakeResponse(500), FakeResponse(500), FakeResponse(500),  # 09-08 fails
        _ok([DVOL_ROWS[1]]),                                   # 09-09
        _ok([DVOL_ROWS[0]]),                                   # 09-10 (partial -> dropped)
    ])
    p = AmberdataOptionsProvider("k", session=s, backoff=0.01, pause=0.5)
    with caplog.at_level("INFO"):
        df = p.get_dvol("btc", START, END)
    _check_daily(df)
    assert df["time"].tolist() == [pd.Timestamp("2026-09-07"), pd.Timestamp("2026-09-09")]
    days = [(c[1]["startDate"], c[1]["endDate"]) for c in s.calls[2:] if c[1]]
    assert days[0] == ("2026-09-07", "2026-09-08")
    assert days[-1] == ("2026-09-10", "2026-09-11")
    assert "capped at 1 day/request" in caplog.text
    assert "day skipped" in caplog.text
    assert 0.5 in no_sleep                       # pause between day calls
    assert p.call_count == len(s.calls) == 9


def test_small_chunk_days_loops_windows(monkeypatch):
    p, calls = _provider(monkeypatch, {"trades-flow/put-call-ratio": PCR_ROWS}, chunk_days=2)
    p.get_put_call_ratio("btc", START, END)
    windows = [(c[1]["startDate"], c[1]["endDate"]) for c in calls if c[0] == "trades-flow/put-call-ratio"]
    assert windows == [("2026-09-07", "2026-09-09"), ("2026-09-09", "2026-09-11")]


def test_non_cap_chunk_failure_is_skipped_not_looped(monkeypatch, caplog):
    p, calls = _provider(monkeypatch, {"trades-flow/put-call-ratio": None}, chunk_days=2)
    with caplog.at_level("WARNING"):
        assert p.get_put_call_ratio("btc", START, END) is None
    assert len([c for c in calls if c[0] == "trades-flow/put-call-ratio"]) == 2
    assert "skipped" in caplog.text


# ----------------------------------------------------------------------
# delta surface / skew
# ----------------------------------------------------------------------

def test_delta_surface_snapshot_skew_is_put_minus_call(monkeypatch):
    p, calls = _provider(monkeypatch, {"volatility/delta-surfaces/constant": SURFACE_SNAPSHOT_ROWS})
    df = p.get_delta_surface("btc")
    assert list(df.columns) == ["days_to_expiration", "atm_iv", "iv_call_10d", "iv_call_25d",
                                "iv_put_10d", "iv_put_25d", "skew_25d", "skew_10d"]
    assert df["days_to_expiration"].tolist() == [7.0, 30.0, 180.0]
    r30 = df[df["days_to_expiration"] == 30].iloc[0]
    assert r30["skew_25d"] == pytest.approx(38.7115 - 38.822)    # puts cheaper -> negative
    assert r30["skew_10d"] == pytest.approx(43.353 - 42.031)     # puts richer  -> positive
    assert df.attrs["snapshot_time"] == pd.Timestamp("2026-09-10 16:09:00")
    assert df.attrs["index_price"] == pytest.approx(76930.9)
    assert calls[-1][1] == {"exchange": "deribit", "currency": "BTC"}   # no dates for the live snapshot


def test_delta_surface_for_a_date_uses_day_range(monkeypatch):
    p, calls = _provider(monkeypatch, {"volatility/delta-surfaces/constant": SURFACE_HISTORY_ROWS})
    df = p.get_delta_surface("btc", timestamp=datetime(2026, 9, 9, 15, 30))
    assert calls[-1][1] == {"exchange": "deribit", "currency": "BTC", "timeInterval": "day",
                            "startDate": "2026-09-09", "endDate": "2026-09-10"}
    # only the latest surface in the payload is kept
    assert df.attrs["snapshot_time"] == pd.Timestamp("2026-09-10")
    assert len(df) == 2


def test_skew_history_daily_from_surface_history(monkeypatch):
    p, calls = _provider(monkeypatch, {"volatility/delta-surfaces/constant": SURFACE_HISTORY_ROWS})
    df = p.get_skew_history("btc", START, END)
    _check_daily(df)
    assert list(df.columns) == ["time", "skew_25d_30d", "skew_10d_30d"]
    assert df["time"].tolist() == [pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-09")]
    assert df["skew_25d_30d"].tolist() == pytest.approx([40.0 - 38.0, 38.757 - 39.583])
    assert df["skew_10d_30d"].tolist() == pytest.approx([44.0 - 41.0, 42.650 - 43.044])
    assert calls[-1][1]["timeInterval"] == "day"
    # a second tenor reuses the same surface download
    df7 = p.get_skew_history("btc", START, END, tenor_days=7)
    assert list(df7.columns) == ["time", "skew_25d_7d", "skew_10d_7d"]
    assert df7["skew_25d_7d"].tolist() == pytest.approx([42.0 - 41.0])
    assert len([c for c in calls if c[0] == "volatility/delta-surfaces/constant"]) == 1


def test_skew_history_unknown_tenor_returns_none(monkeypatch, caplog):
    p, calls = _provider(monkeypatch, {"volatility/delta-surfaces/constant": SURFACE_HISTORY_ROWS})
    with caplog.at_level("WARNING"):
        assert p.get_skew_history("btc", START, END, tenor_days=45) is None
    assert "tenor" in caplog.text
    assert calls == []


# ----------------------------------------------------------------------
# snapshots
# ----------------------------------------------------------------------

def test_term_structure_snapshot(monkeypatch):
    p, calls = _provider(monkeypatch, {"volatility/term-structures/forward-volatility/constant": TERM_ROWS})
    df = p.get_term_structure("btc")
    assert list(df.columns) == ["days_to_expiration", "atm_iv", "fwd_atm_iv"]
    assert df["days_to_expiration"].tolist() == [1.0, 2.0, 180.0]         # row without atm dropped
    assert pd.isna(df["fwd_atm_iv"].iloc[0]) and df["fwd_atm_iv"].iloc[1] == 43.13
    assert df.attrs["snapshot_time"] == pd.Timestamp("2026-09-10 16:09:00")
    assert calls[-1][1] == {"exchange": "deribit", "currency": "BTC"}
    p.get_term_structure("btc", timestamp=datetime(2026, 9, 5))
    assert calls[-1][1]["timestamp"] == "2026-09-05"


def test_gamma_exposure_aggregates_latest_snapshot_by_strike(monkeypatch):
    p, calls = _provider(monkeypatch, {"trades-flow/gamma-exposures-snapshots": GEX_ROWS})
    df = p.get_gamma_exposure("btc")
    assert list(df.columns) == ["strike", "net_dealer_gamma", "total_dealer_gamma", "gamma_level", "index_price"]
    assert df["strike"].tolist() == [71000.0, 98000.0]
    r71 = df.iloc[0]
    assert r71["net_dealer_gamma"] == pytest.approx(-26.6 - 10.0 + 6.6)      # across expiries and puts/calls
    assert r71["total_dealer_gamma"] == pytest.approx(-27.0 - 20.0 + 7.0)
    assert r71["gamma_level"] == pytest.approx(0.001)
    assert (df["index_price"] == 78220.66).all()
    assert df.attrs["snapshot_time"] == pd.Timestamp("2026-09-09 23:00:00")
    assert df.attrs["index_price"] == 78220.66
    assert calls[-1][1] == {"exchange": "deribit", "currency": "BTC"}   # latest snapshot: no dates


def test_gamma_exposure_for_a_date_asks_for_last_hour_then_widens(monkeypatch):
    p = AmberdataOptionsProvider("k", session=FakeSession([]))
    seen = []

    def fake_get(path, params):
        seen.append(dict(params))
        if path == "instruments/information":
            return INSTRUMENT_ROWS
        return [] if params.get("startDate", "").endswith("T23:00:00") else GEX_ROWS

    monkeypatch.setattr(p, "_get", fake_get)
    df = p.get_gamma_exposure("btc", date=datetime(2026, 9, 9))
    assert df is not None
    assert seen[1]["startDate"] == "2026-09-09T23:00:00" and seen[1]["endDate"] == "2026-09-10T00:00:00"
    assert seen[2]["startDate"] == "2026-09-09" and seen[2]["endDate"] == "2026-09-10"


def test_block_trades_sorted_by_premium_top_n_and_aggregated(monkeypatch):
    dup = dict(BLOCK_ROWS[3], contractVolume=10, premiumVolume=1000000.0)  # same (expiry, strike, C) -> summed
    p, calls = _provider(monkeypatch, {"trades-flow/block-volumes": BLOCK_ROWS + [dup]})
    df = p.get_block_trades("btc", START, END, top_n=3)
    assert list(df.columns) == ["expiry", "strike", "put_call", "contract_volume", "premium_volume"]
    assert len(df) == 3
    assert df["premium_volume"].is_monotonic_decreasing
    top = df.iloc[0]
    assert top["strike"] == 79500 and top["contract_volume"] == 50 and top["premium_volume"] == pytest.approx(1001576.046)
    assert top["expiry"] == pd.Timestamp("2026-09-09 08:00:00") and top["expiry"].tz is None
    assert df["put_call"].tolist() == ["C", "C", "C"]
    assert calls[-1][1] == {"exchange": "deribit", "currency": "BTC", "startDate": "2026-09-07", "endDate": "2026-09-11"}


# ----------------------------------------------------------------------
# HTTP behaviour through the shared layer
# ----------------------------------------------------------------------

def test_403_returns_none_with_warning(no_sleep, caplog):
    s = FakeSession([_ok(INSTRUMENT_ROWS), FakeResponse(403, text="Forbidden")])
    p = AmberdataOptionsProvider("k", session=s)
    with caplog.at_level("WARNING"):
        assert p.get_put_call_ratio("btc", START, END) is None
    assert "403" in caplog.text and "auth/tier" in caplog.text
    assert no_sleep == []
    assert s.calls[-1][0] == "https://api.amberdata.com/markets/derivatives/analytics/trades-flow/put-call-ratio"


def test_429_is_retried_with_backoff(no_sleep):
    s = FakeSession([
        _ok(INSTRUMENT_ROWS),
        FakeResponse(429, text="slow down", headers={"Retry-After": "2"}),
        _ok(PCR_ROWS),
    ])
    p = AmberdataOptionsProvider("k", session=s, backoff=0.5)
    df = p.get_put_call_ratio("btc", START, END)
    assert len(df) == 2
    assert no_sleep == [2.0]
    assert len(s.calls) == 3 and p.call_count == 3


def test_cursor_pagination_is_followed():
    s = FakeSession([
        _ok(INSTRUMENT_ROWS),
        _ok(PCR_ROWS[:2], nxt="https://api.amberdata.com/markets/derivatives/analytics/trades-flow/put-call-ratio?cursor=abc"),
        _ok(PCR_ROWS[2:]),
    ])
    p = AmberdataOptionsProvider("k", session=s)
    df = p.get_put_call_ratio("btc", START, END)
    assert len(df) == 2
    assert s.calls[2] == ("https://api.amberdata.com/markets/derivatives/analytics/trades-flow/put-call-ratio?cursor=abc", None)


# ----------------------------------------------------------------------
# memoisation
# ----------------------------------------------------------------------

def test_results_are_memoised_per_method_token_and_range(monkeypatch):
    p, calls = _provider(monkeypatch, {
        "volatility/index": DVOL_ROWS,
        "trades-flow/put-call-ratio": PCR_ROWS,
        "trades-flow/gamma-exposures-snapshots": GEX_ROWS,
    })
    a = p.get_dvol("btc", START, END)
    b = p.get_dvol("BTC", datetime(2026, 9, 7, 12), END)      # same days, different case/time-of-day
    p.get_put_call_ratio("btc", START, END)
    p.get_put_call_ratio("btc", START, END)
    g1 = p.get_gamma_exposure("btc")
    g2 = p.get_gamma_exposure("btc")
    assert [c[0] for c in calls] == ["instruments/information", "volatility/index",
                                     "trades-flow/put-call-ratio", "trades-flow/gamma-exposures-snapshots"]
    assert a.equals(b) and a is not b
    assert g2.attrs == g1.attrs
    # different range -> new fetch; None results are memoised too
    p.get_dvol("btc", datetime(2026, 9, 1), END)
    assert [c[0] for c in calls].count("volatility/index") == 2
    assert p.get_dvol("uni", START, END) is None
    assert p.get_dvol("uni", START, END) is None
    assert len(calls) == 5


def test_memoised_frames_are_copies(monkeypatch):
    p, _ = _provider(monkeypatch, {"volatility/index": DVOL_ROWS})
    a = p.get_dvol("btc", START, END)
    a["dvol_close"] = 0.0
    assert p.get_dvol("btc", START, END)["dvol_close"].tolist() == [38.79, 39.92, 40.2]


# ----------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------

def test_factory_get_options_provider(monkeypatch):
    import providers.factory as factory
    import utils.config as config_mod

    class FakeConfig:
        AMBERDATA_API_KEY = "ad-key"

    monkeypatch.setattr(config_mod, "Config", FakeConfig)
    factory.reset_options_provider()
    try:
        p = factory.get_options_provider()
        assert isinstance(p, AmberdataOptionsProvider)
        assert p.exchange == "deribit"
        assert p._http.session.headers["x-api-key"] == "ad-key"
        assert factory.get_options_provider() is p          # memoised

        factory.reset_provider()                            # also drops the options provider
        FakeConfig.AMBERDATA_API_KEY = None
        assert factory.get_options_provider() is None
        assert factory.get_options_provider() is None       # None memoised, no rebuild
    finally:
        factory.reset_options_provider()
