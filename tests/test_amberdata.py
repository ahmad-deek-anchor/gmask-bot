"""Unit tests for providers.amberdata — no network.

Canned payloads are trimmed copies of real Amberdata responses (Sept 2026),
with timestamps in milliseconds as requested by the provider
(``timeFormat=milliseconds``).
"""

from datetime import datetime

import pandas as pd
import pytest

import providers.amberdata as amb
from providers.amberdata import AmberdataProvider, annualize_funding, to_instrument

START = datetime(2026, 9, 8)
END = datetime(2026, 9, 9)

D08 = 1788825600000  # 2026-09-08T00:00:00Z
D09 = 1788912000000  # 2026-09-09T00:00:00Z
H8 = 8 * 3600 * 1000

OI_ROWS = [
    # 2026-09-09
    {"coin": 139846.28, "exchange": "binance", "timestamp": D09, "type": "swaps", "usd": 11001.75, "volumeMilUSD": 15709.98},
    {"coin": 2727.26, "exchange": "binance", "timestamp": D09, "type": "futures", "usd": 215.97, "volumeMilUSD": 35.74},
    {"coin": 62330.87, "exchange": "bybit", "timestamp": D09, "type": "swaps", "usd": 5026.07, "volumeMilUSD": 4804.41},
    {"coin": 33757.67, "exchange": "okex", "timestamp": D09, "type": "swaps", "usd": 2654.25, "volumeMilUSD": 6636.81},
    {"coin": 491.55, "exchange": "deribit", "timestamp": D09, "type": "swaps", "usd": 896.42, "volumeMilUSD": 456.84},
    {"coin": 29305.02, "exchange": "huobi", "timestamp": D09, "type": "swaps", "usd": 2304.23, "volumeMilUSD": 436.90},  # not supported
    {"coin": None, "exchange": "bitmex", "timestamp": D09, "type": "futures", "usd": 3.75, "volumeMilUSD": 0},
    # 2026-09-08
    {"coin": 140000.0, "exchange": "binance", "timestamp": D08, "type": "swaps", "usd": 11000.0, "volumeMilUSD": 15000.0},
    {"coin": 62000.0, "exchange": "bybit", "timestamp": D08, "type": "swaps", "usd": 5000.0, "volumeMilUSD": 4800.0},
    # out of requested range (2026-09-10) — must be dropped
    {"coin": 1.0, "exchange": "binance", "timestamp": D09 + 86400000, "type": "swaps", "usd": 99999.0, "volumeMilUSD": 1.0},
]

LIQ_ROWS = [
    {"averagePrice": 77931.0, "buyLiquidationsUSD": 11819893, "exchange": "binance", "sellLiquidationsUSD": 27412808, "timestamp": D09},
    {"averagePrice": 77931.0, "buyLiquidationsUSD": 4614677, "exchange": "bybit", "sellLiquidationsUSD": 5851724, "timestamp": D09},
    {"averagePrice": 77931.0, "buyLiquidationsUSD": 0, "exchange": "deribit", "sellLiquidationsUSD": 26010, "timestamp": D09},
    {"averagePrice": 77931.0, "buyLiquidationsUSD": 4330797, "exchange": "okex", "sellLiquidationsUSD": 12070458, "timestamp": D09},
    {"averagePrice": 77931.0, "buyLiquidationsUSD": 5000000, "exchange": "huobi", "sellLiquidationsUSD": 5000000, "timestamp": D09},  # ignored
    # bitget is a supported exchange but its liquidation feed is mis-scaled -> excluded (real value seen live)
    {"averagePrice": 77931.0, "buyLiquidationsUSD": 657260563177, "exchange": "bitget", "sellLiquidationsUSD": 19951340139, "timestamp": D09},
    # 2026-09-08 delivered as two 12h buckets (short-window granularity) — must be summed
    {"averagePrice": 79000.0, "buyLiquidationsUSD": 100, "exchange": "binance", "sellLiquidationsUSD": 200, "timestamp": D08},
    {"averagePrice": 79000.0, "buyLiquidationsUSD": 300, "exchange": "binance", "sellLiquidationsUSD": 400, "timestamp": D08 + 12 * 3600 * 1000},
]


def _fr(exchange, instrument, ts, rate, margin="linear", quote="USDT", hours=8):
    return {
        "exchange": exchange, "fundingRateIntervalHours": hours, "fundingRateNormalized8h": rate,
        "instrument": instrument, "marginType": margin, "quoteAsset": quote,
        "realizedFunding": rate, "realizedFundingCumulated": rate,
        "symbol": f"{exchange}_{instrument}", "timestamp": ts, "underlying": "BTC",
    }


FUNDING_ROWS = [
    # binance BTCUSDT: three 8h events on 09-08 -> mean 0.0001
    _fr("binance", "BTCUSDT", D08, 0.0000),
    _fr("binance", "BTCUSDT", D08 + H8, 0.0001),
    _fr("binance", "BTCUSDT", D08 + 2 * H8, 0.0002),
    # binance inverse + oddball linear instruments must be ignored when BTCUSDT exists
    _fr("binance", "BTCUSD_PERP", D08, 0.0100, margin="inverse", quote="USD"),
    _fr("binance", "BTCU", D08, 0.0100, quote="U"),
    # bybit BTCUSDT: single event -> 0.0003
    _fr("bybit", "BTCUSDT", D08, 0.0003),
    # okex canonical linear swap -> 0.0002 ; inverse ignored
    _fr("okex", "BTC-USDT-SWAP", D08, 0.0002),
    _fr("okex", "BTC-USD-SWAP", D08, 0.0500, margin="inverse", quote="USD"),
    # unsupported exchange
    _fr("bitmart", "BTCUSDT", D08, 0.0900),
    # 09-09: only binance
    _fr("binance", "BTCUSDT", D09, 0.0001),
]


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
    """Minimal requests.Session stand-in: pops responses from a queue."""

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


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(amb.time, "sleep", lambda s: slept.append(s))
    return slept


def _provider_with_get(monkeypatch, table):
    """Provider whose _get returns table[path] (list, or None for HTTP errors)."""
    p = AmberdataProvider("k", session=FakeSession([]))
    calls = []

    def fake_get(path, params):
        calls.append((path, dict(params)))
        v = table.get(path, [])
        return None if v is None else list(v)

    monkeypatch.setattr(p, "_get", fake_get)
    return p, calls


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def test_to_instrument_patterns():
    assert to_instrument("btc", "binance") == "BTCUSDT"
    assert to_instrument("eth", "bybit") == "ETHUSDT"
    assert to_instrument("eth", "okx") == "ETH-USDT-SWAP"   # alias -> okex
    assert to_instrument("sol", "deribit") == "SOL_USDC-PERPETUAL"
    assert to_instrument("btc", "hyperliquid") == "BTC_USDT-PERP"
    assert to_instrument("btc", "kraken") == "PF_XBTUSD"
    assert to_instrument("btc", "nope") is None


def test_asset_symbol_override():
    assert amb.asset_symbol("btc") == "BTC"
    assert amb.asset_symbol("matic") == "POL"


def test_annualize_funding():
    # 0.01% per 8h -> 3 * 365 * 0.01% = 10.95% p.a.
    assert annualize_funding(0.0001) == pytest.approx(10.95)
    assert annualize_funding(-0.0002) == pytest.approx(-21.9)


def test_date_chunks_respect_cap_and_exclusive_end():
    chunks = amb._date_chunks(datetime(2026, 7, 27, 15), datetime(2026, 9, 9), 30)
    assert chunks[0] == (datetime(2026, 7, 27), datetime(2026, 8, 26))
    assert chunks[-1][1] == datetime(2026, 9, 10)          # end day included via exclusive bound
    assert all((e - s).days <= 30 for s, e in chunks)
    assert chunks[1][0] == chunks[0][1]                     # contiguous


def test_spot_methods_not_implemented():
    p = AmberdataProvider("k", session=FakeSession([]))
    with pytest.raises(NotImplementedError):
        p.get_spot_price("btc", START, END)
    with pytest.raises(NotImplementedError):
        p.get_spot_ohlcv("btc", START, END)


def test_session_gets_api_key_header():
    s = FakeSession([])
    AmberdataProvider("secret-key", session=s)
    assert s.headers["x-api-key"] == "secret-key"


# ----------------------------------------------------------------------
# normalisation + aggregation
# ----------------------------------------------------------------------

def _check_time(df):
    assert list(df.columns)[0] == "time"
    assert str(df["time"].dtype) == "datetime64[ns]"
    assert df["time"].dt.tz is None
    assert (df["time"] == df["time"].dt.normalize()).all()
    assert df["time"].is_monotonic_increasing
    assert not df["time"].duplicated().any()


def test_perp_oi_sums_swaps_over_supported_exchanges(monkeypatch):
    p, calls = _provider_with_get(monkeypatch, {"/analytics/futures-perpetuals/open-interest-total": OI_ROWS})
    df = p.get_perp_oi("btc", START, END)
    _check_time(df)
    assert list(df.columns) == ["time", "perp_oi"]
    assert len(df) == 2                                     # 09-10 row dropped
    d09 = df.loc[df["time"] == pd.Timestamp("2026-09-09"), "perp_oi"].item()
    # binance + bybit + okex + deribit swaps only (futures rows and huobi excluded), in USD
    assert d09 == pytest.approx((11001.75 + 5026.07 + 2654.25 + 896.42) * 1e6)
    d08 = df.loc[df["time"] == pd.Timestamp("2026-09-08"), "perp_oi"].item()
    assert d08 == pytest.approx(16000.0 * 1e6)
    path, params = calls[0]
    assert params["asset"] == "BTC"
    assert params["startDate"] == "2026-09-08" and params["endDate"] == "2026-09-10"
    assert params["timeFormat"] == "milliseconds"


def test_perp_volume_uses_swaps_volume_from_oi_endpoint(monkeypatch):
    p, calls = _provider_with_get(monkeypatch, {"/analytics/futures-perpetuals/open-interest-total": OI_ROWS})
    df = p.get_perp_volume("btc", START, END)
    _check_time(df)
    assert list(df.columns) == ["time", "perp_volume"]
    d09 = df.loc[df["time"] == pd.Timestamp("2026-09-09"), "perp_volume"].item()
    assert d09 == pytest.approx((15709.98 + 4804.41 + 6636.81 + 456.84) * 1e6)
    assert [c[0] for c in calls] == ["/analytics/futures-perpetuals/open-interest-total"]


def test_perp_volume_falls_back_to_volumes_endpoint(monkeypatch):
    oi_no_vol = [{k: v for k, v in r.items() if k != "volumeMilUSD"} for r in OI_ROWS]
    vol_rows = [
        {"exchange": "binance", "timestamp": D09, "totalDailyVolume": 199714.2, "totalDailyVolumeMilUSD": 15736.3, "totalDailyVolumeNative": 199714.2, "underlying": "BTC"},
        {"exchange": "bybit", "timestamp": D09, "totalDailyVolume": 61239.2, "totalDailyVolumeMilUSD": 4825.7, "totalDailyVolumeNative": 61239.2, "underlying": "BTC"},
        {"exchange": "huobi", "timestamp": D09, "totalDailyVolume": 1.0, "totalDailyVolumeMilUSD": 999.0, "totalDailyVolumeNative": 1.0, "underlying": "BTC"},
    ]
    p, calls = _provider_with_get(monkeypatch, {
        "/analytics/futures-perpetuals/open-interest-total": oi_no_vol,
        "/analytics/futures-perpetuals/volumes": vol_rows,
    })
    df = p.get_perp_volume("btc", START, END)
    assert len(df) == 1
    assert df["perp_volume"].item() == pytest.approx((15736.3 + 4825.7) * 1e6)
    assert calls[-1][0] == "/analytics/futures-perpetuals/volumes"


def test_volumes_endpoint_is_chunked_to_30_days(monkeypatch):
    p, calls = _provider_with_get(monkeypatch, {"/analytics/futures-perpetuals/volumes": []})
    p._volume_from_volumes_endpoint("btc", datetime(2026, 7, 26), datetime(2026, 9, 9))
    spans = [(pd.Timestamp(c[1]["endDate"]) - pd.Timestamp(c[1]["startDate"])).days for c in calls]
    assert len(calls) == 2 and max(spans) <= 30 and sum(spans) == 46


def test_liquidations_map_sides_and_sum_subdaily_buckets(monkeypatch):
    p, _ = _provider_with_get(monkeypatch, {"/analytics/futures-perpetuals/liquidations-total": LIQ_ROWS})
    df = p.get_liquidations("btc", START, END)
    _check_time(df)
    assert list(df.columns) == ["time", "long_liquidations", "short_liquidations", "total_liquidations"]
    r09 = df[df["time"] == pd.Timestamp("2026-09-09")].iloc[0]
    assert r09["long_liquidations"] == pytest.approx(27412808 + 5851724 + 26010 + 12070458)   # sell-to-close
    assert r09["short_liquidations"] == pytest.approx(11819893 + 4614677 + 0 + 4330797)       # buy-to-close
    assert r09["total_liquidations"] == pytest.approx(r09["long_liquidations"] + r09["short_liquidations"])
    r08 = df[df["time"] == pd.Timestamp("2026-09-08")].iloc[0]
    assert (r08["long_liquidations"], r08["short_liquidations"], r08["total_liquidations"]) == (600, 400, 1000)


def test_liquidations_exclude_bitget_but_other_metrics_keep_it(monkeypatch):
    assert "bitget" in amb.SUPPORTED_EXCHANGES and "bitget" in amb.LIQUIDATION_EXCLUDED_EXCHANGES
    p, _ = _provider_with_get(monkeypatch, {
        "/analytics/futures-perpetuals/liquidations-total": [r for r in LIQ_ROWS if r["exchange"] == "bitget"],
        "/analytics/futures-perpetuals/open-interest-total": [
            {"exchange": "bitget", "timestamp": D09, "type": "swaps", "usd": 3217.19, "volumeMilUSD": 2884.56}],
    })
    assert p.get_liquidations("btc", START, END) is None
    assert p.get_perp_oi("btc", START, END)["perp_oi"].item() == pytest.approx(3217.19e6)


def test_funding_rate_daily_mean_per_exchange_then_cross_exchange_mean_annualised(monkeypatch):
    p, calls = _provider_with_get(monkeypatch, {"/analytics/futures-perpetuals/funding-rates": FUNDING_ROWS})
    df = p.get_funding_rate("btc", START, END)
    _check_time(df)
    assert list(df.columns) == ["time", "funding_rate"]
    assert calls[0][1]["underlying"] == "BTC"
    # 09-08: binance mean(0,1e-4,2e-4)=1e-4 ; bybit 3e-4 ; okex 2e-4 -> mean 2e-4 -> 21.9 % p.a.
    r08 = df.loc[df["time"] == pd.Timestamp("2026-09-08"), "funding_rate"].item()
    assert r08 == pytest.approx(annualize_funding(2e-4)) == pytest.approx(21.9)
    r09 = df.loc[df["time"] == pd.Timestamp("2026-09-09"), "funding_rate"].item()
    assert r09 == pytest.approx(10.95)


def test_funding_falls_back_to_linear_usd_quoted_instrument(monkeypatch):
    rows = [
        _fr("binance", "BTCUSDC", D08, 0.0004, quote="USDC"),                      # no BTCUSDT -> use USDC perp
        _fr("binance", "BTCUSD_PERP", D08, 0.0100, margin="inverse", quote="USD"),  # inverse never used
        _fr("deribit", "BTC-PERPETUAL", D08, 0.0100, margin="inverse", quote="USD", hours=1),  # only inverse -> skipped
    ]
    p, _ = _provider_with_get(monkeypatch, {"/analytics/futures-perpetuals/funding-rates": rows})
    df = p.get_funding_rate("btc", START, END)
    assert len(df) == 1
    assert df["funding_rate"].item() == pytest.approx(annualize_funding(0.0004))


def test_exchanges_override_and_okx_alias(monkeypatch):
    p = AmberdataProvider("k", session=FakeSession([]), exchanges=["OKX", "binance"])
    assert p.exchanges == ("okex", "binance")
    monkeypatch.setattr(p, "_get", lambda path, params: list(OI_ROWS))
    df = p.get_perp_oi("btc", START, END)
    d09 = df.loc[df["time"] == pd.Timestamp("2026-09-09"), "perp_oi"].item()
    assert d09 == pytest.approx((11001.75 + 2654.25) * 1e6)


def test_iso_timestamps_are_also_parsed(monkeypatch):
    rows = [dict(r, timestamp="2026-09-09T00:00:00.000Z") for r in OI_ROWS if r["timestamp"] == D09]
    p, _ = _provider_with_get(monkeypatch, {"/analytics/futures-perpetuals/open-interest-total": rows})
    df = p.get_perp_oi("btc", START, END)
    assert df["time"].tolist() == [pd.Timestamp("2026-09-09")]


# ----------------------------------------------------------------------
# empty / error handling -> None
# ----------------------------------------------------------------------

@pytest.mark.parametrize("method", ["get_funding_rate", "get_perp_oi", "get_perp_volume", "get_liquidations"])
def test_empty_payload_returns_none(monkeypatch, method):
    p, _ = _provider_with_get(monkeypatch, {})  # every path -> []
    assert getattr(p, method)("zzzz", START, END) is None


@pytest.mark.parametrize("method", ["get_funding_rate", "get_perp_oi", "get_perp_volume", "get_liquidations"])
def test_http_error_returns_none(monkeypatch, method):
    table = {path: None for path in (
        "/analytics/futures-perpetuals/open-interest-total",
        "/analytics/futures-perpetuals/volumes",
        "/analytics/futures-perpetuals/liquidations-total",
        "/analytics/futures-perpetuals/funding-rates",
    )}
    p, _ = _provider_with_get(monkeypatch, table)
    assert getattr(p, method)("btc", START, END) is None


def test_only_unsupported_exchanges_returns_none(monkeypatch):
    rows = [r for r in OI_ROWS if r["exchange"] == "huobi"]
    p, _ = _provider_with_get(monkeypatch, {"/analytics/futures-perpetuals/open-interest-total": rows})
    assert p.get_perp_oi("btc", START, END) is None


def test_get_returns_none_on_404_and_403(no_sleep, caplog):
    s = FakeSession([FakeResponse(404, text="not found"), FakeResponse(403, text="unauthorized")])
    p = AmberdataProvider("k", session=s)
    with caplog.at_level("INFO"):
        assert p._get("/x", {"asset": "BTC"}) is None
        assert p._get("/x", {"asset": "BTC"}) is None
    assert no_sleep == []
    assert "403" in caplog.text


def test_get_retries_on_429_and_5xx_with_backoff(no_sleep):
    s = FakeSession([
        FakeResponse(429, text="slow down", headers={"Retry-After": "2"}),
        FakeResponse(502, text="bad gateway"),
        _ok([{"a": 1}]),
    ])
    p = AmberdataProvider("k", session=s, backoff=0.5)
    assert p._get("/x", {}) == [{"a": 1}]
    assert len(s.calls) == 3
    assert no_sleep == [2.0, 1.0]          # max(backoff, Retry-After) then backoff*2


def test_get_gives_up_after_max_retries(no_sleep):
    s = FakeSession([FakeResponse(503)] * 4)
    p = AmberdataProvider("k", session=s, max_retries=3)
    assert p._get("/x", {}) is None
    assert len(s.calls) == 4


def test_get_follows_next_cursor():
    s = FakeSession([
        _ok([{"a": 1}], nxt="https://api.amberdata.com/markets/derivatives/x?cursor=abc"),
        _ok([{"a": 2}]),
    ])
    p = AmberdataProvider("k", session=s)
    assert p._get("/x", {"asset": "BTC"}) == [{"a": 1}, {"a": 2}]
    assert s.calls[1] == ("https://api.amberdata.com/markets/derivatives/x?cursor=abc", None)


def test_partial_chunk_failure_keeps_good_chunks(monkeypatch):
    p = AmberdataProvider("k", session=FakeSession([]))
    seen = []

    def fake_get(path, params):
        seen.append(params["startDate"])
        if params["startDate"] == "2026-08-25":
            return None  # e.g. transient 5xx after retries
        return [{"exchange": "binance", "timestamp": D09, "type": "swaps", "usd": 1.0, "volumeMilUSD": 1.0}]

    monkeypatch.setattr(p, "_get", fake_get)
    monkeypatch.setattr(amb, "_DEFAULT_CHUNK_DAYS", 30)
    df = p.get_perp_oi("btc", datetime(2026, 7, 26), datetime(2026, 9, 9))
    assert seen == ["2026-07-26", "2026-08-25"]
    assert df is not None and df["perp_oi"].item() == pytest.approx(1e6)
