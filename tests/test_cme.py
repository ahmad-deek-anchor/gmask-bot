"""CME futures + BTC ETF on-chain methods and tools against a fake Coin Metrics client. No network.

Reference rows mirror the real CME listing on 2026-09-16 (BTCU6 expiring 2026-09-25, BTCV6,
BTCZ6, the micro MBTV6, a calendar spread, a weekly Bitcoin Friday contract and an already
expired BTCM6), with OI at 21:00 UTC and daily candles.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from providers.coinmetrics import CME_BASES, CoinMetricsProvider, cme_contract_label, cme_product
from tools import cme_tools

NOW = datetime(2026, 9, 16, 14, 0, tzinfo=timezone.utc)
D = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


class _Result:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


REF_ROWS = [
    # market, symbol, base, listing, expiration, size
    ("cme-BTCU6-future", "BTCU6", "btc", "2025-03-28 21:01", "2026-09-25 15:00", 5.0),
    ("cme-BTCV6-future", "BTCV6", "btc", "2026-04-24 21:01", "2026-10-30 16:00", 5.0),
    ("cme-BTCZ6-future", "BTCZ6", "btc", "2024-12-27 22:01", "2026-12-24 16:00", 5.0),
    ("cme-BTCH7-future", "BTCH7", "btc", "2025-09-26 21:01", "2027-03-25 16:00", 5.0),     # listed, never traded
    ("cme-MBTV6-future", "MBTV6", "btc", "2026-04-24 21:01", "2026-10-30 16:00", 0.1),
    ("cme-BTCU6-BTCV6-future", "BTCU6-BTCV6", "btc", "2026-04-24 21:01", "2026-09-25 15:00", 1.0),
    ("cme-BFFU618-future", "BFFU618", "bff", "2026-09-03 21:01", "2026-09-18 20:00", 0.02),
    ("cme-BTCM6-future", "BTCM6", "btc", "2024-06-28 21:01", "2026-06-26 15:00", 5.0),     # expired
    ("cme-BTCZ7-future", "BTCZ7", "btc", "2026-12-26 22:01", "2027-12-31 16:00", 5.0),     # not yet listed
    ("cme-ETHV6-future", "ETHV6", "eth", "2026-04-24 21:01", "2026-10-30 16:00", 50.0),
    ("cme-MSLV6-future", "MSLV6", "msl", "2026-04-24 21:01", "2026-10-30 16:00", 25.0),
    ("cme-SOLV6-future", "SOLV6", "sol", "2026-04-24 21:01", "2026-10-30 16:00", 500.0),
]

CLOSES = {"cme-BTCU6-future": 75_400.0, "cme-BTCV6-future": 76_125.0, "cme-BTCZ6-future": 77_300.0, "cme-MBTV6-future": 76_130.0,
          "cme-BFFU618-future": 75_300.0, "cme-ETHV6-future": 2_410.0, "cme-SOLV6-future": 101.5, "cme-MSLV6-future": 101.6}
OI = {"cme-BTCU6-future": 1_200, "cme-BTCV6-future": 5_457, "cme-BTCZ6-future": 2_100, "cme-MBTV6-future": 40_000,
      "cme-BFFU618-future": 3_000, "cme-ETHV6-future": 9_000, "cme-SOLV6-future": 2_000, "cme-MSLV6-future": 5_000}
SPOT = 75_000.0


class FakeClient:
    def __init__(self, batch_fails=False):
        self.calls = []
        self.batch_fails = batch_fails

    def reference_data_markets(self, **kw):
        self.calls.append("refdata")
        return _Result(pd.DataFrame([{"market": m, "symbol": s, "base": b, "quote": "usd", "listing": D(l), "expiration": D(e),
                                      "contract_size": sz, "size_asset": b if b in ("btc", "eth", "sol") else "x"}
                                     for m, s, b, l, e, sz in REF_ROWS]))

    def _check(self, markets):
        if any(m not in CLOSES for m in markets):
            raise RuntimeError("400 bad_request: market not supported")
        if self.batch_fails and len(markets) > 1:
            raise RuntimeError("400 bad_request: too many markets")

    def get_market_candles(self, markets, frequency="1d", start_time=None, **kw):
        self.calls.append(("candles", tuple(markets)))
        self._check(markets)
        rows = []
        for m in markets:
            for d in range(8, 0, -1):
                day = (NOW - timedelta(days=d)).replace(hour=0, minute=0)
                rows.append({"market": m, "time": day, "price_open": CLOSES[m] - 100, "price_close": CLOSES[m] - 10 * d,
                             "price_high": CLOSES[m] + 50, "price_low": CLOSES[m] - 200, "vwap": CLOSES[m], "volume": 500.0,
                             "candle_usd_volume": 100e6 + d * 1e6, "candle_trades_count": 400})
        return _Result(pd.DataFrame(rows))

    def get_market_open_interest(self, markets, start_time=None, **kw):
        self.calls.append(("oi", tuple(markets)))
        self._check(markets)
        rows = []
        for m in markets:
            for d in range(8, 0, -1):
                day = (NOW - timedelta(days=d)).replace(hour=21, minute=0)
                if day.weekday() >= 5:
                    continue                                     # CME publishes on weekdays only
                rows.append({"market": m, "time": day, "contract_count": OI[m] - 10 * d, "value_usd": (OI[m] - 10 * d) * CLOSES[m] * 5.0})
        return _Result(pd.DataFrame(rows))

    def get_market_trades(self, markets, **kw):
        (m,) = markets
        self.calls.append(("trade", m))
        if not m.startswith("coinbase-"):
            raise RuntimeError("400")
        return _Result(pd.DataFrame({"market": m, "time": [NOW], "price": [str(SPOT)], "amount": ["1.0"], "side": ["buy"]}))

    def get_asset_metrics(self, assets, metrics, frequency="1d", start_time=None, **kw):
        self.calls.append(("asset_metrics", tuple(assets), tuple(metrics), frequency))
        rows = []
        n = 30 if frequency == "1d" else 24 * 3
        for i in range(n, 0, -1):
            t = (NOW - (timedelta(days=i) if frequency == "1d" else timedelta(hours=i)))
            t = t.replace(hour=0, minute=0) if frequency == "1d" else t.replace(minute=0)
            row = {"asset": assets[0], "time": t}
            if "FlowInEtfUSD" in metrics:
                row.update({"FlowInEtfUSD": 100e6 + i * 1e6, "FlowOutEtfUSD": 80e6 if i % 3 else 200e6})
            if "SplyEtfNtv" in metrics:
                row.update({"SplyEtfNtv": 1_070_000 + i, "SplyEtfUSD": (1_070_000 + i) * SPOT})
            if "open_interest_reported_future_usd" in metrics:
                row.update({"open_interest_reported_future_usd": 44e9, "volume_reported_future_usd_1d": 60e9})
            rows.append(row)
        return _Result(pd.DataFrame(rows))


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def provider(client, monkeypatch):
    prov = CoinMetricsProvider(client)
    monkeypatch.setattr(cme_tools._factory, "get_provider", lambda: prov)
    return prov


# --- helpers -----------------------------------------------------------------

def test_symbol_parsing_and_labels():
    assert cme_product("BTCV6") == ("BTC", "V", "6") and cme_product("MBTZ6") == ("MBT", "Z", "6")
    assert cme_product("BFFU618") == ("BFF", "U", "618") and cme_product("BTCU6-BTCV6") is None
    assert cme_contract_label("BTCV6") == "Oct-26" and cme_contract_label("ETHH7") == "Mar-27"
    assert cme_contract_label("BFFU618") == "wk Sep-18" and cme_contract_label("weird") == "weird"
    assert CME_BASES == ("btc", "eth", "sol", "xrp")


# --- provider ----------------------------------------------------------------

def test_cme_contracts_filters_active_outrights(provider, client):
    df = provider.cme_contracts("btc", now=NOW)
    assert df["symbol"].tolist() == ["BTCU6", "BTCV6", "MBTV6", "BTCZ6", "BTCH7"]     # expired / unlisted / spread / weekly dropped
    assert df["is_standard"].tolist() == [True, True, False, True, True]
    assert df["label"].tolist()[:2] == ["Sep-26", "Oct-26"] and df.iloc[0]["days_to_expiry"] == pytest.approx(9.0, abs=0.1)
    std = provider.cme_contracts("btc", include_micro=False, now=NOW)
    assert "MBTV6" not in std["symbol"].tolist()
    wk = provider.cme_contracts("btc", include_weekly=True, now=NOW)
    assert "BFFU618" in wk["symbol"].tolist() and wk[wk["symbol"] == "BFFU618"]["is_weekly"].iloc[0]
    assert provider.cme_contracts("sol", now=NOW)["symbol"].tolist() == ["SOLV6", "MSLV6"]     # micro SOL under base 'msl'
    assert provider.cme_contracts("ada", now=NOW).empty
    provider.cme_contracts("eth", now=NOW)
    assert client.calls.count("refdata") == 1                                              # cached


def test_cme_curve_basis_and_open_interest(provider, client):
    df = provider.cme_curve("btc", now=NOW)
    assert df["symbol"].tolist() == ["BTCU6", "BTCV6", "BTCZ6"]                          # BTCH7 never traded -> dropped
    assert df.attrs["spot"] == SPOT and df.attrs["spot_market"] == "coinbase-btc-usd-spot"
    v6 = df.set_index("symbol").loc["BTCV6"]
    assert v6["close"] == CLOSES["cme-BTCV6-future"] - 10                                  # latest daily close (d=1)
    assert v6["basis_pct"] == pytest.approx((v6["close"] / SPOT - 1) * 100)
    assert v6["basis_ann_pct"] == pytest.approx(v6["basis_pct"] * 365 / v6["days_to_expiry"])
    assert v6["oi_contracts"] == OI["cme-BTCV6-future"] - 10 and v6["oi_base"] == pytest.approx(v6["oi_contracts"] * 5.0)
    assert pd.Timestamp(df.attrs["oi_as_of"]).hour == 21
    # the batch candle call failed on the unlisted contract -> per-market fallback, BTCH7 skipped
    assert ("candles", ("cme-BTCU6-future",)) in client.calls
    micro = provider.cme_curve("btc", include_micro=True, now=NOW)
    assert "MBTV6" in micro["symbol"].tolist()


def test_cme_curve_without_contracts_or_data(provider):
    assert provider.cme_curve("ada", now=NOW) is None


def test_cme_history_aggregate_and_single_contract(provider, client):
    df = provider.cme_history("btc", days=7, now=NOW)
    assert list(df.columns) == ["time", "usd_volume", "oi_contracts", "oi_usd", "all_venue_oi_usd", "all_venue_volume_usd", "cme_share_oi_pct"]
    assert df.attrs["contract"] is None and set(df.attrs["contracts"]) >= {"cme-BTCU6-future", "cme-MBTV6-future", "cme-BFFU618-future"}
    weekday = df.dropna(subset=["oi_contracts"]).iloc[-1]
    # OI summed over every active outright that has data (standard + micro + weekly)
    expected = sum(OI[m] - 10 for m in ("cme-BTCU6-future", "cme-BTCV6-future", "cme-BTCZ6-future", "cme-MBTV6-future", "cme-BFFU618-future"))
    assert weekday["oi_contracts"] == expected
    assert weekday["cme_share_oi_pct"] == pytest.approx(weekday["oi_usd"] / 44e9 * 100)
    assert df["time"].min() >= pd.Timestamp(NOW - timedelta(days=7)).floor("D")
    one = provider.cme_history("btc", contract="btcz6", days=7, now=NOW)
    assert one.attrs["contract"] == "cme-BTCZ6-future" and "close" in one.columns
    assert one.dropna(subset=["close"]).iloc[-1]["close"] == CLOSES["cme-BTCZ6-future"] - 10
    assert provider.cme_history("btc", contract="BTCH7", days=7, now=NOW) is None


def test_etf_onchain_flows(provider, client):
    df = provider.etf_onchain_flows("btc", days=10, now=NOW)
    assert list(df.columns) == ["time", "flow_in_usd", "flow_out_usd", "net_flow_usd", "supply_btc", "supply_usd"]
    assert len(df) == 30 and df["net_flow_usd"].iloc[-1] == pytest.approx((100e6 + 1e6) - 80e6)
    assert df["supply_btc"].iloc[-1] == 1_070_001 and df.attrs["frequency"] == "1d"
    hourly = provider.etf_onchain_flows("btc", days=2, frequency="1h", now=NOW)
    assert len(hourly) == 72 and hourly["supply_btc"].notna().sum() <= 3                  # daily supply lands on the midnight rows only
    assert provider.etf_onchain_flows("eth", now=NOW) is None
    assert provider.etf_onchain_flows("btc", frequency="5m", now=NOW) is None


# --- tools -------------------------------------------------------------------

def test_get_cme_curve_tool(provider):
    out = cme_tools.get_cme_curve.invoke({"token": "btc"})
    assert out.startswith("### CME BTC futures curve (3 contracts, standard size only)")
    assert "spot reference: $75,000 last trade on Coinbase BTC-USD spot" in out
    assert "| BTCV6 (Oct-26) | 5 BTC |" in out and "Ann. basis" in out
    assert "total standard-contract OI" in out and "curve upward-sloping (contango)" in out
    assert "ACT/365" in out
    assert "MBTV6" in cme_tools.get_cme_curve.invoke({"token": "btc", "include_micro": True})
    assert "not one of them" in cme_tools.get_cme_curve.invoke({"token": "doge"})
    assert "No CME futures data" in cme_tools.get_cme_curve.invoke({"token": "xrp"})


def test_get_cme_open_interest_tool(provider):
    out = cme_tools.get_cme_open_interest.invoke({"token": "btc", "days": 7})
    assert out.startswith("### CME BTC open interest, last 7 days - all active BTC outrights")
    assert "latest OI $" in out and "CME share of all-venue BTC futures OI" in out
    assert "| Date | OI (contracts) | OI (USD) | Volume (USD) | All-venue OI | CME share |" in out
    one = cme_tools.get_cme_open_interest.invoke({"token": "btc", "days": 7, "contract": "BTCZ6"})
    assert "- BTCZ6" in one and "| Close |" in one and "CME share" not in one.split("\n")[0]
    assert "No CME open interest" in cme_tools.get_cme_open_interest.invoke({"token": "btc", "contract": "BTCH7"})


def test_get_btc_etf_onchain_flows_tool(provider):
    out = cme_tools.get_btc_etf_onchain_flows.invoke({"days": 10})
    assert out.startswith("### BTC ETF on-chain flows, last 10 days (daily, Coin Metrics)")
    assert "latest day" in out and "net-inflow" in out and "ETF-held supply 1,070,001 BTC" in out
    assert "| Date | Inflow | Outflow | Net | ETF supply (BTC) |" in out and "lag issuer reports" in out
    hourly = cme_tools.get_btc_etf_onchain_flows.invoke({"days": 2, "hourly": True})
    assert "(hourly, Coin Metrics)" in hourly and "UTC" in hourly


def test_tools_registered_in_chat_default_tools():
    from chat import default_tools
    names = [t.name for t in default_tools()]
    assert set(cme_tools.CME_TOOL_NAMES) <= set(names)
    assert names.index("get_cme_curve") > names.index("list_crypto_sectors")
