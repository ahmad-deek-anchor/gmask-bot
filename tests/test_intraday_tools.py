"""Intraday provider methods and tools against a fake Coin Metrics client. No network."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from providers.coinmetrics import CoinMetricsProvider, INTRADAY_FREQUENCIES, spot_markets_for
from tools import intraday_tools

NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0)


class _Result:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


def _bars(market, freq_minutes, n, start_price=100.0):
    times = [NOW - timedelta(minutes=freq_minutes * (n - i)) for i in range(n)]
    closes = [start_price + i for i in range(n)]
    return pd.DataFrame({
        "market": market, "time": times,
        "price_open": [c - 0.5 for c in closes], "price_high": [c + 1 for c in closes],
        "price_low": [c - 1 for c in closes], "price_close": closes,
        "vwap": closes, "volume": [10.0] * n,
        "candle_usd_volume": [1_000_000.0] * n, "candle_trades_count": [100] * n,
    })


class FakeClient:
    """Serves candles for one market at any intraday frequency, plus trades and quotes."""

    def __init__(self, market, fail_markets=()):
        self.market = market
        self.fail_markets = set(fail_markets)
        self.calls = []

    def get_market_candles(self, markets, frequency, **kw):
        (m,) = markets
        self.calls.append(("candles", m, frequency))
        if m != self.market or m in self.fail_markets:
            raise RuntimeError(f"400 bad_request: market {m} not supported")
        minutes = {"1m": 1, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60, "4h": 240}[frequency]
        return _Result(_bars(m, minutes, 30 if frequency != "1m" else 5))

    def get_market_trades(self, markets, **kw):
        (m,) = markets
        self.calls.append(("trades", m))
        if m != self.market:
            raise RuntimeError("400 bad_request")
        if kw.get("paging_from") == "end":
            return _Result(pd.DataFrame({"market": m, "time": [NOW], "price": ["130.5"], "amount": ["2.0"], "side": ["buy"]}))
        return _Result(pd.DataFrame({
            "market": m,
            "time": [NOW - timedelta(seconds=s) for s in (240, 180, 120, 60, 5)],
            "price": ["129.0", "129.5", "130.0", "130.2", "130.5"],
            "amount": ["1.0", "500.0", "2.0", "0.5", "1000.0"],
            "side": ["sell", "buy", "buy", "sell", "buy"],
        }))

    def get_market_quotes(self, markets, **kw):
        (m,) = markets
        self.calls.append(("quotes", m))
        if m != self.market:
            raise RuntimeError("400 bad_request")
        return _Result(pd.DataFrame({"market": m, "time": [NOW], "bid_price": ["130.4"], "ask_price": ["130.6"],
                                     "bid_size": ["1.5"], "ask_size": ["2.5"]}))


@pytest.fixture
def provider(monkeypatch):
    coinbase = spot_markets_for("btc")[0]
    prov = CoinMetricsProvider(FakeClient(coinbase))
    monkeypatch.setattr(intraday_tools._factory, "get_provider", lambda: prov)
    return prov


# --- provider -------------------------------------------------------------

def test_intraday_candles_normalises_columns_and_names_market(provider):
    df = provider.get_intraday_candles("btc", frequency="5m", lookback_minutes=150)
    assert list(df.columns) == ["time", "open", "high", "low", "close", "vwap", "volume", "usd_volume", "trades"]
    assert df.attrs["market"] == spot_markets_for("btc")[0]
    assert df.attrs["frequency"] == "5m"
    assert str(df["time"].dtype).startswith("datetime64[ns, UTC]")
    assert df["close"].is_monotonic_increasing


def test_intraday_candles_rejects_unknown_frequency(provider):
    assert provider.get_intraday_candles("btc", frequency="2m") is None
    assert "1m" in INTRADAY_FREQUENCIES and "1d" not in INTRADAY_FREQUENCIES


def test_intraday_candles_falls_back_to_next_market():
    binance = spot_markets_for("btc")[1]
    prov = CoinMetricsProvider(FakeClient(binance))
    df = prov.get_intraday_candles("btc", frequency="1h", lookback_minutes=600)
    assert df.attrs["market"] == binance
    assert prov._client.calls[0][1] == spot_markets_for("btc")[0]   # coinbase tried first


def test_latest_trade_and_quote(provider):
    t = provider.get_latest_trade("btc")
    assert t["price"] == 130.5 and t["amount"] == 2.0 and t["side"] == "buy" and t["usd"] == 261.0
    q = provider.get_latest_quote("btc")
    assert q["bid"] == 130.4 and q["ask"] == 130.6 and q["mid"] == pytest.approx(130.5)
    assert q["spread_bp"] == pytest.approx(0.2 / 130.5 * 1e4)


def test_recent_trades_adds_usd_and_caps_window(provider):
    df = provider.get_recent_trades("btc", minutes=500)          # capped to MAX_TAPE_MINUTES, still works
    assert list(df.columns) == ["time", "price", "amount", "side", "usd"]
    assert df["usd"].iloc[-1] == pytest.approx(130_500.0)
    assert df.attrs["market"] == spot_markets_for("btc")[0]


# --- tools ----------------------------------------------------------------

def test_get_live_price_names_market_timestamp_and_changes(provider):
    out = intraday_tools.get_live_price.invoke({"token": "btc"})
    assert "BTC live (Coinbase BTC-USD spot)" in out
    assert "last trade: $130.50" in out and "buy" in out
    assert "bid / ask: $130.40 / $130.60" in out
    assert "change: 1h" in out and "24h" in out
    assert "UTC" in out and "not an index" in out


def test_get_live_price_unknown_token(provider):
    assert intraday_tools.get_live_price.invoke({"token": "notatoken"}).startswith("Unknown token")


def test_get_intraday_candles_tool_summary_and_table(provider):
    out = intraday_tools.get_intraday_candles.invoke({"token": "btc", "frequency": "5m", "lookback_minutes": 150})
    assert "BTC 5m candles" in out and "30 bars" in out
    assert "| time (UTC) | open | high | low | close | usd volume |" in out
    assert "volume $30.00M" in out
    assert "Coinbase BTC-USD spot" in out


def test_get_intraday_candles_tool_bad_frequency(provider):
    out = intraday_tools.get_intraday_candles.invoke({"token": "btc", "frequency": "2m"})
    assert out.startswith("Unsupported frequency")


def test_get_recent_trades_tool_buy_sell_split_and_big_prints(provider):
    out = intraday_tools.get_recent_trades.invoke({"token": "btc", "minutes": 5, "min_trade_usd": 60_000})
    assert "5 trades" in out
    assert "taker buys" in out and "taker sells" in out
    assert "largest prints" in out
    assert "| " in out and "130.50" in out          # the 1000 x 130.5 print is listed
    assert "sell | 1 |" not in out                     # small prints are not listed


def test_get_recent_trades_tool_no_big_prints_message(provider):
    out = intraday_tools.get_recent_trades.invoke({"token": "btc", "minutes": 5, "min_trade_usd": 10_000_000})
    assert "no single print at or above $10.00M" in out


def test_tools_registered_in_chat_default_tools():
    from chat import default_tools
    names = {t.name for t in default_tools()}
    assert {"get_live_price", "get_intraday_candles", "get_recent_trades"} <= names
