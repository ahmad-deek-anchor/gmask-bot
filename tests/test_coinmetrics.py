"""CoinMetricsProvider against a fake coinmetrics client. No network."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from providers.coinmetrics import CoinMetricsProvider, spot_markets_for

START, END = datetime(2026, 8, 1), datetime(2026, 8, 5)
DAYS = pd.date_range("2026-08-01", "2026-08-05", freq="D", tz="UTC")


class _Result:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


def _candles(market, usd_volume, close):
    n = len(DAYS)
    return pd.DataFrame({
        "market": market, "time": DAYS,
        "price_open": close, "price_high": close + 1, "price_low": close - 1, "price_close": close,
        "vwap": close, "volume": usd_volume / close,            # base units (coins) - must NOT be used
        "candle_usd_volume": np.full(n, usd_volume, dtype=float),
        "candle_trades_count": 10,
    })


class FakeClient:
    """Returns candles for a configurable subset of markets; others raise like the SDK does."""

    def __init__(self, markets: dict, price_usd=None):
        self.markets = markets
        self.price_usd = price_usd
        self.calls = []

    def get_market_candles(self, markets, **kw):
        (m,) = markets
        self.calls.append(m)
        if m not in self.markets:
            raise RuntimeError(f"400 bad_request: market {m} not supported")
        return _Result(self.markets[m])

    def get_asset_metrics(self, assets, metrics, **kw):
        if self.price_usd is None:
            raise RuntimeError("403 forbidden")
        return _Result(pd.DataFrame({"asset": assets[0], "time": DAYS, metrics[0]: self.price_usd}))


def test_spot_volume_is_summed_in_usd_across_all_markets_with_data():
    coinbase, binance, kraken, bybit = spot_markets_for("btc")
    client = FakeClient({
        coinbase: _candles(coinbase, 400e6, 78_000.0),
        binance: _candles(binance, 1_100e6, 78_010.0),
        kraken: _candles(kraken, 180e6, 78_005.0),
        # bybit missing -> raises -> skipped
    })
    df = CoinMetricsProvider(client).get_spot_ohlcv("btc", START, END)

    assert list(df.columns) == ["time", "open", "high", "low", "close", "spot_volume"]
    assert len(df) == len(DAYS)
    assert df["spot_volume"].tolist() == pytest.approx([1_680e6] * len(DAYS))  # USD, summed over 3 markets
    assert (df["close"] == 78_000.0).all()                            # OHLC from the primary market
    assert str(df["time"].dtype) == "datetime64[ns]"                  # tz-naive UTC days
    assert df["spot_volume"].dtype == np.float64
    assert client.calls == [coinbase, binance, kraken, bybit]         # every market is tried once


def test_spot_ohlcv_none_when_no_market_returns_data():
    assert CoinMetricsProvider(FakeClient({})).get_spot_ohlcv("btc", START, END) is None


def test_spot_price_prefers_price_usd_then_falls_back_to_candle_close():
    coinbase = spot_markets_for("hype")[0]
    with_price = FakeClient({coinbase: _candles(coinbase, 1e6, 40.0)}, price_usd=41.0)
    df = CoinMetricsProvider(with_price).get_spot_price("hype", START, END)
    assert list(df.columns) == ["time", "price"] and (df["price"] == 41.0).all()

    no_price_usd = FakeClient({coinbase: _candles(coinbase, 1e6, 40.0)})   # PriceUSD -> 403
    df = CoinMetricsProvider(no_price_usd).get_spot_price("hype", START, END)
    assert list(df.columns) == ["time", "price"] and (df["price"] == 40.0).all()

    assert CoinMetricsProvider(FakeClient({})).get_spot_price("hype", START, END) is None
