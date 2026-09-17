"""tools/movers_tools.py against a fake Coin Metrics client and a fake universe. No network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from providers.coinmetrics import CoinMetricsProvider
from tools import movers_tools as mv

NOW = datetime(2026, 9, 17, 14, 0, tzinfo=timezone.utc)
PRICES = {"btc": 76000.0, "eth": 2400.0, "sol": 100.0, "hype": 79.5, "near": 3.0, "zec": 1370.0, "drv": 1.2, "dead": None}
CHG24 = {"btc": 0.3, "eth": 0.6, "sol": 2.1, "hype": 0.2, "near": 15.6, "zec": 10.9, "drv": 32.0}
CHG7 = {"btc": -2.4, "eth": -1.7, "sol": -1.8, "hype": -4.5, "near": 16.3, "zec": 9.1, "drv": 40.0}
VOLX = {"btc": 1.0, "eth": 1.0, "sol": 1.1, "hype": 0.9, "near": 2.0, "zec": 4.0, "drv": 3.0}


class _Result:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


class FakeClient:
    def get_market_candles(self, markets, frequency="1h", start_time=None, **kw):
        (m,) = markets
        sym = m.split("-")[1].split("_")[0]
        if PRICES.get(sym) is None:
            raise RuntimeError("400 market not supported")
        hours = 8 * 24 + 2
        times = [NOW - timedelta(hours=h) for h in range(hours, 0, -1)]
        last = PRICES[sym]
        p24 = last / (1 + CHG24[sym] / 100)
        p7 = last / (1 + CHG7[sym] / 100)
        closes = []
        for t in times:
            age = (NOW - timedelta(hours=1) - t).total_seconds() / 3600
            closes.append(last if age <= 0 else p24 if age <= 24 else p7)
        base_vol = 1_000_000.0
        vols = [base_vol * (VOLX[sym] if (NOW - timedelta(hours=1) - t).total_seconds() / 3600 < 24 else 1.0) for t in times]
        return _Result(pd.DataFrame({"market": m, "time": times, "price_open": closes, "price_high": closes, "price_low": closes,
                                     "price_close": closes, "vwap": closes, "volume": 1.0, "candle_usd_volume": vols, "candle_trades_count": 10}))


class FakeUniverse:
    def top_assets(self, n=100, classifier=None, **kw):
        syms = list(PRICES)
        caps = [1.5e12, 3e11, 5e10, 2e10, 3.5e9, 1.9e10, 4e8, 1e8]
        df = pd.DataFrame({"rank": range(1, len(syms) + 1), "asset": syms, "symbol": syms, "market_cap": caps,
                           "spot_volume": 1e9, "as_of": "2026-09-16"})
        df["sector"] = [["Networks"], ["Networks"], ["Others"], ["DeFi"], ["Others"], ["Others"], ["DeFi"], ["Meme"]]
        return df.head(n)

    def resolve(self, s):
        return s


@pytest.fixture
def wired(monkeypatch):
    prov = CoinMetricsProvider(FakeClient(), universe=FakeUniverse())
    monkeypatch.setattr(mv._factory, "get_provider", lambda: prov)
    monkeypatch.setattr(mv._factory, "get_universe", lambda: prov.universe)
    monkeypatch.setattr("tools.chat_tools._sector_classifier", lambda: None)
    mv._memo.clear()
    return prov


def test_movers_frame_changes_volume_ratio_and_errors(wired):
    df = mv.movers_frame(100, now=NOW)
    r = df.set_index("symbol")
    assert r.loc["btc", "chg_24h"] == pytest.approx(0.3, abs=0.01) and r.loc["btc", "chg_7d"] == pytest.approx(-2.4, abs=0.01)
    assert r.loc["near", "chg_24h"] == pytest.approx(15.6, abs=0.01) and r.loc["zec", "vol_ratio"] == pytest.approx(4.0, abs=0.05)
    assert r.loc["dead", "error"] == "no candles" or r.loc["dead", "error"] == "RuntimeError"
    assert r.loc["btc", "sector1"] == "Networks"
    again = mv.movers_frame(100, now=NOW)
    assert len(again) == len(df)                                              # memoised


def test_get_top_movers_tool(wired):
    out = mv.get_top_movers.invoke({"n": 100})
    assert out.startswith("### Movers across the top 8 assets, as of 2026-09-17 13:00 UTC")
    assert "breadth T-24h: 7 up / 0 down" in out
    assert "| 1 | BTC | $76,000 | +0.3% | -2.4% |" in out and "**Majors**" in out
    assert "**Leaders above $1.00B market cap (T-24h)**" in out and "| 5 | NEAR | $3.00 | +15.6% | +16.3% |" in out
    assert "**Smaller names (below $1.00B) moving on unusual volume" in out and "| 7 | DRV |" in out
    assert "no candles for: DEAD" in out and "Sector per Messari" in out


def test_registered_and_in_policy():
    from access.policy import Policy
    from chat import default_tools
    assert "get_top_movers" in {t.name for t in default_tools()}
    assert Policy.load().group_of("get_top_movers") == "live"
