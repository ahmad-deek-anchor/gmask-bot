"""Z-score signal pipeline on synthetic data. No network."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tools.metrics import (
    INSIGNIFICANT_THRESHOLD,
    OUTLIER_THRESHOLD,
    calculate_zscore_with_weekend_separation,
    fetch_token_metrics,
)
from tools.signals import calculate_statistical_signals

END = datetime(2026, 9, 10)  # a Thursday


def _frame(days=45, seed=0, **overrides) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = pd.date_range(end=END, periods=days, freq="D")
    df = pd.DataFrame({
        "time": time,
        "price": 100 + rng.normal(0, 0.5, days),
        "spot_volume": 1_000_000 + rng.normal(0, 20_000, days),
        "perp_volume": 2_000_000 + rng.normal(0, 40_000, days),
        "perp_oi": 5_000_000 + rng.normal(0, 50_000, days),
        "funding_rate": np.full(days, 0.10),
        "long_liquidations": 10_000 + rng.normal(0, 500, days),
        "short_liquidations": 10_000 + rng.normal(0, 500, days),
    })
    df["total_liquidations"] = df["long_liquidations"] + df["short_liquidations"]
    # Pin the last day of every z-scored metric to the median of its history so a
    # "calm" frame is deterministic (z ~ 0) regardless of the random draw.
    for col in ("spot_volume", "perp_volume", "perp_oi", "total_liquidations"):
        df.loc[df.index[-1], col] = df[col].iloc[:-1].median()
    df["price_pct_change"] = df["price"].pct_change() * 100
    for col, fn in overrides.items():
        df[col] = fn(df[col])
    return df


def _spike_last(factor):
    def _apply(s):
        s = s.copy()
        s.iloc[-1] *= factor
        return s
    return _apply


# ----------------------------------------------------------------------------
# calculate_zscore_with_weekend_separation
# ----------------------------------------------------------------------------

def test_zscore_plain_window_flags_spike():
    df = _frame(perp_oi=_spike_last(3))
    z = calculate_zscore_with_weekend_separation(df["perp_oi"], df["time"], window=30, separate_weekends=False)
    assert z.iloc[:9].isna().all()  # rolling min_periods=10 -> first value at index 9
    assert z.iloc[-1] > OUTLIER_THRESHOLD
    assert abs(z.iloc[-2]) < OUTLIER_THRESHOLD


def test_zscore_weekend_separation_uses_same_day_type_history():
    # Weekends at half volume: without separation Saturday looks depressed;
    # with separation it is compared against previous weekends and is normal.
    df = _frame(days=60)
    is_weekend = df["time"].dt.dayofweek >= 5
    df.loc[is_weekend, "spot_volume"] *= 0.5
    sat_idx = df.index[is_weekend][-1]
    # pin the target weekend day to the weekend median: normal vs weekends, depressed vs all days
    df.loc[sat_idx, "spot_volume"] = df.loc[is_weekend, "spot_volume"].iloc[:-1].median()

    z_sep = calculate_zscore_with_weekend_separation(df["spot_volume"], df["time"], separate_weekends=True)
    z_plain = calculate_zscore_with_weekend_separation(df["spot_volume"], df["time"], separate_weekends=False)

    assert abs(z_sep.loc[sat_idx]) < INSIGNIFICANT_THRESHOLD
    assert z_plain.loc[sat_idx] < -INSIGNIFICANT_THRESHOLD


def test_zscore_constant_series_is_zero_not_nan():
    df = _frame()
    const = pd.Series(np.full(len(df), 42.0))
    z = calculate_zscore_with_weekend_separation(const, df["time"], separate_weekends=True)
    assert (z.iloc[10:] == 0).all()


# ----------------------------------------------------------------------------
# calculate_statistical_signals
# ----------------------------------------------------------------------------

def test_calm_token_has_no_signals():
    signals = calculate_statistical_signals({"eth": _frame(seed=1)}, window=30)
    s = signals["eth"]
    assert s["has_outliers"] is False
    assert s["has_significant_moves"] is False
    assert s["latest_date"] == "2026-09-10"
    for m in ("spot_volume", "perp_volume", "perp_oi", "total_liquidations"):
        assert m in s["metrics"]
        assert s["metrics"][m]["z_score"] is not None
        assert s["metrics"][m]["is_outlier"] is False
    # funding_rate column is already an annualised % (0.10 % p.a.); no re-scaling
    assert s["metrics"]["funding_rate"]["value_annual_pct"] == pytest.approx(0.10)
    assert s["metrics"]["funding_rate"]["value_8h_pct"] == pytest.approx(0.10 / 365 / 3)
    assert s["metrics"]["price"]["value"] == pytest.approx(float(_frame(seed=1)["price"].iloc[-1]))


def test_outlier_spike_is_flagged():
    df = _frame(perp_oi=_spike_last(10), perp_volume=_spike_last(1.5))
    s = calculate_statistical_signals({"btc": df}, window=30)["btc"]
    assert s["has_outliers"] is True
    assert s["has_significant_moves"] is True
    oi = s["metrics"]["perp_oi"]
    assert oi["is_outlier"] and oi["is_significant"]
    assert oi["z_score"] >= OUTLIER_THRESHOLD
    assert oi["value"] == pytest.approx(float(df["perp_oi"].iloc[-1]))


def test_significant_but_not_outlier():
    df = _frame(seed=3)
    # push last spot volume to ~1.5 sigma above the median
    sigma = df["spot_volume"].iloc[:-1].std()
    df.loc[df.index[-1], "spot_volume"] = df["spot_volume"].iloc[:-1].median() + 1.6 * sigma
    s = calculate_statistical_signals({"sol": df}, window=30)["sol"]
    sv = s["metrics"]["spot_volume"]
    assert sv["is_significant"] is True
    assert sv["is_outlier"] is False
    assert s["has_significant_moves"] is True
    assert s["has_outliers"] is False


def test_missing_metric_columns_are_skipped():
    df = _frame().drop(columns=["perp_oi", "total_liquidations"])
    df["funding_rate"] = np.nan
    s = calculate_statistical_signals({"x": df}, window=30)["x"]
    assert "perp_oi" not in s["metrics"]
    assert "total_liquidations" not in s["metrics"]
    assert "funding_rate" not in s["metrics"]  # all-NaN funding is dropped
    assert "spot_volume" in s["metrics"]


def test_empty_and_short_frames():
    assert calculate_statistical_signals({"x": pd.DataFrame()}) == {}
    short = _frame(days=8)
    s = calculate_statistical_signals({"y": short})["y"]
    # fewer than 10 observations -> no z-score metrics, but price still reported
    assert not any(k in s["metrics"] for k in ("spot_volume", "perp_volume", "perp_oi", "total_liquidations"))
    assert "price" in s["metrics"]


def test_uses_last_valid_observation_when_latest_is_nan():
    df = _frame(perp_oi=_spike_last(10))
    df.loc[df.index[-1], "spot_volume"] = np.nan
    s = calculate_statistical_signals({"btc": df})["btc"]
    assert s["metrics"]["spot_volume"]["value"] == pytest.approx(float(df["spot_volume"].iloc[-2]))


# ----------------------------------------------------------------------------
# fetch_token_metrics with a fake provider (checks merge + None handling)
# ----------------------------------------------------------------------------

FETCH_END = datetime(2026, 8, 31)  # safely in the past: no "today" row to drop


class FakeProvider:
    def __init__(self, days=45, end=FETCH_END):
        self.time = pd.date_range(end=end, periods=days, freq="D")
        self.n = days

    def get_spot_price(self, token, s, e):
        if token == "nodata":
            return None
        return pd.DataFrame({"time": self.time, "price": np.linspace(100, 110, self.n)})

    def get_spot_ohlcv(self, token, s, e):
        return pd.DataFrame({
            "time": self.time, "open": 1, "high": 1, "low": 1, "close": 1,
            "spot_volume": np.full(self.n, 5.0),
        })

    def get_perp_volume(self, token, s, e):
        return pd.DataFrame({"time": self.time, "perp_volume": np.full(self.n, 7.0)})

    def get_perp_oi(self, token, s, e):
        return None  # provider has no OI for this token

    def get_funding_rate(self, token, s, e):
        # shorter series: only the last 5 days
        return pd.DataFrame({"time": self.time[-5:], "funding_rate": np.full(5, 0.05)})

    def get_liquidations(self, token, s, e):
        return pd.DataFrame({
            "time": self.time,
            "long_liquidations": 1.0, "short_liquidations": 2.0, "total_liquidations": 3.0,
        })


def test_fetch_token_metrics_merges_and_fills(monkeypatch):
    import tools.metrics as metrics

    monkeypatch.setattr(metrics, "get_provider", lambda: FakeProvider())
    out = fetch_token_metrics(["BTC", "nodata"], datetime(2026, 7, 17), FETCH_END)

    assert list(out) == ["btc"]  # lowercased; token without price skipped
    df = out["btc"]
    expected_cols = {
        "time", "price", "price_pct_change", "perp_volume", "perp_oi", "funding_rate",
        "spot_volume", "long_liquidations", "short_liquidations", "total_liquidations",
    }
    assert expected_cols <= set(df.columns)
    assert len(df) == 45
    assert df["perp_oi"].isna().all()                # None -> NaN column
    assert df["funding_rate"].notna().sum() == 5     # partial series left-joined
    assert (df["spot_volume"] == 5.0).all()
    assert (df["total_liquidations"] == 3.0).all()
    assert df["time"].is_monotonic_increasing
    assert df["price_pct_change"].iloc[1] == pytest.approx((110 - 100) / 44 / 100 * 100, rel=1e-6)


def test_funding_rate_uses_last_non_nan_and_annualised_percent_units():
    """Amberdata funding_rate is annualised % (10.95 == 0.01 %/8h). The last row may be NaN
    (partial join); the latest available observation must be used, un-rescaled."""
    df = _frame(seed=2)
    df["funding_rate"] = 10.95
    df.loc[df.index[-1], "funding_rate"] = np.nan
    s = calculate_statistical_signals({"btc": df}, window=30)["btc"]
    fr = s["metrics"]["funding_rate"]
    assert fr["value_annual_pct"] == pytest.approx(10.95)
    assert fr["value_8h_pct"] == pytest.approx(0.01)


def test_fetch_token_metrics_drops_partial_current_day(monkeypatch):
    """A row for today's (incomplete) UTC day must not enter the z-score history."""
    import tools.metrics as metrics

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    provider = FakeProvider(days=20, end=today)
    monkeypatch.setattr(metrics, "get_provider", lambda: provider)
    df = fetch_token_metrics(["btc"], today - timedelta(days=19), today)["btc"]
    assert len(df) == 19
    assert df["time"].max() == pd.Timestamp(today - timedelta(days=1))


def test_fetch_token_metrics_joins_tz_aware_and_intraday_timestamps(monkeypatch):
    """Spot (tz-naive daily) and derivatives (tz-aware / intraday) must join on the UTC day."""
    import tools.metrics as metrics

    class TzProvider(FakeProvider):
        def get_perp_oi(self, token, s, e):
            t = pd.to_datetime(self.time, utc=True) + timedelta(hours=8)  # tz-aware, 08:00Z
            return pd.DataFrame({"time": t, "perp_oi": np.full(self.n, 11.0)})

    monkeypatch.setattr(metrics, "get_provider", lambda: TzProvider())
    df = fetch_token_metrics(["btc"], datetime(2026, 7, 17), FETCH_END)["btc"]
    assert df["perp_oi"].notna().all()
    assert (df["perp_oi"] == 11.0).all()
    assert str(df["time"].dtype) == "datetime64[ns]"
    for c in ("price", "perp_oi", "spot_volume", "funding_rate", "total_liquidations"):
        assert df[c].dtype == np.float64


def test_fetch_token_metrics_one_failing_metric_does_not_drop_token(monkeypatch):
    import tools.metrics as metrics

    class Boom(FakeProvider):
        def get_liquidations(self, token, s, e):
            raise RuntimeError("HTTP 500")

    monkeypatch.setattr(metrics, "get_provider", lambda: Boom())
    out = fetch_token_metrics(["btc"], datetime(2026, 7, 17), FETCH_END)
    assert "btc" in out
    assert out["btc"]["total_liquidations"].isna().all()
    assert (out["btc"]["spot_volume"] == 5.0).all()
