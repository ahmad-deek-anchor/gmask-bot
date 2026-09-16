"""providers.fred + tools/macro_tools.py against a fake requests session. No network.

Observations mimic FRED's JSON: daily series with "." placeholders on holidays, a monthly
series, an unknown id (400 error payload) and a 429 that is retried.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

import pandas as pd
import pytest
import requests

import providers.fred as fred
from providers.fred import MACRO_SERIES, FredError, FredProvider
from tools import macro_tools as mac

TODAY = datetime.now(timezone.utc).date()          # the provider windows are relative to the real clock
M0 = TODAY.replace(day=1)
M1 = (M0 - timedelta(days=1)).replace(day=1)
M2 = (M1 - timedelta(days=1)).replace(day=1)
M3 = (M2 - timedelta(days=1)).replace(day=1)


def _daily(start_value: float, step: float, n: int = 70, holidays=()):
    """Business-day observations ending yesterday; holidays appear as '.'."""
    rows = []
    d = TODAY - timedelta(days=1)
    v = start_value + step * n
    while len(rows) < n:
        if d.weekday() < 5:
            rows.append({"date": d.isoformat(), "value": "." if d.isoformat() in holidays else f"{v:.2f}"})
            v -= step
        d -= timedelta(days=1)
    return list(reversed(rows))


SERIES = {
    "DGS10": _daily(4.00, 0.01, holidays=(_holiday := (TODAY - timedelta(days=9)).isoformat(),)),   # one "." placeholder
    "DGS2": _daily(3.50, 0.005),
    "VIXCLS": _daily(15.0, 0.1),
    "SP500": _daily(6000.0, 5.0),
    "DTWEXBGS": _daily(120.0, 0.05),
    "UNRATE": [{"date": M3.isoformat(), "value": "4.1"}, {"date": M2.isoformat(), "value": "4.2"}, {"date": M1.isoformat(), "value": "4.3"}],
    "WALCL": _daily(6_800_000.0, 1000.0, n=10),
}
INFO = {
    "DGS10": {"id": "DGS10", "title": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity", "units": "Percent",
              "units_short": "%", "frequency": "Daily", "frequency_short": "D", "seasonal_adjustment": "Not Seasonally Adjusted",
              "last_updated": "2026-09-16 15:20:01-05", "observation_start": "1962-01-02", "observation_end": "2026-09-15", "notes": "..."},
    "UNRATE": {"id": "UNRATE", "title": "Unemployment Rate", "units": "Percent", "units_short": "%", "frequency": "Monthly",
               "frequency_short": "M", "seasonal_adjustment": "Seasonally Adjusted", "last_updated": "2026-09-05 07:46:02-05",
               "observation_start": "1948-01-01", "observation_end": "2026-08-01", "notes": ""},
    "WALCL": {"id": "WALCL", "title": "Assets: Total Assets: Total Assets (Less Eliminations from Consolidation): Wednesday Level",
              "units": "Millions of U.S. Dollars", "units_short": "Mil. of U.S. $", "frequency": "Weekly, As of Wednesday", "frequency_short": "W",
              "seasonal_adjustment": "Not Seasonally Adjusted", "last_updated": "2026-09-10 16:31:05-05", "observation_start": "2002-12-18",
              "observation_end": "2026-09-09", "notes": ""},
}


class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text if payload is None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self):
        self.calls = []
        self.rate_limit_once = False

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        path = urlparse(url).path
        self.calls.append((path, params))
        assert params.get("api_key") == "fredkey" and params.get("file_type") == "json"
        if self.rate_limit_once:
            self.rate_limit_once = False
            return FakeResponse(429, text="Too Many Requests")
        if path.endswith("/series/observations"):
            sid = params["series_id"]
            if sid not in SERIES:
                return FakeResponse(400, {"error_code": 400, "error_message": "Bad Request.  The series does not exist."})
            start = params.get("observation_start", "1900-01-01")
            return FakeResponse(200, {"observations": [o for o in SERIES[sid] if o["date"] >= start]})
        if path.endswith("/series/search"):
            q = params["search_text"].lower()
            hits = [INFO[k] | {"popularity": 90} for k in INFO if q.split()[0] in INFO[k]["title"].lower()]
            return FakeResponse(200, {"seriess": hits[: int(params.get("limit", 10))]})
        if path.endswith("/series"):
            sid = params["series_id"]
            if sid not in INFO:
                return FakeResponse(400, {"error_code": 400, "error_message": "Bad Request.  The series does not exist."})
            return FakeResponse(200, {"seriess": [INFO[sid]]})
        return FakeResponse(404, text="nope")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(fred.time, "sleep", lambda s: None)


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def provider(session, monkeypatch):
    prov = FredProvider("fredkey", session=session)
    monkeypatch.setattr(mac, "_get_fred", lambda: prov)
    return prov


# --- provider ---------------------------------------------------------------

def test_observations_drop_placeholders_and_memoise(provider, session):
    df = provider.observations("dgs10", days=60)
    assert list(df.columns) == ["date", "value"] and str(df["date"].dtype).startswith("datetime64[ns, UTC]")
    assert _holiday not in df["date"].dt.strftime("%Y-%m-%d").tolist()               # "." dropped
    assert df["value"].is_monotonic_increasing and df.iloc[-1]["value"] == pytest.approx(4.70)
    n = len(session.calls)
    provider.observations("DGS10", days=60)
    assert len(session.calls) == n                                                   # memoised 15 min
    with pytest.raises(FredError) as e:
        provider.observations("NOPE")
    assert "does not exist" in str(e.value)


def test_retry_on_429_and_key_redaction(provider, session, caplog):
    session.rate_limit_once = True
    df = provider.observations("VIXCLS", days=30)
    assert len(df) > 0 and sum(1 for p, _ in session.calls if p.endswith("observations")) == 2
    with caplog.at_level("WARNING"):
        with pytest.raises(FredError):
            provider.series_info("NOPE")
    assert "fredkey" not in caplog.text and "***" in caplog.text


def test_latest_changes_and_monthly_fallback(provider):
    df = provider.latest(["DGS10", "UNRATE", "NOPE"])
    dgs = df.set_index("series_id").loc["DGS10"]
    assert dgs["name"] == "US 10y Treasury yield" and dgs["units"] == "%" and dgs["group"] == "rates"
    assert dgs["value"] == pytest.approx(4.70) and dgs["chg_1"] == pytest.approx(0.01)
    assert dgs["chg_1w"] == pytest.approx(0.05) and dgs["chg_1m"] == pytest.approx(0.20)   # 5 / 20 business days
    un = df.set_index("series_id").loc["UNRATE"]
    assert un["value"] == 4.3 and str(un["date"])[:10] == M1.isoformat() and un["chg_1"] == pytest.approx(0.1)
    assert un["chg_1m"] == pytest.approx(0.1) and pd.isna(un["error"])                  # 400-day fallback for monthly data
    assert un["chg_1w"] is None or pd.isna(un["chg_1w"])                                  # no one-week change for a monthly print
    bad = df.set_index("series_id").loc["NOPE"]
    assert pd.isna(bad["value"]) and "does not exist" in bad["error"]
    full = provider.latest()
    assert full["series_id"].tolist() == list(MACRO_SERIES)


def test_search_and_info(provider):
    df = provider.search("unemployment rate")
    assert df["id"].tolist() == ["UNRATE"] and df.iloc[0]["frequency"] == "M"
    info = provider.series_info("walcl")
    assert info["units_short"] == "Mil. of U.S. $" and info["observation_end"] == "2026-09-09"
    assert provider.search("zzz").empty


# --- tools -------------------------------------------------------------------

def test_tools_unavailable_without_key(monkeypatch):
    monkeypatch.setattr(mac, "_get_fred", lambda: None)
    assert mac.get_macro_snapshot.invoke({}) == mac.UNAVAILABLE
    assert "fred_api_key" in mac.get_fred_series.invoke({"series_id": "DGS10"})


def test_get_macro_snapshot_tool(provider):
    out = mac.get_macro_snapshot.invoke({"groups": "rates,labour"})
    assert out.startswith("### Macro snapshot (FRED")
    assert "**Rates**" in out and "**Labour**" in out and "**Equities**" not in out
    last_bd = SERIES["DGS10"][-1]["date"]
    assert f"| US 10y Treasury yield (DGS10) | 4.70% | {last_bd} | +1 bp | +5 bp | +20 bp |  |" in out
    assert f"| Unemployment rate (monthly) (UNRATE) | 4.30% | {M1.isoformat()} | +10 bp | n/a | +10 bp |  |" in out
    assert "| US 30y Treasury yield (DGS30) | n/a | n/a | n/a | n/a | n/a | error: FRED HTTP 400" in out
    assert "quote the observation date" in out and "GICS" in out
    assert "Unknown group(s) bonds" in mac.get_macro_snapshot.invoke({"groups": "bonds"})
    full = mac.get_macro_snapshot.invoke({})
    assert f"| S&P 500 (SP500) | 6,350 | {last_bd} | +5.00 | +25.00 | +100.00 | 1w +0.40%, 1m +1.60% |" in full


def test_get_fred_series_and_search_tools(provider):
    out = mac.get_fred_series.invoke({"series_id": "dgs10", "days": 30})
    assert out.startswith("### Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity (DGS10), last 30 days")
    assert "units: %; frequency: Daily; seasonal adjustment: Not Seasonally Adjusted; FRED last updated 2026-09-16 15:20" in out
    last_bd = SERIES["DGS10"][-1]["date"]
    assert f"latest 4.70% as of {last_bd}" in out and f"| {last_bd} | 4.70% |" in out and f"| {_holiday} |" not in out
    wal = mac.get_fred_series.invoke({"series_id": "WALCL", "days": 30})
    assert f"latest 6,810,000 as of {last_bd}" in wal and "+9,000 vs" in wal and "units: Mil. of U.S. $" in wal
    assert "FRED error in get_fred_series: FRED HTTP 400" in mac.get_fred_series.invoke({"series_id": "NOPE"})
    assert mac.get_fred_series.invoke({"series_id": ""}).startswith("Give a FRED series id")
    s = mac.search_fred.invoke({"text": "unemployment"})
    assert "| UNRATE | Unemployment Rate | % | M | 2026-09-05 | 2026-08-01 |" in s
    assert "no series matching" in mac.search_fred.invoke({"text": "zzz"})


def test_tools_registered_in_chat_default_tools():
    from chat import default_tools
    names = [t.name for t in default_tools()]
    assert set(mac.MACRO_TOOL_NAMES) <= set(names)
    assert names.index("get_macro_snapshot") > names.index("get_btc_etf_onchain_flows")
