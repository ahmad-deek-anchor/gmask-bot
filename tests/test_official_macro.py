"""providers.official_macro (US Treasury curve, CBOE VIX) and their tools against a fake session. No network."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import providers.official_macro as om
from providers.official_macro import CboeVix, OfficialFeedError, TreasuryCurve
from tools import macro_tools as mac

TODAY = date(2026, 9, 17)


def _entry(d: date, base: float) -> str:
    fields = {"BC_1MONTH": base - 1.0, "BC_2MONTH": base - 0.9, "BC_3MONTH": base - 0.85, "BC_4MONTH": base - 0.75,
              "BC_6MONTH": base - 0.8, "BC_1YEAR": base - 0.55, "BC_2YEAR": base - 0.25, "BC_3YEAR": base - 0.2,
              "BC_5YEAR": base - 0.15, "BC_7YEAR": base - 0.05, "BC_10YEAR": base, "BC_20YEAR": base + 0.38, "BC_30YEAR": base + 0.35}
    props = "".join(f"<d:{k}>{v:.2f}</d:{k}>" for k, v in fields.items())
    return (f"<entry><content type='application/xml'><m:properties><d:Id>1</d:Id><d:NEW_DATE>{d.isoformat()}T00:00:00</d:NEW_DATE>"
            f"{props}<d:BC_30YEARDISPLAY>{fields['BC_30YEAR']:.2f}</d:BC_30YEARDISPLAY></m:properties></content></entry>")


def _month_xml(month: str) -> str:
    y, m = int(month[:4]), int(month[4:])
    entries = []
    d = date(y, m, 1)
    i = 0
    while d.month == m and d <= TODAY - timedelta(days=1):
        if d.weekday() < 5:
            # 10y climbs 1 bp per business day: 4.50 on 2026-08-03 -> ~4.82 on 09-16
            days_since = sum(1 for k in range((d - date(2026, 8, 3)).days) if (date(2026, 8, 3) + timedelta(days=k)).weekday() < 5)
            entries.append(_entry(d, 4.50 + 0.01 * days_since))
            i += 1
        d += timedelta(days=1)
    return ("<?xml version='1.0' encoding='utf-8'?><feed xmlns='http://www.w3.org/2005/Atom' "
            "xmlns:m='http://schemas.microsoft.com/ado/2007/08/dataservices/metadata' "
            "xmlns:d='http://schemas.microsoft.com/ado/2007/08/dataservices'>" + "".join(entries) + "</feed>")


def _vix_csv() -> str:
    lines = ["DATE,OPEN,HIGH,LOW,CLOSE"]
    d = date(2025, 9, 1)
    v = 15.0
    while d <= TODAY - timedelta(days=1):
        if d.weekday() < 5:
            v = 15.0 + ((d.toordinal() * 7) % 11)          # deterministic wobble 15..25
            lines.append(f"{d.strftime('%m/%d/%Y')},{v - 0.3:.6f},{v + 0.5:.6f},{v - 0.6:.6f},{v:.6f}")
        d += timedelta(days=1)
    return "\n".join(lines) + "\n"


class FakeResponse:
    def __init__(self, status, text):
        self.status_code, self.text = status, text


class FakeSession:
    def __init__(self, fail=False):
        self.headers = {}
        self.calls = []
        self.fail = fail

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if self.fail:
            return FakeResponse(503, "down")
        if url == om.TREASURY_URL:
            return FakeResponse(200, _month_xml(params["field_tdr_date_value_month"]))
        if url == om.CBOE_VIX_URL:
            return FakeResponse(200, _vix_csv())
        return FakeResponse(404, "nope")


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def feeds(session, monkeypatch):
    t, c = TreasuryCurve(session=session), CboeVix(session=session)
    monkeypatch.setattr(mac, "_treasury_feed", lambda: t)
    monkeypatch.setattr(mac, "_cboe_feed", lambda: c)
    return t, c


def test_months_back():
    assert om._months_back(date(2026, 9, 17), 30) == ["202609", "202608"]
    assert om._months_back(date(2026, 9, 17), 5) == ["202609"]
    assert om._months_back(date(2026, 3, 2), 60) == ["202603", "202602", "202601"]


def test_treasury_curve_parses_months_and_memoises(feeds, session):
    t, _ = feeds
    df = t.curve(days=30, today=TODAY)
    assert list(df.columns) == ["date"] + om.TENOR_ORDER
    assert df["date"].iloc[-1] == pd.Timestamp("2026-09-16") and df["date"].min() >= pd.Timestamp("2026-08-18")
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
    assert df.iloc[-1]["10y"] == pytest.approx(4.82) and df.iloc[-1]["2y"] == pytest.approx(4.57)
    months = [p["field_tdr_date_value_month"] for u, p in session.calls if u == om.TREASURY_URL]
    assert months == ["202609", "202608"]
    t.curve(days=30, today=TODAY)
    assert len(session.calls) == 2                                          # memoised
    with pytest.raises(OfficialFeedError):
        TreasuryCurve(session=FakeSession(fail=True)).curve(days=10, today=TODAY)


def test_vix_history_parses_and_windows(feeds, session):
    _, c = feeds
    df = c.history(days=30, today=TODAY)
    assert list(df.columns) == ["date", "open", "high", "low", "close"]
    assert df["date"].iloc[-1] == pd.Timestamp("2026-09-16") and len(df) >= 20
    year = c.history(days=365, today=TODAY)
    assert len(year) > 240 and sum(1 for u, _ in session.calls if u == om.CBOE_VIX_URL) == 1   # one download, memoised


def test_get_treasury_curve_tool(feeds):
    out = mac.get_treasury_curve.invoke({"days": 30})
    assert out.startswith("### US Treasury par yield curve as of 2026-09-16 (US Treasury daily publication)")
    assert "| 10y | 4.82% | +1 bp | +5 bp | +20 bp |" in out
    assert "| 2y | 4.57% |" in out and "| 30y | 5.17% |" in out
    assert "2s10s +25 bp (+0 bp on the day)" in out and "3m10y +85 bp" in out and "5s30s +50 bp" in out
    assert "| Date | 2y | 10y | 30y |" in out and "US Department of the Treasury" in out


def test_get_vix_history_tool(feeds):
    out = mac.get_vix_history.invoke({"days": 30})
    assert out.startswith("### VIX as of 2026-09-16 close (CBOE)")
    assert "on the day" in out and "vs one week ago" in out and "vs one month ago" in out
    assert "one-year context: the current close is above" in out and "1y range 15.00 to 25.00" in out
    assert "| Date | Open | High | Low | Close |" in out and "CBOE VIX history file" in out


def test_tools_report_feed_errors(monkeypatch):
    bad = FakeSession(fail=True)
    monkeypatch.setattr(mac, "_treasury_feed", lambda: TreasuryCurve(session=bad))
    monkeypatch.setattr(mac, "_cboe_feed", lambda: CboeVix(session=bad))
    assert mac.get_treasury_curve.invoke({}).startswith("US Treasury feed error: HTTP 503") and "FRED DGS2" in mac.get_treasury_curve.invoke({})
    assert mac.get_vix_history.invoke({}).startswith("CBOE feed error: HTTP 503")


def test_registered_and_in_policy():
    from access.policy import Policy
    from chat import default_tools
    names = {t.name for t in default_tools()}
    assert {"get_treasury_curve", "get_vix_history"} <= names
    assert Policy.load().group_of("get_treasury_curve") == "macro" and Policy.load().group_of("get_vix_history") == "macro"
