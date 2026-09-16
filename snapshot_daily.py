#!/usr/bin/env python
"""Capture a daily snapshot of desk risk, market signals and spot PnL into the
`snapshots` table of providers.memory_store, so the chat agent can answer "how has X
moved since ..." questions via tools/snapshot_tools.py.

Where the rows go is the store's SNAPSHOT_BACKEND (.env): `sqlite` = the memory database
(MEMORY_DB_URL, default sqlite:///data/memory.db); `bigquery` = the shared table
SNAPSHOT_BQ_TABLE. In production the `signals` source runs as a Cloud Run job
(deploy/deploy_snapshot_job.sh, 23:30 UTC) writing to BigQuery, and the local systemd
timer runs `--sources haruko,sheet` (those need the user's own credentials).
Only the stdlib is imported at module level; each source imports what it needs when it
runs, so a machine without Sheets/desk-BigQuery access can still run `--sources signals`.

Sources (each isolated: one failing never stops the others):

    haruko   latest fct_otc_haruko_pnl_portfolio row per entity via
             providers.factory.get_desk_bigquery() (same query shape as
             tools.desk_tools.get_desk_risk_snapshot). Entities "20" (A1 Ltd),
             "86" (ADSD) and "combined" (sum). Metrics: delta_usd,
             delta_adjusted_usd, gamma_usd, gamma_pct_usd, vega, theta, day_pnl,
             ytd_pnl, gross_notional, equity, valid_pricer_pct, data_quality_flag
             (value 1.0 = Normal / 0.0 = flagged; value_json carries the text).
    signals  every token in tools.metrics.FULL_TOKEN_UNIVERSE: latest value and
             z-score of every metric from tools.signals.calculate_statistical_signals
             (metric `<m>` = value, `<m>_z` = z-score; options level metrics get
             `<m>_chg7d`), plus price / price_pct_change_1d / funding_rate.
    sheet    providers.gsheets A1 Metrics Dashboard: per entity HOLD / A1 / TOTAL
             the current month (mtd_*) and YTD (ytd_*) volume, PnL and take rate,
             plus the latest weekly PnL row (week_*).
    etf      Messari / Blockworks crypto ETF table (providers.messari.etf_assets):
             per underlying (entity "bitcoin", "ethereum", "solana", "xrp", "multi-asset")
             spot / futures AUM, latest published flow, product counts, regional AUM
             and flow, volume; null flows are skipped (not yet published, never 0).
             Plus entity "bitcoin" Coin Metrics on-chain ETF metrics onchain_flow_in_usd,
             onchain_flow_out_usd, onchain_net_flow_usd, etf_supply_btc, etf_supply_usd.
    cme      Coin Metrics CME futures (providers.coinmetrics.cme_curve) for btc, eth,
             sol, xrp: per contract (entity = symbol, e.g. "BTCZ6") close, oi_contracts,
             oi_usd, volume_usd, basis_ann_pct, days_to_expiry; per underlying (entity
             = "btc") cme_oi_usd, cme_volume_usd, front_basis_ann_pct, next_basis_ann_pct,
             spot_ref (value_json names the front contract and the spot market).

Examples
--------
    python snapshot_daily.py                         # all sources, today's UTC date
    python snapshot_daily.py --sources haruko,sheet  # subset
    python snapshot_daily.py --sources etf,cme       # vendor tables (Cloud Run job runs signals,etf,cme)
    python snapshot_daily.py --date 2026-09-10       # store under another date
    python snapshot_daily.py --dry-run -v            # print rows, write nothing

Idempotent: rows are upserted on (snapshot_date, source, entity, metric). Exit code is
non-zero only when every requested source fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("snapshot_daily")

SOURCES = ("haruko", "signals", "sheet", "etf", "cme")
SIGNALS_DAYS = 45
CME_SNAPSHOT_BASES = ("btc", "eth", "sol", "xrp")
ETF_ASSET_METRICS = ("spot_aum_usd", "spot_flow_usd", "spot_products", "futures_aum_usd", "futures_flow_usd",
                     "futures_products", "us_spot_aum_usd", "us_spot_flow_usd", "us_spot_products",
                     "europe_spot_aum_usd", "europe_spot_flow_usd", "apac_spot_aum_usd", "apac_spot_flow_usd",
                     "deltaone_aum_usd", "leveraged_aum_usd", "total_volume_usd")

# ---------------------------------------------------------------------------
# haruko (BigQuery) column mapping
# ---------------------------------------------------------------------------

# snapshot metric -> fct_otc_haruko_pnl_portfolio column
HARUKO_METRICS = {
    "delta_usd": "total_delta_usd",
    "delta_adjusted_usd": "total_delta_adjusted_usd",
    "gamma_usd": "total_gamma_usd",
    "gamma_pct_usd": "total_gamma_percent_usd",
    "vega": "total_vega",
    "theta": "total_theta",
    "day_pnl": "total_portfolio_pnl",
    "ytd_pnl": "total_year_to_date_pnl",
    "gross_notional": "total_abs_size_usd",
    "equity": "total_equity_usd",
}
HARUKO_SUMMED = tuple(HARUKO_METRICS)              # combined = sum over entities
HARUKO_EXTRA_COLS = ("position_timestamp", "entity_id", "valid_pricer_pct", "valid_pricer_count",
                     "invalid_pricer_count", "data_quality_flag")
HARUKO_ENTITIES = {20: "20", 86: "86"}             # entity_id -> snapshot entity key
COMBINED = "combined"
NORMAL_FLAG = "normal"

# sheet entities as the dashboard names them
SHEET_ENTITIES = ("HOLD", "A1", "TOTAL")


@dataclass
class Row:
    entity: str
    metric: str
    value: Optional[float]
    value_json: Optional[str] = None


@dataclass
class SourceResult:
    source: str
    rows: List[Row] = field(default_factory=list)
    calls: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    seconds: float = 0.0
    written: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.rows)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _num(v) -> Optional[float]:
    """Python float or None for NaN / None / non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _iso(v) -> Optional[str]:
    if v is None:
        return None
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))


def today_utc() -> date:
    return datetime.now(timezone.utc).date()


def parse_date(s: Optional[str]) -> date:
    if not s:
        return today_utc()
    return datetime.strptime(s, "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# source: haruko
# ---------------------------------------------------------------------------

def latest_portfolio_sql(bq, columns: Sequence[str]) -> str:
    """Latest fct_otc_haruko_pnl_portfolio row per entity (last 3 days) - the query
    shape used by tools.desk_tools.get_desk_risk_snapshot, restricted to `columns`."""
    from tools.desk_tools import PORTFOLIO
    return f"""
SELECT {", ".join(columns)}
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY position_timestamp DESC) AS rn
  FROM {bq.table(PORTFOLIO)}
  WHERE position_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 DAY)
)
WHERE rn = 1
ORDER BY entity_id
LIMIT 10"""


def _get_bq():
    from providers.factory import get_desk_bigquery
    return get_desk_bigquery()


def capture_haruko(bq=None) -> SourceResult:
    res = SourceResult("haruko")
    bq = bq if bq is not None else _get_bq()
    if bq is None:
        raise RuntimeError("Desk data (BigQuery) unavailable: no client / ADC")
    cols = list(HARUKO_EXTRA_COLS) + list(dict.fromkeys(HARUKO_METRICS.values()))
    df = bq.query(latest_portfolio_sql(bq, cols))
    res.calls["bigquery_queries"] = 1
    if df is None or df.empty:
        raise RuntimeError("No portfolio snapshot in the last 3 days in fct_otc_haruko_pnl_portfolio")

    sums: Dict[str, float] = {m: 0.0 for m in HARUKO_SUMMED}
    flags: Dict[str, Any] = {}
    valid_n = invalid_n = 0.0
    pcts: List[float] = []
    as_ofs: List[str] = []
    for _, r in df.iterrows():
        try:
            eid = int(r["entity_id"])
        except (TypeError, ValueError):
            continue
        entity = HARUKO_ENTITIES.get(eid, str(eid))
        as_of = _iso(r.get("position_timestamp"))
        as_ofs.append(as_of or "")
        for metric, col in HARUKO_METRICS.items():
            v = _num(r.get(col))
            if v is None:
                continue
            sums[metric] += v
            res.rows.append(Row(entity, metric, v, _dumps({"as_of": as_of})))
        pct = _num(r.get("valid_pricer_pct"))
        flag = r.get("data_quality_flag")
        flag = None if flag is None or (isinstance(flag, float) and math.isnan(flag)) else str(flag)
        flags[entity] = flag
        if pct is not None:
            pcts.append(pct)
            res.rows.append(Row(entity, "valid_pricer_pct", pct, _dumps({"as_of": as_of})))
        vc, ic = _num(r.get("valid_pricer_count")), _num(r.get("invalid_pricer_count"))
        if vc is not None and ic is not None:
            valid_n += vc
            invalid_n += ic
        res.rows.append(Row(entity, "data_quality_flag", _flag_value(flag),
                            _dumps({"flag": flag, "valid_pricer_pct": pct, "as_of": as_of})))

    if len(flags) > 1:
        as_of = max(as_ofs) or None
        for metric in HARUKO_SUMMED:
            res.rows.append(Row(COMBINED, metric, sums[metric], _dumps({"as_of": as_of, "entities": sorted(flags)})))
        if valid_n + invalid_n > 0:
            comb_pct: Optional[float] = valid_n / (valid_n + invalid_n) * 100.0
        elif pcts:
            comb_pct = sum(pcts) / len(pcts)
        else:
            comb_pct = None
        if comb_pct is not None:
            res.rows.append(Row(COMBINED, "valid_pricer_pct", comb_pct, _dumps({"as_of": as_of})))
        all_normal = all((f or "").strip().lower() == NORMAL_FLAG for f in flags.values())
        res.rows.append(Row(COMBINED, "data_quality_flag", 1.0 if all_normal else 0.0,
                            _dumps({"flags": flags, "valid_pricer_pct": comb_pct, "as_of": as_of})))
    return res


def _flag_value(flag: Optional[str]) -> float:
    return 1.0 if (flag or "").strip().lower() == NORMAL_FLAG else 0.0


# ---------------------------------------------------------------------------
# source: signals
# ---------------------------------------------------------------------------

def capture_signals(tokens: Optional[Sequence[str]] = None, days: int = SIGNALS_DAYS,
                    fetch: Optional[Callable] = None, calc: Optional[Callable] = None,
                    include_options: bool = True) -> SourceResult:
    from tools.metrics import FULL_TOKEN_UNIVERSE
    if fetch is None or calc is None:
        from tools.metrics import fetch_token_metrics
        from tools.signals import calculate_statistical_signals
        fetch = fetch or fetch_token_metrics
        calc = calc or calculate_statistical_signals

    res = SourceResult("signals")
    tokens = [t.lower() for t in (tokens or FULL_TOKEN_UNIVERSE)]
    end = datetime.now()
    start = end - timedelta(days=days)
    data = fetch(tokens, start, end, include_options=include_options)
    res.calls["tokens_requested"] = len(tokens)
    res.calls["tokens_with_data"] = len(data)
    signals = calc(data)
    res.calls["tokens_with_signals"] = len(signals)

    for token, ts in signals.items():
        as_of = ts.get("latest_date")
        for metric, entry in (ts.get("metrics") or {}).items():
            if not isinstance(entry, dict):
                continue
            if "z_score" in entry:
                v, z = _num(entry.get("value")), _num(entry.get("z_score"))
                if v is not None:
                    res.rows.append(Row(token, metric, v, _dumps({
                        "as_of": as_of, "z_score": z,
                        "is_outlier": bool(entry.get("is_outlier")),
                        "is_significant": bool(entry.get("is_significant"))})))
                if z is not None:
                    res.rows.append(Row(token, f"{metric}_z", z, _dumps({"as_of": as_of})))
            elif metric == "funding_rate":
                v = _num(entry.get("value_annual_pct"))
                if v is not None:
                    res.rows.append(Row(token, "funding_rate", v, _dumps({
                        "as_of": as_of, "unit": "annualised %", "value_8h_pct": _num(entry.get("value_8h_pct"))})))
            elif metric == "price":
                v, chg = _num(entry.get("value")), _num(entry.get("pct_change_1d"))
                if v is not None:
                    res.rows.append(Row(token, "price", v, _dumps({"as_of": as_of})))
                if chg is not None:
                    res.rows.append(Row(token, "price_pct_change_1d", chg, _dumps({"as_of": as_of})))
            elif "change_7d" in entry:
                v, chg = _num(entry.get("value")), _num(entry.get("change_7d"))
                if v is not None:
                    res.rows.append(Row(token, metric, v, _dumps({"as_of": as_of})))
                if chg is not None:
                    res.rows.append(Row(token, f"{metric}_chg7d", chg, _dumps({"as_of": as_of})))
            else:
                v = _num(entry.get("value"))
                if v is not None:
                    res.rows.append(Row(token, metric, v, _dumps({"as_of": as_of})))
    return res


# ---------------------------------------------------------------------------
# source: sheet
# ---------------------------------------------------------------------------

def _get_sheet():
    from providers.factory import get_a1_metrics_sheet
    return get_a1_metrics_sheet()


def capture_sheet(sheet=None, today: Optional[date] = None) -> SourceResult:
    import pandas as pd

    res = SourceResult("sheet")
    sheet = sheet if sheet is not None else _get_sheet()
    if sheet is None:
        raise RuntimeError("Spot desk sheet unavailable: google-auth / ADC missing")
    today = today or today_utc()
    errors: List[str] = []

    # --- monthly (MTD + YTD) -------------------------------------------------
    try:
        df = sheet.monthly_volume_pnl()
        res.calls["sheet_ranges"] = res.calls.get("sheet_ranges", 0) + 1
        meta = {"tab": df.attrs.get("tab"), "data_as_of": df.attrs.get("data_as_of")}
        if df.empty:
            errors.append("monthly: no rows")
        else:
            months = df[~df["is_ytd"]]
            cur = months[months["month"] <= today.month] if len(months) else months
            if cur.empty:
                cur = months
            if len(cur):
                m = int(cur["month"].max())
                mdf = cur[cur["month"] == m].set_index("entity")
                for ent in SHEET_ENTITIES:
                    if ent not in mdf.index:
                        continue
                    r = mdf.loc[ent]
                    mj = _dumps({**meta, "month": m, "year": _num(r.get("year"))})
                    for metric, col in (("mtd_volume_usd", "volume_usd"), ("mtd_pnl_usd", "pnl_usd"),
                                        ("mtd_take_rate_bps", "take_rate_bps")):
                        v = _num(r.get(col))
                        if v is not None:
                            res.rows.append(Row(ent, metric, v, mj))
            ytd = df[df["is_ytd"]].set_index("entity") if df["is_ytd"].any() else pd.DataFrame()
            yj = _dumps({**meta, "basis": "sheet YTD row" if len(ytd) else "sum of populated months"})
            for ent in SHEET_ENTITIES:
                if len(ytd) and ent in ytd.index:
                    r = ytd.loc[ent]
                    vol, pnl, tr = _num(r.get("volume_usd")), _num(r.get("pnl_usd")), _num(r.get("take_rate_bps"))
                    target, pct = _num(r.get("target_pnl_usd")), _num(r.get("pct_of_target"))
                else:
                    sub = months[months["entity"] == ent]
                    if sub.empty:
                        continue
                    vol, pnl = _num(sub["volume_usd"].sum()), _num(sub["pnl_usd"].sum())
                    tr = pnl / vol * 1e4 if vol and pnl is not None else None
                    target = pct = None
                for metric, v in (("ytd_volume_usd", vol), ("ytd_pnl_usd", pnl), ("ytd_take_rate_bps", tr),
                                  ("ytd_target_pnl_usd", target), ("ytd_pct_of_target", pct)):
                    if v is not None:
                        res.rows.append(Row(ent, metric, v, yj))
    except Exception as e:  # noqa: BLE001 - weekly may still work
        errors.append(f"monthly: {type(e).__name__}: {e}")

    # --- latest weekly row ---------------------------------------------------
    try:
        wk = sheet.weekly_pnl()
        res.calls["sheet_ranges"] = res.calls.get("sheet_ranges", 0) + 1
        weeks = wk[~wk["is_total"]] if len(wk) else wk
        if weeks.empty:
            errors.append("weekly: no rows")
        else:
            last = weeks.sort_values("week").iloc[-1]
            wj = _dumps({"tab": wk.attrs.get("tab"), "data_as_of": wk.attrs.get("data_as_of"),
                         "week": int(last["week"]), "start_date": _iso(last.get("start_date")),
                         "end_date": _iso(last.get("end_date"))})
            for ent, metric, col in (("HOLD", "week_pnl_usd", "hold_pnl"),
                                     ("A1", "week_pnl_usd", "a1_total"),
                                     ("A1", "week_realized_pnl_usd", "a1_realized"),
                                     ("A1", "week_unrealized_pnl_usd", "a1_unrealized"),
                                     ("TOTAL", "week_pnl_usd", "total")):
                v = _num(last.get(col))
                if v is not None:
                    res.rows.append(Row(ent, metric, v, wj))
    except Exception as e:  # noqa: BLE001
        errors.append(f"weekly: {type(e).__name__}: {e}")

    if errors and not res.rows:
        raise RuntimeError("; ".join(errors))
    if errors:
        logger.warning("sheet: partial capture (%s)", "; ".join(errors))
    return res


# ---------------------------------------------------------------------------
# source: etf (Messari / Blockworks + Coin Metrics on-chain)
# ---------------------------------------------------------------------------

def _get_messari():
    from providers.factory import get_messari_provider
    return get_messari_provider()


def _get_spot():
    """The Coin Metrics provider (composite .spot)."""
    from providers.factory import get_provider
    provider = get_provider()
    return getattr(provider, "spot", provider)


def capture_etf(messari=None, spot=None) -> SourceResult:
    import pandas as pd

    res = SourceResult("etf")
    messari = messari if messari is not None else _get_messari()
    errors: List[str] = []

    # --- Messari / Blockworks per-asset table ---------------------------------
    if messari is None:
        errors.append("messari: provider unavailable (no API key)")
    else:
        try:
            df = messari.etf_assets()
            res.calls["messari_requests"] = res.calls.get("messari_requests", 0) + 1
            if df is None:
                errors.append("messari: etf_assets returned nothing")
            else:
                for _, r in df.iterrows():
                    ent = str(r.get("id") or r.get("slug") or "").strip().lower()
                    if not ent:
                        continue
                    as_of = r.get("as_of")
                    vj = _dumps({"as_of": _iso(as_of) if not pd.isna(as_of) else None, "vendor": "Messari / Blockworks Research",
                                 "note": "flows missing when not yet published"})
                    for metric in ETF_ASSET_METRICS:
                        v = _num(r.get(metric)) if metric in df.columns else None
                        if v is not None:
                            res.rows.append(Row(ent, metric, v, vj))
        except Exception as e:  # noqa: BLE001
            errors.append(f"messari: {type(e).__name__}: {e}")

    # --- Coin Metrics on-chain BTC ETF metrics ---------------------------------
    try:
        spot = spot if spot is not None else _get_spot()
        oc = spot.etf_onchain_flows("btc", days=4)
        res.calls["coinmetrics_requests"] = res.calls.get("coinmetrics_requests", 0) + 2
        if oc is None or oc.empty:
            errors.append("coinmetrics: no on-chain ETF rows")
        else:
            last = oc.iloc[-1]
            vj = _dumps({"as_of": _iso(last["time"]), "vendor": "Coin Metrics on-chain (ETF-labelled addresses)"})
            for metric, col in (("onchain_flow_in_usd", "flow_in_usd"), ("onchain_flow_out_usd", "flow_out_usd"),
                                ("onchain_net_flow_usd", "net_flow_usd")):
                v = _num(last.get(col))
                if v is not None:
                    res.rows.append(Row("bitcoin", metric, v, vj))
            sup = oc.dropna(subset=["supply_btc"])
            if len(sup):
                s_last = sup.iloc[-1]
                sj = _dumps({"as_of": _iso(s_last["time"]), "vendor": "Coin Metrics on-chain (ETF-labelled addresses)"})
                for metric, col in (("etf_supply_btc", "supply_btc"), ("etf_supply_usd", "supply_usd")):
                    v = _num(s_last.get(col))
                    if v is not None:
                        res.rows.append(Row("bitcoin", metric, v, sj))
    except Exception as e:  # noqa: BLE001
        errors.append(f"coinmetrics: {type(e).__name__}: {e}")

    if errors and not res.rows:
        raise RuntimeError("; ".join(errors))
    if errors:
        logger.warning("etf: partial capture (%s)", "; ".join(errors))
    return res


# ---------------------------------------------------------------------------
# source: cme (Coin Metrics CME futures)
# ---------------------------------------------------------------------------

def capture_cme(spot=None, bases: Sequence[str] = CME_SNAPSHOT_BASES) -> SourceResult:
    import pandas as pd

    res = SourceResult("cme")
    spot = spot if spot is not None else _get_spot()
    errors: List[str] = []
    for base in bases:
        res.calls["curve_calls"] = res.calls.get("curve_calls", 0) + 1
        try:
            curve = spot.cme_curve(base, include_micro=True)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{base}: {type(e).__name__}: {e}")
            continue
        if curve is None or curve.empty:
            errors.append(f"{base}: no contracts with data")
            continue
        spot_ref = curve.attrs.get("spot")
        common = {"base": base, "spot_ref": spot_ref, "spot_market": curve.attrs.get("spot_market"),
                  "spot_time": _iso(curve.attrs.get("spot_time")) if curve.attrs.get("spot_time") is not None else None,
                  "basis": "close / spot last trade - 1, annualised x 365 / days (ACT/365)"}
        for _, r in curve.iterrows():
            cj = _dumps({**common, "as_of": _iso(r.get("close_time")) if not pd.isna(r.get("close_time")) else None,
                         "oi_as_of": _iso(r.get("oi_time")) if not pd.isna(r.get("oi_time")) else None,
                         "expiration": _iso(r.get("expiration")), "product": r.get("product"),
                         "contract_size": _num(r.get("contract_size")), "is_standard": bool(r.get("is_standard"))})
            for metric, col in (("close", "close"), ("oi_contracts", "oi_contracts"), ("oi_usd", "oi_usd"),
                                ("volume_usd", "usd_volume"), ("basis_ann_pct", "basis_ann_pct"),
                                ("days_to_expiry", "days_to_expiry")):
                v = _num(r.get(col))
                if v is not None:
                    res.rows.append(Row(str(r["symbol"]), metric, v, cj))
        std = curve[curve["is_standard"]]
        front = std.iloc[0] if len(std) else None
        nxt = std.iloc[1] if len(std) > 1 else None
        aj = _dumps({**common, "as_of": _iso(curve.attrs.get("oi_as_of")) if curve.attrs.get("oi_as_of") is not None else None,
                     "front_contract": None if front is None else str(front["symbol"]),
                     "next_contract": None if nxt is None else str(nxt["symbol"]),
                     "contracts": [str(x) for x in curve["symbol"].tolist()]})
        for metric, v in (("cme_oi_usd", _num(curve["oi_usd"].sum(min_count=1))),
                          ("cme_volume_usd", _num(curve["usd_volume"].sum(min_count=1))),
                          ("front_basis_ann_pct", None if front is None else _num(front.get("basis_ann_pct"))),
                          ("next_basis_ann_pct", None if nxt is None else _num(nxt.get("basis_ann_pct"))),
                          ("spot_ref", _num(spot_ref))):
            if v is not None:
                res.rows.append(Row(base, metric, v, aj))
    if errors and not res.rows:
        raise RuntimeError("; ".join(errors))
    if errors:
        logger.warning("cme: partial capture (%s)", "; ".join(errors))
    return res


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

CAPTURES: Dict[str, Callable[[], SourceResult]] = {
    "haruko": capture_haruko,
    "signals": capture_signals,
    "sheet": capture_sheet,
    "etf": capture_etf,
    "cme": capture_cme,
}


def _get_store():
    from providers.factory import get_memory_store
    return get_memory_store()


def write_rows(store, snapshot_date: date, source: str, rows: Sequence[Row]) -> int:
    """Upsert a source's rows. One bulk ``put_snapshots`` call when the store offers it (the
    BigQuery backend turns that into a single MERGE per batch); per-row otherwise."""
    if hasattr(store, "put_snapshots"):
        return int(store.put_snapshots([
            {"snapshot_date": snapshot_date, "source": source, "entity": r.entity, "metric": r.metric,
             "value": r.value, "value_json": r.value_json} for r in rows]))
    n = 0
    for r in rows:
        store.put_snapshot(snapshot_date, source, r.entity, r.metric, r.value, value_json=r.value_json)
        n += 1
    return n


def run(sources: Sequence[str], snapshot_date: date, store=None, dry_run: bool = False,
        captures: Optional[Dict[str, Callable[[], SourceResult]]] = None) -> Dict[str, SourceResult]:
    """Capture each source in turn and upsert into the store. Never raises for a
    single source; the summary carries per-source errors, row counts and timings."""
    captures = captures or CAPTURES
    results: Dict[str, SourceResult] = {}
    for source in sources:
        t0 = time.monotonic()
        try:
            res = captures[source]()
            res.source = source
        except Exception as e:  # noqa: BLE001 - isolate sources
            res = SourceResult(source, error=f"{type(e).__name__}: {e}")
            logger.error("%s: capture failed: %s", source, res.error,
                         exc_info=logger.isEnabledFor(logging.DEBUG))
        if res.error is None and not res.rows:
            res.error = "no rows captured"
            logger.error("%s: %s", source, res.error)
        if res.rows and not dry_run:
            try:
                if store is None:
                    store = _get_store()
                res.written = write_rows(store, snapshot_date, source, res.rows)
            except Exception as e:  # noqa: BLE001
                res.error = f"store write failed: {type(e).__name__}: {e}"
                logger.error("%s: %s", source, res.error, exc_info=logger.isEnabledFor(logging.DEBUG))
        res.seconds = time.monotonic() - t0
        results[source] = res
        calls = ", ".join(f"{k}={v}" for k, v in res.calls.items()) or "-"
        logger.info("%s: %d rows%s in %.1fs (calls: %s)%s", source, len(res.rows),
                    "" if dry_run else f", {res.written} written", res.seconds, calls,
                    f" ERROR {res.error}" if res.error else "")
    return results


def summary_lines(results: Dict[str, SourceResult], snapshot_date: date, dry_run: bool) -> List[str]:
    lines = [f"Snapshot {snapshot_date.isoformat()}{' (dry run)' if dry_run else ''}"]
    for source, res in results.items():
        status = "ok" if res.error is None else f"FAILED ({res.error})"
        calls = ", ".join(f"{k}={v}" for k, v in res.calls.items()) or "-"
        lines.append(f"- {source}: {len(res.rows)} rows"
                     + ("" if dry_run else f", {res.written} written")
                     + f", {res.seconds:.1f}s, calls {calls}: {status}")
    return lines


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily data snapshots into the memory store.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--sources", default=",".join(SOURCES),
                   help=f"Comma-separated subset of {','.join(SOURCES)} (default: all)")
    p.add_argument("--date", metavar="YYYY-MM-DD", help="Snapshot date (default: today UTC)")
    p.add_argument("--dry-run", action="store_true", help="Capture and print rows; write nothing")
    p.add_argument("--verbose", "-v", action="store_true", help="INFO logging")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from dotenv import load_dotenv

    load_dotenv()  # SNAPSHOT_BACKEND etc. from .env, like every other entry point
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "google", "cm_client", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)

    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    bad = [s for s in sources if s not in SOURCES]
    if bad or not sources:
        print(f"Unknown source(s): {', '.join(bad) or '(none given)'}; choose from {', '.join(SOURCES)}", file=sys.stderr)
        return 2
    try:
        snapshot_date = parse_date(args.date)
    except ValueError:
        print(f"Bad --date {args.date!r}; expected YYYY-MM-DD", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    results = run(sources, snapshot_date, dry_run=args.dry_run)
    if args.dry_run:
        for source, res in results.items():
            for r in res.rows:
                print(f"{snapshot_date} {source:8s} {r.entity:10s} {r.metric:32s} "
                      f"{'' if r.value is None else f'{r.value:,.6g}':>18s}  {r.value_json or ''}")
    print("\n".join(summary_lines(results, snapshot_date, args.dry_run)))
    print(f"Total {time.monotonic() - t0:.1f}s")
    return 0 if any(r.error is None for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
