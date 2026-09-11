"""@tool functions over the daily `snapshots` table written by snapshot_daily.py
(read-only; storage is providers.memory_store via providers.factory.get_memory_store()).

Sources / entities / metrics (see snapshot_daily.py for the full list):

* ``haruko`` - desk risk from Haruko. Entities ``combined`` (default), ``20`` (A1 Ltd;
  aliases a1, "a1 ltd"), ``86`` (ADSD; alias adsd). Metrics delta_usd,
  delta_adjusted_usd, gamma_usd, gamma_pct_usd, vega, theta, day_pnl, ytd_pnl,
  gross_notional, equity, valid_pricer_pct, data_quality_flag (1 = Normal).
* ``signals`` - market metrics per token (entity = lowercase token, required).
  ``<metric>`` is the latest value, ``<metric>_z`` its z-score (spot_volume,
  perp_volume, perp_oi, total_liquidations, dvol_close, atm_iv_30d, pcr_oi,
  options_notional_volume, options_block_notional_volume), ``skew_25d_30d`` /
  ``pcr_volume_24h`` (+ ``_chg7d``), ``price``, ``price_pct_change_1d``,
  ``funding_rate`` (annualised %).
* ``sheet`` - spot desk PnL sheet. Entities ``TOTAL`` (default), ``HOLD``, ``A1``.
  Metrics mtd_volume_usd, mtd_pnl_usd, mtd_take_rate_bps, ytd_volume_usd, ytd_pnl_usd,
  ytd_take_rate_bps, ytd_target_pnl_usd, ytd_pct_of_target (TOTAL only), week_pnl_usd,
  week_realized_pnl_usd / week_unrealized_pnl_usd (A1 only).

Every tool returns compact markdown and states the snapshot dates it used. Snapshots
are taken once a day (23:30 UTC timer); the ``signals`` values are as of the last
complete UTC day before the snapshot.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.tools import tool

from tools.desk_tools import _md_table, _usd

logger = logging.getLogger(__name__)

SOURCES = ("haruko", "signals", "sheet")
DEFAULT_ENTITY = {"haruko": "combined", "signals": None, "sheet": "TOTAL"}
HARUKO_ALIASES = {
    "combined": "combined", "all": "combined", "desk": "combined", "total": "combined",
    "20": "20", "a1": "20", "a1 ltd": "20", "a1ltd": "20", "a1_ltd": "20",
    "86": "86", "adsd": "86", "swap dealer": "86",
}
HARUKO_LABELS = {"20": "A1 Ltd (entity 20)", "86": "ADSD (entity 86)", "combined": "combined desk"}
MAX_DAYS = 3650
DEFAULT_DAYS = 30
FALLBACK_DAYS = 31   # compare_to_snapshot: how far before the requested date to look for a snapshot

UNAVAILABLE = ("Snapshot store is not available (providers.memory_store / MEMORY_DB_URL). "
               "Run `python snapshot_daily.py` to start collecting daily snapshots.")

_USD_HINTS = ("usd", "pnl", "notional", "equity", "delta", "gamma", "vega", "theta",
              "volume", "price", "liquidations", "oi", "target")


# ---------------------------------------------------------------------------
# store access + normalisation (tolerant of the store's row representation)
# ---------------------------------------------------------------------------

def _get_store():
    """MemoryStore or None (tests monkeypatch this)."""
    try:
        from providers.factory import get_memory_store
        return get_memory_store()
    except Exception as e:  # noqa: BLE001
        logger.warning("memory store unavailable: %s", e)
        return None


def _guarded(name: str, body: Callable) -> str:
    store = _get_store()
    if store is None:
        return UNAVAILABLE
    try:
        return body(store)
    except Exception as e:  # noqa: BLE001 - surface to the model, never crash the agent
        logger.error("%s failed: %s", name, e, exc_info=logger.isEnabledFor(logging.DEBUG))
        return f"Error reading snapshots in {name}: {type(e).__name__}: {e}"


def _to_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v)[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _pick(obj, *names, index: Optional[int] = None):
    if isinstance(obj, dict):
        for n in names:
            if n in obj:
                return obj[n]
        return None
    if isinstance(obj, (tuple, list)):
        return obj[index] if index is not None and index < len(obj) else None
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _norm_row(r) -> Optional[Tuple[date, Optional[float], Optional[dict]]]:
    """(snapshot_date, value, value_json as dict) from a dict / tuple / row object."""
    d = _to_date(_pick(r, "snapshot_date", "date", index=0))
    if d is None:
        return None
    v = _pick(r, "value", index=1)
    try:
        v = None if v is None else float(v)
    except (TypeError, ValueError):
        v = None
    vj = _pick(r, "value_json", "json", index=2)
    if isinstance(vj, str):
        try:
            vj = json.loads(vj)
        except ValueError:
            vj = {"raw": vj}
    elif vj is not None and not isinstance(vj, dict):
        vj = None
    return d, v, vj


def _series(store, source: str, entity: str, metric: str, days: int) -> List[Tuple[date, Optional[float], Optional[dict]]]:
    raw = store.get_snapshot_series(source, entity, metric, days) or []
    rows = [x for x in (_norm_row(r) for r in raw) if x is not None]
    rows.sort(key=lambda x: x[0])
    return rows


def _metrics(store, source: str) -> Dict[str, List[str]]:
    """entity -> sorted metric names, from whatever list_snapshot_metrics returns
    (strings, (entity, metric) pairs or dicts)."""
    out: Dict[str, set] = {}
    for item in store.list_snapshot_metrics(source) or []:
        if isinstance(item, str):
            out.setdefault("*", set()).add(item)
        elif isinstance(item, dict):
            out.setdefault(str(item.get("entity", "*")), set()).add(str(item.get("metric")))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            out.setdefault(str(item[0]), set()).add(str(item[1]))
        elif isinstance(item, (tuple, list)) and item:
            out.setdefault("*", set()).add(str(item[0]))
        else:
            e, m = _pick(item, "entity"), _pick(item, "metric")
            if m is not None:
                out.setdefault(str(e) if e is not None else "*", set()).add(str(m))
    return {e: sorted(ms) for e, ms in sorted(out.items())}


# ---------------------------------------------------------------------------
# argument handling / formatting
# ---------------------------------------------------------------------------

def _source(s: Optional[str]) -> Optional[str]:
    s = (s or "").strip().lower()
    return s if s in SOURCES else None


def _entity(source: str, entity: Optional[str]) -> Optional[str]:
    e = (entity or "").strip()
    if not e:
        return DEFAULT_ENTITY[source]
    if source == "haruko":
        return HARUKO_ALIASES.get(e.lower(), e)
    if source == "sheet":
        return e.upper()
    return e.lower()


def _entity_label(source: str, entity: str) -> str:
    if source == "haruko":
        return HARUKO_LABELS.get(entity, entity)
    return entity.upper() if source == "signals" else entity


def _clamp(v, default: int, lo: int, hi: int) -> int:
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = default
    return min(max(v, lo), hi)


def _fmt(metric: str, v: Optional[float], vj: Optional[dict] = None) -> str:
    if metric == "data_quality_flag":
        if vj:
            flags = vj.get("flags") or ({"": vj.get("flag")} if vj.get("flag") is not None else None)
            if flags:
                return "; ".join(f"{k}: {f}" if k else str(f) for k, f in flags.items())
        return "n/a" if v is None else ("Normal" if v >= 1 else "flagged")
    if v is None:
        return "n/a"
    m = metric.lower()
    if m.endswith("_z"):
        return f"{v:+.2f}"
    if m.endswith("_bps"):
        return f"{v:,.2f} bps"
    if "pct" in m or m == "funding_rate":
        return f"{v:,.2f}%"
    if m.startswith("pcr") or m.startswith("skew"):
        return f"{v:,.3f}"
    if any(h in m for h in _USD_HINTS):
        return _usd(v)
    return f"{v:,.4f}" if abs(v) < 1 else f"{v:,.2f}"


def _is_relative(metric: str) -> bool:
    """Metrics already expressed as a score / % / bps: show absolute change only."""
    m = metric.lower()
    return m.endswith(("_z", "_bps")) or "pct" in m or m == "funding_rate"


def _delta(metric: str, a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "n/a"
    d = b - a
    s = _fmt(metric, d)
    if not s.startswith(("-", "+")):
        s = "+" + s
    if a != 0 and not _is_relative(metric):
        s += f" ({d / abs(a) * 100:+.1f}%)"
    return s


def _no_data(source: str, entity: str, metric: str, days: int) -> str:
    return (f"No `{source}` snapshots for entity `{entity}`, metric `{metric}` in the last {days} day(s). "
            f"Use list_snapshot_metrics('{source}') to see what has been captured.")


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

@tool("get_snapshot_history")
def get_snapshot_history(source: str, metric: str, entity: Optional[str] = None, days: int = DEFAULT_DAYS) -> str:
    """Daily history of one snapshotted metric: desk risk from Haruko ('haruko': delta_usd, gamma_usd, vega, theta, day_pnl, ytd_pnl, gross_notional, equity, valid_pricer_pct, data_quality_flag), market signals per token ('signals': price, funding_rate, spot_volume, perp_oi, <metric>_z z-scores ...) or spot desk PnL from the sheet ('sheet': mtd_pnl_usd, ytd_pnl_usd, week_pnl_usd ...).

    Use for "how has the desk delta moved since the first snapshot / over the last
    two weeks", "trend of BTC funding in our snapshots", "spot desk MTD PnL day by
    day". Returns a compact date | value table plus change first -> last and min / max.
    Snapshots are captured once a day by snapshot_daily.py; the series is only as long
    as the collection history.

    Args:
        source: 'haruko' | 'signals' | 'sheet'.
        metric: Snapshot metric name (see list_snapshot_metrics).
        entity: haruko: 'combined' (default), '20' / 'a1' (A1 Ltd), '86' / 'adsd';
            signals: the token, e.g. 'btc' (required); sheet: 'TOTAL' (default), 'HOLD', 'A1'.
        days: Look-back window in days (default 30, max 3650).
    """
    src = _source(source)
    if src is None:
        return f"Unknown source {source!r}; choose from {', '.join(SOURCES)}."
    ent = _entity(src, entity)
    if ent is None:
        return "For source 'signals' pass the token as `entity`, e.g. entity='btc'."
    met = (metric or "").strip()
    n_days = _clamp(days, DEFAULT_DAYS, 1, MAX_DAYS)

    def body(store) -> str:
        rows = _series(store, src, ent, met, n_days)
        if not rows:
            return _no_data(src, ent, met, n_days)
        first_d, first_v, _ = rows[0]
        last_d, last_v, last_j = rows[-1]
        valued = [(d, v) for d, v, _ in rows if v is not None]
        out = [f"**{src} / {_entity_label(src, ent)} / {met}** - {len(rows)} daily snapshot(s), "
               f"{first_d} to {last_d}"]
        if met == "data_quality_flag":
            table = [[str(d), _fmt(met, v, vj)] for d, v, vj in rows]
            out += _md_table(["Date", "Data quality"], table)
            return "\n".join(out)
        out.append(f"Latest ({last_d}): {_fmt(met, last_v, last_j)}; first ({first_d}): {_fmt(met, first_v)}; "
                   f"change {_delta(met, first_v, last_v)}.")
        if valued:
            lo = min(valued, key=lambda x: x[1])
            hi = max(valued, key=lambda x: x[1])
            out.append(f"Min {_fmt(met, lo[1])} on {lo[0]}; max {_fmt(met, hi[1])} on {hi[0]}.")
        if last_j:
            extras = {k: v for k, v in last_j.items() if k in ("as_of", "z_score", "unit", "week", "month", "data_as_of")
                      and v is not None}
            if extras:
                out.append("Latest row context: " + ", ".join(f"{k}={v}" for k, v in extras.items()) + ".")
        table = [[str(d), _fmt(met, v)] for d, v, _ in rows]
        out.append("")
        out += _md_table(["Date", met], table)
        return "\n".join(out)

    return _guarded("get_snapshot_history", body)


@tool("compare_to_snapshot")
def compare_to_snapshot(source: str, metric: str, entity: Optional[str] = None, date: str = "") -> str:
    """Compare the latest snapshotted value of a metric with its value on a given past date ("today vs 2026-09-01", "how much has our gamma changed since the start of the month", "BTC funding now vs two weeks ago in the snapshots").

    If no snapshot exists on that exact date the closest earlier one is used and said
    so. Sources / entities / metrics as in get_snapshot_history.

    Args:
        source: 'haruko' | 'signals' | 'sheet'.
        metric: Snapshot metric name.
        entity: haruko: 'combined' (default) / '20' / '86'; signals: token (required); sheet: 'TOTAL' / 'HOLD' / 'A1'.
        date: Past snapshot date, YYYY-MM-DD.
    """
    src = _source(source)
    if src is None:
        return f"Unknown source {source!r}; choose from {', '.join(SOURCES)}."
    ent = _entity(src, entity)
    if ent is None:
        return "For source 'signals' pass the token as `entity`, e.g. entity='btc'."
    met = (metric or "").strip()
    target = _to_date(date)
    if target is None:
        return f"Bad date {date!r}; expected YYYY-MM-DD."
    today = datetime.now(timezone.utc).date()
    # fetch a little further back so a missing day can fall back to the closest earlier snapshot
    n_days = _clamp((today - target).days + 1 + FALLBACK_DAYS, 1, 1, MAX_DAYS)

    def body(store) -> str:
        rows = _series(store, src, ent, met, n_days)
        if not rows:
            return _no_data(src, ent, met, n_days)
        last_d, last_v, last_j = rows[-1]
        on_or_before = [r for r in rows if r[0] <= target]
        if not on_or_before:
            return (f"No `{src}` snapshot of `{met}` for {_entity_label(src, ent)} on or before {target}; "
                    f"earliest available is {rows[0][0]} ({_fmt(met, rows[0][1])}).")
        base_d, base_v, base_j = on_or_before[-1]
        out = [f"**{src} / {_entity_label(src, ent)} / {met}** - latest {last_d} vs {base_d}"
               + (f" (closest snapshot on or before {target})" if base_d != target else "")]
        if base_d == last_d:
            out.append(f"Only one snapshot in range ({last_d}): {_fmt(met, last_v, last_j)}. Nothing to compare yet.")
            return "\n".join(out)
        out += _md_table(["", "Date", met], [["then", str(base_d), _fmt(met, base_v, base_j)],
                                             ["latest", str(last_d), _fmt(met, last_v, last_j)]])
        if met != "data_quality_flag":
            out.append(f"Change: {_delta(met, base_v, last_v)} over {(last_d - base_d).days} day(s).")
        return "\n".join(out)

    return _guarded("compare_to_snapshot", body)


@tool("list_snapshot_metrics")
def list_snapshot_metrics(source: str = "") -> str:
    """List which snapshot metrics (and entities / tokens) have been captured for a source, and the latest snapshot date. Call this before get_snapshot_history / compare_to_snapshot when unsure of a metric name.

    Args:
        source: 'haruko' | 'signals' | 'sheet'; empty for all three.
    """
    srcs = [_source(source)] if source else list(SOURCES)
    if srcs == [None]:
        return f"Unknown source {source!r}; choose from {', '.join(SOURCES)}."

    def body(store) -> str:
        out: List[str] = []
        for src in srcs:
            latest = _to_date(store.latest_snapshot_date(src))
            by_entity = _metrics(store, src)
            out.append(f"**{src}** - latest snapshot: {latest or 'none yet'}")
            if not by_entity:
                out.append("- no snapshots captured yet (run `python snapshot_daily.py`)")
                continue
            if src == "signals" and len(by_entity) > 1:
                tokens = sorted(e for e in by_entity if e != "*")
                metrics = sorted({m for ms in by_entity.values() for m in ms})
                out.append(f"- tokens ({len(tokens)}): {', '.join(tokens)}")
                out.append(f"- metrics ({len(metrics)}): {', '.join(metrics)}")
            else:
                for ent, ms in by_entity.items():
                    label = "all entities" if ent == "*" else _entity_label(src, ent)
                    out.append(f"- {label}: {', '.join(ms)}")
        return "\n".join(out)

    return _guarded("list_snapshot_metrics", body)


def get_snapshot_tools() -> list:
    """Read-only snapshot tools for the chat agent (register via chat.default_tools())."""
    return [get_snapshot_history, compare_to_snapshot, list_snapshot_metrics]


SNAPSHOT_TOOL_NAMES = [t.name for t in get_snapshot_tools()]

__all__ = ["get_snapshot_tools", "SNAPSHOT_TOOL_NAMES", "get_snapshot_history",
           "compare_to_snapshot", "list_snapshot_metrics"]
