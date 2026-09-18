"""The derivatives desk = three Haruko portfolios. Every desk tool filters to them.

Ahmad Deek's standing rule (2026-09-18): any question about the derivatives desk
("derivs", "derivatives", desk PnL / risk / positions / greeks / funding / fees) is
answered for exactly these Haruko portfolios and nothing else:

    portfolio (strategy_name)   Haruko venue account / BigQuery entity_id   legal entity
    Derivs Risk                 20                                          A1 Ltd
    ADSD                        86                                          Anchorage Digital Swap Dealer, LLC
    AD Hedge Co                 87                                          AD Hedge Co

Other Haruko venue accounts (counterparty books, HRP, spot accounts) belong to other
businesses and distort the totals, so they are excluded by default rather than on
request. The BigQuery export (``anc-global-markets.brokerage_a1.fct_otc_haruko_*``)
carries entities 20 and 86 only as of 2026-09-18; account 87 has no rows there yet,
which the tools say in their scope footer.

Overrides (comma separated, for tests or a future change of book):

    DESK_PORTFOLIOS   default "Derivs Risk,ADSD,AD Hedge Co"   Haruko strategy_name values
    DESK_ENTITY_IDS   default "20,86,87"                        BigQuery entity_id values

The SQL helpers return plain literals built from these constants (never from user
input), so they stay inside the read-only guard's rules.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Tuple

DEFAULT_PORTFOLIOS: Tuple[str, ...] = ("Derivs Risk", "ADSD", "AD Hedge Co")
DEFAULT_ENTITY_IDS: Tuple[int, ...] = (20, 86, 87)

# entity_id -> label printed by the tools (portfolio first, legal entity in brackets)
ENTITY_NAMES: Dict[int, str] = {
    20: "Derivs Risk (A1 Ltd)",
    86: "ADSD (Anchorage Digital Swap Dealer)",
    87: "AD Hedge Co",
}
ENTITY_PORTFOLIO: Dict[int, str] = {20: "Derivs Risk", 86: "ADSD", 87: "AD Hedge Co"}

# entities present in the BigQuery export (verified 2026-09-18); the rest are named in the footer
BQ_EXPORTED_ENTITY_IDS: Tuple[int, ...] = (20, 86)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _./&-]{0,63}$")
_OTC_ENTITY_UUID = "00000000-0000-0000-0000-{:012d}"     # dim_otcderivatives_entities.id shape


def _csv(name: str) -> Tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(v.strip() for v in raw.split(",") if v.strip())


def portfolios() -> Tuple[str, ...]:
    """Haruko strategy / portfolio names in scope (env ``DESK_PORTFOLIOS`` or the default three)."""
    vals = _csv("DESK_PORTFOLIOS") or DEFAULT_PORTFOLIOS
    bad = [v for v in vals if not _NAME_RE.match(v)]
    if bad:
        raise ValueError(f"DESK_PORTFOLIOS contains an invalid portfolio name: {bad!r}")
    return tuple(vals)


def entity_ids() -> Tuple[int, ...]:
    """BigQuery ``entity_id`` values in scope (env ``DESK_ENTITY_IDS`` or 20, 86, 87)."""
    vals = _csv("DESK_ENTITY_IDS")
    if not vals:
        return DEFAULT_ENTITY_IDS
    try:
        return tuple(int(v) for v in vals)
    except ValueError as e:
        raise ValueError(f"DESK_ENTITY_IDS must be comma-separated integers: {e}") from e


def entity_name(eid) -> str:
    try:
        eid = int(eid)
    except (TypeError, ValueError):
        return f"entity {eid}"
    return ENTITY_NAMES.get(eid, f"entity {eid}")


# -- SQL fragments (literals from the constants above, never from user input) -----------

def entity_sql(column: str = "entity_id") -> str:
    """``entity_id IN (20, 86, 87)`` for the portfolio / greeks / position tables."""
    return f"{column} IN ({', '.join(str(int(e)) for e in entity_ids())})"


def strategy_sql(column: str = "strategy_name") -> str:
    """``strategy_name IN ('Derivs Risk', 'ADSD', 'AD Hedge Co')`` for position-level tables."""
    return f"{column} IN ({', '.join(_lit(p) for p in portfolios())})"


def otc_entity_sql(column: str = "t.entity_id") -> str:
    """Entity filter for ``fct_otcderivatives_trades`` whose ``entity_id`` is a UUID ending in the entity number."""
    return f"{column} IN ({', '.join(_lit(_OTC_ENTITY_UUID.format(int(e))) for e in entity_ids())})"


def strategies_literal(sep: str = "|") -> str:
    """The portfolios joined for a single-string DECLARE (``sql/haruko_eod_pnl.sql`` splits it)."""
    return sep.join(portfolios())


def _lit(s: str) -> str:
    return "'" + s.replace("'", "\\'") + "'"


# -- wording ------------------------------------------------------------------------------

def scope_label() -> str:
    """'Derivs Risk, ADSD and AD Hedge Co' - for headings."""
    p = list(portfolios())
    return p[0] if len(p) == 1 else ", ".join(p[:-1]) + " and " + p[-1]


def scope_note() -> str:
    """One-line footer every desk tool appends so the answer states its scope."""
    ids = ", ".join(str(e) for e in entity_ids())
    missing = [ENTITY_PORTFOLIO.get(e, str(e)) for e in entity_ids() if e not in BQ_EXPORTED_ENTITY_IDS]
    note = f"_Scope: {scope_label()} portfolios only (Haruko entities {ids}); other Haruko accounts are excluded by default."
    if missing:
        note += f" {', '.join(missing)} has no rows in the BigQuery export yet, so figures cover the other portfolios."
    return note + "_"


__all__ = ["BQ_EXPORTED_ENTITY_IDS", "DEFAULT_ENTITY_IDS", "DEFAULT_PORTFOLIOS", "ENTITY_NAMES", "ENTITY_PORTFOLIO",
           "entity_ids", "entity_name", "entity_sql", "otc_entity_sql", "portfolios", "scope_label", "scope_note",
           "strategies_literal", "strategy_sql"]
