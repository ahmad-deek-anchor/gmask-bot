#!/usr/bin/env python
"""One-off: copy the local SQLite `snapshots` rows into the shared BigQuery table.

    python scripts/migrate_snapshots_to_bq.py                     # data/memory.db -> SNAPSHOT_BQ_TABLE
    python scripts/migrate_snapshots_to_bq.py --db other.db --sources haruko,sheet --dry-run

Idempotent: rows go through BigQuerySnapshotBackend.put_snapshots (MERGE on the primary
key) and keep their original captured_at. Prints per-source counts: local rows, rows sent,
and the BigQuery count afterwards.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("migrate_snapshots")

DEFAULT_DB = "data/memory.db"
SELECT_SQL = ("SELECT snapshot_date, source, entity, metric, value, value_json, captured_at "
              "FROM snapshots ORDER BY snapshot_date, source, entity, metric")


def _captured(v) -> Optional[datetime]:
    if v is None or v == "":
        return None
    try:
        dt = datetime.fromisoformat(str(v))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def read_sqlite_rows(db_path: str, sources: Optional[Sequence[str]] = None) -> List[dict]:
    """All snapshot rows of a memory database as put_snapshots() dicts."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"{db_path} does not exist")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(SELECT_SQL).fetchall()
    finally:
        conn.close()
    wanted = {s.lower() for s in sources} if sources else None
    out = []
    for d, source, entity, metric, value, value_json, captured_at in rows:
        if wanted is not None and source not in wanted:
            continue
        out.append({"snapshot_date": d, "source": source, "entity": entity, "metric": metric,
                    "value": value, "value_json": value_json, "captured_at": _captured(captured_at)})
    return out


def migrate(rows: Sequence[dict], backend, batch_size: int = 500, dry_run: bool = False) -> Dict[str, dict]:
    """Send rows source by source; returns {source: {local, sent, bigquery}}."""
    by_source: Dict[str, List[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    report: Dict[str, dict] = {}
    for source in sorted(by_source):
        chunk_rows = by_source[source]
        sent = 0
        if not dry_run:
            for i in range(0, len(chunk_rows), batch_size):
                sent += int(backend.put_snapshots(chunk_rows[i:i + batch_size]))
        after = None if dry_run else backend.count_snapshots(source)
        report[source] = {"local": len(chunk_rows), "sent": sent, "bigquery": after}
        logger.info("%s: %d local rows, %d sent, %s in BigQuery", source, len(chunk_rows), sent,
                    "n/a (dry run)" if after is None else after)
    return report


def _default_backend(table: Optional[str]):
    from providers.snapshot_bq import BigQuerySnapshotBackend

    return BigQuerySnapshotBackend(table=table)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=DEFAULT_DB, help=f"SQLite memory database (default {DEFAULT_DB})")
    p.add_argument("--table", default=None, help="Target table (default env SNAPSHOT_BQ_TABLE)")
    p.add_argument("--sources", default="", help="Comma-separated subset of sources (default: all)")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--dry-run", action="store_true", help="Count rows, write nothing")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None, backend_factory: Optional[Callable] = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()] or None
    try:
        rows = read_sqlite_rows(args.db, sources)
    except (FileNotFoundError, sqlite3.Error) as e:
        print(f"Cannot read {args.db}: {e}", file=sys.stderr)
        return 2
    if not rows:
        print(f"No snapshot rows in {args.db}" + (f" for sources {', '.join(sources)}" if sources else ""))
        return 0
    backend = (backend_factory or _default_backend)(args.table)
    report = migrate(rows, backend, batch_size=max(1, args.batch_size), dry_run=args.dry_run)
    print(f"Migrated {args.db} -> {getattr(backend, 'table', '?')}{' (dry run)' if args.dry_run else ''}")
    for source, c in report.items():
        print(f"- {source}: {c['local']} local rows, {c['sent']} sent, "
              f"{'n/a' if c['bigquery'] is None else c['bigquery']} now in BigQuery")
    print(f"Total {sum(c['local'] for c in report.values())} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
