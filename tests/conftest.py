"""Test-suite wide environment.

The developer .env sets SNAPSHOT_BACKEND=bigquery for the bot; providers.memory_store loads
.env at import, so pin the suite to the sqlite snapshot backend before any test module
imports it. Tests that exercise the BigQuery backend construct it explicitly with a fake
client (tests/test_snapshot_bq.py) or monkeypatch the variable themselves.
"""

import os

os.environ["SNAPSHOT_BACKEND"] = "sqlite"
