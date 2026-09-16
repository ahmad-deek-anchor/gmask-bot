"""Test-suite wide environment.

The developer .env sets SNAPSHOT_BACKEND=bigquery for the bot; providers.memory_store loads
.env at import, so pin the suite to the sqlite snapshot backend before any test module
imports it. Tests that exercise the BigQuery backend construct it explicitly with a fake
client (tests/test_snapshot_bq.py) or monkeypatch the variable themselves.
"""

import os
import tempfile

os.environ["SNAPSHOT_BACKEND"] = "sqlite"
# Never build a real market-data provider for token resolution inside the suite (see providers.factory.get_universe)
os.environ["UNIVERSE_DISABLED"] = "1"
# providers.universe reads UNIVERSE_CACHE_PATH at import: keep the test suite away from data/universe_cache.json
os.environ.setdefault("UNIVERSE_CACHE_PATH", os.path.join(tempfile.mkdtemp(prefix="universe-test-"), "universe_cache.json"))
# providers.messari reads MESSARI_CACHE_PATH at import: keep the suite away from data/messari_assets_cache.json
os.environ.setdefault("MESSARI_CACHE_PATH", os.path.join(tempfile.mkdtemp(prefix="messari-test-"), "messari_assets_cache.json"))
# The Slack bot tests exercise handle() with arbitrary users / channels: keep gating off unless a test installs
# its own AccessControl (tests/test_access.py), and never touch data/access.db.
os.environ["ACCESS_CONTROL"] = "off"
os.environ.setdefault("ACCESS_BACKEND", "memory")
# Channel replies are prefixed with <@asker> in production (SLACK_TAG_ASKER=1); the existing handle() tests assert
# exact texts, so keep it off here and test the prefix explicitly in tests/test_slack_bot.py.
os.environ["SLACK_TAG_ASKER"] = "0"
