"""Runtime configuration.

Values are read from the environment (a `.env` file is loaded via python-dotenv
on first use). API keys fall back to GCP Secret Manager through
`utils.secrets.get_secret` when the env var is absent. Secret lookups are lazy:
nothing touches GCP until the corresponding attribute is read.

Attributes
----------
COINMETRICS_API_KEY   env COINMETRICS_API_KEY  | secret coinmetrics_trial_api
AMBERDATA_API_KEY     env AMBERDATA_API_KEY    | secret amberdata_key
MESSARI_API_KEY       env MESSARI_API_KEY      | secret messari_api_key2  (Enterprise; news, taxonomy, ETF, Intel, signals)
MESSARI_CACHE_PATH    env MESSARI_CACHE_PATH   (default data/messari_assets_cache.json; 24 h ranked-asset table, read by providers.messari at import)
FRED_API_KEY          env FRED_API_KEY         | secret fred_api_key      (free key from fred.stlouisfed.org; macro series)
SLACK_BOT_TOKEN       env SLACK_BOT_TOKEN      | secret trading_signals_slack_bot_token   (xoxb-)
SLACK_APP_TOKEN       env SLACK_APP_TOKEN      | secret trading_signals_slack_app_token   (xapp-)
SLACK_CHANNEL_ID      env SLACK_CHANNEL_ID     | secret trading_signals_slack_channel_id  (daily post target)
SLACK_WEBHOOK_URL     env SLACK_WEBHOOK_URL    only (legacy incoming webhook; never read from Secret Manager)
GCP_SECRETS_PROJECT   env GCP_SECRETS_PROJECT  (default anchorage-trading-solutions)
VERTEX_PROJECT        env VERTEX_PROJECT       (default anchorage-ai-development)
VERTEX_LOCATION       env VERTEX_LOCATION      (default us-east5)
VERTEX_MODEL          env VERTEX_MODEL         (default claude-sonnet-4-6)

Desk data in BigQuery (providers/bigquery.py, read-only):
BQ_DATA_PROJECT       env BQ_DATA_PROJECT      (default anc-global-markets; where the tables live)
BQ_BILLING_PROJECT    env BQ_BILLING_PROJECT   (default anchorage-corp-eng-playground; jobs run and are billed here)
BQ_ALLOWED_DATASETS   env BQ_ALLOWED_DATASETS  (default "brokerage_a1,pricing"; comma separated)
BQ_MAX_BYTES_BILLED   env BQ_MAX_BYTES_BILLED  (default 20_000_000_000 = 20 GB per query; Carson Levy's
                                               EOW derivatives PnL script scans ~15 GB, 20 GB is ~$0.13 worst case)
BQ_MAX_ROWS           env BQ_MAX_ROWS          (default 200; LIMIT enforced on every query)
BQ_TIMEOUT_S          env BQ_TIMEOUT_S         (default 60)
BQ_CATALOG_PATH       env BQ_CATALOG_PATH      (default data/bq_catalog.json; 24 h table-metadata cache)

Long-term memory + daily snapshots (providers/memory_store.py):
MEMORY_DB_URL         env MEMORY_DB_URL        (default sqlite:///data/memory.db; postgresql://... needs psycopg)
MEMORY_EPISODE_TTL_DAYS env MEMORY_EPISODE_TTL_DAYS (default 90; conversation summaries expire)
MEMORY_EMBEDDINGS     env MEMORY_EMBEDDINGS    (default vertex = text-embedding-005 with keyword fallback; off = keyword only)
SNAPSHOT_BACKEND      env SNAPSHOT_BACKEND     (default sqlite = snapshots table inside MEMORY_DB_URL; bigquery = shared table below)
SNAPSHOT_BQ_TABLE     env SNAPSHOT_BQ_TABLE    (default anchorage-corp-eng-playground.gmask_bot.snapshots; written by the Cloud Run job)

Access control for the Slack bot (access/):
ACCESS_CONTROL        env ACCESS_CONTROL       (default on; off = every Slack user gets every tool)
ACCESS_POLICY_PATH    env ACCESS_POLICY_PATH   (default access/policy.yaml; roles, tool groups, bootstrap admins)
ACCESS_BACKEND        env ACCESS_BACKEND       (default = SNAPSHOT_BACKEND: sqlite locally, bigquery on Cloud Run)
ACCESS_DB_PATH        env ACCESS_DB_PATH       (default data/access.db; sqlite backend)
ACCESS_BQ_TABLE       env ACCESS_BQ_TABLE      (default anchorage-corp-eng-playground.gmask_bot.access)

Spot desk PnL in the "A1 Metrics Dashboard" Google Sheet (providers/gsheets.py, read-only):
A1_METRICS_SHEET_ID   env A1_METRICS_SHEET_ID  (default 1BksNxC2QXHLjFJNCuv-GC9JOBHeb8EyoGwzTuNwqNSY)
GSHEETS_QUOTA_PROJECT env GSHEETS_QUOTA_PROJECT (default anchorage-corp-eng-playground; x-goog-user-project)
GSHEETS_CACHE_TTL_S   env GSHEETS_CACHE_TTL_S  (default 300; in-process cache per range)
"""

from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import Optional

from dotenv import load_dotenv

from utils.net import prefer_ipv4  # noqa: E402

prefer_ipv4()

logger = logging.getLogger(__name__)

DEFAULT_SECRETS_PROJECT = "anchorage-trading-solutions"
DEFAULT_VERTEX_PROJECT = "anchorage-ai-development"
DEFAULT_VERTEX_LOCATION = "us-east5"
DEFAULT_VERTEX_MODEL = "claude-sonnet-4-6"

# Desk data (BigQuery). The user/service account has table data access only on
# these datasets of the data project and no jobs.create there, so queries are
# submitted to (and billed to) BQ_BILLING_PROJECT.
DEFAULT_BQ_DATA_PROJECT = "anc-global-markets"
DEFAULT_BQ_BILLING_PROJECT = "anchorage-corp-eng-playground"
DEFAULT_BQ_ALLOWED_DATASETS = "brokerage_a1,pricing"
DEFAULT_BQ_MAX_BYTES_BILLED = 20_000_000_000  # ~$0.13 worst case; Carson's EOW PnL script ~15 GB
DEFAULT_BQ_MAX_ROWS = 200
DEFAULT_BQ_TIMEOUT_S = 60
DEFAULT_BQ_CATALOG_PATH = "data/bq_catalog.json"

# Long-term memory + daily snapshots (providers/memory_store.py)
DEFAULT_MEMORY_DB_URL = "sqlite:///data/memory.db"
DEFAULT_MEMORY_EPISODE_TTL_DAYS = 90
DEFAULT_MEMORY_EMBEDDINGS = "vertex"
# Daily snapshots: "sqlite" keeps them in the memory database; "bigquery" uses the shared
# table that the Cloud Run job (deploy/deploy_snapshot_job.sh) writes and the bot reads.
DEFAULT_SNAPSHOT_BACKEND = "sqlite"
DEFAULT_SNAPSHOT_BQ_TABLE = "anchorage-corp-eng-playground.gmask_bot.snapshots"
SNAPSHOT_BACKENDS = ("sqlite", "bigquery")

# Spot desk PnL sheet ("A1 Metrics Dashboard", maintained daily by the desk). Read
# with Application Default Credentials (user account, spreadsheets.readonly scope);
# API calls carry x-goog-user-project so quota is charged to the ADC quota project.
DEFAULT_A1_METRICS_SHEET_ID = "1BksNxC2QXHLjFJNCuv-GC9JOBHeb8EyoGwzTuNwqNSY"
DEFAULT_GSHEETS_QUOTA_PROJECT = "anchorage-corp-eng-playground"
DEFAULT_GSHEETS_CACHE_TTL_S = 300

# Secret Manager names (project DEFAULT_SECRETS_PROJECT)
SECRET_COINMETRICS = "coinmetrics_trial_api"
SECRET_AMBERDATA = "amberdata_key"
SECRET_MESSARI = "messari_api_key2"         # messari_api_key (v1) is dead: 403 on every endpoint
SECRET_FRED = "fred_api_key"
SECRET_SLACK_BOT_TOKEN = "trading_signals_slack_bot_token"
SECRET_SLACK_APP_TOKEN = "trading_signals_slack_app_token"
SECRET_SLACK_CHANNEL_ID = "trading_signals_slack_channel_id"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.replace("_", ""))
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; using %s", name, raw, default)
        return default


class Config:

    def __init__(self):
        load_dotenv()

        self.GCP_SECRETS_PROJECT = os.getenv("GCP_SECRETS_PROJECT", DEFAULT_SECRETS_PROJECT)

        # Vertex AI (Claude via Model Garden)
        self.VERTEX_PROJECT = os.getenv("VERTEX_PROJECT", DEFAULT_VERTEX_PROJECT)
        self.VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION)
        self.VERTEX_MODEL = os.getenv("VERTEX_MODEL", DEFAULT_VERTEX_MODEL)

        # Desk data in BigQuery (read-only; see providers/bigquery.py)
        self.BQ_DATA_PROJECT = os.getenv("BQ_DATA_PROJECT", DEFAULT_BQ_DATA_PROJECT)
        self.BQ_BILLING_PROJECT = os.getenv("BQ_BILLING_PROJECT", DEFAULT_BQ_BILLING_PROJECT)
        self.BQ_ALLOWED_DATASETS = tuple(
            d.strip() for d in os.getenv("BQ_ALLOWED_DATASETS", DEFAULT_BQ_ALLOWED_DATASETS).split(",") if d.strip()
        )
        self.BQ_MAX_BYTES_BILLED = _int_env("BQ_MAX_BYTES_BILLED", DEFAULT_BQ_MAX_BYTES_BILLED)
        self.BQ_MAX_ROWS = _int_env("BQ_MAX_ROWS", DEFAULT_BQ_MAX_ROWS)
        self.BQ_TIMEOUT_S = _int_env("BQ_TIMEOUT_S", DEFAULT_BQ_TIMEOUT_S)
        self.BQ_CATALOG_PATH = os.getenv("BQ_CATALOG_PATH", DEFAULT_BQ_CATALOG_PATH)

        # Long-term memory + snapshots (see providers/memory_store.py)
        self.MEMORY_DB_URL = os.getenv("MEMORY_DB_URL", DEFAULT_MEMORY_DB_URL)
        self.MEMORY_EPISODE_TTL_DAYS = _int_env("MEMORY_EPISODE_TTL_DAYS", DEFAULT_MEMORY_EPISODE_TTL_DAYS)
        self.MEMORY_EMBEDDINGS = os.getenv("MEMORY_EMBEDDINGS", DEFAULT_MEMORY_EMBEDDINGS)
        self.SNAPSHOT_BACKEND = (os.getenv("SNAPSHOT_BACKEND") or DEFAULT_SNAPSHOT_BACKEND).strip().lower()
        self.SNAPSHOT_BQ_TABLE = os.getenv("SNAPSHOT_BQ_TABLE") or DEFAULT_SNAPSHOT_BQ_TABLE

        # Spot desk PnL Google Sheet (read-only; see providers/gsheets.py)
        self.A1_METRICS_SHEET_ID = os.getenv("A1_METRICS_SHEET_ID", DEFAULT_A1_METRICS_SHEET_ID)
        self.GSHEETS_QUOTA_PROJECT = os.getenv("GSHEETS_QUOTA_PROJECT", DEFAULT_GSHEETS_QUOTA_PROJECT)
        self.GSHEETS_CACHE_TTL_S = _int_env("GSHEETS_CACHE_TTL_S", DEFAULT_GSHEETS_CACHE_TTL_S)

    # ------------------------------------------------------------------
    # Secrets (env override, then Secret Manager; resolved lazily, cached)
    # ------------------------------------------------------------------

    def _resolve(self, env_var: str, secret_name: str) -> Optional[str]:
        value = os.getenv(env_var)
        if value:
            return value
        try:
            from utils.secrets import get_secret
        except ImportError as e:  # utils.secrets unavailable
            logger.warning(f"Cannot import utils.secrets ({e}); {env_var} unresolved")
            return None
        return get_secret(secret_name, project=self.GCP_SECRETS_PROJECT, env_var=env_var)

    @cached_property
    def COINMETRICS_API_KEY(self) -> Optional[str]:
        return self._resolve("COINMETRICS_API_KEY", SECRET_COINMETRICS)

    @cached_property
    def AMBERDATA_API_KEY(self) -> Optional[str]:
        return self._resolve("AMBERDATA_API_KEY", SECRET_AMBERDATA)

    @cached_property
    def MESSARI_API_KEY(self) -> Optional[str]:
        return self._resolve("MESSARI_API_KEY", SECRET_MESSARI)

    @cached_property
    def FRED_API_KEY(self) -> Optional[str]:
        return self._resolve("FRED_API_KEY", SECRET_FRED)

    # Slack bot (Bolt Socket Mode) + daily post target. Lazy: only read by
    # slack_bot.py and by run_signals.py --post-slack.
    @cached_property
    def SLACK_BOT_TOKEN(self) -> Optional[str]:
        return self._resolve("SLACK_BOT_TOKEN", SECRET_SLACK_BOT_TOKEN)

    @cached_property
    def SLACK_APP_TOKEN(self) -> Optional[str]:
        return self._resolve("SLACK_APP_TOKEN", SECRET_SLACK_APP_TOKEN)

    @cached_property
    def SLACK_CHANNEL_ID(self) -> Optional[str]:
        """Channel for the daily signals post. None until the secret gets a version."""
        return self._resolve("SLACK_CHANNEL_ID", SECRET_SLACK_CHANNEL_ID)

    @cached_property
    def SLACK_WEBHOOK_URL(self) -> Optional[str]:
        """Legacy incoming webhook. Env var only - deliberately never fetched from
        Secret Manager, so --post-slack cannot post anywhere unless the operator
        set SLACK_WEBHOOK_URL explicitly (or the bot token + channel id exist)."""
        return os.getenv("SLACK_WEBHOOK_URL") or None
