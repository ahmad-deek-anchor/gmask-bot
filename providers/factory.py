"""Provider factory.

`get_provider()` returns a process-wide CompositeProvider:

  spot        -> CoinMetricsProvider (Coin Metrics API)
  derivatives -> AmberdataProvider   (Amberdata API)

`get_options_provider()` returns a process-wide AmberdataOptionsProvider
(Deribit options analytics), or None when no Amberdata key is configured.

`get_desk_bigquery()` returns a process-wide providers.bigquery.DeskBigQuery
(read-only desk data in BigQuery), or None when google-cloud-bigquery or
Application Default Credentials are unavailable.

`get_a1_metrics_sheet()` returns a process-wide providers.gsheets.A1MetricsSheet
(read-only spot desk PnL from the "A1 Metrics Dashboard" Google Sheet), or None
when google-auth cannot resolve Application Default Credentials.

There is no local cache; every call hits the upstream API. API keys come from
utils.config.Config (env var override, else GCP Secret Manager).
"""

import logging
import threading
from typing import Optional

from providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

_provider: Optional[MarketDataProvider] = None


def build_provider() -> MarketDataProvider:
    """Construct a fresh CompositeProvider from config (no memoisation)."""
    from coinmetrics.api_client import CoinMetricsClient

    from providers.amberdata import AmberdataProvider
    from providers.coinmetrics import CoinMetricsProvider
    from providers.composite import CompositeProvider
    from utils.config import Config

    cfg = Config()

    cm_key = cfg.COINMETRICS_API_KEY
    if not cm_key:
        logger.warning(
            "No Coin Metrics API key (env COINMETRICS_API_KEY or secret coinmetrics_trial_api); "
            "spot requests will use the unauthenticated community tier."
        )
    spot = CoinMetricsProvider(CoinMetricsClient(api_key=cm_key, debug_mode=False))

    ad_key = cfg.AMBERDATA_API_KEY
    if not ad_key:
        logger.warning(
            "No Amberdata API key (env AMBERDATA_API_KEY or secret amberdata_key); "
            "derivative metrics will be unavailable."
        )
    derivatives = AmberdataProvider(ad_key or "")

    logger.info("Market data provider: Coin Metrics (spot) + Amberdata (derivatives)")
    return CompositeProvider(spot=spot, derivatives=derivatives)


def get_provider() -> MarketDataProvider:
    """Return the shared provider, building it on first use."""
    global _provider
    if _provider is None:
        _provider = build_provider()
    return _provider


def reset_provider() -> None:
    """Force re-initialisation on next get_provider() call (useful for tests)."""
    global _provider
    _provider = None
    reset_options_provider()


# ----------------------------------------------------------------------
# Options (Amberdata, Deribit)
# ----------------------------------------------------------------------

_options_provider = None
_options_provider_built = False


def get_options_provider():
    """Shared AmberdataOptionsProvider, or None when no Amberdata key is available.

    The result (including None) is memoised; call reset_options_provider() to rebuild.
    """
    global _options_provider, _options_provider_built
    if _options_provider_built:
        return _options_provider
    from providers.amberdata_options import AmberdataOptionsProvider
    from utils.config import Config

    key = Config().AMBERDATA_API_KEY
    if not key:
        logger.warning(
            "No Amberdata API key (env AMBERDATA_API_KEY or secret amberdata_key); "
            "options metrics will be unavailable."
        )
        _options_provider = None
    else:
        _options_provider = AmberdataOptionsProvider(key)
        logger.info("Options data provider: Amberdata (%s)", _options_provider.exchange)
    _options_provider_built = True
    return _options_provider


def reset_options_provider() -> None:
    """Drop the memoised options provider so the next get_options_provider() rebuilds it."""
    global _options_provider, _options_provider_built
    _options_provider = None
    _options_provider_built = False


# ----------------------------------------------------------------------
# Desk data (BigQuery, read-only)
# ----------------------------------------------------------------------

_desk_bq = None
_desk_bq_built = False
_desk_bq_lock = threading.Lock()


def get_desk_bigquery():
    """Shared DeskBigQuery, or None when the library / ADC credentials are unavailable.

    The result (including None) is memoised; call reset_desk_bigquery() to retry.
    Nothing is queried here - the BigQuery client is only constructed to prove that
    credentials resolve. Thread-safe: the agent runs parallel tool calls in threads,
    so the instance is built under a lock and published only once it exists.
    """
    global _desk_bq, _desk_bq_built
    if _desk_bq_built:
        return _desk_bq
    with _desk_bq_lock:
        if _desk_bq_built:
            return _desk_bq
        bq = None
        try:
            from providers.bigquery import DeskBigQuery
            from utils.config import Config

            candidate = DeskBigQuery.from_config(Config())
            candidate.client  # forces client construction -> BigQueryUnavailable on missing lib / ADC
            bq = candidate
            logger.info("Desk data: BigQuery %s (datasets %s; billed to %s)",
                        bq.data_project, ", ".join(bq.allowed_datasets), bq.billing_project)
        except Exception as e:
            logger.warning("Desk data (BigQuery) unavailable: %s", e)
        _desk_bq = bq
        _desk_bq_built = True
        return _desk_bq


def reset_desk_bigquery() -> None:
    """Drop the memoised DeskBigQuery so the next get_desk_bigquery() rebuilds it."""
    global _desk_bq, _desk_bq_built
    with _desk_bq_lock:
        _desk_bq = None
        _desk_bq_built = False


# ----------------------------------------------------------------------
# Spot desk PnL (Google Sheet "A1 Metrics Dashboard", read-only)
# ----------------------------------------------------------------------

_a1_sheet = None
_a1_sheet_built = False
_a1_sheet_lock = threading.Lock()


def get_a1_metrics_sheet():
    """Shared A1MetricsSheet, or None when google-auth / ADC are unavailable.

    The result (including None) is memoised; call reset_a1_metrics_sheet() to retry.
    Nothing is fetched here - credentials are only resolved (no network) to prove
    ADC exists; the first tool call does the HTTP round trip. Thread-safe.
    """
    global _a1_sheet, _a1_sheet_built
    if _a1_sheet_built:
        return _a1_sheet
    with _a1_sheet_lock:
        if _a1_sheet_built:
            return _a1_sheet
        sheet = None
        try:
            from providers.gsheets import A1MetricsSheet
            from utils.config import Config

            candidate = A1MetricsSheet.from_config(Config())
            candidate._get_credentials()  # SheetsUnavailable when ADC is missing
            sheet = candidate
            logger.info("Spot desk PnL: Google Sheet %s (quota project %s)",
                        sheet.spreadsheet_id, sheet.quota_project)
        except Exception as e:
            logger.warning("Spot desk PnL sheet unavailable: %s", e)
        _a1_sheet = sheet
        _a1_sheet_built = True
        return _a1_sheet


def reset_a1_metrics_sheet() -> None:
    """Drop the memoised A1MetricsSheet so the next get_a1_metrics_sheet() rebuilds it."""
    global _a1_sheet, _a1_sheet_built
    with _a1_sheet_lock:
        _a1_sheet = None
        _a1_sheet_built = False


# ----------------------------------------------------------------------
# Long-term memory + daily snapshots (providers/memory_store.py)
# ----------------------------------------------------------------------

_memory_store = None
_memory_store_lock = threading.Lock()


def get_memory_store():
    """Shared providers.memory_store.MemoryStore (env MEMORY_DB_URL, default sqlite:///data/memory.db).

    Memoised and lock-guarded: the agent runs tool calls in threads and the Slack bot writes
    episodes from background tasks. Opening the store touches only the local SQLite file;
    embeddings (Vertex) are probed lazily on the first put/search and fall back to keyword search.
    """
    global _memory_store
    if _memory_store is not None:
        return _memory_store
    with _memory_store_lock:
        if _memory_store is None:
            from providers.memory_store import MemoryStore

            _memory_store = MemoryStore()
        return _memory_store


def set_memory_store(store) -> None:
    """Install a specific store (tests: in-memory SQLite with a hash embedder)."""
    global _memory_store
    with _memory_store_lock:
        _memory_store = store


def reset_memory_store() -> None:
    """Drop the memoised store so the next get_memory_store() reopens it (closes the old one)."""
    global _memory_store
    with _memory_store_lock:
        old, _memory_store = _memory_store, None
    if old is not None:
        try:
            old.close()
        except Exception:  # noqa: BLE001
            pass
