"""Provider factory.

`get_provider()` returns a process-wide CompositeProvider:

  spot        -> CoinMetricsProvider (Coin Metrics API)
  derivatives -> AmberdataProvider   (Amberdata API)

`get_options_provider()` returns a process-wide AmberdataOptionsProvider
(Deribit options analytics), or None when no Amberdata key is configured.

`get_desk_bigquery()` returns a process-wide providers.bigquery.DeskBigQuery
(read-only desk data in BigQuery), or None when google-cloud-bigquery or
Application Default Credentials are unavailable.

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
