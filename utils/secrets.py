"""Secret lookup: environment override first, then Google Secret Manager.

Usage:
    from utils.secrets import get_secret
    key = get_secret("coinmetrics_trial_api")           # env COINMETRICS_TRIAL_API wins if set
    key = get_secret("amberdata_key", env_var="AMBERDATA_API_KEY")

Auth for Secret Manager is Application Default Credentials
(`gcloud auth application-default login`). The google client is imported
lazily so importing this module never requires GCP libraries or credentials.
Any failure (no ADC, missing permission, secret not found) logs a warning and
returns None; this module never raises on lookup.
"""

from __future__ import annotations

import logging
import os
import threading

from utils.net import prefer_ipv4  # noqa: E402

prefer_ipv4()

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "anchorage-trading-solutions"

_cache: dict[tuple[str, str], str | None] = {}
_cache_lock = threading.Lock()


def _fetch_from_secret_manager(name: str, project: str) -> str:
    """Read `projects/{project}/secrets/{name}/versions/latest`. Raises on failure."""
    from google.cloud import secretmanager  # lazy: keep import cheap and GCP-free

    # ADC has no default project unless gcloud config / GOOGLE_CLOUD_PROJECT set one;
    # default it so google.auth stops warning on every client construction.
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project)
    client = secretmanager.SecretManagerServiceClient()
    resource = f"projects/{project}/secrets/{name}/versions/latest"
    response = client.access_secret_version(request={"name": resource})
    return response.payload.data.decode("utf-8")


def get_secret(
    name: str,
    project: str = DEFAULT_PROJECT,
    env_var: str | None = None,
) -> str | None:
    """Return a secret value, or None if it cannot be resolved.

    Resolution order:
      1. ``os.environ[env_var]`` (or ``os.environ[name.upper()]`` when env_var is None),
         if set and non-empty.
      2. Google Secret Manager, latest version, cached in-process per (project, name).

    Never raises on lookup failure; logs a warning and returns None instead.
    """
    env_key = env_var or name.upper()
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val

    cache_key = (project, name)
    with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

    try:
        value: str | None = _fetch_from_secret_manager(name, project)
    except Exception as exc:  # noqa: BLE001 - by contract we never raise here
        logger.warning(
            "Could not read secret %r from project %r: %s: %s",
            name, project, type(exc).__name__, exc,
        )
        value = None

    with _cache_lock:
        _cache[cache_key] = value
    return value


def clear_cache() -> None:
    """Drop all cached Secret Manager lookups (for tests / credential rotation)."""
    with _cache_lock:
        _cache.clear()
