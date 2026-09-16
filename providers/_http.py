"""Shared HTTP plumbing for the Amberdata providers.

Both ``providers.amberdata`` (perp analytics) and ``providers.amberdata_options``
(options analytics) talk to ``api.amberdata.com`` with the same conventions:

* ``x-api-key`` header; ``Accept-Encoding: gzip`` (the options endpoints reject
  uncompressed requests with HTTP 400 "Compression required").
* Retry with exponential backoff on 429 / 5xx (honouring ``Retry-After``).
* Responses wrap rows in ``payload.data`` with an optional cursor URL in
  ``payload.metadata.next``.
* ``startDate`` inclusive, ``endDate`` exclusive; timestamps arrive as epoch
  milliseconds on some endpoints and ISO-8601 strings on others.

This module hosts the pieces that are identical for both providers so neither
copies the other.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}
AUTH_STATUSES = {401, 403}
DEFAULT_MAX_PAGES = 50


def day_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def to_utc_day(values: pd.Series) -> pd.Series:
    """Parse Amberdata timestamps (ms ints or ISO strings) into tz-naive UTC days."""
    if pd.api.types.is_numeric_dtype(values):
        ts = pd.to_datetime(values.astype("int64"), unit="ms", utc=True)
    else:
        try:
            ts = pd.to_datetime(values, utc=True, format="ISO8601")
        except (ValueError, TypeError):
            ts = pd.to_datetime(values, utc=True)
    return ts.dt.tz_convert(None).dt.floor("D")


def to_utc_timestamp(values: pd.Series) -> pd.Series:
    """Parse Amberdata timestamps (ms ints or ISO strings) into tz-naive UTC datetimes."""
    if pd.api.types.is_numeric_dtype(values):
        ts = pd.to_datetime(values.astype("int64"), unit="ms", utc=True)
    else:
        try:
            ts = pd.to_datetime(values, utc=True, format="ISO8601")
        except (ValueError, TypeError):
            ts = pd.to_datetime(values, utc=True)
    return ts.dt.tz_convert(None)


def date_chunks(start: datetime, end: datetime, max_days: int) -> list[tuple[datetime, datetime]]:
    """Split [start, end] (inclusive days) into (startDate, endDate-exclusive) windows."""
    start_day = datetime(start.year, start.month, start.day)
    end_excl = datetime(end.year, end.month, end.day) + timedelta(days=1)
    chunks = []
    cur = start_day
    while cur < end_excl:
        nxt = min(cur + timedelta(days=max_days), end_excl)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


class AmberdataHTTP:
    """Authenticated session with retry/backoff and cursor pagination.

    ``last_status`` / ``last_error`` describe the most recent failed request
    (reset to ``None`` / ``""`` on success) so callers can react to specific
    4xx replies, e.g. the "range over the maximum allowed" 400.
    ``call_count`` counts every HTTP request issued (including retries and
    cursor pages) for cost accounting.
    """

    def __init__(
        self,
        api_key: str,
        session: Optional[requests.Session] = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff: float = 1.0,
        header_name: str = "x-api-key",
        retry_statuses: Optional[set] = None,
        label: str = "Amberdata",
    ):
        """``header_name`` / ``retry_statuses`` / ``label`` let other vendors with the same
        envelope-and-retry needs (providers.messari) reuse this class unchanged."""
        self.session = session or requests.Session()
        self.session.headers.update({
            header_name: api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        })
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.retry_statuses = set(retry_statuses) if retry_statuses is not None else set(RETRY_STATUSES)
        self.label = label
        self.call_count = 0
        self.last_status: Optional[int] = None
        self.last_error: str = ""

    # ------------------------------------------------------------------

    def request(self, url: str, params: Optional[dict], timeout: Optional[float] = None,
                retries: Optional[int] = None) -> Optional[dict]:
        """GET with retry/backoff on 429/5xx. Returns parsed JSON or None.

        ``timeout`` / ``retries`` override the instance defaults for this one call (a
        caller that knows an endpoint is flaky can fail fast and fall back)."""
        attempt = 0
        timeout = self.timeout if timeout is None else timeout
        max_retries = self.max_retries if retries is None else max(0, int(retries))
        while True:
            self.call_count += 1
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                if attempt >= max_retries:
                    logger.warning("%s request failed %s: %s", self.label, url, e)
                    self.last_status, self.last_error = None, str(e)
                    return None
                attempt += 1
                time.sleep(self.backoff * (2 ** (attempt - 1)))
                continue

            status = resp.status_code
            if status == 200:
                try:
                    data = resp.json()
                except ValueError as e:
                    logger.warning("%s non-JSON response %s: %s", self.label, url, e)
                    self.last_status, self.last_error = status, "non-JSON response"
                    return None
                self.last_status, self.last_error = None, ""
                return data
            if status in self.retry_statuses and attempt < max_retries:
                attempt += 1
                delay = self.backoff * (2 ** (attempt - 1))
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                logger.info("%s HTTP %s on %s — retry %d/%d in %.1fs",
                            self.label, status, url, attempt, max_retries, delay)
                time.sleep(delay)
                continue

            body = resp.text[:300] if resp.text else ""
            self.last_status, self.last_error = status, _error_message(resp, body)
            if status in AUTH_STATUSES:
                logger.warning("%s HTTP %s (auth/tier) on %s params=%s: %s",
                               self.label, status, url, params, body)
            elif status == 404:
                logger.info("%s HTTP 404 on %s params=%s", self.label, url, params)
            else:
                logger.warning("%s HTTP %s on %s params=%s: %s", self.label, status, url, params, body)
            return None

    def get_rows(self, url: str, params: dict, max_pages: int = DEFAULT_MAX_PAGES) -> Optional[list]:
        """Fetch ``payload.data`` for an analytics endpoint, following cursors.

        Returns a list of row dicts (possibly empty) or None on error.
        """
        rows: list = []
        page_params: Optional[dict] = dict(params)
        for _ in range(max_pages):
            data = self.request(url, page_params)
            if data is None:
                return None if not rows else rows
            payload = data.get("payload") if isinstance(data, dict) else None
            if isinstance(payload, dict):
                page = payload.get("data") or []
                nxt = (payload.get("metadata") or {}).get("next")
            elif isinstance(payload, list):
                page, nxt = payload, None
            else:
                page, nxt = [], None
            rows.extend(r for r in page if isinstance(r, dict))
            if not nxt:
                break
            url, page_params = nxt, None  # cursor URL is fully qualified
        return rows


def _error_message(resp, body: str) -> str:
    try:
        j = resp.json()
        if isinstance(j, dict):
            return str(j.get("message") or j.get("description") or body)
    except ValueError:
        pass
    return body
