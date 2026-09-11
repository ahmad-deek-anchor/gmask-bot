"""Network hygiene shared by every entry point.

`prefer_ipv4()` makes Python resolve hostnames to IPv4 addresses only. Some hosts
(notably this dev container) advertise a global IPv6 address whose upstream is
broken: Python tries the AAAA record first and hangs for the full connect
timeout, while curl falls back to IPv4 instantly. That stalls every Google token
refresh, Vertex call and Secret Manager read. Set FORCE_IPV4=0 to disable.
"""

from __future__ import annotations

import logging
import os
import socket

log = logging.getLogger(__name__)
_applied = False
_original_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002 - socket signature
    if family in (0, socket.AF_UNSPEC):
        try:
            res = _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
            if res:
                return res
        except socket.gaierror:
            pass  # no A record: fall through to the default behaviour
    return _original_getaddrinfo(host, port, family, type, proto, flags)


def prefer_ipv4() -> bool:
    """Install the IPv4-first resolver once. Returns True if active."""
    global _applied
    if _applied:
        return True
    if os.getenv("FORCE_IPV4", "1").lower() in ("0", "false", "no"):
        return False
    socket.getaddrinfo = _ipv4_getaddrinfo
    # grpc (used by google-cloud-* clients) resolves on its own; make it prefer v4 too.
    os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
    _applied = True
    log.debug("IPv4-first name resolution enabled (FORCE_IPV4=0 to disable)")
    return True


def reset_for_tests() -> None:
    global _applied
    socket.getaddrinfo = _original_getaddrinfo
    _applied = False
