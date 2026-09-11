import socket

from utils import net


def test_prefer_ipv4_filters_to_af_inet(monkeypatch):
    net.reset_for_tests()
    monkeypatch.delenv("FORCE_IPV4", raising=False)
    calls = []

    def fake(host, port, family=0, type=0, proto=0, flags=0):
        calls.append(family)
        if family == socket.AF_INET:
            return [(socket.AF_INET, type, proto, "", ("1.2.3.4", port))]
        return [(socket.AF_INET6, type, proto, "", ("::1", port, 0, 0)), (socket.AF_INET, type, proto, "", ("1.2.3.4", port))]

    monkeypatch.setattr(net, "_original_getaddrinfo", fake)
    assert net.prefer_ipv4() is True
    res = socket.getaddrinfo("example.com", 443, 0, socket.SOCK_STREAM)
    assert [r[0] for r in res] == [socket.AF_INET]
    assert calls[0] == socket.AF_INET
    net.reset_for_tests()


def test_prefer_ipv4_can_be_disabled(monkeypatch):
    net.reset_for_tests()
    monkeypatch.setenv("FORCE_IPV4", "0")
    assert net.prefer_ipv4() is False
    assert socket.getaddrinfo is net._original_getaddrinfo
    net.reset_for_tests()


def test_prefer_ipv4_is_idempotent(monkeypatch):
    net.reset_for_tests()
    monkeypatch.delenv("FORCE_IPV4", raising=False)
    assert net.prefer_ipv4() and net.prefer_ipv4()
    net.reset_for_tests()
