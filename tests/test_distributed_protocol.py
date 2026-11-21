import socket
import threading
import pytest
from federated.process_fedavg import accept_update, receive, send


def test_serialized_frame_and_accounting():
    a, b = socket.socketpair()
    try:
        sent = send(a, {"type": "update"}, b"weight bytes")
        meta, payload, consumed = receive(b)
        assert meta["type"] == "update" and sent == consumed and payload == b"weight bytes"
    finally:
        a.close()
        b.close()


def test_version_and_duplicate_checks():
    m = dict(type="update", run="p0", version=1, client=0, samples=96, parent="base")
    assert accept_update(m, "p0", 0, 1, "base", {})
    assert not accept_update(m, "p0", 0, 1, "base", {0: 1})
    assert not accept_update(m, "p0", 0, 2, "base", {})
    for field, val in [
        ("run", "p20"),
        ("client", 1),
        ("parent", "other"),
        ("samples", 2),
        ("version", 4),
    ]:
        with pytest.raises(ValueError):
            accept_update({**m, field: val}, "p0", 0, 1, "base", {})
