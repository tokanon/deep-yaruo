from __future__ import annotations

import socket

from backend.desktop import find_available_port


def test_find_available_port_skips_an_occupied_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        assert find_available_port(port) == port + 1
