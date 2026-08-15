from __future__ import annotations

import socket
import os
import threading
import webbrowser

import uvicorn


def find_available_port(preferred: int = 8000) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No available local port was found.")


def main() -> None:
    port = find_available_port()
    url = f"http://127.0.0.1:{port}/"
    if os.environ.get("YARUOAA_NO_BROWSER") != "1":
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
