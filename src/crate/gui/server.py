"""Serve the Callima window on localhost and open it."""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from .service import Desk, DeskError

STATIC = Path(__file__).resolve().parent / "static"
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


class _Server(ThreadingHTTPServer):
    desk: Desk


class Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send(200, self.server.desk.snapshot())
            return
        rel = "index.html" if path in {"", "/"} else path.lstrip("/")
        file = (STATIC / rel).resolve()
        if not str(file).startswith(str(STATIC.resolve())) or not file.is_file():
            self.send_error(404)
            return
        data = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _TYPES.get(file.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "The request was not valid JSON."})
            return
        if not isinstance(payload, dict):
            self._send(400, {"error": "Expected an object."})
            return
        body = cast(dict[str, object], payload)
        try:
            if path == "/api/pick":
                prompt = str(body.get("prompt") or "Choose a folder")
                result: dict[str, object] = self.server.desk.pick(prompt)
            elif path == "/api/action":
                result = self.server.desk.action(str(body.get("action") or ""), body)
            else:
                self._send(404, {"error": "Unknown request."})
                return
        except DeskError as exc:
            self._send(400, {"error": str(exc)})
            return
        self._send(200, result)

    def _send(self, status: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def make_server(port: int, home: Path | None = None, config_path: Path | None = None) -> _Server:
    desk = Desk(home, config_path)
    server = _Server(("127.0.0.1", port), Handler)
    server.desk = desk
    return server


def serve(port: int = 8765, *, open_browser: bool = True, home: Path | None = None) -> None:
    server: _Server | None = None
    bound = port
    for offset in range(10):
        try:
            server = make_server(port + offset, home)
            bound = port + offset
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"No free port from {port} to {port + 9}.")
    thread = threading.Thread(target=_watch_loop, args=(server.desk,), daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{bound}"
    print(f"Callima is open at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def _watch_loop(desk: Desk) -> None:
    while True:
        time.sleep(15)
        try:
            desk.watch_tick()
        except Exception:
            continue
