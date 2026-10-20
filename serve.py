"""Static server for the dashboard, with caching disabled.

Serves the repo root on :8000 over all interfaces (so it's reachable via Tailscale), and
sends ``Cache-Control: no-store`` on every response so the dashboard always shows fresh
artifacts. Use this instead of plain ``python -m http.server``.

Run:  python3 serve.py
"""

from __future__ import annotations

import http.server
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()


def main() -> None:
    os.chdir(ROOT)
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    httpd.allow_reuse_address = True
    print(f"Serving {ROOT} on http://0.0.0.0:{PORT} (no-store)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
