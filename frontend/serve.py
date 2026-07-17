#!/usr/bin/env python3
"""Serve wasm-doc-boost frontend with COOP/COEP headers for SharedArrayBuffer."""

import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        # Prevent caching of model/config files during development
        path = self.path
        if any(path.endswith(ext) for ext in ('.onnx', '.json', '.wasm', '.mjs')):
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
        super().end_headers()


print(f"Serving {DIR} on http://localhost:{PORT}")
print("Press Ctrl+C to stop.")
http.server.HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
