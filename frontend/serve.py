#!/usr/bin/env python3
"""Serve wasm-doc-boost frontend with COOP/COEP headers for SharedArrayBuffer."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.requests import Request
from starlette.responses import Response

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DIR = Path(__file__).resolve().parent

app = FastAPI()

NO_CACHE_EXTS = {".onnx", ".json", ".wasm", ".mjs"}


@app.middleware("http")
async def add_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    # Prevent caching of model/config files during development
    ext = Path(request.url.path).suffix
    if ext in NO_CACHE_EXTS:
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@app.get("/{full_path:path}")
async def serve(full_path: str):
    path = full_path or "index.html"
    file_path = (DIR / path).resolve()

    # Security: prevent directory traversal
    if not str(file_path).startswith(str(DIR)):
        return Response(status_code=403, content="Forbidden")

    if not file_path.is_file():
        return Response(status_code=404, content="Not found")

    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
