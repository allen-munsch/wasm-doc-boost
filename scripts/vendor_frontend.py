#!/usr/bin/env python3
"""
Bootstrap vendored third-party frontend assets for wasm-doc-boost.

Fetches:
  - tesseract.js UMD bundle + worker + WASM core + eng.traineddata
  - zod ESM bundle (pinned, no CDN at runtime)

Outputs:
  frontend/vendor/tesseract/   — tesseract.min.js, worker.min.js, core WASM, eng.traineddata
  frontend/vendor/zod/         — index.mjs (self-contained ESM)
"""

import sys
from pathlib import Path
from urllib.request import Request, urlopen

# ── Version pins ─────────────────────────────────────────────────────────
TESSERACT_JS_VERSION = "6.0.1"
TESSERACT_CORE_VERSION = "6.0.0"
TESSERACT_DATA_ENG_VERSION = "4.1.1"
ZOD_VERSION = "3.24.2"

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
VENDOR = FRONTEND / "vendor"
TESS_DIR = VENDOR / "tesseract"
ZOD_DIR = VENDOR / "zod"

# ── Assets ───────────────────────────────────────────────────────────────
ASSETS = [
    (
        f"https://cdn.jsdelivr.net/npm/tesseract.js@{TESSERACT_JS_VERSION}/dist/tesseract.min.js",
        TESS_DIR / "tesseract.min.js",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/tesseract.js@{TESSERACT_JS_VERSION}/dist/worker.min.js",
        TESS_DIR / "worker.min.js",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/tesseract.js-core@{TESSERACT_CORE_VERSION}/tesseract-core-simd-lstm.wasm.js",
        TESS_DIR / "tesseract-core-simd-lstm.wasm.js",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/tesseract.js-core@{TESSERACT_CORE_VERSION}/tesseract-core-simd-lstm.wasm",
        TESS_DIR / "tesseract-core-simd-lstm.wasm",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/@tesseract.js-data/eng@{TESSERACT_DATA_ENG_VERSION}/eng.traineddata",
        TESS_DIR / "eng.traineddata",
    ),
    (
        f"https://esm.sh/zod@{ZOD_VERSION}/es2022/zod.mjs",
        ZOD_DIR / "index.mjs",
    ),
]


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  SKIP (exists) {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  FETCH {url}  ->  {dest.relative_to(ROOT)}")
    req = Request(url, headers={"User-Agent": "wasm-doc-boost/vendor_frontend"})
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)
    size_kb = len(data) / 1024
    print(f"    {size_kb:.0f} KB")


def main() -> int:
    print("wasm-doc-boost: vendoring frontend assets\n")
    ok, fail = 0, 0
    for url, dest in ASSETS:
        try:
            download(url, dest)
            ok += 1
        except Exception as e:
            print(f"  FAIL {url}: {e}")
            fail += 1
    print(f"\nDone: {ok} fetched/skipped, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
