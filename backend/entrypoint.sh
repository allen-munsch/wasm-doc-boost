#!/usr/bin/env bash
# ── wasm-doc-boost backend entrypoint ──────────────
# Routes subcommand to the right Python script.
set -euo pipefail

CMD="${1:-server}"
shift 2>/dev/null || true

case "$CMD" in
  server)
    exec python backend/server.py "$@"
    ;;
  train)
    exec python scripts/train_model.py "$@"
    ;;
  export-features)
    exec python scripts/export_features.py "$@"
    ;;
  export-features-parallel)
    exec python scripts/export_features_parallel.py "$@"
    ;;
  download-datasets)
    exec python scripts/download_datasets.py "$@"
    ;;
  build-labels)
    exec python scripts/build_labels.py "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    echo "usage: $0 {server|train|export-features|export-features-parallel|download-datasets|build-labels|shell} [...]"
    exit 1
    ;;
esac
