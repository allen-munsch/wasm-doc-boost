#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv-gpu/bin/python
SHIM="$HOME/.cache/nvidia-shim"

log() { echo "[retrain-batch] $(date +%H:%M:%S) $*"; }

log "== [1/4] export features (16 workers, ~29k base x16) =="
$PY scripts/export_parallel.py --images data/images --labels data/labels.csv \
    --output data/features_v9.npz --workers 16

log "== [2/4] train (xgboost cuda) =="
LD_LIBRARY_PATH="$SHIM" $PY scripts/train_model.py --features data/features_v9.npz \
    --output data/model_v9.json --report data/report_v9.json --device cuda

log "== [3/4] benchmark (cardinal + tilt + latency) =="
$PY scripts/benchmark_v052.py --model data/model_v9.json --report data/report_v9.json \
    --out data/metrics_v052.json

log "== [4/4] goal gate =="
$PY scripts/goal_gate.py
log "== BATCH COMPLETE =="
