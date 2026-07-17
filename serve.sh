#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

FRONTEND_PORT="${1:-8000}"
BACKEND_PORT="${2:-8765}"

echo "=== wasm-doc-boost ==="
echo "Frontend:  http://localhost:$FRONTEND_PORT"
echo "GLM-OCR:   http://localhost:$BACKEND_PORT"
echo "Press Ctrl+C to stop both."
echo

cleanup() {
  echo
  echo "Shutting down..."
  kill $FRONTEND_PID $BACKEND_PID 2>/dev/null || true
  wait
}
trap cleanup EXIT INT TERM

# Start GLM-OCR backend
cd "$DIR/backend"
python3 server.py --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Start frontend
cd "$DIR/frontend"
python3 serve.py "$FRONTEND_PORT" &
FRONTEND_PID=$!

wait
