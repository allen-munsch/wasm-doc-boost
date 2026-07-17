#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8000}"
echo "Serving $DIR on http://localhost:$PORT"
echo "Press Ctrl+C to stop."
cd "$DIR"
python3 serve.py "$PORT"
