#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Pick free ports if defaults are busy
pick_port() {
  local default_port="$1"
  local port="$default_port"
  while lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; do
    port=$((port+1))
  done
  echo "$port"
}

BACKEND_PORT=$(pick_port 8000)
FRONTEND_PORT=$(pick_port 8080)

export VITE_BACKEND_URL="http://localhost:${BACKEND_PORT}"
export VITE_DEV_PORT="$FRONTEND_PORT"

# Start backend
(
  cd "$ROOT_DIR/backend"
  chmod +x run_backend.sh
  PORT="$BACKEND_PORT" ./run_backend.sh &
  echo "Backend starting on http://localhost:${BACKEND_PORT}"
)

# Start frontend
(
  cd "$ROOT_DIR/frontend"
  chmod +x run.sh
  VITE_DEV_PORT="$FRONTEND_PORT" VITE_BACKEND_URL="$VITE_BACKEND_URL" ./run.sh &
  echo "Frontend starting on http://localhost:${FRONTEND_PORT}"
)

echo "Both servers started. Press Ctrl+C to stop."
wait
