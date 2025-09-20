#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

# Install deps only once (touch a marker after success)
if [ ! -f .venv/.deps_installed ]; then
  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  touch .venv/.deps_installed
fi

# If marker exists but a new dependency is missing, install requirements again
python - <<'PY'
try:
    import pydantic_settings  # noqa: F401
except Exception:
    raise SystemExit(1)
else:
    raise SystemExit(0)
PY
if [ $? -ne 0 ]; then
  pip install -r requirements.txt
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Pick a free port if the chosen one is busy
pick_port() {
  local port="$1"
  while lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; do
    port=$((port+1))
  done
  echo "$port"
}

PORT="$(pick_port "$PORT")"

if [ "${RELOAD:-1}" = "1" ]; then
  # Restrict reload to source dirs; ignore venv
  echo "Backend starting (reload) on http://localhost:${PORT}"
  exec python -m uvicorn app.main:app \
    --host "$HOST" --port "$PORT" --reload \
    --reload-dir app --reload-dir code \
    --reload-exclude ".venv/*"
else
  echo "Backend starting on http://localhost:${PORT}"
  exec python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
fi
