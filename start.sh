#!/usr/bin/env bash
# Start the local transcription web app (API + frontend).
# Usage: ./start.sh [--port-api PORT] [--port-ui PORT] [--no-ollama]
set -euo pipefail

API_PORT=${TRANSCRIBER_PORT:-8000}
UI_PORT=${FRONTEND_PORT:-3000}
SKIP_OLLAMA=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --port-api) API_PORT=$2; shift 2 ;;
        --port-ui)  UI_PORT=$2;  shift 2 ;;
        --no-ollama) SKIP_OLLAMA=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n'  "$*"; }

# ── Prerequisites ──────────────────────────────────────────────────────────────

check_dep() {
    command -v "$1" &>/dev/null || { red "Missing: $1 — $2"; exit 1; }
}

check_dep ffmpeg  "install with: sudo apt install ffmpeg"
check_dep node    "install from: https://nodejs.org"

if command -v uv &>/dev/null; then
    PYTHON_RUN="uv run"
else
    check_dep python3 "install from: https://www.python.org"
    PYTHON_RUN="python3 -m"
fi

if [[ $SKIP_OLLAMA -eq 0 ]] && ! command -v ollama &>/dev/null; then
    yellow "Warning: ollama not found — LLM correction will be unavailable."
    yellow "         Install from https://ollama.com or pass --no-ollama to suppress this."
fi

# ── Python environment ─────────────────────────────────────────────────────────

bold "Setting up Python environment…"
cd "$ROOT"
if command -v uv &>/dev/null; then
    uv sync --quiet
else
    if [[ ! -d .venv ]]; then
        python3 -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -e ".[dev]" --quiet
fi

# ── Frontend dependencies ──────────────────────────────────────────────────────

bold "Installing frontend dependencies…"
cd "$ROOT/frontend"
npm install --silent

# Write .env.local if it doesn't exist yet.
if [[ ! -f .env.local ]]; then
    echo "NEXT_PUBLIC_API_URL=http://localhost:${API_PORT}" > .env.local
fi

# ── Launch ─────────────────────────────────────────────────────────────────────

API_PID=""
cleanup() {
    [[ -n $API_PID ]] && kill "$API_PID" 2>/dev/null && wait "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

bold "Starting API server on port ${API_PORT}…"
cd "$ROOT"
TRANSCRIBER_PORT=$API_PORT $PYTHON_RUN uvicorn transcriber.api:app \
    --host 0.0.0.0 --port "$API_PORT" --log-level warning &
API_PID=$!

# Wait until the API is reachable before opening the UI.
echo -n "Waiting for API"
for _ in $(seq 1 20); do
    sleep 0.5
    curl -sf "http://localhost:${API_PORT}/api/health" &>/dev/null && break
    echo -n "."
done
echo ""

if ! kill -0 "$API_PID" 2>/dev/null; then
    red "API server failed to start. Check output above."
    exit 1
fi
green "API ready → http://localhost:${API_PORT}"

bold "Starting frontend on port ${UI_PORT}…"
cd "$ROOT/frontend"
green "App ready → http://localhost:${UI_PORT}"
NEXT_PUBLIC_API_URL="http://localhost:${API_PORT}" PORT=$UI_PORT npm run dev
