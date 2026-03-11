#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT_DIR/.run"
BACKEND_LOG="$RUN_DIR/backend.log"
FRONTEND_LOG="$RUN_DIR/frontend.log"
BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"

mkdir -p "$RUN_DIR"

if [[ -f "$BACKEND_PID_FILE" ]] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
  echo "Backend already running with PID $(cat "$BACKEND_PID_FILE")"
else
  source "$ROOT_DIR/.venv/bin/activate"
  cd "$ROOT_DIR"
  nohup uvicorn backend_api.app.main:app --host 0.0.0.0 --port 8000 >"$BACKEND_LOG" 2>&1 &
  echo $! >"$BACKEND_PID_FILE"
  echo "Started backend on http://localhost:8000 (PID $(cat "$BACKEND_PID_FILE"))"
fi

if [[ -f "$FRONTEND_PID_FILE" ]] && kill -0 "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null; then
  echo "Frontend already running with PID $(cat "$FRONTEND_PID_FILE")"
else
  cd "$ROOT_DIR/web"
  nohup npm run dev -- -H 0.0.0.0 -p 3000 >"$FRONTEND_LOG" 2>&1 &
  echo $! >"$FRONTEND_PID_FILE"
  echo "Started frontend on http://localhost:3000 (PID $(cat "$FRONTEND_PID_FILE"))"
fi

echo "Logs:"
echo "  Backend: $BACKEND_LOG"
echo "  Frontend: $FRONTEND_LOG"
