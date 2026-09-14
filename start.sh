#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT_DIR/.run"
RUN_HISTORY_DIR="$RUN_DIR/history"
BACKEND_LOG="$RUN_DIR/backend.log"
FRONTEND_LOG="$RUN_DIR/frontend.log"
BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
TAIL_LINES="${TAIL_LINES:-40}"

log_event() {
  local file="$1"
  local message="$2"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" >>"$file"
}

wait_for_backend() {
  local attempts=60
  local tracked_pid=""
  if [[ -f "$BACKEND_PID_FILE" ]]; then
    tracked_pid="$(cat "$BACKEND_PID_FILE")"
  fi

  while (( attempts > 0 )); do
    if [[ -n "$tracked_pid" ]] && ! kill -0 "$tracked_pid" 2>/dev/null; then
      return 2
    fi

    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
      return 0
    fi

    sleep 0.5
    ((attempts--))
  done
  return 1
}

mkdir -p "$RUN_DIR"
mkdir -p "$RUN_HISTORY_DIR"

rotate_log() {
  local file="$1"
  local label="$2"
  if [[ -f "$file" ]] && [[ -s "$file" ]]; then
    local ts
    ts="$(date '+%Y%m%d_%H%M%S')"
    mv "$file" "$RUN_HISTORY_DIR/${label}_${ts}.log"
  fi
  : >"$file"
}


touch "$BACKEND_LOG" "$FRONTEND_LOG"

log_event "$BACKEND_LOG" "start.sh invoked"
log_event "$FRONTEND_LOG" "start.sh invoked"


if [[ -f "$BACKEND_PID_FILE" ]] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
  echo "Backend already running on http://localhost:8000 (PID $(cat "$BACKEND_PID_FILE"))"
  log_event "$BACKEND_LOG" "Backend already running with PID $(cat "$BACKEND_PID_FILE")"
else
  rotate_log "$BACKEND_LOG" "backend"
  log_event "$BACKEND_LOG" "New log session started"
  log_event "$BACKEND_LOG" "start.sh invoked"
  source "$ROOT_DIR/.venv/bin/activate"
  cd "$ROOT_DIR"
  nohup uvicorn backend_api.app.main:app --host 127.0.0.1 --port 8000 >>"$BACKEND_LOG" 2>&1 &
  echo $! >"$BACKEND_PID_FILE"
  echo "Started backend on http://localhost:8000 (PID $(cat "$BACKEND_PID_FILE"))"
  log_event "$BACKEND_LOG" "Started backend with PID $(cat "$BACKEND_PID_FILE")"
fi

if [[ -f "$FRONTEND_PID_FILE" ]] && kill -0 "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null; then
  echo "Frontend already running on http://localhost:3000 (PID $(cat "$FRONTEND_PID_FILE"))"
  log_event "$FRONTEND_LOG" "Frontend already running with PID $(cat "$FRONTEND_PID_FILE")"
else
  rotate_log "$FRONTEND_LOG" "frontend"
  log_event "$FRONTEND_LOG" "New log session started"
  log_event "$FRONTEND_LOG" "start.sh invoked"
  nohup npm --prefix "$ROOT_DIR/web" run dev -- -H 127.0.0.1 -p 3000 >>"$FRONTEND_LOG" 2>&1 &
  echo $! >"$FRONTEND_PID_FILE"
  echo "Started frontend on http://localhost:3000 (PID $(cat "$FRONTEND_PID_FILE"))"
  log_event "$FRONTEND_LOG" "Started frontend with PID $(cat "$FRONTEND_PID_FILE")"
fi

echo "Running services:"
echo "  Backend URL: http://localhost:8000"
echo "  Frontend URL: http://localhost:3000"

echo "Logs:"
echo "  Backend: $BACKEND_LOG"
echo "  Frontend: $FRONTEND_LOG"

if ! wait_for_backend; then
  echo
  echo "Backend health check did not pass within timeout. Recent backend logs:"
  tail -n "$TAIL_LINES" "$BACKEND_LOG" || true

  # Clear stale PID files when processes are no longer alive.
  if [[ -f "$BACKEND_PID_FILE" ]] && ! kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
    rm -f "$BACKEND_PID_FILE"
    log_event "$BACKEND_LOG" "Removed stale backend PID file after failed startup"
  fi

  if [[ -f "$FRONTEND_PID_FILE" ]] && ! kill -0 "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null; then
    rm -f "$FRONTEND_PID_FILE"
    log_event "$FRONTEND_LOG" "Removed stale frontend PID file after failed startup"
  fi

  exit 1
fi

if [[ -f "$BACKEND_LOG" ]]; then
  echo
  echo "Most recent backend logs (last $TAIL_LINES lines):"
  tail -n "$TAIL_LINES" "$BACKEND_LOG" || true
else
  echo
  echo "Backend log file not found yet."
fi

if [[ -f "$FRONTEND_LOG" ]]; then
  echo
  echo "Most recent frontend logs (last 20 lines):"
  tail -n 20 "$FRONTEND_LOG" || true
fi
