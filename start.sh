#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT_DIR/.run"
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
  local attempts=30
  local tracked_pid=""
  if [[ -f "$BACKEND_PID_FILE" ]]; then
    tracked_pid="$(cat "$BACKEND_PID_FILE")"
  fi

  while (( attempts > 0 )); do
    if [[ -n "$tracked_pid" ]] && ! kill -0 "$tracked_pid" 2>/dev/null; then
      return 2
    fi

    if [[ -n "$(find_port_pid 8000)" ]]; then
      return 0
    fi

    sleep 0.5
    ((attempts--))
  done
  return 1
}

find_port_pid() {
  local port="$1"
  local pid=""

  if command -v lsof >/dev/null 2>&1; then
    pid="$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
    if [[ -n "$pid" ]]; then
      echo "$pid"
      return
    fi
  fi

  if command -v ss >/dev/null 2>&1; then
    pid="$(ss -ltnpH "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n 1 || true)"
    if [[ -n "$pid" ]]; then
      echo "$pid"
      return
    fi
    return
  fi

  echo ""
}

kill_conflicting_port_process() {
  local port="$1"
  local service_name="$2"
  local tracked_pid_file="$3"
  local tracked_pid=""

  if [[ -f "$tracked_pid_file" ]]; then
    tracked_pid="$(cat "$tracked_pid_file")"
  fi

  local port_pid
  port_pid="$(find_port_pid "$port")"

  if [[ -z "$port_pid" ]]; then
    return
  fi

  if [[ -n "$tracked_pid" ]] && [[ "$tracked_pid" == "$port_pid" ]]; then
    return
  fi

  echo "$service_name port $port is occupied by PID $port_pid. Stopping conflicting process."
  kill "$port_pid" 2>/dev/null || true
  sleep 1

  if kill -0 "$port_pid" 2>/dev/null; then
    kill -9 "$port_pid" 2>/dev/null || true
  fi
}

mkdir -p "$RUN_DIR"
touch "$BACKEND_LOG" "$FRONTEND_LOG"

kill_conflicting_port_process 8000 "Backend" "$BACKEND_PID_FILE"
kill_conflicting_port_process 3000 "Frontend" "$FRONTEND_PID_FILE"

log_event "$BACKEND_LOG" "start.sh invoked"
log_event "$FRONTEND_LOG" "start.sh invoked"

if [[ -f "$BACKEND_PID_FILE" ]] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
  echo "Backend already running on http://localhost:8000 (PID $(cat "$BACKEND_PID_FILE"))"
  log_event "$BACKEND_LOG" "Backend already running with PID $(cat "$BACKEND_PID_FILE")"
else
  source "$ROOT_DIR/.venv/bin/activate"
  cd "$ROOT_DIR"
  nohup stdbuf -oL -eL uvicorn backend_api.app.main:app --host 0.0.0.0 --port 8000 >>"$BACKEND_LOG" 2>&1 &
  echo $! >"$BACKEND_PID_FILE"
  echo "Started backend on http://localhost:8000 (PID $(cat "$BACKEND_PID_FILE"))"
  log_event "$BACKEND_LOG" "Started backend with PID $(cat "$BACKEND_PID_FILE")"
fi

if [[ -f "$FRONTEND_PID_FILE" ]] && kill -0 "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null; then
  echo "Frontend already running on http://localhost:3000 (PID $(cat "$FRONTEND_PID_FILE"))"
  log_event "$FRONTEND_LOG" "Frontend already running with PID $(cat "$FRONTEND_PID_FILE")"
else
  nohup npm --prefix "$ROOT_DIR/web" run dev -- -H 0.0.0.0 -p 3000 >>"$FRONTEND_LOG" 2>&1 &
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
