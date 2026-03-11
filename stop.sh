#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT_DIR/.run"
BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"

stop_process() {
  local pid_file="$1"
  local name="$2"

  if [[ ! -f "$pid_file" ]]; then
    echo "$name is not running (no PID file)."
    return
  fi

  local pid
  pid="$(cat "$pid_file")"

  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "Stopped $name (PID $pid)."
  else
    echo "$name process not found (stale PID $pid)."
  fi

  rm -f "$pid_file"
}

stop_process "$BACKEND_PID_FILE" "Backend"
stop_process "$FRONTEND_PID_FILE" "Frontend"
