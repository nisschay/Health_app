#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT_DIR/.run"
BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"

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

for spec in "8000:Backend" "3000:Frontend"; do
  port="${spec%%:*}"
  name="${spec##*:}"
  pid="$(find_port_pid "$port")"

  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    echo "Stopped $name listener on port $port (PID $pid)."
  fi
done
