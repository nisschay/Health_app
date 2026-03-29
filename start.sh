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
CLOCK_SYNC_ENABLED="${CLOCK_SYNC_ENABLED:-1}"
CLOCK_DRIFT_MAX_SECONDS="${CLOCK_DRIFT_MAX_SECONDS:-120}"

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

get_http_date_epoch() {
  local url="$1"
  if ! command -v curl >/dev/null 2>&1; then
    return
  fi

  local date_header
  date_header="$(curl -fsSI --max-time 5 "$url" 2>/dev/null | sed -n 's/^date:[[:space:]]*//Ip' | head -n 1 | tr -d '\r' || true)"
  if [[ -z "$date_header" ]]; then
    return
  fi

  date -u -d "$date_header" +%s 2>/dev/null || true
}

get_rtc_date_epoch() {
  if ! command -v timedatectl >/dev/null 2>&1; then
    return
  fi

  local rtc_raw
  rtc_raw="$(timedatectl status 2>/dev/null | sed -n 's/^[[:space:]]*RTC time:[[:space:]]*//p' | head -n 1 || true)"
  if [[ -z "$rtc_raw" ]]; then
    return
  fi

  date -u -d "$rtc_raw" +%s 2>/dev/null || true
}

sync_system_clock_if_drifted() {
  if [[ "$CLOCK_SYNC_ENABLED" != "1" ]]; then
    return
  fi

  local local_epoch
  local reference_epoch=""
  local source=""

  local_epoch="$(date -u +%s)"

  for url in "https://www.google.com" "https://cloudflare.com" "https://www.microsoft.com"; do
    reference_epoch="$(get_http_date_epoch "$url")"
    if [[ -n "$reference_epoch" ]]; then
      source="$url"
      break
    fi
  done

  if [[ -z "$reference_epoch" ]]; then
    reference_epoch="$(get_rtc_date_epoch)"
    if [[ -n "$reference_epoch" ]]; then
      source="RTC"
    fi
  fi

  if [[ -z "$reference_epoch" ]]; then
    echo "Clock preflight skipped: no reference time source available."
    log_event "$BACKEND_LOG" "Clock preflight skipped: no reference source available"
    return
  fi

  local drift_seconds=$((reference_epoch - local_epoch))
  local abs_drift="$drift_seconds"
  if (( abs_drift < 0 )); then
    abs_drift=$(( -abs_drift ))
  fi

  if (( abs_drift <= CLOCK_DRIFT_MAX_SECONDS )); then
    log_event "$BACKEND_LOG" "Clock preflight drift=${abs_drift}s source=${source} (within threshold)"
    return
  fi

  local reference_iso
  reference_iso="$(date -u -d "@$reference_epoch" '+%Y-%m-%d %H:%M:%S UTC')"

  echo "Clock drift detected (${abs_drift}s, source: ${source}). Adjusting system time..."
  if date -u -s "$reference_iso" >/dev/null 2>&1; then
    echo "Clock adjusted to $reference_iso"
    log_event "$BACKEND_LOG" "Clock adjusted by preflight: drift=${abs_drift}s source=${source} target=${reference_iso}"
  else
    echo "Clock adjustment failed. Authentication may fail until system time is corrected."
    log_event "$BACKEND_LOG" "Clock adjustment failed: drift=${abs_drift}s source=${source}"
  fi
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

kill_conflicting_port_process 8000 "Backend" "$BACKEND_PID_FILE"
kill_conflicting_port_process 3000 "Frontend" "$FRONTEND_PID_FILE"

touch "$BACKEND_LOG" "$FRONTEND_LOG"

log_event "$BACKEND_LOG" "start.sh invoked"
log_event "$FRONTEND_LOG" "start.sh invoked"

sync_system_clock_if_drifted

if [[ -f "$BACKEND_PID_FILE" ]] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
  echo "Backend already running on http://localhost:8000 (PID $(cat "$BACKEND_PID_FILE"))"
  log_event "$BACKEND_LOG" "Backend already running with PID $(cat "$BACKEND_PID_FILE")"
else
  rotate_log "$BACKEND_LOG" "backend"
  log_event "$BACKEND_LOG" "New log session started"
  log_event "$BACKEND_LOG" "start.sh invoked"
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
  rotate_log "$FRONTEND_LOG" "frontend"
  log_event "$FRONTEND_LOG" "New log session started"
  log_event "$FRONTEND_LOG" "start.sh invoked"
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
