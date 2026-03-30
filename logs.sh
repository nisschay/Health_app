#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT_DIR/.run"
BACKEND_LOG="$RUN_DIR/backend.log"
FRONTEND_LOG="$RUN_DIR/frontend.log"
TAIL_LINES="${TAIL_LINES:-80}"
MODE="${1:-backend}"

show_log() {
  local file="$1"
  local label="$2"
  if [[ ! -f "$file" ]]; then
    echo "$label log not found: $file"
    exit 1
  fi

  echo "Showing latest $TAIL_LINES lines from $label log:"
  tail -n "$TAIL_LINES" "$file"
}

case "$MODE" in
  backend)
    show_log "$BACKEND_LOG" "backend"
    ;;
  frontend)
    show_log "$FRONTEND_LOG" "frontend"
    ;;
  all)
    show_log "$BACKEND_LOG" "backend"
    echo
    show_log "$FRONTEND_LOG" "frontend"
    ;;
  follow-backend)
    [[ -f "$BACKEND_LOG" ]] || { echo "backend log not found: $BACKEND_LOG"; exit 1; }
    echo "Following backend log: $BACKEND_LOG"
    tail -n "$TAIL_LINES" -f "$BACKEND_LOG"
    ;;
  follow-frontend)
    [[ -f "$FRONTEND_LOG" ]] || { echo "frontend log not found: $FRONTEND_LOG"; exit 1; }
    echo "Following frontend log: $FRONTEND_LOG"
    tail -n "$TAIL_LINES" -f "$FRONTEND_LOG"
    ;;
  follow-all)
    [[ -f "$BACKEND_LOG" ]] || { echo "backend log not found: $BACKEND_LOG"; exit 1; }
    [[ -f "$FRONTEND_LOG" ]] || { echo "frontend log not found: $FRONTEND_LOG"; exit 1; }
    echo "Following backend and frontend logs..."
    tail -n "$TAIL_LINES" -f "$BACKEND_LOG" "$FRONTEND_LOG"
    ;;
  *)
    echo "Usage: $0 [backend|frontend|all|follow-backend|follow-frontend|follow-all]"
    exit 1
    ;;
esac
