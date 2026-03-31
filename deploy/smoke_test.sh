#!/usr/bin/env bash
set -euo pipefail

require_var() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required environment variable: $var_name"
    exit 1
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    exit 1
  fi
}

require_cmd curl
require_var BACKEND_URL
require_var FRONTEND_URL

BACKEND_URL="${BACKEND_URL%/}"
FRONTEND_URL="${FRONTEND_URL%/}"

echo "Checking backend health endpoint"
curl -fsS "${BACKEND_URL}/health" >/dev/null

echo "Checking frontend root"
curl -fsS "${FRONTEND_URL}" >/dev/null

echo "Smoke test passed."
echo "Backend: ${BACKEND_URL}/health"
echo "Frontend: ${FRONTEND_URL}"
