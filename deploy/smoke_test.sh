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

echo "Checking analyze endpoint reachability"
analyze_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BACKEND_URL}/api/v1/reports/analyze")
if [[ "${analyze_code}" != "400" && "${analyze_code}" != "401" && "${analyze_code}" != "415" ]]; then
  echo "Unexpected status from /api/v1/reports/analyze: ${analyze_code}"
  exit 1
fi

echo "Checking analyze stream endpoint reachability"
stream_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BACKEND_URL}/api/v1/reports/analyze/stream")
if [[ "${stream_code}" != "400" && "${stream_code}" != "401" && "${stream_code}" != "415" ]]; then
  echo "Unexpected status from /api/v1/reports/analyze/stream: ${stream_code}"
  exit 1
fi

echo "Smoke test passed."
echo "Backend: ${BACKEND_URL}/health"
echo "Frontend: ${FRONTEND_URL}"
