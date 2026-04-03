#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERIFY_FILE="${VERIFY_FILE:-$ROOT_DIR/backend_api/sql/verify_postgres_schema.sql}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    exit 1
  fi
}

require_cmd psql

if [[ ! -f "$VERIFY_FILE" ]]; then
  echo "Verification SQL file not found: $VERIFY_FILE"
  exit 1
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  if [[ -n "${GCP_PROJECT_ID:-}" ]] && [[ -n "${DATABASE_SECRET_NAME:-}" ]]; then
    require_cmd gcloud
    echo "Reading DATABASE_URL from Secret Manager"
    DATABASE_URL="$(gcloud secrets versions access latest --secret "$DATABASE_SECRET_NAME" --project "$GCP_PROJECT_ID")"
  else
    echo "DATABASE_URL is required (or set GCP_PROJECT_ID + DATABASE_SECRET_NAME)."
    exit 1
  fi
fi

echo "Running schema verification: $VERIFY_FILE"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$VERIFY_FILE"

echo "Schema verification completed successfully."
