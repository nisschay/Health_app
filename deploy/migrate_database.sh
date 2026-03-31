#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MIGRATION_FILE="${MIGRATION_FILE:-$ROOT_DIR/backend_api/sql/2026_03_24_study_management.sql}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    exit 1
  fi
}

require_cmd psql

if [[ ! -f "$MIGRATION_FILE" ]]; then
  echo "Migration file not found: $MIGRATION_FILE"
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

echo "Applying migration: $MIGRATION_FILE"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$MIGRATION_FILE"

echo "Migration completed successfully."
