#!/usr/bin/env bash
# Apply every migration in backend_api/sql in filename order, once each.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SQL_DIR="$ROOT_DIR/backend_api/sql"

if ! command -v psql >/dev/null 2>&1; then
  echo "Missing required command: psql"
  exit 1
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is required."
  exit 1
fi

# SQLAlchemy dialect prefixes are valid for the app but not for psql.
PSQL_URL="${DATABASE_URL/postgresql+psycopg2:\/\//postgresql://}"
PSQL_URL="${PSQL_URL/postgres+psycopg2:\/\//postgres://}"

psql "$PSQL_URL" -v ON_ERROR_STOP=1 -q -c "
  CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
  );"

applied=0
skipped=0

# verify_postgres_schema.sql is a read-only check, not a migration.
for migration in $(find "$SQL_DIR" -maxdepth 1 -name '*.sql' ! -name 'verify_*' | sort); do
  filename="$(basename "$migration")"
  already="$(psql "$PSQL_URL" -tAc "SELECT 1 FROM schema_migrations WHERE filename = '$filename'")"

  if [[ "$already" == "1" ]]; then
    echo "skip    $filename"
    skipped=$((skipped + 1))
    continue
  fi

  echo "apply   $filename"
  psql "$PSQL_URL" -v ON_ERROR_STOP=1 -q -f "$migration"
  psql "$PSQL_URL" -v ON_ERROR_STOP=1 -q \
    -c "INSERT INTO schema_migrations (filename) VALUES ('$filename') ON CONFLICT DO NOTHING;"
  applied=$((applied + 1))
done

echo "Migrations complete: $applied applied, $skipped already present."
