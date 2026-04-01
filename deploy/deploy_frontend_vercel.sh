#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="$ROOT_DIR/web"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    exit 1
  fi
}

require_cmd npm
require_cmd vercel

if [[ ! -f "$WEB_DIR/package.json" ]]; then
  echo "Could not find web/package.json."
  exit 1
fi

cd "$WEB_DIR"

echo "Installing frontend dependencies"
npm ci

if [[ "${SKIP_LOCAL_BUILD:-0}" != "1" ]]; then
  echo "Running local frontend build verification"
  npm run build
fi

echo "Deploying frontend to Vercel production"
vercel deploy --prod --yes

echo "Frontend deployment command completed."
