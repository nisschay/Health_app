#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    exit 1
  fi
}

require_var() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required environment variable: $var_name"
    exit 1
  fi
}

upsert_secret_from_stdin() {
  local project="$1"
  local secret_name="$2"

  if ! gcloud secrets describe "$secret_name" --project "$project" >/dev/null 2>&1; then
    gcloud secrets create "$secret_name" \
      --project "$project" \
      --replication-policy="automatic" >/dev/null
  fi

  gcloud secrets versions add "$secret_name" \
    --project "$project" \
    --data-file=- >/dev/null
}

upsert_secret_from_file() {
  local project="$1"
  local secret_name="$2"
  local file_path="$3"

  if [[ ! -f "$file_path" ]]; then
    echo "Secret file not found: $file_path"
    exit 1
  fi

  if ! gcloud secrets describe "$secret_name" --project "$project" >/dev/null 2>&1; then
    gcloud secrets create "$secret_name" \
      --project "$project" \
      --replication-policy="automatic" >/dev/null
  fi

  gcloud secrets versions add "$secret_name" \
    --project "$project" \
    --data-file="$file_path" >/dev/null
}

require_cmd gcloud

require_var GCP_PROJECT_ID
require_var FIREBASE_PROJECT_ID
require_var API_CORS_ORIGINS
require_var GEMINI_API_KEY
require_var DATABASE_URL
require_var FIREBASE_SERVICE_ACCOUNT_FILE

GCP_REGION="${GCP_REGION:-us-central1}"
AR_REPOSITORY="${AR_REPOSITORY:-medical-project}"
CLOUD_RUN_SERVICE="${CLOUD_RUN_SERVICE:-medical-backend}"
IMAGE_NAME="${IMAGE_NAME:-medical-backend}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
CLOUD_SQL_CONNECTION="${CLOUD_SQL_CONNECTION:-}"
FIREBASE_CLOCK_SKEW_SECONDS="${FIREBASE_CLOCK_SKEW_SECONDS:-60}"

GEMINI_SECRET_NAME="${GEMINI_SECRET_NAME:-medical-gemini-api-key}"
DATABASE_SECRET_NAME="${DATABASE_SECRET_NAME:-medical-database-url}"
FIREBASE_SECRET_NAME="${FIREBASE_SECRET_NAME:-medical-firebase-admin}"

IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Configuring gcloud project: ${GCP_PROJECT_ID}"
gcloud config set project "$GCP_PROJECT_ID" >/dev/null

echo "Enabling required Google Cloud services"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  --project "$GCP_PROJECT_ID" >/dev/null

echo "Ensuring Artifact Registry repository exists"
if ! gcloud artifacts repositories describe "$AR_REPOSITORY" \
  --location "$GCP_REGION" \
  --project "$GCP_PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPOSITORY" \
    --location "$GCP_REGION" \
    --project "$GCP_PROJECT_ID" \
    --repository-format docker >/dev/null
fi

echo "Updating secrets in Secret Manager"
printf "%s" "$GEMINI_API_KEY" | upsert_secret_from_stdin "$GCP_PROJECT_ID" "$GEMINI_SECRET_NAME"
printf "%s" "$DATABASE_URL" | upsert_secret_from_stdin "$GCP_PROJECT_ID" "$DATABASE_SECRET_NAME"
upsert_secret_from_file "$GCP_PROJECT_ID" "$FIREBASE_SECRET_NAME" "$FIREBASE_SERVICE_ACCOUNT_FILE"

echo "Building backend image: ${IMAGE_URI}"
gcloud builds submit "$ROOT_DIR" --tag "$IMAGE_URI" --project "$GCP_PROJECT_ID"

echo "Deploying Cloud Run service: ${CLOUD_RUN_SERVICE}"
DEPLOY_ARGS=(
  run deploy "$CLOUD_RUN_SERVICE"
  --project "$GCP_PROJECT_ID"
  --region "$GCP_REGION"
  --image "$IMAGE_URI"
  --allow-unauthenticated
  --quiet
  --set-env-vars "API_REQUIRE_AUTH=true,FIREBASE_PROJECT_ID=${FIREBASE_PROJECT_ID},API_CORS_ORIGINS=${API_CORS_ORIGINS},FIREBASE_CLOCK_SKEW_SECONDS=${FIREBASE_CLOCK_SKEW_SECONDS},FIREBASE_CREDENTIALS_PATH=/secrets/firebase/serviceAccountKey.json"
  --set-secrets "GEMINI_API_KEY=${GEMINI_SECRET_NAME}:latest,DATABASE_URL=${DATABASE_SECRET_NAME}:latest,/secrets/firebase/serviceAccountKey.json=${FIREBASE_SECRET_NAME}:latest"
)

if [[ -n "$CLOUD_SQL_CONNECTION" ]]; then
  DEPLOY_ARGS+=(--add-cloudsql-instances "$CLOUD_SQL_CONNECTION")
fi

gcloud "${DEPLOY_ARGS[@]}"

SERVICE_URL="$(gcloud run services describe "$CLOUD_RUN_SERVICE" --project "$GCP_PROJECT_ID" --region "$GCP_REGION" --format='value(status.url)')"

echo
echo "Backend deployed successfully."
echo "Cloud Run URL: ${SERVICE_URL}"
echo "Health check: ${SERVICE_URL}/health"
