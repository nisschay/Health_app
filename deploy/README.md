# Deployment Toolkit

This folder contains repeatable deployment helpers for this project.

## 1) Deploy Backend (Cloud Run)

Script: `deploy/deploy_backend_cloudrun.sh`

Required environment variables:

- `GCP_PROJECT_ID`
- `FIREBASE_PROJECT_ID`
- `API_CORS_ORIGINS`
- `GEMINI_API_KEY`
- `DATABASE_URL`
- `FIREBASE_SERVICE_ACCOUNT_FILE` (path to Firebase Admin JSON)

Optional variables:

- `GCP_REGION` (default: `us-central1`)
- `AR_REPOSITORY` (default: `medical-project`)
- `CLOUD_RUN_SERVICE` (default: `medical-backend`)
- `IMAGE_NAME` (default: `medical-backend`)
- `IMAGE_TAG` (default: timestamp)
- `CLOUD_SQL_CONNECTION` (for Cloud SQL socket attach)
- `FIREBASE_CLOCK_SKEW_SECONDS` (default: `60`)
- `GEMINI_SECRET_NAME` (default: `medical-gemini-api-key`)
- `DATABASE_SECRET_NAME` (default: `medical-database-url`)
- `FIREBASE_SECRET_NAME` (default: `medical-firebase-admin`)

Example:

```bash
export GCP_PROJECT_ID="your-project-id"
export FIREBASE_PROJECT_ID="your-project-id"
export API_CORS_ORIGINS="https://your-app.vercel.app"
export GEMINI_API_KEY="your-gemini-api-key"
export DATABASE_URL="postgresql+psycopg2://user:password@/medical_project?host=/cloudsql/PROJECT:REGION:INSTANCE"
export FIREBASE_SERVICE_ACCOUNT_FILE="$PWD/backend_api/serviceAccountKey.json"
export CLOUD_SQL_CONNECTION="PROJECT:REGION:INSTANCE"

./deploy/deploy_backend_cloudrun.sh
```

## 2) Run Database Migration

Script: `deploy/migrate_database.sh`

Requirements:

- `DATABASE_URL`, or
- `GCP_PROJECT_ID` + `DATABASE_SECRET_NAME` (to read URL from Secret Manager)

Example:

```bash
export DATABASE_URL="postgresql://user:password@host:5432/medical_project"
./deploy/migrate_database.sh
```

## 3) Deploy Frontend (Vercel)

Script: `deploy/deploy_frontend_vercel.sh`

Before running:

- Configure Vercel project root as `web`
- Ensure Vercel env vars are set (`NEXT_PUBLIC_*` values)

Example:

```bash
./deploy/deploy_frontend_vercel.sh
```

Skip local build check:

```bash
SKIP_LOCAL_BUILD=1 ./deploy/deploy_frontend_vercel.sh
```

## 4) Smoke Test Production

Script: `deploy/smoke_test.sh`

Required variables:

- `BACKEND_URL`
- `FRONTEND_URL`

Example:

```bash
BACKEND_URL="https://your-backend.run.app" \
FRONTEND_URL="https://your-app.vercel.app" \
./deploy/smoke_test.sh
```

## 5) Env Templates

- Backend template: `backend_api/.env.example`
- Frontend template: `web/.env.example`

Use these as starting points for local and production environment configuration.
