# Deployment

One target, all free tier:

| Piece | Host |
|---|---|
| Frontend | Vercel, project root `web/` |
| Backend API | Hugging Face Space, Docker SDK, port 7860 |
| Database | Neon PostgreSQL |
| Auth | Firebase Authentication (Spark) |

The full runbook, including first-time setup, is
[`PRODUCTION_DEPLOYMENT_VERCEL_FIREBASE.md`](../PRODUCTION_DEPLOYMENT_VERCEL_FIREBASE.md).
This file covers the scripts in this folder.

## Backend

The Space builds the root `Dockerfile` on push. There is no deploy script:
push to the branch the Space tracks and it rebuilds.

Secrets to set in the Space (Settings, then Variables and secrets):

- `GEMINI_API_KEY`
- `DATABASE_URL` (Neon connection string, `sslmode=require`)
- `FIREBASE_PROJECT_ID`
- `FIREBASE_SERVICE_ACCOUNT_JSON` (the whole Admin SDK JSON, pasted)
- `API_CORS_ORIGINS` (comma-separated, including the Vercel URL)

Optional limits, with their defaults:

- `MAX_UPLOAD_FILES` (20), `MAX_UPLOAD_FILE_MB` (10), `MAX_UPLOAD_TOTAL_MB` (25)
- `RATE_LIMIT_PER_MINUTE` (20)
- `PDF_OCR_FALLBACK_ENABLED` (true), `PDF_OCR_MAX_PAGES` (5)
- `GEMINI_EXTRACTION_MODELS`, `GEMINI_CHAT_MODELS` (gemini-3-flash), `GEMINI_RPM` (10), `EXTRACTION_WORKERS` (4)

## Keeping the Space awake

A free Space sleeps when idle, and the first visitor afterwards waits through a
cold start. Point a free uptime monitor at the health endpoint every 5 minutes:

```
https://<user>-<space>.hf.space/health
```

[UptimeRobot](https://uptimerobot.com) and [cron-job.org](https://cron-job.org)
both do this on a free plan. The endpoint returns the database status too, so a
sleeping Neon branch shows up as `{"status":"ok","database":"unreachable"}`.

## Frontend

```bash
VERCEL_TOKEN=... ./deploy/deploy_frontend_vercel.sh
```

## Database

Migrations live in `backend_api/sql/` and are applied in filename order, each
recorded in a `schema_migrations` table so re-running is safe:

```bash
DATABASE_URL="postgresql://..." ./deploy/migrate_database.sh
```

Verify the result:

```bash
DATABASE_URL="postgresql://..." ./deploy/verify_database_schema.sh
```

## Smoke test

```bash
BACKEND_URL="https://<user>-<space>.hf.space" \
FRONTEND_URL="https://<app>.vercel.app" \
./deploy/smoke_test.sh
```
