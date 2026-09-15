# Medical Project Backend API

This folder contains the first migration slice away from the Streamlit-only application.

## What It Does

- Wraps the current Python medical-analysis pipeline in a FastAPI service.
- Preserves the existing report workflow: PDF upload, merge with prior Excel/CSV data, normalized records, chatbot responses, and PDF export.
- Adds Firebase-ready authentication hooks for the future Next.js frontend on Vercel.

## Environment Variables

- `GEMINI_API_KEY`: required for report analysis, chatbot responses, and PDF export summaries.
- `FIREBASE_CREDENTIALS_PATH`: path to a Firebase service account JSON file.
- `FIREBASE_PROJECT_ID`: optional Firebase project identifier.
- `API_CORS_ORIGINS`: comma-separated allowed origins (for example `http://localhost:3000,https://your-app.vercel.app`).

## Run Locally

```bash
uvicorn backend_api.app.main:app --reload
```

## Database Schema (Study Management)

Schema changes are Alembic revisions under `migrations/`; the API upgrades the
database to head at startup. See `deploy/README.md`.

## Frontend Contract

- `POST /api/v1/reports/jobs`, `GET /api/v1/reports/jobs/{id}`
- `POST /api/v1/reports/chat`
- `POST /api/v1/reports/export/pdf`
- `GET /api/v1/auth/me`
