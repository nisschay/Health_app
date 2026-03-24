# Medical Project Backend API

This folder contains the first migration slice away from the Streamlit-only application.

## What It Does

- Wraps the current Python medical-analysis pipeline in a FastAPI service.
- Preserves the existing report workflow: PDF upload, merge with prior Excel/CSV data, normalized records, chatbot responses, and PDF export.
- Adds Firebase-ready authentication hooks for the future Next.js frontend on Vercel.

## Environment Variables

- `GEMINI_API_KEY`: required for report analysis, chatbot responses, and PDF export summaries.
- `API_REQUIRE_AUTH`: set to `true` to require Firebase bearer tokens.
- `FIREBASE_CREDENTIALS_PATH`: path to a Firebase service account JSON file.
- `FIREBASE_PROJECT_ID`: optional Firebase project identifier.
- `API_CORS_ORIGINS`: comma-separated allowed origins (for example `http://localhost:3000,https://your-app.vercel.app`).

## Run Locally

```bash
uvicorn backend_api.app.main:app --reload
```

## Database Schema (Study Management)

The API now includes additive schema support for three new tables:

- `profiles`
- `studies`
- `reports`

Apply SQL migration manually when needed:

```bash
psql "$DATABASE_URL" -f backend_api/sql/2026_03_24_study_management.sql
```

Notes:

- Existing tables are preserved (`users`, `report_analyses`).
- FastAPI startup still calls SQLAlchemy `create_all` for ORM-managed table creation.
- The SQL migration includes indexes and triggers to keep `studies.updated_at` in sync.
- The migration currently validates that `users.id` is an integer type before creating `profiles.account_owner_id`.
- On auth sync, backend user upsert now auto-creates a default `self` profile if one does not exist.

## Planned Frontend Contract

- `POST /api/v1/reports/analyze`
- `POST /api/v1/reports/chat`
- `POST /api/v1/reports/insights`
- `POST /api/v1/reports/export/pdf`
- `GET /api/v1/auth/me`
