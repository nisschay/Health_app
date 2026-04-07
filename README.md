# Medical Report Analyzer

Production URL: https://health-app-lovat-eta.vercel.app/

Medical Report Analyzer is a full-stack clinical data intelligence system that ingests raw lab PDFs, extracts structured findings with Gemini, normalizes noisy test nomenclature, persists longitudinal records in PostgreSQL, and serves report-aware analytics and assistant responses through a secured web application.

This repository contains both:

- A modern production stack: Next.js frontend + FastAPI backend + Firebase Auth + PostgreSQL.
- A legacy Streamlit stack that still powers core extraction helpers and remains useful for experimentation.

## 1) System Scope

Implemented capabilities across the project:

- Multi-file PDF ingestion with optional merge against prior Excel/CSV exports.
- AI extraction and parsing into a structured medical record schema.
- Two-pass normalization of categories, test names, statuses, and duplicates.
- Longitudinal study model: account owner -> profiles -> studies -> reports.
- Combined-report views for trend analysis across report timelines.
- Clinical assistant workflow with report context and retrieval-grounded guideline augmentation.
- Export to PDF and Excel.
- Firebase-based authentication and backend token verification.
- Admin-facing validation metrics UI and telemetry contract.
- Deployment scripts for Vercel frontend + containerized backend hosting.

## 2) Technical Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/310d8357-d821-4659-af9d-ee7bc2b01fb4" />


Primary architecture decisions:

- Frontend and backend are decoupled by HTTP API contracts and env-configurable base URLs.
- Authentication is delegated to Firebase; authorization and ownership checks are enforced in backend routes.
- Durable state is in PostgreSQL, not in runtime filesystem.
- Parsing and feature engineering are centralized in Python services and normalization modules.
- Frontend includes defensive fallback logic to retry direct API base URLs on proxy/edge failures.

## 3) Repository Components

Top-level modules and responsibilities:

- web/: Next.js frontend, App Router pages, auth context, API client, dashboard components, clinical assistant route, metrics client/tests.
- backend_api/: FastAPI service, SQLAlchemy models/repositories, auth verification, parsing/analysis services, normalization, SQL migrations.
- deploy/: Production deployment helpers for Cloud Run, Vercel, DB migration, schema verification, smoke test.
- scripts/: Data maintenance utilities, including normalization migration for persisted JSON payloads.
- main.py + Medical_Project.py + Helper_Functions.py: legacy Streamlit app and extraction/business logic origin.
- Medical_Project-Backend/: mirrored backend slice used in some deployment and migration workflows.

## 4) Ingestion and Parsing Pipeline

### 4.1 Upload and orchestration

- Frontend dashboard sends multipart uploads to:
	- POST /api/v1/reports/analyze
	- POST /api/v1/reports/analyze/stream (SSE progress mode)
- Streamed status includes stage and per-file events:
	- stages: validating, uploading, processing, saving
	- file steps: queued, extracting, parsing, done, failed

### 4.2 PDF extraction and LLM parsing

Backend service path:

- Extract raw text from PDF bytes using extract_text_from_pdf.
- Call Gemini extraction workflow through analyze_medical_report_with_gemini.
- Transform LLM output into tabular records via create_structured_dataframe.
- Merge with optional existing CSV/XLSX data via process_existing_excel_csv.
- Consolidate patient identity and demographics across reports.

Core implementation: backend_api/app/services.py and Helper_Functions.py.

### 4.3 Data engineering and normalization

- Parse numeric values and dates.
- Normalize categories and test names to canonical forms.
- Resolve aliases and reduce semantic duplicates.
- Recompute health summary and body-system aggregations from normalized records.

Normalization entry points:

- backend_api/app/normalization.py
- web/lib/normalizeTest.ts
- scripts/migrateNormalize.ts (one-time DB payload cleanup)

### 4.4 Persistence and study-level storage

When saved to a study, analysis payload is scoped per source report and stored into reports.analysis_data (JSONB), then later recombined for timeline analysis through GET /api/v1/studies/{study_id}/combined-report.

## 5) Data Model and Storage Design

Main relational entities (PostgreSQL):

- users: Firebase identity mapping, admin flag, login metadata.
- report_analyses: saved analysis history snapshots (JSON text payload).
- profiles: patient profiles under account owner.
- studies: logical longitudinal group under profile.
- reports: uploaded report instances per study with normalized analysis_data (JSONB).
- metrics: telemetry table for validation and reliability metrics.

Schema assets:

- backend_api/sql/2026_03_24_study_management.sql
- backend_api/sql/2026_03_31_firebase_postgres_bootstrap.sql
- backend_api/sql/verify_postgres_schema.sql

## 6) Backend API Surface

Implemented FastAPI route groups under /api/v1:

- Auth
	- GET /auth/me
	- POST /auth/sync
- Profile and study management
	- GET /studies/profiles
	- POST /studies/profiles
	- GET /studies/profiles/{profile_id}/studies
	- POST /studies
	- GET /studies/dashboard
	- POST /studies/{study_id}/reports/save-analysis
	- GET /studies/{study_id}/combined-report
- Report analysis and history
	- POST /reports/analyze
	- POST /reports/analyze/stream
	- POST /reports/save
	- GET /reports/history
	- GET /reports/history/{analysis_id}
- Clinical and export
	- POST /reports/chat
	- POST /reports/insights
	- POST /reports/export/pdf
	- POST /reports/export/excel
- Utility
	- GET /health

## 7) Frontend Architecture and Component Model

Frontend framework:

- Next.js 15 + React 19 + TypeScript App Router.
- Global auth provider in web/app/layout.tsx.
- Route protection via middleware cookie gate plus server-side token validation in backend.

Major frontend modules:

- Authentication
	- web/lib/auth-context.tsx
	- Firebase sign-in (Google + email/password), token provider wiring, backend sync.
- Dashboard and workflows
	- web/app/dashboard/page.tsx
	- Study flow modal, uploads, SSE progress UI, history retrieval, save-to-study.
- Visual components
	- TrendChart, AlertsByCategory, OrganizedDataTree, ClinicalChatPanel.
- Profile report view
	- web/app/dashboard/reports/[profileId]/page.tsx
	- Combined report access and study-level summarization.
- Admin metrics UI
	- web/app/admin/metrics/page.tsx
	- Displays quality and reliability cards, sparkline trends, failed PDFs, token usage.

## 8) Clinical Assistant and Retrieval Layer

Clinical assistant flow:

- Frontend sends user question + session history + analysis identifier to /api/clinical-assistant.
- Route handler (web/app/api/clinical-assistant/route.ts):
	- Optionally fetches full analysis from backend (history or study-combined endpoint).
	- Aggregates report timeline and latest findings.
	- Builds a structured system prompt with chronology and persistent concerns.
	- Retrieves guideline snippets from local KB using scoring-based retrieval.
	- Forwards enriched request to backend /api/v1/reports/chat.

RAG assets:

- web/lib/medicalKnowledge.ts (curated guidelines and interpretation bands)
- web/lib/ragRetrieval.ts (token/intent scoring retrieval engine)

## 9) Security and Authentication

Authentication model:

- Firebase issues client ID tokens.
- Frontend includes Authorization: Bearer <token> in protected calls.
- Backend verifies token with Firebase Admin SDK and enforces project/audience checks.

Security controls in codebase:

- CORS allowlist controlled by API_CORS_ORIGINS.
- Auth-required endpoints reject missing/invalid tokens.
- Clock-skew tolerant token verification via FIREBASE_CLOCK_SKEW_SECONDS (0..60).
- Middleware redirects unauthenticated route access to /login.

## 10) Deployment and Hosting

Live frontend:

- https://health-app-lovat-eta.vercel.app/

Deployment strategies implemented in repository:

- Frontend hosting
	- Vercel with web/ as project root.
	- Configured by web/vercel.json and deploy/deploy_frontend_vercel.sh.
- Backend hosting
	- Cloud Run workflow script: deploy/deploy_backend_cloudrun.sh.
	- Dockerized backend images via root Dockerfile and backend_api/Dockerfile.
- Database
	- PostgreSQL-compatible environments (including Supabase/Neon patterns).
	- Migration and verification scripts in deploy/.

Additional runbook:

- PRODUCTION_DEPLOYMENT_VERCEL_FIREBASE.md includes a Hugging Face Docker Space + Neon deployment option.

## 11) Local Development

### 11.1 Prerequisites

- Node.js 18+
- Python 3.11+
- PostgreSQL
- Firebase project and service account
- Gemini API key

### 11.2 Setup

```bash
git clone <repo-url>
cd Medical_Project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd web
npm ci
cd ..

cp .env.example .env
cp web/.env.example web/.env.local
cp backend_api/.env.example backend_api/.env
```

### 11.3 Run

Recommended:

```bash
./start.sh
```

Manual:

```bash
# backend
source .venv/bin/activate
uvicorn backend_api.app.main:app --reload --port 8000

# frontend
cd web
npm run dev
```

Logs:

```bash
./logs.sh backend
./logs.sh frontend
./logs.sh follow-backend
```

Stop local services:

```bash
./stop.sh
```

## 12) Testing and Validation

Frontend and integration-oriented tests:

```bash
cd web
npx vitest
npx vitest run tests/clinical-assistant-context-retention.test.ts
npx vitest run tests/extraction-f1.test.ts
```

Optional metric-log test prerequisites:

```bash
export METRICS_TEST_TOKEN="<firebase-id-token>"
```

Operational checks:

```bash
BACKEND_URL="https://your-backend" \
FRONTEND_URL="https://your-frontend" \
./deploy/smoke_test.sh
```

## 13) Environment Variables

Most important variables by layer:

- Frontend
	- NEXT_PUBLIC_API_URL
	- NEXT_PUBLIC_API_BASE_URL
	- NEXT_PUBLIC_DIRECT_API_URL
	- NEXT_PUBLIC_FIREBASE_API_KEY
	- NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN
	- NEXT_PUBLIC_FIREBASE_PROJECT_ID
	- NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET
	- NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID
	- NEXT_PUBLIC_FIREBASE_APP_ID
- Backend
	- DATABASE_URL
	- GEMINI_API_KEY
	- API_REQUIRE_AUTH
	- API_CORS_ORIGINS
	- FIREBASE_PROJECT_ID
	- FIREBASE_CREDENTIALS_PATH or FIREBASE_SERVICE_ACCOUNT_JSON
	- FIREBASE_CLOCK_SKEW_SECONDS

## 14) Legacy Stack and Migration Status

Legacy artifacts retained in repository:

- Streamlit application entry points: main.py and Medical_Project.py.
- Shared extraction and analytics utilities: Helper_Functions.py.
- Legacy normalization helpers and mapping tests: unify_test_names.py, test_category_mapping.py.

Current production-oriented stack is the Next.js + FastAPI path, while legacy files remain useful for model experimentation and reference implementations.

## 15) Known Constraints

- Extraction quality is constrained by PDF text quality and OCR characteristics.
- Lab-specific reference ranges can vary and must be interpreted in per-report context.
- Clinical assistant outputs are informational support, not medical diagnosis.

## 16) License

Add your preferred license file and update this section.
