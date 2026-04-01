# Medical Report Analyzer

> Upload medical lab reports, track biomarker trends across time,
> and get AI-powered clinical context in one secure workspace.

## What It Does

- Extracts structured data from PDF lab reports using Google Gemini AI
- Tracks test results across multiple reports and labs over time
- Identifies concerning trends across repeated abnormalities
- Provides a Clinical Assistant chatbot grounded in uploaded report data
- Includes an admin metrics dashboard for extraction quality and reliability

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, TypeScript, React 19, MUI |
| Backend | FastAPI (Python), Google Gemini API |
| Auth | Firebase Authentication |
| Database | PostgreSQL (Supabase-compatible) |
| Hosting | Vercel (frontend) + Cloud Run (backend) |

## Project Structure

medical-project/
├── web/                    # Next.js frontend
│   ├── app/                # App Router pages
│   ├── lib/                # Utilities (auth, API client, normalization)
│   └── tests/              # Vitest test suite
├── backend_api/            # FastAPI backend
│   ├── app/
│   │   ├── main.py         # API entry point + route registration
│   │   ├── services.py     # Business logic (PDF extraction, AI calls)
│   │   ├── database.py     # DB models/session + query helpers
│   │   └── auth.py         # Firebase token verification
├── scripts/                # Shared utility scripts
├── deploy/                 # Deployment helpers (Cloud Run, Vercel, smoke tests)
├── .env                    # Local env (gitignored)
└── README.md

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.11+
- PostgreSQL database
- Firebase project (for auth)
- Google Gemini API key

### 1. Clone and install

```bash
git clone <repo-url>
cd medical-project

# Install frontend
cd web && npm install && cd ..

# Install backend (root requirements for current repo layout)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment setup

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Required variables:

Frontend (copy into web/.env.local)

```bash
NEXT_PUBLIC_FIREBASE_API_KEY=
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
NEXT_PUBLIC_FIREBASE_PROJECT_ID=
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=
NEXT_PUBLIC_FIREBASE_APP_ID=
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_DIRECT_API_URL=http://127.0.0.1:8000
```

Backend (root .env or backend_api/.env)

```bash
DATABASE_URL=postgresql://...
GEMINI_API_KEY=your_gemini_key
API_REQUIRE_AUTH=true
FIREBASE_PROJECT_ID=
FIREBASE_CREDENTIALS_PATH=/absolute/path/to/serviceAccountKey.json
API_CORS_ORIGINS=http://localhost:3000
```

### 3. Database setup

Run schema setup/migrations using one of these approaches:

```bash
# Option A: deployment migration helper
./deploy/migrate_database.sh

# Option B: apply SQL directly
psql "$DATABASE_URL" -f backend_api/sql/2026_03_24_study_management.sql
```

### 4. Run locally

```bash
# Option A: using start script (recommended)
./start.sh

# Option B: manual
# Terminal 1 — Backend
source .venv/bin/activate
uvicorn backend_api.app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd web && npm run dev
```

Open: http://localhost:3000

## Key Features

### PDF Report Processing

Upload one or more medical lab PDFs. The system:

1. Extracts test names, values, units, and reference ranges
2. Normalizes test names across different lab formats
3. Classifies result severity and trends across reports
4. Saves structured findings into profile-linked studies

### Clinical Intelligence Panel

Aggregates findings across reports to show:

- Top concerning findings ranked by severity and persistence
- Trend views across studies and profile timelines
- Category-level comparisons across body systems
- Consolidated risk-oriented summary cards

### Clinical Assistant

AI chatbot with historical report context:

- Answers temporal questions such as marker movement over time
- Uses normalized data from combined report context
- Returns grounded clinical context and follow-up guidance

### Admin Metrics Dashboard

Available at /admin/metrics for admin users. Tracks:

- JSON Validity Rate
- PDF Processing Success Rate
- Hallucination Detection
- API Reliability and latency
- Context Retention across conversation turns
- Extraction F1 Score against fixtures

## Running Tests

```bash
cd web
npx vitest
npx vitest run tests/clinical-assistant-context-retention.test.ts
npx vitest run tests/extraction-f1.test.ts
```

If metrics tests require secure context, set these first:

```bash
export METRICS_TEST_TOKEN="<firebase-id-token>"
export TEST_PROFILE_ID="<profile-uuid-from-db>"
```

## Making a User Admin

```bash
psql "$DATABASE_URL" -c \
	"UPDATE users SET is_admin = true WHERE email = 'your@email.com';"
```

Then sign out and sign back in.

## Deployment

```bash
# Backend (Cloud Run helper)
./deploy/deploy_backend_cloudrun.sh

# Frontend (Vercel helper)
./deploy/deploy_frontend_vercel.sh

# Smoke test
BACKEND_URL="https://your-backend" FRONTEND_URL="https://your-frontend" ./deploy/smoke_test.sh
```

## Environment Variables Reference

Auto-generate this section with:

```bash
grep -Rho --exclude-dir=node_modules --exclude-dir=.next --include='*.ts' --include='*.tsx' 'process\.env\.[A-Z0-9_]*' web | sed 's/process\.env\.//' | sort -u
grep -Rho --exclude-dir=__pycache__ --include='*.py' 'os\.getenv("[A-Z0-9_]*"' backend_api | sed -E 's/.*"([A-Z0-9_]+)".*/\1/' | sort -u
```

## Known Limitations

- PDF extraction quality depends on source document text quality
- Reference ranges vary by lab and are interpreted per report context
- Clinical Assistant responses are informational and not medical advice

## License

[Choose and add your license here]
