# Medical Project

Medical report analysis platform with:
- Python FastAPI backend for PDF parsing, Gemini-powered extraction, insights, chat, and PDF export
- Next.js frontend intended for Vercel deployment
- Firebase-ready auth flow using bearer tokens

## Project Layout

- backend_api/: FastAPI service
- web/: Next.js frontend (deploy this on Vercel)
- archive/: moved legacy docs and notebooks
- start.sh / stop.sh: local testing helpers
- Medical_Project.py: legacy Streamlit prototype (kept for reference)

## 1. Local Setup

### Backend

1. Create/activate virtual env and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create backend env file:

```bash
cp backend_api/.env.example backend_api/.env
```

3. Fill values in backend_api/.env:

- GEMINI_API_KEY: required for analysis/chat/export
- API_REQUIRE_AUTH: false for local open mode, true for Firebase-protected mode
- FIREBASE_CREDENTIALS_PATH: service account JSON path (required when auth=true)
- FIREBASE_PROJECT_ID: Firebase project id
- API_CORS_ORIGINS: comma-separated allowed origins (example: http://localhost:3000)

4. Run backend:

```bash
source .venv/bin/activate
uvicorn backend_api.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

1. Install dependencies:

```bash
cd web
npm ci
```

2. Create frontend env file:

```bash
cp .env.example .env.local
```

3. Fill values in web/.env.local:

- NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
- NEXT_PUBLIC_FIREBASE_* values from Firebase console (if using frontend auth)

4. Run frontend:

```bash
npm run dev
```

Open:
- Frontend: http://localhost:3000
- Backend docs: http://localhost:8000/docs

## 2. Quick Start/Stop Scripts

From project root:

```bash
./start.sh
```

This starts:
- FastAPI on port 8000
- Next.js dev server on port 3000

Logs and PIDs are stored in .run/.

Stop both services:

```bash
./stop.sh
```

## 3. Vercel Deployment (Frontend)

Deploy web/ as the Vercel project root.

### Vercel Project Settings

- Framework Preset: Next.js
- Root Directory: web
- Build Command: npm run build
- Install Command: npm ci
- Output Directory: .next

### Vercel Environment Variables

Set these in Vercel project settings:

- NEXT_PUBLIC_API_BASE_URL=https://<your-backend-domain>
- NEXT_PUBLIC_FIREBASE_API_KEY
- NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN
- NEXT_PUBLIC_FIREBASE_PROJECT_ID
- NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET
- NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID
- NEXT_PUBLIC_FIREBASE_APP_ID

## 4. Backend Deployment Notes

The backend should be deployed as a separate Python service (for example: Render, Railway, Fly.io, or container host).

Required backend env vars in production:
- GEMINI_API_KEY
- API_REQUIRE_AUTH=true (recommended)
- FIREBASE_CREDENTIALS_PATH (or mounted credentials)
- FIREBASE_PROJECT_ID
- API_CORS_ORIGINS=https://<your-vercel-domain>

## 5. Completed Next Steps

- Dashboard now uploads PDFs to /api/v1/reports/analyze
- Dashboard now supports conversational follow-up via /api/v1/reports/chat
- Optional bearer token field added for Firebase-secured backend mode
- CORS configuration added to FastAPI for deployed frontend domains
- Legacy docs and notebook moved under archive/

## 6. Archived Files

Moved to keep runtime root clean:

- archive/docs/GENAI_PROJECT_JUSTIFICATION.md
- archive/docs/PROJECT_ABSTRACT.md
- archive/docs/PRESENTATION_SLIDES_CONTENT.md
- archive/notebooks/Medical_Report_Analysis_(V2).ipynb
