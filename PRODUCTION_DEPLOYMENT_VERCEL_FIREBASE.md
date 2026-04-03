# Production Runbook: Vercel Frontend + Hugging Face Docker Backend + Neon Postgres + Firebase Auth

This runbook is tailored to your exact target architecture:
- Frontend: Vercel
- Backend API: Hugging Face Space (Docker SDK)
- Database: Neon Postgres
- Authentication: Firebase Authentication (Spark plan)

Current repo fit:
- Frontend already exists in web
- Backend already exists in backend_api (FastAPI + SQLAlchemy)
- Backend already uses PostgreSQL patterns, so Neon is a direct fit

---

## 1) Final architecture and data flow

- [ ] Keep Firebase only for Auth (ID token issuance)
- [ ] Deploy web on Vercel
- [ ] Deploy backend API as Docker Space on Hugging Face
- [ ] Use Neon as primary PostgreSQL database
- [ ] Verify Firebase ID token on backend before DB operations

Request flow:
1. User signs in from frontend using Firebase Auth.
2. Frontend attaches Bearer token to API requests.
3. Backend validates token with Firebase Admin SDK.
4. Backend executes business logic and reads/writes Neon Postgres.

---

## 2) Free-tier reality and constraints

## 2.1 Firebase Spark (Auth only)

- [ ] Spark is fine for authentication use case
- [ ] Do not use Firebase backend products that require billing upgrade

## 2.2 Neon Free

- [ ] 100 CU-hours per project per month
- [ ] 0.5 GB storage per project
- [ ] 5 GB public egress per month
- [ ] Scale-to-zero may suspend compute after inactivity

## 2.3 Hugging Face Spaces Free (Docker)

- [ ] Free CPU hardware can sleep when idle
- [ ] Startup after sleep can add cold-start delay
- [ ] Disk is not persistent for app runtime state
- [ ] Keep state in Neon and object storage, not local disk

## 2.4 Vercel Hobby

- [ ] Works for personal/small-scale projects
- [ ] Enforce limits by monitoring function and bandwidth usage

---

## 3) Prerequisites

- [ ] Firebase project created
- [ ] Neon project created
- [ ] Hugging Face account created
- [ ] Vercel account and project created
- [ ] CLI tools installed locally:
  - Node.js 20+
  - npm
  - git
  - python 3.11+

---

## 4) Firebase Auth setup

- [ ] Enable Firebase Authentication providers:
  - Email/Password
  - Google
- [ ] Add authorized domains:
  - your Vercel domain
  - your custom domain
- [ ] Download Firebase service account JSON from Firebase Console
- [ ] Keep service account JSON out of git

Frontend env values required:
- [ ] NEXT_PUBLIC_FIREBASE_API_KEY
- [ ] NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN
- [ ] NEXT_PUBLIC_FIREBASE_PROJECT_ID
- [ ] NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET
- [ ] NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID
- [ ] NEXT_PUBLIC_FIREBASE_APP_ID

---

## 5) Neon Postgres setup

- [ ] Create Neon project and database
- [ ] Create a dedicated app DB user with strong password
- [ ] Copy pooled connection string for serverless workload
- [ ] Enforce SSL in connection string

Schema bootstrapping:
- [ ] Apply SQL files used by backend:
  - backend_api/sql/2026_03_24_study_management.sql

If migrating existing production data:
- [ ] Export old Postgres
- [ ] Import into Neon
- [ ] Validate row counts and key entities

---

## 6) Backend deployment on Hugging Face Docker Space

## 6.1 Create Space

- [ ] Create a new Space
- [ ] Choose SDK: Docker
- [ ] Choose visibility based on your need
- [ ] Connect Space to backend source repository (or push Docker backend files)

## 6.2 Backend container requirements

- [ ] Docker image runs FastAPI app via uvicorn
- [ ] Container listens on the port expected by Hugging Face Space runtime
- [ ] Health endpoint stays available:
  - /health

## 6.3 Environment secrets in Space settings

Set as Hugging Face Space Secrets:
- [ ] DATABASE_URL (Neon pooled URL)
- [ ] GEMINI_API_KEY
- [ ] API_REQUIRE_AUTH=true
- [ ] FIREBASE_PROJECT_ID
- [ ] API_CORS_ORIGINS=https://your-frontend-domain
- [ ] FIREBASE_CLOCK_SKEW_SECONDS=60
- [ ] FIREBASE_SERVICE_ACCOUNT_JSON (raw JSON string)

Important code adjustment:
- [ ] Backend currently expects FIREBASE_CREDENTIALS_PATH file path.
- [ ] Add support for FIREBASE_SERVICE_ACCOUNT_JSON so backend can initialize Firebase Admin directly from env secret.

## 6.4 Backend CORS and auth

- [ ] Restrict CORS to Vercel frontend domain only
- [ ] Verify every protected API route requires valid Firebase Bearer token
- [ ] Return 401 for missing/invalid token

---

## 7) Frontend deployment on Vercel

## 7.1 Vercel project

- [ ] Set project root to web
- [ ] Build command: npm run build
- [ ] Install command: npm ci

## 7.2 Frontend environment variables

- [ ] NEXT_PUBLIC_FIREBASE_API_KEY
- [ ] NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN
- [ ] NEXT_PUBLIC_FIREBASE_PROJECT_ID
- [ ] NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET
- [ ] NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID
- [ ] NEXT_PUBLIC_FIREBASE_APP_ID
- [ ] NEXT_PUBLIC_API_URL
- [ ] NEXT_PUBLIC_API_BASE_URL
- [ ] NEXT_PUBLIC_DIRECT_API_URL

API base URL setup:
- [ ] Point all three NEXT_PUBLIC_API variables to your Hugging Face Space backend URL

---

## 8) File upload and processing strategy

- [ ] Do not rely on local backend disk for durable file storage
- [ ] Persist all important extracted data in Neon
- [ ] If original PDFs must be retained long-term, use object storage (for example Firebase Storage)
- [ ] Ensure backend logic tolerates ephemeral filesystem and restarts

---

## 9) Security checklist

- [ ] Remove committed credential files from repository and rotate any exposed keys
- [ ] Keep all secrets in Vercel and Hugging Face secret stores
- [ ] Never expose service account JSON to frontend
- [ ] Enforce HTTPS-only URLs everywhere
- [ ] Apply least privilege to DB user permissions

Repo note:
- [ ] backend_api/serviceAccountKey.json should not be used in production deployment

---

## 10) Validation before go-live

Authentication:
- [ ] Email sign-in works
- [ ] Google sign-in works
- [ ] Token refresh works
- [ ] Logout works

Backend security:
- [ ] No token returns 401
- [ ] Invalid token returns 401
- [ ] Token with wrong project returns 401

Core product paths:
- [ ] Report analyze endpoint works
- [ ] Chat endpoint works with context retention
- [ ] Insights endpoint works
- [ ] Export endpoints work

Data integrity:
- [ ] User/profile/study/report creation writes to Neon correctly
- [ ] History endpoints return expected records

Operational checks:
- [ ] Frontend home page loads
- [ ] Backend /health returns ok
- [ ] CORS allows only production frontend

---

## 11) Go-live checklist

- [ ] Deploy backend Space production image
- [ ] Deploy frontend Vercel production build
- [ ] Set production env vars in both platforms
- [ ] Run smoke tests end-to-end
- [ ] Monitor first 24 hours closely for sleep/wakeup latency and DB usage

---

## 12) Monitoring and cost control

- [ ] Track Neon CU-hours, storage, and egress daily in early phase
- [ ] Track Hugging Face Space uptime and sleep behavior
- [ ] Add backend request throttling to protect free-tier quotas
- [ ] Keep logs for auth failures, API errors, and DB connection errors

When to upgrade:
- [ ] If Space sleep latency hurts UX, move backend to paid always-on hosting
- [ ] If Neon free limits are hit, upgrade Neon first before changing architecture

---

## 13) Rollout sequence (recommended)

- [ ] Phase 1: Configure Neon and run schema
- [ ] Phase 2: Deploy backend to Hugging Face Docker Space
- [ ] Phase 3: Point frontend API env vars to Space URL and deploy Vercel
- [ ] Phase 4: Run full auth and API test suite
- [ ] Phase 5: Announce production and monitor usage limits

This sequence keeps architecture changes minimal while meeting your exact platform choice.
