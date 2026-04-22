# Medical Report Analyzer: Deep-Dive Q&A

Date: April 19, 2026

This document answers the architecture, data model, ingestion, clinical assistant, and testing questions in depth, based on the current codebase.

## 1) Architecture and design

### Q1. Your README mentions a /reports/analyze/stream endpoint with SSE. How does SSE work at the protocol level? Why is it one-directional? Why choose it over WebSockets?

**Answer**

Server-Sent Events (SSE) is HTTP response streaming from server to client.

At protocol level:
1. The client opens a normal HTTP request.
2. The server returns `Content-Type: text/event-stream` and keeps the connection open.
3. The server pushes event frames in text format, typically lines that begin with `data:` and end with a blank line.
4. Each event arrives as incremental bytes; the client parses each frame and updates UI state.
5. The stream closes when processing completes or fails.

In this project, the analysis stream endpoint sends JSON payloads wrapped as SSE `data:` frames and terminates on `done` or `error` events.

Relevant implementation:
- [backend_api/app/main.py](../backend_api/app/main.py#L743)
- [backend_api/app/main.py](../backend_api/app/main.py#L820)
- [web/lib/api.ts](../web/lib/api.ts#L348)

Why one-directional:
- SSE is server to client only on that open stream.
- Client to server communication happens via separate HTTP requests.
- This model is ideal for long-running tasks where the server reports progress, status, and final output.

Why SSE over WebSockets here:
1. Simpler infrastructure path: still plain HTTP semantics.
2. Easier proxy/load balancer compatibility for progress updates.
3. No need for full duplex messaging in the analysis path.
4. Easier to reason about request lifecycle: one upload request in, many progress events out.

WebSockets are stronger when you need frequent two-way low-latency signaling on the same connection (for example collaborative editing or near-real-time bi-directional control). For batch analysis progress, SSE is a clean fit.

---

### Q2. You have both /reports/analyze and /reports/analyze/stream. When should a client use the non-streaming version?

**Answer**

Use `/reports/analyze` when you want a simple request-response contract and do not need live progress updates.

Best use-cases:
1. CLI or backend integrations that only care about final output.
2. Environments where streaming is blocked or fragile.
3. Simpler clients where incremental UI state is unnecessary.
4. Retries with idempotent orchestration around final payload only.

Use `/reports/analyze/stream` when you need:
1. Stage-level UX (`validating`, `uploading`, `processing`, `saving`).
2. Per-file progress and failure visibility.
3. Better user trust during long-running analysis.

Relevant implementation:
- Non-stream endpoint: [backend_api/app/main.py](../backend_api/app/main.py#L679)
- Stream endpoint: [backend_api/app/main.py](../backend_api/app/main.py#L743)
- Frontend stream consumption: [web/app/dashboard/page.tsx](../web/app/dashboard/page.tsx#L981)

---

### Q3. Firebase token verification happens in backend with Firebase Admin SDK. Walk through the full auth flow from Google sign-in click to protected API success.

**Answer**

End-to-end flow:

1. User initiates sign-in in the UI.
- The login page triggers `signInWithGoogle`.
- [web/app/login/page.tsx](../web/app/login/page.tsx#L92)

2. Firebase client SDK handles Google popup auth.
- Auth state is managed in the auth provider.
- [web/lib/auth-context.tsx](../web/lib/auth-context.tsx#L166)

3. On auth state change, frontend stores user session signals and sets token provider.
- The app registers a token getter used by all API calls.
- [web/lib/auth-context.tsx](../web/lib/auth-context.tsx#L72)
- [web/lib/api.ts](../web/lib/api.ts#L7)

4. Frontend syncs the user with backend via `/api/v1/auth/sync`.
- Sends `Authorization: Bearer <firebase_id_token>`.
- Backend upserts user record in PostgreSQL.
- [web/lib/auth-context.tsx](../web/lib/auth-context.tsx#L97)
- [backend_api/app/main.py](../backend_api/app/main.py#L103)

5. Future protected API calls automatically attach Bearer token.
- `authFetch` injects auth header.
- [web/lib/api.ts](../web/lib/api.ts#L49)

6. Backend validates token with Firebase Admin SDK.
- Extract Bearer token.
- Verify cryptographically and validate claims.
- Enforce project/audience consistency.
- [backend_api/app/auth.py](../backend_api/app/auth.py#L32)
- [backend_api/app/auth.py](../backend_api/app/auth.py#L103)
- [backend_api/app/auth.py](../backend_api/app/auth.py#L145)

7. Request is authorized and proceeds to endpoint logic.
- If token invalid/expired/revoked/mismatched, backend returns 401/503 with actionable messages.

Additional gate:
- Middleware uses a lightweight auth-presence cookie to protect frontend routes before backend validation occurs.
- [web/middleware.ts](../web/middleware.ts#L14)

---

### Q4. You mention FIREBASE_CLOCK_SKEW_SECONDS. What is clock skew and why does it break token verification?

**Answer**

Clock skew is time mismatch between systems involved in JWT issuance and validation.

Why it matters:
- ID tokens contain time-based claims (`iat`, `nbf`, `exp`).
- If backend clock is too far behind/ahead, valid tokens can appear “used too early” or expired.
- This causes false auth failures.

How this system mitigates it:
1. `verify_id_token` is called with `clock_skew_seconds`.
2. Config is bounded for safety.
3. Startup script has optional drift detection and correction logic.

Relevant implementation:
- [backend_api/app/auth.py](../backend_api/app/auth.py#L103)
- [backend_api/app/config.py](../backend_api/app/config.py#L69)
- [backend_api/.env.example](../backend_api/.env.example#L13)
- [start.sh](../start.sh#L111)

Operationally, skew tolerance avoids brittle failures while preserving token expiry discipline.

---

### Q5. Your backend has a metrics table for validation/reliability telemetry. What metrics are tracked and how do you use them to improve the system?

**Answer**

Schema-level telemetry support exists via `metrics(metric_name, value, metadata, created_at, user_id)`.

Relevant schema:
- [backend_api/sql/2026_03_31_firebase_postgres_bootstrap.sql](../backend_api/sql/2026_03_31_firebase_postgres_bootstrap.sql#L146)

Frontend metric taxonomy includes:
1. `json_validity`
2. `pdf_processing_success`
3. `hallucination_detection`
4. `api_reliability`
5. `context_retention`
6. `extraction_f1`

Relevant code:
- [web/lib/metrics.ts](../web/lib/metrics.ts#L6)
- [web/app/admin/metrics/page.tsx](../web/app/admin/metrics/page.tsx#L23)

How to use these metrics for continuous improvement:
1. **Reliability triage**: track failure rate by file characteristics and identify top failure modes.
2. **Model quality monitoring**: watch extraction F1 and hallucination trends over deploys.
3. **Latency SLO control**: use p95 response times to trigger scaling/caching work.
4. **Safety drift detection**: alert when hallucination or context retention worsens.
5. **Release gating**: require thresholds before production rollout.

Important codebase nuance:
- The active runtime backend (wired by `start.sh`) is `backend_api`, which currently does not expose `/api/v1/metrics/log` or `/api/v1/admin/metrics` routes.
- Those endpoints exist in the parallel `Medical_Project-Backend` variant.

Reference points:
- Active backend launch path: [start.sh](../start.sh#L209)
- Implemented metrics routes in parallel backend: [Medical_Project-Backend/app/main.py](../Medical_Project-Backend/app/main.py#L975)

## 2) Data model

### Q6. You have report_analyses and reports. What is the difference and why both?

**Answer**

They serve different persistence purposes.

`report_analyses`:
- User-centric analysis history snapshots.
- Stores one saved analysis payload per save action.
- Useful for quick historical retrieval by user.

`reports`:
- Study-centric longitudinal storage.
- Represents uploaded report instances tied to a study.
- Contains JSONB `analysis_data` and normalization fields.
- Designed for timeline assembly and study-level analytics.

Relevant models:
- [backend_api/app/database.py](../backend_api/app/database.py#L57)
- [backend_api/app/database.py](../backend_api/app/database.py#L113)

Why both:
1. Snapshot history UX and study timeline UX are distinct needs.
2. Snapshot table supports quick “analysis history” retrieval.
3. Study table supports composable per-report timeline construction and normalization governance.

---

### Q7. Why is study separate from profile? What real-world concept does it model?

**Answer**

`profile` models a person (self/family member). `study` models a clinically meaningful tracking context for that person.

Real-world examples:
1. Same person, separate studies for diabetes follow-up and liver workup.
2. A new episode or focused monitoring period gets its own study.
3. Avoids mixing unrelated test journeys into one undifferentiated timeline.

This maps cleanly to:
- users -> profiles -> studies -> reports

Relevant schema:
- [backend_api/sql/2026_03_31_firebase_postgres_bootstrap.sql](../backend_api/sql/2026_03_31_firebase_postgres_bootstrap.sql#L78)
- [backend_api/sql/2026_03_31_firebase_postgres_bootstrap.sql](../backend_api/sql/2026_03_31_firebase_postgres_bootstrap.sql#L95)

---

### Q8. You store analysis_data as JSONB in reports. What queries do you run against JSONB and how do you index it?

**Answer**

Current active backend primarily retrieves rows by relational keys (`study_id`, `report_date`, `uploaded_at`) and processes JSON in Python.

Current index posture:
- Relational indexes exist for study/time access patterns.
- No dedicated GIN index on `analysis_data` in active migration.

Relevant schema:
- [backend_api/sql/2026_03_31_firebase_postgres_bootstrap.sql](../backend_api/sql/2026_03_31_firebase_postgres_bootstrap.sql#L140)

What this implies:
1. Timeline retrieval is efficient.
2. Deep JSON predicate queries are not optimized yet.
3. For heavy JSON analytics, adding targeted JSONB indexes is recommended.

Examples of future useful indexes:
1. GIN index on `analysis_data` for containment/path existence queries.
2. Expression indexes for high-frequency JSON fields (for example extracted date or lab name if queried directly in SQL).

## 3) Ingestion pipeline

### Q9. Ingestion stages are validating -> uploading -> processing -> saving. What can go wrong at each stage and how are errors surfaced to frontend?

**Answer**

**Validating**
- Failures: no files, malformed inputs.
- Surface: stage event + terminal `error` event with message.

**Uploading**
- Failures: network interruptions, gateway limits/timeouts.
- Surface: request failure before processing stage or stream interruption.

**Processing**
- Failures: no text extraction, OCR failure, model extraction failure, quota/rate limits, parsing issues.
- Surface: per-file `failed` events and final `error` event; includes reason.

**Saving**
- In stream endpoint, saving stage is progress signaling.
- Actual persistence happens in subsequent frontend save calls.
- Failures there are surfaced as save-state error in dashboard UI.

Relevant implementation:
- Stream stage/file events: [backend_api/app/main.py](../backend_api/app/main.py#L774)
- File processing and failures: [backend_api/app/services.py](../backend_api/app/services.py#L97)
- Frontend stream event handling: [web/app/dashboard/page.tsx](../web/app/dashboard/page.tsx#L901)
- Frontend save status handling: [web/app/dashboard/page.tsx](../web/app/dashboard/page.tsx#L984)

---

### Q10. You call extract_text_from_pdf before Gemini. What about scanned/image PDFs? How does the system handle that?

**Answer**

It uses a two-step extraction strategy:
1. Try native PDF text extraction via PyPDF2.
2. If text is empty or below threshold, attempt OCR fallback using pdf2image + pytesseract.

Relevant implementation:
- OCR fallback function: [Helper_Functions.py](../Helper_Functions.py#L292)
- Main extraction with fallback thresholds: [Helper_Functions.py](../Helper_Functions.py#L317)

Important behavior:
1. OCR can be disabled via env.
2. OCR depends on optional system/python dependencies.
3. If both native and OCR paths fail, the file is marked failed.

---

### Q11. You merge with existing Excel/CSV via process_existing_excel_csv. If incoming PDF has a test already in Excel with different value, which wins?

**Answer**

The pipeline concatenates existing data first, then new PDF data, then normalizes and deduplicates.

Flow:
1. Existing rows and new rows are combined.
2. Test names are canonicalized.
3. Dedupe merges same canonical test on same date.
4. Winner row is chosen by completeness scoring (numeric value, reference range, unit, status quality).

Relevant logic:
- Concatenation order: [backend_api/app/services.py](../backend_api/app/services.py#L280)
- Merge rule by date/test and row scoring: [backend_api/app/normalization.py](../backend_api/app/normalization.py#L832)

Practical interpretation:
- It is not hard-coded “PDF always wins” or “Excel always wins.”
- The most complete row wins.
- If rows are equally complete, earlier ordering can influence tie outcomes (so existing data may win ties).

## 4) Clinical assistant

### Q12. RAG flow is frontend -> /api/clinical-assistant -> backend /api/v1/reports/chat. Why two-hop? Why not call backend directly?

**Answer**

Two-hop architecture enables server-side context orchestration before LLM call.

What the Next route adds:
1. Resolves/fetches full analysis context when only IDs are provided.
2. Builds timeline and latest-values summaries.
3. Retrieves relevant guideline snippets from local knowledge base.
4. Constructs a stronger system prompt with grounding constraints.
5. Normalizes streaming response behavior to frontend.

Relevant implementation:
- Route handler orchestration: [web/app/api/clinical-assistant/route.ts](../web/app/api/clinical-assistant/route.ts#L567)
- Backend chat endpoint: [backend_api/app/main.py](../backend_api/app/main.py#L914)

Why this is useful:
- Keeps frontend clients thin.
- Centralizes prompt and safety policy logic.
- Allows incremental improvements without changing mobile/web clients.

---

### Q13. What is in medicalKnowledge.ts? Static guidelines? How do you decide what to retrieve for a question?

**Answer**

Yes, it is a curated static knowledge base of medical interpretation guidance.

Entry content includes:
1. Canonical test names and aliases.
2. Keywords and related tests.
3. Interpretation bands.
4. Trend signals and confounders.
5. Escalation triggers and patient-friendly actions.
6. Source citations and evidence levels.

Relevant data:
- [web/lib/medicalKnowledge.ts](../web/lib/medicalKnowledge.ts#L1)

Retrieval logic combines:
1. Active findings in patient data.
2. Current user question and recent conversation history.
3. Concept expansion (for example “sugar” -> glucose/HbA1c family).
4. Intent signals (trend, urgency, action).
5. Evidence-level and recency weighting.
6. Category diversity constraints.

Relevant algorithm:
- [web/lib/ragRetrieval.ts](../web/lib/ragRetrieval.ts#L77)

---

### Q14. How do you prevent the assistant from making up lab values not in reports?

**Answer**

Current guardrails are primarily prompt-grounding and context-shaping:
1. Structured report context is passed in.
2. Prompt instructs strict behavior for missing tests.
3. Timeline and latest findings are explicitly embedded into system prompt.

Relevant prompt constraints:
- [web/app/api/clinical-assistant/route.ts](../web/app/api/clinical-assistant/route.ts#L421)

Additional safety support:
- Hallucination detection utility exists for telemetry and quality monitoring.
- [web/lib/metrics.ts](../web/lib/metrics.ts#L151)

Current limitation:
- There is no deterministic post-generation fact-check blocker in active path yet.
- Best next step is an answer-verification pass that rejects unsupported value claims before returning output.

## 5) Testing

### Q15. You have extraction-f1.test.ts. What does F1 measure here? What is ground truth? How was dataset built?

**Answer**

F1 measures extraction quality as harmonic mean of precision and recall:

F1 = 2PR / (P + R)

In this code:
1. Extracted findings are matched to expected findings by normalized test name and numeric value.
2. True positives, false positives, false negatives are computed.
3. Precision, recall, and F1 are derived.

Relevant implementation:
- F1 calculation: [web/lib/metrics.ts](../web/lib/metrics.ts#L320)
- Test: [web/tests/extraction-f1.test.ts](../web/tests/extraction-f1.test.ts#L11)
- Ground truth fixture: [web/tests/fixtures/extractionGroundTruth.ts](../web/tests/fixtures/extractionGroundTruth.ts#L3)

Ground truth currently:
- A static fixture list of expected findings for one sample report set.
- The test intentionally includes one synonym variant to verify canonical name matching robustness.

Dataset maturity note:
- This is currently a focused regression fixture, not yet a large benchmark corpus.
- For stronger statistical confidence, expand to many reports across labs/formats and track per-category error slices.

## 6) Practical recommendations

Based on current implementation and architecture, high-impact next steps are:

1. **Unify backend variants**
- Ensure active backend (`backend_api`) includes metrics endpoints used by admin UI.

2. **Add JSONB query optimization where needed**
- Introduce GIN/expression indexes only after query patterns are confirmed in production.

3. **Harden clinical safety**
- Add post-generation value-verification gate against source records.

4. **Expand extraction evaluation set**
- Build a broader ground-truth corpus with report diversity and longitudinal repeats.

5. **Operational observability**
- Add stage-level failure dashboards segmented by parser/OCR/model/rate-limit categories.

---

If needed, this document can be converted into a slide-ready version with one section per interview/demo question and architecture sequence diagrams.