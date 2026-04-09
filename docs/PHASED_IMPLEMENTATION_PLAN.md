# Medical Project Performance Implementation Phases

Last updated: 2026-04-09

## Goal

Reduce end-user latency while preserving medical safety:
- Chat: p95 < 10s, time-to-first-token (TTFT) < 2s
- Dashboard combined report view: p95 < 2s
- Ingestion: avoid repeated heavy compute and reduce duplicate model calls
- Normalization: safer canonicalization with strict clinical guardrails

## Phase A - Day 1 Fast Path (current phase)

Status: In progress (implemented core streaming + chat path trim)

### Implemented in this phase

1. Chat streaming (SSE) in Next route
- started, keepalive, delta, done, error events
- keeps chat synchronous from user to model path (no queue hop)
- backend timeout guard added on chat request

2. Frontend live rendering of assistant draft
- partial response displayed while stream is in progress
- disables input during active stream

3. Chat payload trim and request-path de-dup
- reduced history window to lower token load
- reduced timeline/snapshot sizes in prompt
- short-circuit full backend fetch when reportContext already has records

4. Basic latency instrumentation
- client logs TTFT and total stream time
- server emits latency metadata in done event and JSON headers

### Files touched in Phase A

- web/app/api/clinical-assistant/route.ts
- web/lib/api.ts
- web/app/dashboard/ClinicalAssistantChat.tsx
- Helper_Functions.py
- Medical_Project-Backend/Helper_Functions.py

### Phase A test gates

- Chat regression test:
  - web/tests/clinical-assistant-context-retention.test.ts
- Type checks on touched TS files
- Manual smoke:
  - Ask a question in dashboard chat and verify streaming starts immediately

## Phase B - Persistence + Aggregation Correctness

Status: Planned

### Scope

1. Persist normalized data with metadata
- Add report-level fields for raw and normalized forms
- Add normalization_version and is_normalized state

2. Explicit combined-report aggregation bug fix
- Fix add-reports grouping/aggregation behavior
- Replace fragile filename-only slicing logic with deterministic attribution
- Keep safe fallback for legacy rows

3. Read-path optimization
- Use persisted normalized records directly for versioned rows
- Normalize only legacy/unversioned rows

4. Golden-set normalization regression suite (mandatory in Phase B)
- fixed corpus: raw test names -> expected canonical output
- must-not-merge clinical pairs enforced
- examples: TSH vs T3H, Hb vs HbA1c, protein vs C-reactive protein

5. Frontend normalization fallback gate
- keep client normalization behind feature flag as fallback
- default to backend-normalized payload after validation

### Target files for Phase B

- backend_api/app/database.py
- backend_api/sql/2026_03_31_firebase_postgres_bootstrap.sql
- backend_api/app/main.py
- backend_api/app/normalization.py
- backend_api/scripts/migrate_normalized_records.py
- web/lib/api.ts
- web/lib/normalizeTest.ts
- web/tests/* (new regression tests)

### Phase B test gates

- New regression test for combined-report add-reports scenario
- Golden-set normalization tests must pass
- Migration dry-run parity checks

## Phase C - Batch Throughput + Safe Normalization Expansion

Status: Planned

### Scope

1. Redis queue/workers for batch analysis only
- apply to heavy PDF ingestion workloads (10+ PDFs)
- do not queue chat requests

2. Hybrid OCR fallback
- OCR only for empty/near-empty native text extraction
- keep Gemini for structured medical extraction

3. Safe normalization expansion
- alias-whitelist driven fuzzy-like matching only
- unknown names are never auto-merged by generic edit distance

4. Rollout and tuning
- canary rollout by cohort
- monitor TTFT, chat p95, combined-report p95, normalization regression pass rate

### Target files for Phase C

- backend_api/app/services.py
- Helper_Functions.py
- backend_api/app/normalization.py
- backend_api/app/main.py
- deploy/smoke_test.sh

### Phase C test gates

- Batch upload load test (10 PDFs)
- Concurrent chat validation without queue latency
- OCR fallback success/failure instrumentation

## Operational Notes

- Chat path must stay direct request-response for lowest latency.
- Redis is for background batch ingestion orchestration only.
- Any normalization mapping change requires golden-set pass before deploy.
