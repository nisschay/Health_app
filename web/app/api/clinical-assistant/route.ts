import { NextRequest, NextResponse } from "next/server";
import { retrieveRelevantGuidelines } from "@/lib/ragRetrieval";
import { buildApiUrl, getServerBackendBaseUrl } from "@/lib/apiBaseUrl";

type ChatTurnPayload = {
  role: "user" | "assistant";
  content: string;
};

type ReportContextPayload = {
  patientInfo?: {
    name?: unknown;
    [key: string]: unknown;
  };
  records?: unknown[];
  totalRecords?: unknown;
  reportsIncluded?: unknown;
  sourceFileNames?: unknown;
  [key: string]: unknown;
};

type ReportRecord = {
  Test_Date?: unknown;
  Lab_Name?: unknown;
  Test_Name?: unknown;
  Original_Test_Name?: unknown;
  Result?: unknown;
  Unit?: unknown;
  Status?: unknown;
  Reference_Range?: unknown;
  [key: string]: unknown;
};

type ClinicalAssistantPayload = {
  analysisId?: string;
  sessionId?: string;
  reportContext?: ReportContextPayload;
  history?: unknown;
  message?: string;
  question?: string; // Backward compatibility for older clients.
  stream?: boolean;
};

type AnalysisFinding = {
  name: string;
  canonicalName: string;
  value: string;
  unit: string;
  severity: string;
  referenceRange: string;
};

type AnalysisReport = {
  date: string;
  labName: string;
  findings: AnalysisFinding[];
};

type LatestFinding = {
  canonicalName: string;
  latestValue: string;
  unit: string;
  latestStatus: string;
  latestDate: string;
};

type AggregatedAnalysis = {
  patientName: string;
  reports: AnalysisReport[];
  findings: LatestFinding[];
  dateRange: {
    start: string;
    end: string;
  };
};

type BackendAnalysisPayload = {
  patient_info?: {
    name?: unknown;
    [key: string]: unknown;
  };
  records?: unknown[];
  total_records?: unknown;
  reports_with_data?: unknown;
  combined_report_file_names?: unknown;
};

const sessionHistoryStore = new Map<string, ChatTurnPayload[]>();

const CHAT_HISTORY_LIMIT = 8;
const REPORT_TIMELINE_LIMIT = 12;
const FINDINGS_SNAPSHOT_LIMIT = 40;
const CHAT_BACKEND_TIMEOUT_MS = 55_000;
const STREAM_CHUNK_SIZE = 140;

export const runtime = "nodejs";

function resolveBackendBaseUrl(): string {
  return getServerBackendBaseUrl();
}

function normalizeRole(value: unknown): "user" | "assistant" | null {
  const role = String(value ?? "").trim().toLowerCase();
  if (role === "user" || role === "assistant") return role;
  return null;
}

function sanitizeHistory(value: unknown): ChatTurnPayload[] {
  if (!Array.isArray(value)) return [];

  return value
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const role = "role" in item ? normalizeRole((item as { role?: unknown }).role) : null;
      const content = "content" in item ? String((item as { content?: unknown }).content ?? "").trim() : "";
      if (!role || !content) return null;
      return { role, content };
    })
    .filter((item): item is ChatTurnPayload => item !== null);
}

function decodeBase64Url(value: string): string {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized + "=".repeat((4 - (normalized.length % 4 || 4)) % 4);
  return Buffer.from(padded, "base64").toString("utf8");
}

function resolveUserId(authHeader: string): string {
  const token = authHeader.replace(/^Bearer\s+/i, "").trim();
  if (!token) return "authenticated-user";

  try {
    const payloadSegment = token.split(".")[1];
    if (!payloadSegment) return "authenticated-user";
    const payload = JSON.parse(decodeBase64Url(payloadSegment)) as Record<string, unknown>;
    const value = payload.user_id ?? payload.uid ?? payload.sub ?? payload.email;
    return asText(value, "authenticated-user");
  } catch {
    return "authenticated-user";
  }
}

function asText(value: unknown, fallback = ""): string {
  if (value === null || value === undefined) return fallback;
  const text = String(value).trim();
  return text || fallback;
}

function parseSortableDate(value: string): number {
  const clean = value.trim();
  if (!clean) return Number.MAX_SAFE_INTEGER;

  const ddmmyyyy = clean.match(/^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})$/);
  if (ddmmyyyy) {
    const day = ddmmyyyy[1]!.padStart(2, "0");
    const month = ddmmyyyy[2]!.padStart(2, "0");
    const year = ddmmyyyy[3]!.length === 2 ? `20${ddmmyyyy[3]}` : ddmmyyyy[3]!;
    const parsed = new Date(`${year}-${month}-${day}`).getTime();
    return Number.isNaN(parsed) ? Number.MAX_SAFE_INTEGER : parsed;
  }

  const parsed = new Date(clean).getTime();
  return Number.isNaN(parsed) ? Number.MAX_SAFE_INTEGER : parsed;
}

function canonicalizeTestName(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeSeverity(value: unknown): string {
  const raw = asText(value, "").toLowerCase();
  if (!raw) return "UNKNOWN";
  if (raw.includes("normal") || raw.includes("negative") || raw.includes("within")) return "NORMAL";
  if (raw.includes("high")) return "HIGH";
  if (raw.includes("low")) return "LOW";
  if (raw.includes("critical")) return "CRITICAL";
  if (raw.includes("borderline")) return "BORDERLINE";
  if (raw.includes("positive")) return "POSITIVE";
  return "ABNORMAL";
}

function summarizeRecords(reportContext: ReportContextPayload): AggregatedAnalysis {
  const records = Array.isArray(reportContext.records) ? (reportContext.records as ReportRecord[]) : [];
  const patientName = asText(reportContext.patientInfo?.name, "Patient");

  const groupedReports = new Map<string, AnalysisReport>();
  const latestByCanonical = new Map<string, { finding: LatestFinding; sortKey: number }>();

  for (const record of records) {
    const date = asText(record.Test_Date, "Unknown date");
    const labName = asText(record.Lab_Name, "Unknown Lab");
    const testName = asText(record.Test_Name ?? record.Original_Test_Name, "Unknown test");
    const value = asText(record.Result, "N/A");
    const unit = asText(record.Unit, "");
    const severity = normalizeSeverity(record.Status);
    const referenceRange = asText(record.Reference_Range, "N/A");
    const canonicalName = canonicalizeTestName(testName) || testName.toLowerCase();

    const reportKey = `${date}::${labName}`;
    if (!groupedReports.has(reportKey)) {
      groupedReports.set(reportKey, {
        date,
        labName,
        findings: [],
      });
    }

    groupedReports.get(reportKey)!.findings.push({
      name: testName,
      canonicalName,
      value,
      unit,
      severity,
      referenceRange,
    });

    const sortKey = parseSortableDate(date);
    const existing = latestByCanonical.get(canonicalName);
    if (!existing || sortKey >= existing.sortKey) {
      latestByCanonical.set(canonicalName, {
        finding: {
          canonicalName,
          latestValue: value,
          unit,
          latestStatus: severity,
          latestDate: date,
        },
        sortKey,
      });
    }
  }

  const reports = [...groupedReports.values()].sort(
    (a, b) => parseSortableDate(a.date) - parseSortableDate(b.date),
  );

  const datedReports = reports
    .map((report) => parseSortableDate(report.date))
    .filter((sortKey) => Number.isFinite(sortKey) && sortKey !== Number.MAX_SAFE_INTEGER);

  const dateRange = {
    start: reports.length === 0
      ? "Unknown"
      : datedReports.length > 0
        ? reports.find((report) => parseSortableDate(report.date) === Math.min(...datedReports))?.date ?? reports[0]!.date
        : reports[0]!.date,
    end: reports.length === 0
      ? "Unknown"
      : datedReports.length > 0
        ? reports.find((report) => parseSortableDate(report.date) === Math.max(...datedReports))?.date ?? reports[reports.length - 1]!.date
        : reports[reports.length - 1]!.date,
  };

  const findings = [...latestByCanonical.values()]
    .sort((a, b) => b.sortKey - a.sortKey)
    .map((entry) => entry.finding);

  return {
    patientName,
    reports,
    findings,
    dateRange,
  };
}

function toReportContextFromBackend(analysis: BackendAnalysisPayload): ReportContextPayload {
  const patientInfo = analysis.patient_info && typeof analysis.patient_info === "object"
    ? analysis.patient_info
    : {};

  const sourceFileNames = Array.isArray(analysis.combined_report_file_names)
    ? analysis.combined_report_file_names
    : [];

  return {
    patientInfo,
    records: Array.isArray(analysis.records) ? analysis.records : [],
    totalRecords: analysis.total_records,
    reportsIncluded: analysis.reports_with_data,
    sourceFileNames,
  };
}

function buildAnalysisFetchUrl(analysisId: string, backendBaseUrl: string): string | null {
  const clean = analysisId.trim();
  if (!clean) return null;

  if (clean.startsWith("study-")) {
    const studyId = clean.slice("study-".length).trim();
    if (!studyId) return null;
    return `${backendBaseUrl}/api/v1/studies/${encodeURIComponent(studyId)}/combined-report`;
  }

  if (clean.startsWith("history-")) {
    const historyId = clean.slice("history-".length).trim();
    if (!/^\d+$/.test(historyId)) return null;
    return `${backendBaseUrl}/api/v1/reports/history/${historyId}`;
  }

  if (/^\d+$/.test(clean)) {
    return `${backendBaseUrl}/api/v1/reports/history/${clean}`;
  }

  return null;
}

async function fetchFullAnalysis(
  analysisId: string,
  backendBaseUrl: string,
  authHeader: string,
): Promise<BackendAnalysisPayload | null> {
  const url = buildAnalysisFetchUrl(analysisId, backendBaseUrl);
  if (!url) return null;

  try {
    const response = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: authHeader,
      },
      cache: "no-store",
    });
    if (!response.ok) return null;

    const payload = (await response.json()) as BackendAnalysisPayload;
    return payload && typeof payload === "object" ? payload : null;
  } catch {
    return null;
  }
}

async function getFullAnalysis(
  analysisId: string,
  userId: string,
  reportContext: ReportContextPayload,
  backendBaseUrl: string,
  authHeader: string,
): Promise<AggregatedAnalysis> {
  // userId is part of the function contract for profile-scoped resolution and auditing context.
  void userId;

  const hasContextRecords = Array.isArray(reportContext.records)
    && reportContext.records.some((row) => row && typeof row === "object");
  if (hasContextRecords) {
    return summarizeRecords(reportContext);
  }

  const fetched = await fetchFullAnalysis(analysisId, backendBaseUrl, authHeader);
  if (fetched) {
    return summarizeRecords(toReportContextFromBackend(fetched));
  }
  return summarizeRecords(reportContext);
}

function buildBaseSystemPrompt(
  analysis: AggregatedAnalysis,
  analysisId: string,
  userId: string,
): string {

  const reportSlice = analysis.reports.slice(-REPORT_TIMELINE_LIMIT);

  const reportTimeline = reportSlice
    .map((r) => {
      const abnormalFindings = r.findings
        .filter((f) => f.severity !== "NORMAL")
        .slice(0, 8)
        .map((f) => `${f.name}: ${f.value}${f.unit ? ` ${f.unit}` : ""} (${f.severity})`)
        .join(", ");

      return `
      Date: ${r.date} | Lab: ${r.labName}
      Tests performed: ${r.findings.length}
      Abnormal findings: ${abnormalFindings || "None"}
    `;
    })
    .join("\n---\n");

  const concernMap = new Map<string, number>();
  for (const r of reportSlice) {
    for (const f of r.findings) {
      if (f.severity !== "NORMAL") {
        concernMap.set(f.canonicalName, (concernMap.get(f.canonicalName) || 0) + 1);
      }
    }
  }

  const persistentConcerns = [...concernMap.entries()]
    .filter(([, count]) => count >= 2)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => `${name} (abnormal in ${count} reports)`)
    .join("\n");

  const latestValues = analysis.findings
    .slice(0, FINDINGS_SNAPSHOT_LIMIT)
    .map((f) => `${f.canonicalName}: ${f.latestValue}${f.unit ? ` ${f.unit}` : ""} [${f.latestStatus}] as of ${f.latestDate}`)
    .join("\n");

  return `
You are a Clinical Assistant helping a patient named ${analysis.patientName}
understand their medical test results. You have access to their complete
medical history across ${analysis.reports.length} reports from
${analysis.dateRange.start} to ${analysis.dateRange.end}.

REQUEST CONTEXT:
- Analysis ID: ${analysisId}
- User ID: ${userId}

PATIENT PROFILE:
- Name: ${analysis.patientName}
- Total reports: ${analysis.reports.length}
- Labs used: ${[...new Set(analysis.reports.map((r) => r.labName))].join(", ") || "Unknown"}
- Date range: ${analysis.dateRange.start} to ${analysis.dateRange.end}

CHRONOLOGICAL REPORT HISTORY:
${reportTimeline || "No report timeline available"}

PERSISTENT CONCERNS (abnormal in multiple reports):
${persistentConcerns || "None identified across multiple reports"}

LATEST VALUES SNAPSHOT:
${latestValues || "No latest values available"}

INSTRUCTIONS:
- Answer in clear, plain language a non-medical person can understand
- When citing values, always include the number, unit, and date
- Distinguish between "historically abnormal but recently normal" vs
  "currently abnormal" - these are very different clinical situations
- For tests not re-evaluated recently, explicitly flag them as
  "last tested on [date] - current status unknown"
- When a value is borderline, explain what it means in plain English
- Never diagnose. Always recommend consulting a clinician for
  interpretation and treatment decisions
- Format responses in markdown: use **bold** for important values,
  bullet lists for multiple findings, headers for sections
- Keep responses under 300 words unless the question requires detail
- If asked about a test not in the data, say clearly
  "This test was not found in your uploaded reports"

IMPORTANT - about reference ranges:
Different labs in this patient's history use slightly different reference
ranges. Always use the reference range from the SAME report as the value
being discussed. If ranges conflict across reports, note the discrepancy.
  `.trim();
}

function wantsEventStream(request: NextRequest, payload: ClinicalAssistantPayload): boolean {
  if (payload.stream === true) return true;
  const accept = request.headers.get("accept") ?? "";
  return accept.toLowerCase().includes("text/event-stream");
}

function sseEvent(event: Record<string, unknown>): string {
  return `data: ${JSON.stringify(event)}\n\n`;
}

function streamAnswerChunks(answer: string): string[] {
  const normalized = answer.replace(/\s+/g, " ").trim();
  if (!normalized) return [];

  const chunks: string[] = [];
  const sentences = normalized.split(/(?<=[.!?])\s+/).filter(Boolean);
  let pending = "";

  for (const sentence of sentences) {
    const candidate = pending ? `${pending} ${sentence}` : sentence;
    if (candidate.length <= STREAM_CHUNK_SIZE) {
      pending = candidate;
      continue;
    }
    if (pending) {
      chunks.push(pending);
      pending = "";
    }

    if (sentence.length <= STREAM_CHUNK_SIZE) {
      pending = sentence;
      continue;
    }

    for (let index = 0; index < sentence.length; index += STREAM_CHUNK_SIZE) {
      chunks.push(sentence.slice(index, index + STREAM_CHUNK_SIZE));
    }
  }

  if (pending) {
    chunks.push(pending);
  }

  return chunks;
}

function buildGuidelinesSection(
  userMessage: string,
  activeFindings: string[],
  conversationHistory: ChatTurnPayload[],
): string {
  const relevantGuidelines = retrieveRelevantGuidelines(
    userMessage,
    activeFindings,
    {
      conversationHistory: conversationHistory.map((item) => item.content),
      maxResults: 3,
      minScore: 15,
    },
  );
  if (relevantGuidelines.length === 0) {
    return "";
  }

  return `
RELEVANT MEDICAL GUIDELINES (use these to ground your answer):
${relevantGuidelines
  .map(({ entry, score, matchedTerms, rationale }) => `
[${entry.source}]  [retrieval score: ${score}]
${entry.title}
Category: ${entry.category}
Core content:
${entry.content}

Interpretation bands:
${entry.interpretationBands
  .map((band) => `- ${band.label} (${band.range}): ${band.interpretation}. Typical action: ${band.typicalAction}`)
  .join("\n")}

Trend interpretation rules:
${entry.trendSignals.map((signal) => `- ${signal}`).join("\n")}

Known confounders:
${entry.confounders.map((factor) => `- ${factor}`).join("\n")}

Escalation triggers:
${entry.escalationTriggers.map((trigger) => `- ${trigger}`).join("\n")}

Patient-friendly action points:
${entry.patientFriendlyActions.map((action) => `- ${action}`).join("\n")}

Matched retrieval terms: ${matchedTerms.join(", ") || "none"}
Retrieval rationale: ${rationale.join(", ") || "semantic overlap"}
Evidence level: ${entry.evidenceLevel}
Cite as: "${entry.source}" - ${entry.sourceUrl}
`)
  .join("\n")}

IMPORTANT: When your answer is informed by one of the above guidelines,
end your response with a "Sources" section listing the citation(s) used.
Format: "**Sources:** [Source Name](URL)"
  `.trim();
}

function isRateLimitMessage(text: string): boolean {
  const lowered = text.toLowerCase();
  return lowered.includes("rate limit") || lowered.includes("quota") || lowered.includes("429");
}

function buildRateLimitedFallbackAnswer(
  analysis: AggregatedAnalysis,
  userQuestion: string,
): string {
  const abnormal = analysis.findings
    .filter((finding) => finding.latestStatus !== "NORMAL")
    .slice(0, 5)
    .map((finding) => {
      const valueText = finding.unit
        ? `${finding.latestValue} ${finding.unit}`
        : finding.latestValue;
      return `- ${finding.canonicalName}: **${valueText}** (${finding.latestStatus}) on ${finding.latestDate}`;
    });

  const summaryLines = abnormal.length > 0
    ? abnormal.join("\n")
    : "- No currently abnormal markers were detected in the latest snapshot.";

  return [
    "The live AI model is temporarily rate-limited, so I cannot generate a full Gemini answer right now.",
    "",
    "### Quick report-based summary",
    summaryLines,
    "",
    "### What to discuss with your doctor",
    `- Share your exact question: \"${userQuestion}\"`,
    "- Prioritize the abnormal markers listed above, especially persistent or worsening values.",
    `- Ask for trend interpretation across your report window (${analysis.dateRange.start} to ${analysis.dateRange.end}).`,
    "",
    "Retry in a little while for a full model-generated response.",
  ].join("\n");
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) return payload.detail;
  } catch {
    // Ignore JSON parsing issues and use fallback below.
  }
  return response.statusText || "Request failed.";
}

export async function POST(request: NextRequest) {
  const requestStartedAt = Date.now();
  const authHeader = request.headers.get("authorization");
  if (!authHeader) {
    return NextResponse.json(
      { detail: "Missing authentication token. Please sign in again." },
      { status: 401 },
    );
  }

  let payload: ClinicalAssistantPayload;
  try {
    payload = (await request.json()) as ClinicalAssistantPayload;
  } catch {
    return NextResponse.json({ detail: "Invalid JSON payload." }, { status: 400 });
  }

  const message = typeof payload.message === "string"
    ? payload.message.trim()
    : typeof payload.question === "string"
      ? payload.question.trim()
      : "";
  if (!message) {
    return NextResponse.json({ detail: "Message is required." }, { status: 400 });
  }

  const reportContext = payload.reportContext && typeof payload.reportContext === "object"
    ? payload.reportContext
    : {};
  const records = Array.isArray(reportContext.records) ? reportContext.records : [];
  const shouldStream = wantsEventStream(request, payload);
  const backendBaseUrl = resolveBackendBaseUrl();

  const analysisId = asText(payload.analysisId, "unknown-analysis");
  const sessionId = asText(payload.sessionId, "session-default");
  const userId = resolveUserId(authHeader);

  const incomingHistory = sanitizeHistory(payload.history).slice(-CHAT_HISTORY_LIMIT);
  const storedHistory = sessionHistoryStore.get(sessionId) ?? [];
  const cappedHistory = (incomingHistory.length > 0 ? incomingHistory : storedHistory).slice(-CHAT_HISTORY_LIMIT);
  const messages = [
    ...cappedHistory.map((h) => ({ role: h.role, content: h.content })),
    { role: "user" as const, content: message },
  ];

  const analysis = await getFullAnalysis(
    analysisId,
    userId,
    reportContext,
    backendBaseUrl,
    authHeader,
  );
  const baseSystemPrompt = buildBaseSystemPrompt(analysis, analysisId, userId);
  const guidelinesSection = buildGuidelinesSection(
    message,
    analysis.findings.map((finding) => finding.canonicalName),
    cappedHistory,
  );
  const fullSystemPrompt = guidelinesSection
    ? `${baseSystemPrompt}\n\n${guidelinesSection}`
    : baseSystemPrompt;

  const backendPayload = {
    records,
    question: message,
    history: cappedHistory,
    analysis_id: analysisId,
    session_id: sessionId,
    system_prompt: fullSystemPrompt,
    messages,
    report_context: reportContext,
  };

  const executeChatRequest = async (): Promise<{ answer: string; status: number; backendLatencyMs: number }> => {
    const backendCallStartedAt = Date.now();
    const abortController = new AbortController();
    const timeout = setTimeout(() => abortController.abort(), CHAT_BACKEND_TIMEOUT_MS);
    try {
      const response = await fetch(buildApiUrl(backendBaseUrl, "/api/v1/reports/chat"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: authHeader,
        },
        body: JSON.stringify(backendPayload),
        cache: "no-store",
        signal: abortController.signal,
      });

      if (response.status === 401 || response.status === 403) {
        throw Object.assign(new Error("Session expired or token invalid. Please sign in again."), { status: 401 });
      }

      if (!response.ok) {
        const detail = await readErrorDetail(response);
        if (response.status === 429 || isRateLimitMessage(detail)) {
          return {
            answer: buildRateLimitedFallbackAnswer(analysis, message),
            status: 200,
            backendLatencyMs: Date.now() - backendCallStartedAt,
          };
        }
        if (response.status >= 500) {
          throw Object.assign(new Error(`Clinical assistant model error: ${detail}`), { status: 500 });
        }
        throw Object.assign(new Error(detail), { status: response.status });
      }

      const data = (await response.json()) as { answer?: unknown };
      if (typeof data.answer !== "string") {
        throw Object.assign(new Error("Clinical assistant returned an invalid response."), { status: 500 });
      }

      const answerText = data.answer.trim();
      const finalAnswer = isRateLimitMessage(answerText)
        ? buildRateLimitedFallbackAnswer(analysis, message)
        : answerText;

      const updatedHistory = [
        ...cappedHistory,
        { role: "user" as const, content: message },
        { role: "assistant" as const, content: finalAnswer },
      ].slice(-CHAT_HISTORY_LIMIT);
      sessionHistoryStore.set(sessionId, updatedHistory);

      if (sessionHistoryStore.size > 500) {
        const firstKey = sessionHistoryStore.keys().next().value;
        if (firstKey) {
          sessionHistoryStore.delete(firstKey);
        }
      }

      return {
        answer: finalAnswer,
        status: 200,
        backendLatencyMs: Date.now() - backendCallStartedAt,
      };
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        throw Object.assign(new Error("Clinical assistant timed out. Please retry with a shorter question."), {
          status: 504,
        });
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  };

  if (!shouldStream) {
    try {
      const { answer, backendLatencyMs } = await executeChatRequest();
      return NextResponse.json(
        { answer },
        {
          headers: {
            "X-Chat-Backend-Latency-Ms": String(backendLatencyMs),
            "X-Chat-Total-Latency-Ms": String(Date.now() - requestStartedAt),
          },
        },
      );
    } catch (error) {
      const status = typeof (error as { status?: unknown })?.status === "number"
        ? Number((error as { status: number }).status)
        : 500;
      const detail = error instanceof Error ? error.message : "Unknown assistant service error.";
      return NextResponse.json(
        { detail },
        { status },
      );
    }
  }

  const encoder = new TextEncoder();
  let keepAlive: ReturnType<typeof setInterval> | null = null;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (event: Record<string, unknown>) => {
        controller.enqueue(encoder.encode(sseEvent(event)));
      };

      send({
        type: "started",
        sessionId,
        analysisId,
        serverReceivedAt: requestStartedAt,
      });
      keepAlive = setInterval(() => {
        send({ type: "keepalive", ts: Date.now() });
      }, 10_000);

      (async () => {
        try {
          const { answer, backendLatencyMs } = await executeChatRequest();
          const chunks = streamAnswerChunks(answer);
          for (const chunk of chunks) {
            send({ type: "delta", text: chunk });
          }
          send({
            type: "done",
            answer,
            backendLatencyMs,
            totalLatencyMs: Date.now() - requestStartedAt,
          });
        } catch (error) {
          const status = typeof (error as { status?: unknown })?.status === "number"
            ? Number((error as { status: number }).status)
            : 500;
          const message = error instanceof Error ? error.message : "Unknown assistant service error.";
          send({ type: "error", status, message });
        } finally {
          if (keepAlive) {
            clearInterval(keepAlive);
            keepAlive = null;
          }
          controller.close();
        }
      })();
    },
    cancel() {
      if (keepAlive) {
        clearInterval(keepAlive);
        keepAlive = null;
      }
    },
  });

  return new NextResponse(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
