import { NextRequest, NextResponse } from "next/server";
import { retrieveRelevantGuidelines } from "@/lib/ragRetrieval";
import { buildApiUrl, getServerBackendBaseUrl } from "@/lib/apiBaseUrl";
import { parseMedicalDate } from "@/lib/medicalDate";

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

const CHAT_HISTORY_LIMIT = 8;
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

function asText(value: unknown, fallback = ""): string {
  if (value === null || value === undefined) return fallback;
  const text = String(value).trim();
  return text || fallback;
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

    const sortKey = parseMedicalDate(date);
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
    (a, b) => parseMedicalDate(a.date) - parseMedicalDate(b.date),
  );

  const datedReports = reports
    .map((report) => parseMedicalDate(report.date))
    .filter((sortKey) => Number.isFinite(sortKey) && sortKey !== Number.MAX_SAFE_INTEGER);

  const dateRange = {
    start: reports.length === 0
      ? "Unknown"
      : datedReports.length > 0
        ? reports.find((report) => parseMedicalDate(report.date) === Math.min(...datedReports))?.date ?? reports[0]!.date
        : reports[0]!.date,
    end: reports.length === 0
      ? "Unknown"
      : datedReports.length > 0
        ? reports.find((report) => parseMedicalDate(report.date) === Math.max(...datedReports))?.date ?? reports[reports.length - 1]!.date
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

async function resolveReportContext(
  analysisId: string,
  reportContext: ReportContextPayload,
  backendBaseUrl: string,
  authHeader: string,
): Promise<ReportContextPayload> {
  const hasContextRecords = Array.isArray(reportContext.records)
    && reportContext.records.some((row) => row && typeof row === "object");
  if (hasContextRecords) {
    return reportContext;
  }

  const fetched = await fetchFullAnalysis(analysisId, backendBaseUrl, authHeader);
  return fetched ? toReportContextFromBackend(fetched) : reportContext;
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

function buildGuidelineSnippets(
  userMessage: string,
  activeFindings: string[],
  conversationHistory: ChatTurnPayload[],
): string[] {
  const relevantGuidelines = retrieveRelevantGuidelines(
    userMessage,
    activeFindings,
    {
      conversationHistory: conversationHistory.map((item) => item.content),
      maxResults: 3,
      minScore: 15,
    },
  );
  return relevantGuidelines.map(({ entry, matchedTerms }) => `[${entry.source}] ${entry.title} (${entry.category})
${entry.content}
Interpretation bands:
${entry.interpretationBands
  .map((band) => `- ${band.label} (${band.range}): ${band.interpretation}. Typical action: ${band.typicalAction}`)
  .join("\n")}
Trend rules: ${entry.trendSignals.join("; ")}
Confounders: ${entry.confounders.join("; ")}
Escalation triggers: ${entry.escalationTriggers.join("; ")}
Patient actions: ${entry.patientFriendlyActions.join("; ")}
Matched terms: ${matchedTerms.join(", ") || "none"}
Evidence level: ${entry.evidenceLevel}
Cite as: "${entry.source}" - ${entry.sourceUrl}`.trim());
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

  const cappedHistory = sanitizeHistory(payload.history).slice(-CHAT_HISTORY_LIMIT);
  const resolvedContext = await resolveReportContext(
    analysisId,
    reportContext,
    backendBaseUrl,
    authHeader,
  );
  const analysis = summarizeRecords(resolvedContext);
  const guidelines = buildGuidelineSnippets(
    message,
    analysis.findings.map((finding) => finding.canonicalName),
    cappedHistory,
  );

  const backendPayload = {
    records: Array.isArray(resolvedContext.records) ? resolvedContext.records : records,
    question: message,
    history: cappedHistory,
    analysis_id: analysisId,
    session_id: sessionId,
    guidelines,
    report_context: resolvedContext,
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
