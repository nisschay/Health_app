import { NextRequest, NextResponse } from "next/server";
import { retrieveRelevantGuidelines } from "@/lib/ragRetrieval";

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

export const runtime = "nodejs";

function resolveBackendBaseUrl(): string {
  const raw =
    process.env.NEXT_PUBLIC_DIRECT_API_URL?.replace(/\/$/, "") ||
    process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
    "http://127.0.0.1:8000";

  if (raw.startsWith("/")) {
    return "http://127.0.0.1:8000";
  }
  return raw;
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

  const reportTimeline = analysis.reports
    .map((r) => {
      const abnormalFindings = r.findings
        .filter((f) => f.severity !== "NORMAL")
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
  for (const r of analysis.reports) {
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

function buildGuidelinesSection(
  userMessage: string,
  activeFindings: string[],
): string {
  const relevantGuidelines = retrieveRelevantGuidelines(userMessage, activeFindings);
  if (relevantGuidelines.length === 0) {
    return "";
  }

  return `
RELEVANT MEDICAL GUIDELINES (use these to ground your answer):
${relevantGuidelines
  .map((guideline) => `
[${guideline.source}]
${guideline.title}
${guideline.content}
Cite as: "${guideline.source}" - ${guideline.sourceUrl}
`)
  .join("\n")}

IMPORTANT: When your answer is informed by one of the above guidelines,
end your response with a "Sources" section listing the citation(s) used.
Format: "**Sources:** [Source Name](URL)"
  `.trim();
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
  const backendBaseUrl = resolveBackendBaseUrl();

  const analysisId = asText(payload.analysisId, "unknown-analysis");
  const sessionId = asText(payload.sessionId, "session-default");
  const userId = resolveUserId(authHeader);

  const incomingHistory = sanitizeHistory(payload.history).slice(-20);
  const storedHistory = sessionHistoryStore.get(sessionId) ?? [];
  const cappedHistory = (incomingHistory.length > 0 ? incomingHistory : storedHistory).slice(-20);
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

  try {
    const response = await fetch(`${backendBaseUrl}/api/v1/reports/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: authHeader,
      },
      body: JSON.stringify(backendPayload),
      cache: "no-store",
    });

    if (response.status === 401 || response.status === 403) {
      return NextResponse.json(
        { detail: "Session expired or token invalid. Please sign in again." },
        { status: 401 },
      );
    }

    if (!response.ok) {
      const detail = await readErrorDetail(response);
      if (response.status >= 500) {
        return NextResponse.json(
          { detail: `Clinical assistant model error: ${detail}` },
          { status: 500 },
        );
      }
      return NextResponse.json({ detail }, { status: response.status });
    }

    const data = (await response.json()) as { answer?: unknown };
    if (typeof data.answer !== "string") {
      return NextResponse.json(
        { detail: "Clinical assistant returned an invalid response." },
        { status: 500 },
      );
    }

    const updatedHistory = [
      ...cappedHistory,
      { role: "user" as const, content: message },
      { role: "assistant" as const, content: data.answer },
    ].slice(-20);
    sessionHistoryStore.set(sessionId, updatedHistory);

    if (sessionHistoryStore.size > 500) {
      const firstKey = sessionHistoryStore.keys().next().value;
      if (firstKey) {
        sessionHistoryStore.delete(firstKey);
      }
    }

    return NextResponse.json({ answer: data.answer });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown assistant service error.";
    return NextResponse.json(
      { detail: `Clinical assistant model error: ${detail}` },
      { status: 500 },
    );
  }
}
