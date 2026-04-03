import { normalizeAnalysisPayload } from "./normalizeTest";
import { getAuth } from "firebase/auth";

const HF_SPACE_BACKEND_URL = "https://nisschay-medical-project-backend.hf.space";

const API_BASE_URL = (() => {
  const url = process.env.NEXT_PUBLIC_API_URL ?? process.env.NEXT_PUBLIC_API_BASE_URL;
  if (!url || url === "/backend") {
    if (process.env.NODE_ENV === "production") {
      console.error("NEXT_PUBLIC_API_URL not set in production!");
      return HF_SPACE_BACKEND_URL;
    }
    return "/backend";
  }
  return url.replace(/\/$/, "");
})();

const DIRECT_API_BASE_URL = (
  process.env.NEXT_PUBLIC_DIRECT_API_URL?.replace(/\/$/, "")
  ?? (process.env.NODE_ENV === "production" ? HF_SPACE_BACKEND_URL : "http://localhost:8000")
);

if (process.env.NODE_ENV === "production") {
  console.log("[API] Base URL:", API_BASE_URL);
}

function shouldRetryDirect(response: Response): boolean {
  return API_BASE_URL.startsWith("/") && response.status >= 500;
}

function redirectToLogin(): never {
  if (typeof window !== "undefined") {
    window.location.href = "/login";
  }
  throw new Error("Auth failed");
}

async function getBearerToken(forceRefresh = false): Promise<string> {
  const auth = getAuth();
  const user = auth.currentUser;
  if (!user) {
    throw new Error("No user");
  }
  return user.getIdToken(forceRefresh);
}

function withAuthHeaders(options: RequestInit, token: string): Headers {
  const headers = new Headers(options.headers ?? {});
  headers.set("Authorization", `Bearer ${token}`);
  const hasBody = options.body !== undefined && options.body !== null;
  if (hasBody && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return headers;
}

async function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
  let token: string;
  try {
    token = await getBearerToken(false);
  } catch {
    try {
      token = await getBearerToken(true);
    } catch {
      return redirectToLogin();
    }
  }

  let response = await fetch(url, {
    ...options,
    headers: withAuthHeaders(options, token),
  });

  if (response.status !== 401) {
    return response;
  }

  try {
    const freshToken = await getBearerToken(true);
    response = await fetch(url, {
      ...options,
      headers: withAuthHeaders(options, freshToken),
    });
    if (response.status === 401) {
      return redirectToLogin();
    }
    return response;
  } catch {
    return redirectToLogin();
  }
}

async function authBackendFetch(path: string, options: RequestInit = {}): Promise<Response> {
  let response = await authFetch(`${API_BASE_URL}${path}`, options);
  if (shouldRetryDirect(response)) {
    response = await authFetch(`${DIRECT_API_BASE_URL}${path}`, options);
  }
  return response;
}

export type PatientInfo = {
  name: string;
  age: string;
  gender: string;
  patient_id: string;
  date: string;
  lab_name: string;
};

export type MedicalRecord = {
  Source_Filename?: string | null;
  Patient_ID?: string | null;
  Patient_Name?: string | null;
  Age?: string | null;
  Gender?: string | null;
  Test_Date?: string | null;
  Lab_Name?: string | null;
  Test_Category?: string | null;
  Original_Test_Name?: string | null;
  Test_Name?: string | null;
  Aliases?: string[] | null;
  Result?: string | number | null;
  Unit?: string | null;
  Reference_Range?: string | null;
  Status?: string | null;
  Processed_Date?: string | null;
  Result_Numeric?: number | null;
  Test_Date_dt?: string | null;
};

export type AnalysisConcern = {
  test_name: string;
  result: string | number;
  status: string;
  reference: string;
  category: string;
  date: string;
};

export type CategoryScore = {
  score: number;
  total_tests: number;
  abnormal_count: number;
};

export type HealthSummary = {
  overall_score: number;
  category_scores: Record<string, CategoryScore>;
  concerns: AnalysisConcern[];
};

export type BodySystem = {
  system: string;
  emoji: string;
  concern_level: string;
  concern_score: number;
  abnormal_count: number;
  total_count: number;
  abnormal_ratio: number;
  categories: string[];
  tests: Array<{
    name: string;
    result: string | number;
    status: string;
    category: string;
  }>;
};

export type RawTextPreview = {
  name: string;
  text: string;
};

export type AnalysisResponse = {
  user: {
    user_id: string;
    email?: string | null;
    authenticated: boolean;
    is_admin?: boolean;
  };
  patient_info: PatientInfo;
  total_records: number;
  records: MedicalRecord[];
  health_summary: HealthSummary;
  body_systems: BodySystem[];
  raw_texts: RawTextPreview[];
  combined_report_file_names?: string[];
  reports_with_data?: number | null;
};

export type AnalyzeStageId = "validating" | "uploading" | "processing" | "saving";

export type AnalyzeStreamEvent =
  | { type: "stage"; step: AnalyzeStageId; status: "active" | "complete" }
  | {
      type: "file";
      file: string;
      step: "queued" | "extracting" | "parsing" | "done" | "failed";
      percent: number;
      processed: number;
      total: number;
      eta_seconds?: number;
      error?: string;
    }
  | { type: "done"; result: AnalysisResponse }
  | { type: "error"; status?: number; message: string };

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

export type ChatResponse = {
  answer: string;
};

export type ClinicalAssistantReportContext = {
  patientInfo?: PatientInfo;
  totalRecords?: number;
  reportsIncluded?: number | null;
  sourceFileNames?: string[];
  records: MedicalRecord[];
};

export type ClinicalAssistantRequest = {
  analysisId: string;
  sessionId: string;
  reportContext: ClinicalAssistantReportContext;
  history: ChatTurn[];
  message: string;
};

export type InsightsResponse = {
  health_summary: HealthSummary;
  body_systems: BodySystem[];
};

export type AnalysisHistoryItem = {
  id: number;
  patient_name: string | null;
  patient_age: string | null;
  patient_gender: string | null;
  lab_name: string | null;
  report_date: string | null;
  total_records: number;
  source_filenames: string[];
  created_at: string;
};

export type ProfileItem = {
  id: string;
  account_owner_id: number;
  full_name: string;
  relationship: string;
  date_of_birth: string | null;
  created_at: string;
};

export type UserProfile = {
  firebase_uid: string;
  email: string | null;
  display_name: string | null;
  is_admin: boolean;
};

export type StudySummary = {
  id: string;
  profile_id: string;
  name: string;
  description: string | null;
  report_count: number;
  range_start: string | null;
  range_end: string | null;
  last_updated: string;
  created_at: string;
};

export type SaveStudyAnalysisResponse = {
  study_id: string;
  added_reports: number;
  total_reports: number;
  study_name: string;
};

export type DashboardStudyItem = {
  id: string;
  name: string;
  description: string | null;
  report_count: number;
  range_start: string | null;
  range_end: string | null;
  consistent_lab_name: string | null;
  has_alerts: boolean;
  alerts_count: number;
  last_updated: string;
};

export type DashboardProfileGroup = {
  profile_id: string;
  full_name: string;
  relationship: string;
  studies: DashboardStudyItem[];
};

export type DashboardSummary = {
  total_reports: number;
  total_alerts: number;
  profiles_tracked: number;
  profiles: DashboardProfileGroup[];
};

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = "Request failed.";
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        message = payload.detail;
      }
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export async function analyzeReports(
  formData: FormData,
): Promise<AnalysisResponse> {
  async function sendAnalyze(baseUrl: string): Promise<Response> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8 * 60 * 1000);
    return authFetch(`${baseUrl}/api/v1/reports/analyze`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    }).catch((error: unknown) => {
      if (error instanceof Error && error.name === "AbortError") {
        throw new Error("Analysis timed out after 8 minutes. Please try fewer PDFs at once.");
      }
      throw error;
    }).finally(() => {
      clearTimeout(timeout);
    });
  }

  let response = await sendAnalyze(API_BASE_URL);
  if (shouldRetryDirect(response)) {
    response = await sendAnalyze(DIRECT_API_BASE_URL);
  }

  const parsed = await parseJsonResponse<AnalysisResponse>(response);
  return normalizeAnalysisPayload(parsed);
}

export async function analyzeReportsStream(
  formData: FormData,
  onEvent: (event: AnalyzeStreamEvent) => void,
): Promise<AnalysisResponse> {
  async function sendStream(baseUrl: string): Promise<Response> {
    return authFetch(`${baseUrl}/api/v1/reports/analyze/stream`, {
      method: "POST",
      headers: {
        Accept: "text/event-stream",
      },
      body: formData,
    });
  }

  const streamBases = Array.from(new Set(
    API_BASE_URL.startsWith("/")
      ? [DIRECT_API_BASE_URL, API_BASE_URL]
      : [API_BASE_URL, DIRECT_API_BASE_URL],
  ));

  let response: Response | null = null;
  let streamFetchError: unknown = null;

  for (let index = 0; index < streamBases.length; index += 1) {
    const baseUrl = streamBases[index]!;
    const isLastAttempt = index === streamBases.length - 1;

    console.log("[STREAM] fetch starting at", new Date().toISOString(), "base:", baseUrl);
    try {
      const candidate = await sendStream(baseUrl);
      console.log("[STREAM] response received, status:", candidate.status);
      console.log("[STREAM] content-type:", candidate.headers.get("content-type"));

      response = candidate;
      if (candidate.ok || candidate.status < 500 || isLastAttempt) {
        break;
      }
    } catch (error) {
      streamFetchError = error;
      console.error("[STREAM] fetch failed:", error);
      if (isLastAttempt) {
        throw error;
      }
    }
  }

  if (!response) {
    throw streamFetchError instanceof Error
      ? streamFetchError
      : new Error("Analysis stream could not be started.");
  }

  if (!response.ok) {
    const parsed = await parseJsonResponse<AnalysisResponse>(response);
    return normalizeAnalysisPayload(parsed);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    console.error("[STREAM] NO READER - response.body is null");
    throw new Error("Stream was not available from the server.");
  }
  console.log("[STREAM] reader acquired, starting read loop");

  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: AnalysisResponse | null = null;
  let eventCount = 0;

  while (true) {
    const { value, done } = await reader.read();
    console.log("[STREAM] read chunk:", { done, byteLength: value?.length });

    if (done) {
      console.log("[STREAM] stream ended, total events parsed:", eventCount);
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const normalizedBuffer = buffer.replace(/\r\n/g, "\n");
    const chunks = normalizedBuffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      console.log("[STREAM] raw chunk text:", chunk.slice(0, 200));

      const dataLines = chunk
        .split("\n")
        .filter((line) => line.startsWith("data:"));
      if (dataLines.length === 0) continue;

      const rawPayload = dataLines
        .map((line) => line.replace(/^data:\s*/, ""))
        .join("\n")
        .trim();
      if (!rawPayload) continue;

      let event: AnalyzeStreamEvent;
      try {
        event = JSON.parse(rawPayload) as AnalyzeStreamEvent;
      } catch {
        continue;
      }

        eventCount += 1;
        console.log(`[STREAM] event #${eventCount}:`, rawPayload.slice(0, 100));

      onEvent(event);

      if (event.type === "done") {
        finalResult = event.result;
      }
      if (event.type === "error") {
        throw new Error(event.message || "Analysis failed.");
      }
    }
  }

  if (!finalResult) {
    throw new Error("Analysis stream ended without a final result.");
  }

  return normalizeAnalysisPayload(finalResult);
}

export async function fetchInsights(
  records: unknown[],
): Promise<InsightsResponse> {
  const response = await authBackendFetch("/api/v1/reports/insights", {
    method: "POST",
    body: JSON.stringify({ records }),
  });

  return parseJsonResponse<InsightsResponse>(response);
}

export async function sendChatMessage(
  payload: ClinicalAssistantRequest,
): Promise<ChatResponse> {
  const response = await authFetch(`/api/clinical-assistant`, {
    method: "POST",
    body: JSON.stringify(payload),
  });

  return parseJsonResponse<ChatResponse>(response);
}

export async function fetchReportHistory(): Promise<AnalysisHistoryItem[]> {
  const response = await authBackendFetch("/api/v1/reports/history");
  return parseJsonResponse<AnalysisHistoryItem[]>(response);
}

export async function fetchReportById(id: number): Promise<AnalysisResponse> {
  const response = await authBackendFetch(`/api/v1/reports/history/${id}`);
  const parsed = await parseJsonResponse<AnalysisResponse>(response);
  return normalizeAnalysisPayload(parsed);
}

export async function saveAnalysis(
  analysis: AnalysisResponse,
  sourceFilenames: string[],
): Promise<AnalysisHistoryItem> {
  const response = await authBackendFetch("/api/v1/reports/save", {
    method: "POST",
    body: JSON.stringify({ analysis, source_filenames: sourceFilenames }),
  });
  return parseJsonResponse<AnalysisHistoryItem>(response);
}

export async function fetchProfiles(): Promise<ProfileItem[]> {
  const response = await authBackendFetch("/api/v1/studies/profiles");
  return parseJsonResponse<ProfileItem[]>(response);
}

export async function createProfile(
  payload: { full_name: string; relationship: string; date_of_birth?: string | null },
): Promise<ProfileItem> {
  const response = await authBackendFetch("/api/v1/studies/profiles", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return parseJsonResponse<ProfileItem>(response);
}

export async function fetchStudiesForProfile(profileId: string): Promise<StudySummary[]> {
  const response = await authBackendFetch(`/api/v1/studies/profiles/${profileId}/studies`);
  return parseJsonResponse<StudySummary[]>(response);
}

export async function createStudy(
  payload: { profile_id: string; name: string; description?: string | null },
): Promise<StudySummary> {
  const response = await authBackendFetch("/api/v1/studies", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return parseJsonResponse<StudySummary>(response);
}

export async function saveStudyAnalysis(
  studyId: string,
  analysis: AnalysisResponse,
  sourceFilenames: string[],
): Promise<SaveStudyAnalysisResponse> {
  const response = await authBackendFetch(`/api/v1/studies/${studyId}/reports/save-analysis`, {
    method: "POST",
    body: JSON.stringify({
      analysis,
      source_filenames: sourceFilenames,
    }),
  });
  return parseJsonResponse<SaveStudyAnalysisResponse>(response);
}

export async function fetchStudiesDashboard(): Promise<DashboardSummary> {
  const response = await authBackendFetch("/api/v1/studies/dashboard");
  return parseJsonResponse<DashboardSummary>(response);
}

export async function fetchStudyCombinedReport(studyId: string): Promise<AnalysisResponse> {
  const response = await authBackendFetch(`/api/v1/studies/${studyId}/combined-report`);
  const parsed = await parseJsonResponse<AnalysisResponse>(response);
  return normalizeAnalysisPayload(parsed);
}

export async function exportPdf(
  records: MedicalRecord[],
  patientInfo: PatientInfo,
): Promise<Blob> {
  const response = await authBackendFetch("/api/v1/reports/export/pdf", {
    method: "POST",
    body: JSON.stringify({ records, patient_info: patientInfo }),
  });
  if (!response.ok) throw new Error("PDF export failed.");
  return response.blob();
}

export async function exportExcel(
  records: MedicalRecord[],
  patientInfo: PatientInfo,
): Promise<Blob> {
  const response = await authBackendFetch("/api/v1/reports/export/excel", {
    method: "POST",
    body: JSON.stringify({ records, patient_info: patientInfo }),
  });
  if (!response.ok) throw new Error("Excel export failed.");
  return response.blob();
}