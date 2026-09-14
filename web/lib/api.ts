import { getDirectApiBaseUrl, getPublicApiBaseUrl } from "./apiBaseUrl";

// Token provider - set by AuthContext on login
let _getTokenFn: (() => Promise<string | null>) | null = null;

export function setAuthTokenProvider(fn: () => Promise<string | null>) {
  _getTokenFn = fn;
}

const API_BASE_URL = (() => {
  const url = getPublicApiBaseUrl();
  if (!url || (process.env.NODE_ENV === "production" && url === "/backend")) {
    console.error("NEXT_PUBLIC_API_URL not set correctly for production.");
    return getDirectApiBaseUrl();
  }
  return url.replace(/\/$/, "");
})();

const DIRECT_API_BASE_URL = getDirectApiBaseUrl().replace(/\/$/, "");

function shouldRetryDirect(response: Response): boolean {
  if (DIRECT_API_BASE_URL === API_BASE_URL) {
    return false;
  }
  return response.status === 404 || response.status >= 500;
}

/** The backend rejected the token. Callers refresh once, then sign out; nothing navigates from here. */
export class AuthError extends Error {
  constructor(message = "Authentication failed") {
    super(message);
    this.name = "AuthError";
  }
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
  let token: string | null = null;

  if (_getTokenFn) {
    token = await _getTokenFn();
  }

  if (!token) {
    throw new AuthError("No auth token available");
  }

  const response = await fetch(url, {
    ...options,
    headers: withAuthHeaders(options, token),
  });

  if (response.status === 401) {
    throw new AuthError("Session expired");
  }

  return response;
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

export type JobStatus = "queued" | "running" | "done" | "failed" | "interrupted";
export type FileStep = "queued" | "extracting" | "parsing" | "done" | "failed";

export type JobProgress = {
  stage: "processing" | "saving" | "done";
  files: Record<string, { step: FileStep; percent: number; error?: string }>;
  processed: number;
  total: number;
  eta_seconds?: number | null;
};

export type ReportJob = {
  id: string;
  status: JobStatus;
  progress: JobProgress;
  error: string | null;
  study_id: string | null;
  analysis_id: number | null;
  source_filenames: string[];
  created_at: string;
  finished_at: string | null;
};

export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

export type ChatResponse = {
  answer: string;
};

export type ChatStreamEvent =
  | { type: "started"; sessionId?: string; analysisId?: string; serverReceivedAt?: number }
  | { type: "keepalive"; ts?: number }
  | { type: "delta"; text: string }
  | { type: "done"; answer: string; backendLatencyMs?: number; totalLatencyMs?: number }
  | { type: "error"; status?: number; message: string };

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

export async function startAnalysisJob(formData: FormData): Promise<ReportJob> {
  const response = await authBackendFetch("/api/v1/reports/jobs", { method: "POST", body: formData });
  return parseJsonResponse<ReportJob>(response);
}

export async function fetchJob(jobId: string): Promise<ReportJob> {
  const response = await authBackendFetch(`/api/v1/reports/jobs/${jobId}`);
  return parseJsonResponse<ReportJob>(response);
}

const JOB_POLL_INTERVAL_MS = 2_000;
const JOB_SETTLED: JobStatus[] = ["done", "failed", "interrupted"];

/** Polls until the job settles. Only the id is needed, so a refreshed tab can pick a job back up. */
export async function waitForJob(jobId: string, onProgress: (job: ReportJob) => void): Promise<ReportJob> {
  for (;;) {
    const job = await fetchJob(jobId);
    onProgress(job);
    if (JOB_SETTLED.includes(job.status)) return job;
    await new Promise((resolve) => setTimeout(resolve, JOB_POLL_INTERVAL_MS));
  }
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

export async function sendChatMessageStream(
  payload: ClinicalAssistantRequest,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<ChatResponse> {
  const response = await authFetch(`/api/clinical-assistant`, {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      ...payload,
      stream: true,
    }),
  });

  if (!response.ok) {
    return parseJsonResponse<ChatResponse>(response);
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("text/event-stream")) {
    return parseJsonResponse<ChatResponse>(response);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Chat stream was unavailable from the server.");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let finalAnswer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const normalizedBuffer = buffer.replace(/\r\n/g, "\n");
    const chunks = normalizedBuffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const dataLines = chunk
        .split("\n")
        .filter((line) => line.startsWith("data:"));
      if (dataLines.length === 0) continue;

      const rawPayload = dataLines
        .map((line) => line.replace(/^data:\s*/, ""))
        .join("\n")
        .trim();
      if (!rawPayload) continue;

      let event: ChatStreamEvent;
      try {
        event = JSON.parse(rawPayload) as ChatStreamEvent;
      } catch {
        continue;
      }

      onEvent(event);

      if (event.type === "delta") {
        finalAnswer += event.text;
      }

      if (event.type === "done") {
        finalAnswer = event.answer;
      }

      if (event.type === "error") {
        throw new Error(event.message || "Clinical assistant request failed.");
      }
    }
  }

  if (!finalAnswer.trim()) {
    throw new Error("Clinical assistant stream ended without an answer.");
  }

  return { answer: finalAnswer };
}

export async function fetchReportHistory(): Promise<AnalysisHistoryItem[]> {
  const response = await authBackendFetch("/api/v1/reports/history");
  return parseJsonResponse<AnalysisHistoryItem[]>(response);
}

export async function fetchReportById(id: number): Promise<AnalysisResponse> {
  const response = await authBackendFetch(`/api/v1/reports/history/${id}`);
  const parsed = await parseJsonResponse<AnalysisResponse>(response);
  return parsed;
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

export async function fetchStudiesDashboard(): Promise<DashboardSummary> {
  const response = await authBackendFetch("/api/v1/studies/dashboard");
  return parseJsonResponse<DashboardSummary>(response);
}

export async function fetchStudyCombinedReport(studyId: string): Promise<AnalysisResponse> {
  const response = await authBackendFetch(`/api/v1/studies/${studyId}/combined-report`);
  const parsed = await parseJsonResponse<AnalysisResponse>(response);
  return parsed;
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