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
  // Only reads are retried: repeating a POST could start a second upload job.
  const isRead = (options.method ?? "GET").toUpperCase() === "GET";
  if (isRead && shouldRetryDirect(response)) {
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
  /** One entry per file that was skipped while the rest of the upload was saved. */
  warnings?: string[];
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

/** The server loads the records for the study or saved analysis named here, after checking they are yours. */
export type ChatRequest = {
  question: string;
  history: ChatTurn[];
  sessionId: string;
} & ({ studyId: string } | { analysisId: number });

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

export async function downloadMyData(): Promise<Blob> {
  const response = await authBackendFetch("/api/v1/me/export");
  if (!response.ok) await parseJsonResponse(response);
  return response.blob();
}

export async function deleteMyAccount(): Promise<void> {
  const response = await authBackendFetch("/api/v1/me", { method: "DELETE" });
  if (!response.ok) await parseJsonResponse(response);
}

export async function startAnalysisJob(formData: FormData): Promise<ReportJob> {
  const response = await authBackendFetch("/api/v1/reports/jobs", { method: "POST", body: formData });
  return parseJsonResponse<ReportJob>(response);
}

/** The job no longer exists for this user (deleted, or another account's id). Polling cannot recover. */
export class JobNotFoundError extends Error {
  constructor() {
    super("This upload could not be found. Please upload the files again.");
    this.name = "JobNotFoundError";
  }
}

/** A poll that failed for a reason worth retrying: network down, server waking up, or overloaded. */
class TransientPollError extends Error {}

export async function fetchJob(jobId: string, signal?: AbortSignal): Promise<ReportJob> {
  let response: Response;
  try {
    response = await authBackendFetch(`/api/v1/reports/jobs/${jobId}`, { signal });
  } catch (error) {
    if (error instanceof AuthError || isAbortError(error)) throw error;
    throw new TransientPollError(error instanceof Error ? error.message : "Network error");
  }
  if (response.status === 404 || response.status === 403) throw new JobNotFoundError();
  if (response.status >= 500 || response.status === 429 || response.status === 408) {
    throw new TransientPollError(`Server returned ${response.status}`);
  }
  return parseJsonResponse<ReportJob>(response);
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

const JOB_POLL_INTERVAL_MS = 2_000;
const JOB_POLL_MAX_INTERVAL_MS = 15_000;
// Longer than the server's own stale-job limit (30 min), which settles a lost job as failed first.
const JOB_POLL_DEADLINE_MS = 40 * 60_000;
const JOB_SETTLED: JobStatus[] = ["done", "failed", "interrupted"];

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Polls until the job settles. Only the id is needed, so a refreshed tab can pick a job back up.
 * A failed poll is not a failed job: network blips and a waking server are retried with backoff,
 * because the job keeps running on the server either way.
 */
export async function waitForJob(
  jobId: string,
  onProgress: (job: ReportJob) => void,
  signal?: AbortSignal,
): Promise<ReportJob> {
  const deadline = Date.now() + JOB_POLL_DEADLINE_MS;
  let delay = JOB_POLL_INTERVAL_MS;
  for (;;) {
    try {
      const job = await fetchJob(jobId, signal);
      onProgress(job);
      if (JOB_SETTLED.includes(job.status)) return job;
      delay = JOB_POLL_INTERVAL_MS;
    } catch (error) {
      if (!(error instanceof TransientPollError)) throw error;
      delay = Math.min(delay * 2, JOB_POLL_MAX_INTERVAL_MS);
    }
    if (Date.now() + delay > deadline) {
      throw new Error("Lost contact with the server while it was processing your upload. Check your reports list; it may still have been saved.");
    }
    await sleep(delay, signal);
  }
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await authBackendFetch("/api/v1/reports/chat", {
    method: "POST",
    body: JSON.stringify({
      question: request.question,
      history: request.history,
      session_id: request.sessionId,
      study_id: "studyId" in request ? request.studyId : null,
      analysis_id: "analysisId" in request ? request.analysisId : null,
    }),
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