"use client";

import { useState, useTransition, useEffect, useCallback, useRef, useMemo } from "react";
import { useRouter } from "next/navigation";
import dayjs from "dayjs";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { DatePicker } from "@mui/x-date-pickers/DatePicker";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { useAuth } from "@/lib/auth-context";
import {
  type AnalysisResponse,
  type AnalyzeStreamEvent,
  type AnalysisHistoryItem,
  type ChatTurn,
  type DashboardSummary,
  type ProfileItem,
  type StudySummary,
  analyzeReportsStream,
  createProfile,
  createStudy,
  fetchProfiles,
  sendChatMessage,
  fetchStudiesForProfile,
  fetchStudiesDashboard,
  fetchStudyCombinedReport,
  fetchReportHistory,
  fetchReportById,
  saveAnalysis,
  saveStudyAnalysis,
  exportExcel,
} from "@/lib/api";
import TrendChart from "./TrendChart";
import AlertsByCategory from "./AlertsByCategory";
import OrganizedDataTree from "./OrganizedDataTree";
import ClinicalChatPanel from "./ClinicalChatPanel";
import { generateClinicalPdfReport } from "@/lib/pdf";
import { CANONICAL_CATEGORIES, canonicalizeCategory } from "@/lib/categoryMap";

type AnalyzeStep = "idle" | "preparing" | "uploading" | "processing" | "saving" | "error";
type StudyAction = "add-existing" | "start-new";

type StageState = "pending" | "active" | "complete" | "error";

type FileStatus = "queued" | "extracting" | "parsing" | "done" | "failed";

type FileProgressItem = {
  file: string;
  percent: number;
  status: FileStatus;
  error?: string;
};

type AnalyzeContext = {
  profile: ProfileItem;
  study: StudySummary;
  mode: "existing" | "new";
};

type ChatRole = "user" | "assistant" | "system";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  isLoading?: boolean;
};

const MIN_CHAT_LOADING_MS = 400;

function createChatMessageId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function toApiChatHistory(messages: ChatMessage[]): ChatTurn[] {
  return messages
    .filter((message) => !message.isLoading && (message.role === "user" || message.role === "assistant"))
    .map((message) => ({
      role: message.role === "assistant" ? "assistant" : "user",
      content: message.content,
    }));
}

function parseMedicalDate(value: string | null | undefined): number {
  if (!value) return Number.MAX_SAFE_INTEGER;
  const raw = value.trim();
  const ddmmyyyy = raw.match(/^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})$/);
  if (ddmmyyyy) {
    const day = ddmmyyyy[1]!.padStart(2, "0");
    const month = ddmmyyyy[2]!.padStart(2, "0");
    const year = ddmmyyyy[3]!.length === 2 ? `20${ddmmyyyy[3]}` : ddmmyyyy[3]!;
    return new Date(`${year}-${month}-${day}`).getTime();
  }
  const parsed = new Date(raw).getTime();
  return Number.isNaN(parsed) ? Number.MAX_SAFE_INTEGER : parsed;
}

function isConcerningStatus(status: string | null | undefined): boolean {
  const normalized = status?.toLowerCase();
  return normalized === "high" || normalized === "low" || normalized === "critical" || normalized === "positive" || normalized === "flagged";
}

function fmtRelativeDate(iso: string) {
  try {
    const value = new Date(iso).getTime();
    if (Number.isNaN(value)) return "Unknown";
    const deltaMs = Date.now() - value;
    const days = Math.floor(deltaMs / 86400000);
    if (days <= 0) return "Today";
    if (days === 1) return "1 day ago";
    if (days < 30) return `${days} days ago`;
    const months = Math.floor(days / 30);
    if (months === 1) return "1 month ago";
    return `${months} months ago`;
  } catch {
    return "Unknown";
  }
}

function fmtRange(start: string | null, end: string | null): string {
  if (!start && !end) return "No reports yet";
  const toLabel = (v: string) => {
    try {
      return new Date(v).toLocaleDateString("en-IN", { month: "short", year: "numeric" });
    } catch {
      return v;
    }
  };
  if (start && end) return `${toLabel(start)} - ${toLabel(end)}`;
  return toLabel(start ?? end ?? "");
}

function relationshipLabel(value: string): string {
  const clean = (value || "").trim();
  if (!clean) return "Family";
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

function normalizeSourceFileName(value: string | null | undefined): string {
  if (!value) return "";
  const normalized = value.replace(/\\/g, "/").split("/").pop() ?? value;
  return normalized.trim().toLowerCase();
}

function shouldRetryWithFreshToken(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  return /invalid firebase token|firebase token expired|firebase token revoked|firebase token project mismatch/i.test(
    error.message,
  );
}

function formatChatTimestamp(createdAt: number): string {
  if (Date.now() - createdAt < 60_000) {
    return "Just now";
  }
  return new Date(createdAt).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
  });
}

function buildUserInitials(displayName?: string | null, email?: string | null): string {
  const fallback = (displayName || email || "User").trim();
  if (!fallback) return "U";

  const fromWords = fallback
    .replace(/@.*$/, "")
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");

  const initials = fromWords.slice(0, 2);
  if (initials) {
    return initials;
  }
  return fallback.slice(0, 2).toUpperCase();
}

function NavBar({ user, onLogout, onHome }: { user: { displayName?: string | null; email?: string | null } | null; onLogout: () => void; onHome: () => void }) {
  return (
    <nav className="top-nav">
      <button className="nav-brand" onClick={onHome} type="button">Medical Report Analyzer</button>
      <div className="nav-right">
        {user && <span className="nav-user">{user.displayName ?? user.email ?? "User"}</span>}
        <button className="secondary-button nav-logout" onClick={onLogout} type="button">Sign Out</button>
      </div>
    </nav>
  );
}

function StudyFlowModal({
  open,
  userName,
  profiles,
  selectedProfile,
  studies,
  selectedAction,
  selectedStudy,
  step,
  loading,
  formError,
  onClose,
  onSelectProfile,
  onSelectAction,
  onSelectStudy,
  onConfirmAdd,
  onBack,
  newStudyName,
  newStudyDescription,
  onChangeNewStudyName,
  onChangeNewStudyDescription,
  onCreateStudy,
  newMemberName,
  newMemberRelationship,
  newMemberOtherRelationship,
  newMemberDob,
  onChangeMemberName,
  onChangeMemberRelationship,
  onChangeMemberOtherRelationship,
  onChangeMemberDob,
  onCreateMember,
  onStartNewMember,
}: {
  open: boolean;
  userName: string;
  profiles: ProfileItem[];
  selectedProfile: ProfileItem | null;
  studies: StudySummary[];
  selectedAction: StudyAction | null;
  selectedStudy: StudySummary | null;
  step: "who" | "what" | "pick-study" | "confirm-add" | "new-study" | "new-member";
  loading: boolean;
  formError: string | null;
  onClose: () => void;
  onSelectProfile: (profile: ProfileItem) => void;
  onSelectAction: (action: StudyAction) => void;
  onSelectStudy: (study: StudySummary) => void;
  onConfirmAdd: () => void;
  onBack: () => void;
  newStudyName: string;
  newStudyDescription: string;
  onChangeNewStudyName: (value: string) => void;
  onChangeNewStudyDescription: (value: string) => void;
  onCreateStudy: () => void;
  newMemberName: string;
  newMemberRelationship: string;
  newMemberOtherRelationship: string;
  newMemberDob: string;
  onChangeMemberName: (value: string) => void;
  onChangeMemberRelationship: (value: string) => void;
  onChangeMemberOtherRelationship: (value: string) => void;
  onChangeMemberDob: (value: string) => void;
  onCreateMember: () => void;
  onStartNewMember: () => void;
}) {
  if (!open) return null;

  const sortedProfiles = [...profiles].sort((a, b) => {
    const aSelf = a.relationship.toLowerCase() === "self" ? 0 : 1;
    const bSelf = b.relationship.toLowerCase() === "self" ? 0 : 1;
    if (aSelf !== bSelf) return aSelf - bSelf;
    return a.full_name.localeCompare(b.full_name);
  });

  const stepMeta: Record<typeof step, { index: number; total: number }> = {
    who: { index: 1, total: 3 },
    what: { index: 2, total: 3 },
    "pick-study": { index: 3, total: 4 },
    "confirm-add": { index: 4, total: 4 },
    "new-study": { index: 3, total: 3 },
    "new-member": { index: 2, total: 3 },
  };
  const { index, total } = stepMeta[step];

  return (
    <div className="study-flow-backdrop" role="dialog" aria-modal="true">
      <div className="study-flow-modal">
        <div className="study-flow-head">
          <div className="study-flow-progress" aria-label="Step indicator">
            <span>Step {index} of {total}</span>
            <div className="study-flow-dots">
              {Array.from({ length: total }).map((_, idx) => (
                <i key={idx} className={idx + 1 <= index ? "active" : ""} />
              ))}
            </div>
          </div>
          <button className="secondary-button" type="button" onClick={onClose}>Close</button>
        </div>

        {step === "who" && (
          <section className="study-step-panel">
            <h3>Who are these reports for?</h3>
            <p>Select a profile first, then choose whether to append or start a fresh study.</p>
            <div className="study-card-grid">
              {sortedProfiles.map((profile) => {
                const isSelf = profile.relationship.toLowerCase() === "self";
                const label = isSelf
                  ? `Myself - ${userName || profile.full_name}`
                  : `${profile.full_name} - ${relationshipLabel(profile.relationship)}`;
                return (
                  <button
                    key={profile.id}
                    className={`study-choice-card ${selectedProfile?.id === profile.id ? "active" : ""}`}
                    type="button"
                    onClick={() => onSelectProfile(profile)}
                    disabled={loading}
                  >
                    <strong>{label}</strong>
                    <span>{isSelf ? "Primary account profile" : "Family profile"}</span>
                  </button>
                );
              })}
              <button className="study-choice-card add" type="button" onClick={onStartNewMember} disabled={loading}>
                <strong>Add a New Family Member</strong>
                <span>Create profile and continue to first study setup</span>
              </button>
            </div>
            {sortedProfiles.length === 0 && !loading && (
              <p className="muted-copy">
                We could not load your existing profiles yet. Try closing this dialog and reopening it after your session refreshes.
              </p>
            )}
          </section>
        )}

        {step === "what" && selectedProfile && (
          <section className="study-step-panel">
            <h3>What would you like to do for {selectedProfile.full_name}?</h3>
            <div className="study-card-grid two-col">
              <button
                className={`study-choice-card ${selectedAction === "add-existing" ? "active" : ""}`}
                type="button"
                onClick={() => onSelectAction("add-existing")}
                disabled={loading}
              >
                <strong>Add to an existing study</strong>
                <span>Upload new reports and merge trends into an existing study.</span>
              </button>
              <button
                className={`study-choice-card ${selectedAction === "start-new" ? "active" : ""}`}
                type="button"
                onClick={() => onSelectAction("start-new")}
                disabled={loading}
              >
                <strong>Start a new study</strong>
                <span>Begin an independent study for this same profile.</span>
              </button>
            </div>
            <div className="study-step-actions">
              <button className="secondary-button" type="button" onClick={onBack}>Back</button>
            </div>
          </section>
        )}

        {step === "pick-study" && selectedProfile && (
          <section className="study-step-panel">
            <h3>Select an existing study for {selectedProfile.full_name}</h3>
            <div className="study-list-stack">
              {studies.map((study) => (
                <button
                  key={study.id}
                  className={`study-list-card ${selectedStudy?.id === study.id ? "active" : ""}`}
                  type="button"
                  onClick={() => onSelectStudy(study)}
                >
                  <strong>{study.name}</strong>
                  <p>{study.report_count} reports • {fmtRange(study.range_start, study.range_end)}</p>
                  <span>Last updated {fmtRelativeDate(study.last_updated)}</span>
                </button>
              ))}
            </div>
            <div className="study-step-actions">
              <button className="secondary-button" type="button" onClick={onBack}>Back</button>
            </div>
          </section>
        )}

        {step === "confirm-add" && selectedStudy && (
          <section className="study-step-panel">
            <h3>Add to {selectedStudy.name}?</h3>
            <p>
              New reports will be analyzed together with {selectedStudy.report_count} existing reports in this study to strengthen trend tracking.
            </p>
            <div className="study-step-actions">
              <button className="primary-button" type="button" onClick={onConfirmAdd}>Yes, Add to This Study</button>
              <button className="secondary-button" type="button" onClick={onBack}>Go Back</button>
            </div>
          </section>
        )}

        {step === "new-study" && selectedProfile && (
          <section className="study-step-panel">
            <h3>Create a new study for {selectedProfile.full_name}</h3>
            <label className="field-block">
              <span>Study Name</span>
              <input
                className="input"
                maxLength={60}
                placeholder="e.g. Cardiac Monitoring 2025"
                value={newStudyName}
                onChange={(e) => onChangeNewStudyName(e.target.value)}
              />
            </label>
            <label className="field-block">
              <span>Description (optional)</span>
              <textarea
                className="textarea"
                maxLength={200}
                placeholder="What is this study tracking?"
                value={newStudyDescription}
                onChange={(e) => onChangeNewStudyDescription(e.target.value)}
              />
            </label>
            <div className="study-step-actions">
              <button className="primary-button" type="button" onClick={onCreateStudy} disabled={loading}>
                Create Study and Upload Reports
              </button>
              <button className="secondary-button" type="button" onClick={onBack}>Back</button>
            </div>
          </section>
        )}

        {step === "new-member" && (
          <section className="study-step-panel">
            <h3>Add a new family member</h3>
            <label className="field-block">
              <span>Full Name</span>
              <input
                className="input"
                placeholder="Enter their full name"
                value={newMemberName}
                onChange={(e) => onChangeMemberName(e.target.value)}
              />
            </label>
            <label className="field-block">
              <span>Relationship</span>
              <select
                className="select-input"
                value={newMemberRelationship}
                onChange={(e) => onChangeMemberRelationship(e.target.value)}
              >
                <option value="father">Father</option>
                <option value="mother">Mother</option>
                <option value="spouse">Spouse</option>
                <option value="child">Child</option>
                <option value="sibling">Sibling</option>
                <option value="grandparent">Grandparent</option>
                <option value="other">Other</option>
              </select>
            </label>
            {newMemberRelationship === "other" && (
              <label className="field-block">
                <span>Specify Relationship</span>
                <input
                  className="input"
                  placeholder="e.g. Cousin"
                  value={newMemberOtherRelationship}
                  onChange={(e) => onChangeMemberOtherRelationship(e.target.value)}
                />
              </label>
            )}
            <label className="field-block">
              <span>Date of Birth (optional)</span>
              <LocalizationProvider dateAdapter={AdapterDayjs}>
                <DatePicker
                  format="DD/MM/YYYY"
                  value={newMemberDob ? dayjs(newMemberDob) : null}
                  onChange={(value) => onChangeMemberDob(value ? value.format("YYYY-MM-DD") : "")}
                  slotProps={{
                    textField: {
                      size: "small",
                      fullWidth: true,
                      className: "mui-dob-field",
                    },
                    popper: {
                      placement: "bottom-start",
                    },
                  }}
                />
              </LocalizationProvider>
            </label>
            <small className="muted-copy">Helps with age-specific reference ranges in analysis.</small>
            <div className="study-step-actions">
              <button className="primary-button" type="button" onClick={onCreateMember} disabled={loading}>
                Create Profile and Continue
              </button>
              <button className="secondary-button" type="button" onClick={onBack}>Back</button>
            </div>
          </section>
        )}

        {formError && <p className="status-text error-text">{formError}</p>}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { user, loading, getToken, logout } = useAuth();
  const router = useRouter();

  useEffect(() => { if (!loading && !user) router.replace("/login"); }, [user, loading, router]);

  const [view, setView] = useState<"home" | "analyze" | "result">("home");
  const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [dashboardSummary, setDashboardSummary] = useState<DashboardSummary | null>(null);
  const [pdfFiles, setPdfFiles] = useState<File[]>([]);
  const [existingDataFile, setExistingDataFile] = useState<File | null>(null);
  const [uploadMode, setUploadMode] = useState<"new" | "append">("new");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeStep, setAnalyzeStep] = useState<AnalyzeStep>("idle");
  const [analysisStartedAt, setAnalysisStartedAt] = useState<number | null>(null);
  const [analysisElapsedSeconds, setAnalysisElapsedSeconds] = useState(0);
  const [estimatedRemainingSeconds, setEstimatedRemainingSeconds] = useState<number | null>(null);
  const [currentParsingFile, setCurrentParsingFile] = useState<string | null>(null);
  const [fileProgress, setFileProgress] = useState<FileProgressItem[]>([]);
  const [processedFilesCount, setProcessedFilesCount] = useState(0);
  const [stageStates, setStageStates] = useState<Record<"validating" | "uploading" | "processing" | "saving", StageState>>({
    validating: "pending",
    uploading: "pending",
    processing: "pending",
    saving: "pending",
  });
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [chatSessionId, setChatSessionId] = useState(() => createChatMessageId("session"));
  const [activeAnalysisId, setActiveAnalysisId] = useState<string | null>(null);
  const [showAssistantPrompts, setShowAssistantPrompts] = useState(true);
  const [chatError, setChatError] = useState<string | null>(null);
  const [isChatPending, startChatTransition] = useTransition();
  const [selectedBodySystem, setSelectedBodySystem] = useState<string>("all");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedTest, setSelectedTest] = useState<string>("");
  const [isExporting, setIsExporting] = useState<"pdf" | "excel" | null>(null);
  const [studyFlowOpen, setStudyFlowOpen] = useState(false);
  const [studyFlowStep, setStudyFlowStep] = useState<"who" | "what" | "pick-study" | "confirm-add" | "new-study" | "new-member">("who");
  const [flowLoading, setFlowLoading] = useState(false);
  const [flowError, setFlowError] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<ProfileItem[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<ProfileItem | null>(null);
  const [selectedAction, setSelectedAction] = useState<StudyAction | null>(null);
  const [profileStudies, setProfileStudies] = useState<StudySummary[]>([]);
  const [selectedStudy, setSelectedStudy] = useState<StudySummary | null>(null);
  const [studyContext, setStudyContext] = useState<AnalyzeContext | null>(null);
  const [studySuccessMessage, setStudySuccessMessage] = useState<string | null>(null);
  const [newStudyName, setNewStudyName] = useState("");
  const [newStudyDescription, setNewStudyDescription] = useState("");
  const [newMemberName, setNewMemberName] = useState("");
  const [newMemberRelationship, setNewMemberRelationship] = useState("father");
  const [newMemberOtherRelationship, setNewMemberOtherRelationship] = useState("");
  const [newMemberDob, setNewMemberDob] = useState("");
  const [viewLoadingStudyId, setViewLoadingStudyId] = useState<string | null>(null);
  const [resultStudyReportCount, setResultStudyReportCount] = useState<number | null>(null);
  const reauthRedirectedRef = useRef(false);
  const assistantMessagesEndRef = useRef<HTMLDivElement>(null);
  const userInitials = useMemo(() => buildUserInitials(user?.displayName, user?.email), [user?.displayName, user?.email]);

  useEffect(() => {
    assistantMessagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const forceReauth = useCallback(async (): Promise<never> => {
    if (!reauthRedirectedRef.current) {
      reauthRedirectedRef.current = true;
      try {
        await logout();
      } catch {
        // Ignore logout failures and still push user to login for re-auth.
      }
      router.replace("/login");
    }
    throw new Error("Your session is no longer valid. Please sign in again.");
  }, [logout, router]);

  const runWithTokenRetry = useCallback(async <T,>(operation: (token: string) => Promise<T>): Promise<T> => {
    const token = await getToken();
    if (!token) {
      return forceReauth();
    }

    try {
      return await operation(token);
    } catch (error) {
      if (!shouldRetryWithFreshToken(error)) {
        throw error;
      }

      const refreshedToken = await getToken(true);
      if (!refreshedToken) {
        return forceReauth();
      }

      try {
        return await operation(refreshedToken);
      } catch (retryError) {
        if (shouldRetryWithFreshToken(retryError)) {
          return forceReauth();
        }
        throw retryError;
      }
    }
  }, [forceReauth, getToken]);

  const resetChatState = useCallback(() => {
    setChatQuestion("");
    setChatError(null);
    setChatHistory([]);
    setChatSessionId(createChatMessageId("session"));
    setShowAssistantPrompts(true);
  }, []);

  const loadDashboardData = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const [items, summary] = await runWithTokenRetry((token) => Promise.all([
        fetchReportHistory(token),
        fetchStudiesDashboard(token),
      ]));
      setHistory(items);
      setDashboardSummary(summary);
    } catch { /* non-critical */ } finally { setHistoryLoading(false); }
  }, [runWithTokenRetry]);

  useEffect(() => { if (user) loadDashboardData(); }, [user, loadDashboardData]);

  function resetStudyFlow() {
    setStudyFlowStep("who");
    setFlowError(null);
    setSelectedProfile(null);
    setSelectedAction(null);
    setProfileStudies([]);
    setSelectedStudy(null);
    setNewStudyName("");
    setNewStudyDescription("");
    setNewMemberName("");
    setNewMemberRelationship("father");
    setNewMemberOtherRelationship("");
    setNewMemberDob("");
  }

  async function openStudyFlow() {
    setStudyFlowOpen(true);
    resetStudyFlow();
    setFlowLoading(true);
    try {
      let rows = await runWithTokenRetry((token) => fetchProfiles(token));
      if (!rows.some((item) => item.relationship.trim().toLowerCase() === "self")) {
        const fallbackName = (user?.displayName || user?.email?.split("@")[0] || "My Profile").trim() || "My Profile";
        try {
          const createdSelf = await runWithTokenRetry((token) => createProfile({
            full_name: fallbackName,
            relationship: "self",
          }, token));
          rows = [...rows, createdSelf];
        } catch {
          // If self profile bootstrap fails, continue with fetched profiles.
        }
      }
      setProfiles(rows);
    } catch (err) {
      setFlowError(err instanceof Error ? err.message : "Unable to load profiles.");
    } finally {
      setFlowLoading(false);
    }
  }

  async function handleSelectProfile(profile: ProfileItem) {
    setSelectedProfile(profile);
    setSelectedAction(null);
    setSelectedStudy(null);
    setFlowError(null);
    setFlowLoading(true);
    try {
      const studies = await runWithTokenRetry((token) => fetchStudiesForProfile(profile.id, token));
      setProfileStudies(studies);
      if (studies.length > 0) {
        setStudyFlowStep("what");
      } else {
        setStudyFlowStep("new-study");
      }
    } catch (err) {
      setFlowError(err instanceof Error ? err.message : "Unable to load studies.");
    } finally {
      setFlowLoading(false);
    }
  }

  function handleSelectAction(action: StudyAction) {
    setSelectedAction(action);
    setFlowError(null);
    if (action === "add-existing") {
      setStudyFlowStep("pick-study");
      return;
    }
    setStudyFlowStep("new-study");
  }

  function proceedToAnalyze(context: AnalyzeContext) {
    setStudyContext(context);
    setStudyFlowOpen(false);
    setPdfFiles([]);
    setExistingDataFile(null);
    setUploadMode("new");
    setErrorMessage(null);
    setStudySuccessMessage(null);
    setView("analyze");
  }

  function handleConfirmAddToStudy() {
    if (!selectedProfile || !selectedStudy) {
      setFlowError("Please select a study first.");
      return;
    }
    proceedToAnalyze({
      profile: selectedProfile,
      study: selectedStudy,
      mode: "existing",
    });
  }

  async function handleCreateStudyAndContinue() {
    if (!selectedProfile) {
      setFlowError("Please select a profile first.");
      return;
    }
    if (!newStudyName.trim()) {
      setFlowError("Study name is required.");
      return;
    }
    if (newStudyName.trim().length > 60) {
      setFlowError("Study name must be 60 characters or less.");
      return;
    }
    if (newStudyDescription.trim().length > 200) {
      setFlowError("Description must be 200 characters or less.");
      return;
    }

    setFlowLoading(true);
    setFlowError(null);
    try {
      const created = await runWithTokenRetry((token) => createStudy(
        {
          profile_id: selectedProfile.id,
          name: newStudyName.trim(),
          description: newStudyDescription.trim() || undefined,
        },
        token,
      ));
      proceedToAnalyze({
        profile: selectedProfile,
        study: created,
        mode: "new",
      });
    } catch (err) {
      setFlowError(err instanceof Error ? err.message : "Failed to create study.");
    } finally {
      setFlowLoading(false);
    }
  }

  async function handleCreateFamilyMemberAndContinue() {
    if (!newMemberName.trim()) {
      setFlowError("Family member name is required.");
      return;
    }
    const relationship = newMemberRelationship === "other"
      ? newMemberOtherRelationship.trim()
      : newMemberRelationship;
    if (!relationship) {
      setFlowError("Relationship is required.");
      return;
    }

    setFlowLoading(true);
    setFlowError(null);
    try {
      const created = await runWithTokenRetry((token) => createProfile(
        {
          full_name: newMemberName.trim(),
          relationship,
          date_of_birth: newMemberDob || undefined,
        },
        token,
      ));
      setProfiles((prev) => [...prev, created]);
      setSelectedProfile(created);
      setProfileStudies([]);
      setStudyFlowStep("new-study");
    } catch (err) {
      setFlowError(err instanceof Error ? err.message : "Failed to create profile.");
    } finally {
      setFlowLoading(false);
    }
  }

  function handleStudyFlowBack() {
    setFlowError(null);
    if (studyFlowStep === "what") setStudyFlowStep("who");
    else if (studyFlowStep === "pick-study") setStudyFlowStep("what");
    else if (studyFlowStep === "confirm-add") setStudyFlowStep("pick-study");
    else if (studyFlowStep === "new-study") {
      if (selectedProfile && profileStudies.length > 0) setStudyFlowStep("what");
      else setStudyFlowStep("who");
    }
    else if (studyFlowStep === "new-member") setStudyFlowStep("who");
  }

  function startAddReportsShortcut(profile: { profile_id: string; full_name: string; relationship: string }, study: {
    id: string;
    name: string;
    report_count: number;
    range_start: string | null;
    range_end: string | null;
    last_updated: string;
  }) {
    const profileLike: ProfileItem = {
      id: profile.profile_id,
      account_owner_id: 0,
      full_name: profile.full_name,
      relationship: profile.relationship,
      date_of_birth: null,
      created_at: new Date().toISOString(),
    };
    const studyLike: StudySummary = {
      id: study.id,
      profile_id: profile.profile_id,
      name: study.name,
      description: null,
      report_count: study.report_count,
      range_start: study.range_start,
      range_end: study.range_end,
      last_updated: study.last_updated,
      created_at: study.last_updated,
    };

    setStudyFlowOpen(true);
    setStudyFlowStep("confirm-add");
    setFlowError(null);
    setSelectedProfile(profileLike);
    setSelectedStudy(studyLike);
    setSelectedAction("add-existing");
    setProfileStudies([studyLike]);
  }

  function handleAnalyzeStreamEvent(event: AnalyzeStreamEvent) {
    if (event.type === "stage") {
      const mapStep: Record<string, AnalyzeStep> = {
        validating: "preparing",
        uploading: "uploading",
        processing: "processing",
        saving: "saving",
      };
      if (event.status === "active") {
        setAnalyzeStep(mapStep[event.step]);
      }
      setStageStates((prev) => ({
        ...prev,
        [event.step]: event.status === "complete" ? "complete" : "active",
      }));
      return;
    }

    if (event.type === "file") {
      setFileProgress((prev) => {
        const next = [...prev];
        const idx = next.findIndex((item) => item.file === event.file);
        const item: FileProgressItem = {
          file: event.file,
          percent: event.percent,
          status: event.step,
          error: event.error,
        };
        if (idx === -1) next.push(item);
        else next[idx] = item;
        return next;
      });

      setProcessedFilesCount(event.processed);
      if (typeof event.eta_seconds === "number") {
        setEstimatedRemainingSeconds(event.eta_seconds);
      }
      if (event.step === "extracting" || event.step === "parsing") {
        setCurrentParsingFile(event.file);
      }
      if (event.step === "done" || event.step === "failed") {
        setCurrentParsingFile((prev) => (prev === event.file ? null : prev));
      }
    }
  }

  async function submitAnalysis() {
    setErrorMessage(null);
    setIsAnalyzing(true);
    setAnalyzeStep("preparing");
    setAnalysisStartedAt(Date.now());
    setAnalysisElapsedSeconds(0);
    setEstimatedRemainingSeconds(null);
    setCurrentParsingFile(null);
    setProcessedFilesCount(0);
    setStageStates({
      validating: "pending",
      uploading: "pending",
      processing: "pending",
      saving: "pending",
    });

    if (pdfFiles.length === 0) {
      setErrorMessage("Please select at least one PDF file.");
      setIsAnalyzing(false);
      setAnalyzeStep("error");
      return;
    }

    setFileProgress(
      pdfFiles.map((file) => ({ file: file.name, percent: 0, status: "queued" }))
    );

    const fd = new FormData();
    for (const f of pdfFiles) fd.append("pdf_files", f);
    if (existingDataFile) fd.append("existing_data", existingDataFile);

    try {
      const result = await runWithTokenRetry((token) => analyzeReportsStream(fd, token, handleAnalyzeStreamEvent));
      setAnalysis(result);
      setActiveAnalysisId(studyContext ? `study-${studyContext.study.id}` : `analysis-${Date.now()}`);
      resetChatState();
      setSelectedBodySystem("all");
      setSelectedCategory("all");
      setSelectedTest("");
      setView("result");

      setAnalyzeStep("saving");
      setSaveStatus("saving");
      try {
        if (studyContext) {
          const saved = await runWithTokenRetry((token) => saveStudyAnalysis(
            studyContext.study.id,
            result,
            pdfFiles.map((f) => f.name),
            token,
          ));
          setStudySuccessMessage(
            studyContext.mode === "existing"
              ? `${saved.added_reports} new reports added to ${saved.study_name}. Dashboard updated with new trends.`
              : `New study ${saved.study_name} created for ${studyContext.profile.full_name}.`,
          );
          setResultStudyReportCount(saved.total_reports);
        } else {
          await runWithTokenRetry((token) => saveAnalysis(result, pdfFiles.map((f) => f.name), token));
          setStudySuccessMessage(null);
          setResultStudyReportCount(null);
        }
        setSaveStatus("saved");
        loadDashboardData();
      } catch {
        setSaveStatus("error");
      }

      setAnalyzeStep("idle");
    } catch (err) {
      setAnalyzeStep("error");
      setStageStates((prev) => ({
        validating: prev.validating,
        uploading: prev.uploading,
        processing: prev.processing === "complete" ? "complete" : "error",
        saving: prev.saving,
      }));
      setErrorMessage(err instanceof Error ? err.message : "Analysis failed.");
    }
    finally {
      setIsAnalyzing(false);
      setAnalysisStartedAt(null);
    }
  }

  async function loadHistoryItem(id: number) {
    try {
      const result = await runWithTokenRetry((token) => fetchReportById(id, token));
      setStudyContext(null);
      setStudySuccessMessage(null);
      setResultStudyReportCount(null);
      setAnalysis(result);
      setActiveAnalysisId(`history-${id}`);
      resetChatState();
      setSelectedBodySystem("all");
      setSelectedCategory("all");
      setSelectedTest("");
      setView("result");
    } catch (err) { setErrorMessage(err instanceof Error ? err.message : "Failed to load."); }
  }

  async function openStudyCombinedAnalysis(studyId: string, studyName: string, reportCount: number) {
    setErrorMessage(null);
    setViewLoadingStudyId(studyId);
    try {
      const result = await runWithTokenRetry((token) => fetchStudyCombinedReport(studyId, token));
      setStudyContext(null);
      setStudySuccessMessage(`Loaded combined analysis for ${studyName}.`);
      setResultStudyReportCount(reportCount);
      setAnalysis(result);
      setActiveAnalysisId(`study-${studyId}`);
      resetChatState();
      setSelectedBodySystem("all");
      setSelectedCategory("all");
      setSelectedTest("");
      setView("result");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to open study analysis.");
    } finally {
      setViewLoadingStudyId(null);
    }
  }

  async function sendAssistantQuestion(rawQuestion: string) {
    if (!analysis) return;
    const cleanQuestion = rawQuestion.trim();
    if (!cleanQuestion) return;

    setChatError(null);
    setChatQuestion("");
    setShowAssistantPrompts(false);

    const userMessage: ChatMessage = {
      id: createChatMessageId("user"),
      role: "user",
      content: cleanQuestion,
      createdAt: Date.now(),
    };
    const loadingMessage: ChatMessage = {
      id: createChatMessageId("assistant-loading"),
      role: "assistant",
      content: "",
      createdAt: Date.now(),
      isLoading: true,
    };

    const existingThread = chatHistory.filter(
      (message) => !message.isLoading && (message.role === "user" || message.role === "assistant"),
    );
    setChatHistory([...existingThread, userMessage, loadingMessage]);

    const startedAt = Date.now();
    try {
      const resp = await runWithTokenRetry((token) => sendChatMessage(
        {
          analysisId: activeAnalysisId ?? `analysis-${analysis.patient_info.patient_id || "current"}`,
          sessionId: chatSessionId,
          reportContext: {
            patientInfo: analysis.patient_info,
            totalRecords: analysis.total_records,
            reportsIncluded: resultStudyReportCount ?? analysis.reports_with_data ?? null,
            sourceFileNames: analysis.combined_report_file_names ?? [],
            records: analysis.records,
          },
          history: toApiChatHistory(existingThread).slice(-20),
          message: cleanQuestion,
        },
        token,
      ));

      const elapsed = Date.now() - startedAt;
      if (elapsed < MIN_CHAT_LOADING_MS) {
        await sleep(MIN_CHAT_LOADING_MS - elapsed);
      }

      setChatHistory((prev) => [
        ...prev.filter((message) => !message.isLoading),
        {
          id: createChatMessageId("assistant"),
          role: "assistant",
          content: resp.answer,
          createdAt: Date.now(),
        },
      ]);
    } catch (err) {
      const elapsed = Date.now() - startedAt;
      if (elapsed < MIN_CHAT_LOADING_MS) {
        await sleep(MIN_CHAT_LOADING_MS - elapsed);
      }

      const msg = err instanceof Error ? err.message : "Unknown chat error.";
      const normalizedMsg = msg.replace(/^error:\s*/i, "").replace(/\.+\s*$/, "");
      setChatError(msg);
      setChatHistory((prev) => [
        ...prev.filter((message) => !message.isLoading),
        {
          id: createChatMessageId("system"),
          role: "system",
          content: `Error: ${normalizedMsg}. Please try again.`,
          createdAt: Date.now(),
        },
      ]);
    }
  }

  async function submitChatQuestion() {
    if (!chatQuestion.trim() || !analysis) return;
    await sendAssistantQuestion(chatQuestion);
  }

  function handleQuickQuestion(q: string) {
    startChatTransition(() => {
      void sendAssistantQuestion(q);
    });
  }

  async function handleDownloadReport() {
    if (!analysis) return;
    setIsExporting("pdf");
    try {
      const blob = generateClinicalPdfReport(analysis);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url;
      a.download = `health-report-${analysis.patient_info.name || "patient"}.pdf`; a.click();
      URL.revokeObjectURL(url);
    } catch { /* graceful */ } finally { setIsExporting(null); }
  }

  async function handleExportExcel() {
    if (!analysis) return;
    setIsExporting("excel");
    try {
      const blob = await runWithTokenRetry((token) => exportExcel(analysis.records, analysis.patient_info, token));
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url;
      a.download = `medical-data-${analysis.patient_info.name || "patient"}.xlsx`; a.click();
      URL.revokeObjectURL(url);
    } catch { /* graceful */ } finally { setIsExporting(null); }
  }

  const records = analysis?.records ?? [];
  const categoryOf = (record: AnalysisResponse["records"][number]) => canonicalizeCategory(record.Test_Category);
  const allBodySystems = CANONICAL_CATEGORIES.filter((category) => records.some((record) => categoryOf(record) === category));
  const filteredBySystem = selectedBodySystem === "all" ? records : records.filter((record) => categoryOf(record) === selectedBodySystem);
  const allCategories = CANONICAL_CATEGORIES.filter((category) => filteredBySystem.some((record) => categoryOf(record) === category));
  const filteredByCategory = selectedCategory === "all" ? filteredBySystem : filteredBySystem.filter((record) => categoryOf(record) === selectedCategory);
  const allTests = Array.from(new Set(filteredByCategory.map((r) => r.Test_Name ?? "").filter(Boolean))).sort();
  const selectedTestData = selectedTest ? filteredByCategory.filter((r) => r.Test_Name === selectedTest) : [];
  const sourceReportCount = new Set(
    records
      .map((record) => (record.Source_Filename ?? "").trim())
      .filter((name) => name.length > 0),
  ).size;
  const normalCount = records.filter((record) => {
    const normalized = record.Status?.toLowerCase();
    return normalized === "normal" || normalized === "negative";
  }).length;
  const concerningCount = records.filter((record) => isConcerningStatus(record.Status)).length;
  const trendByDateMap = new Map<string, { total: number; concerning: number; good: number }>();
  for (const record of records) {
    const dateKey = record.Test_Date ?? "Unknown";
    if (!trendByDateMap.has(dateKey)) {
      trendByDateMap.set(dateKey, { total: 0, concerning: 0, good: 0 });
    }
    const bucket = trendByDateMap.get(dateKey)!;
    bucket.total += 1;
    if (isConcerningStatus(record.Status)) bucket.concerning += 1;
    if ((record.Status ?? "").toLowerCase() === "normal" || (record.Status ?? "").toLowerCase() === "negative") {
      bucket.good += 1;
    }
  }
  const trendByDate = Array.from(trendByDateMap.entries())
    .map(([date, values]) => ({ date, ...values }))
    .sort((a, b) => parseMedicalDate(a.date) - parseMedicalDate(b.date));
  const reportsIncludedCount = resultStudyReportCount ?? (sourceReportCount > 0 ? sourceReportCount : trendByDate.length);
  const combinedReportFileNames = analysis?.combined_report_file_names ?? [];
  const reportTrendRows = combinedReportFileNames.length > 0
    ? combinedReportFileNames.map((fileName, index) => {
        const normalizedFileName = normalizeSourceFileName(fileName);
        const matchedRows = records.filter(
          (record) => normalizeSourceFileName(record.Source_Filename) === normalizedFileName,
        );
        const concerning = matchedRows.filter((row) => isConcerningStatus(row.Status)).length;
        const good = matchedRows.filter((row) => {
          const normalized = row.Status?.toLowerCase();
          return normalized === "normal" || normalized === "negative";
        }).length;
        return {
          id: `${fileName}-${index}`,
          label: fileName,
          total: matchedRows.length,
          concerning,
          good,
        };
      })
    : trendByDate.map((point) => ({
        id: point.date,
        label: point.date,
        total: point.total,
        concerning: point.concerning,
        good: point.good,
      }));
  const reportsWithDataCount = analysis?.reports_with_data ?? sourceReportCount;

  useEffect(() => {
    setSelectedCategory("all");
    setSelectedTest("");
  }, [selectedBodySystem]);

  useEffect(() => {
    setSelectedTest("");
  }, [selectedCategory]);

  useEffect(() => {
    if (!isAnalyzing || analysisStartedAt === null) return;
    const timer = window.setInterval(() => {
      setAnalysisElapsedSeconds(Math.floor((Date.now() - analysisStartedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [isAnalyzing, analysisStartedAt]);

  const analyzeSteps = [
    { id: "validating", label: "Validating files", detail: "Checking selected PDFs and upload mode" },
    { id: "uploading", label: "Uploading reports", detail: "Sending files securely to backend" },
    { id: "processing", label: "Running AI extraction", detail: "Parsing PDFs and building structured data" },
    { id: "saving", label: "Saving to your history", detail: "Storing the analysis in your account" },
  ] as const;
  const elapsedMinutes = Math.floor(analysisElapsedSeconds / 60);
  const elapsedSeconds = analysisElapsedSeconds % 60;
  const etaMinutes = estimatedRemainingSeconds !== null ? Math.floor(estimatedRemainingSeconds / 60) : 0;
  const etaSeconds = estimatedRemainingSeconds !== null ? estimatedRemainingSeconds % 60 : 0;
  const overallProgressPercent = fileProgress.length > 0
    ? Math.round((processedFilesCount / fileProgress.length) * 100)
    : 0;
  const assistantPrompts = [
    "Which trends worsened over time?",
    "What are the top risk findings right now?",
    "Summarize the latest report changes.",
    "What should I discuss with my doctor first?",
  ];

  if (loading || !user) {
    return <main className="auth-shell"><div className="auth-loading">Loading…</div></main>;
  }

  return (
    <div className="dashboard-root">
      <NavBar user={user} onLogout={async () => { await logout(); router.replace("/login"); }} onHome={() => setView("home")} />

      {/* HOME */}
      {view === "home" && (
        <main className="home-shell">
          <section className="dashboard-header">
            <div>
              <h1>Your Health Reports</h1>
              <p>Track, analyze, and revisit your medical history.</p>
            </div>
            <button className="primary-button" onClick={() => { void openStudyFlow(); }} type="button">
              {(dashboardSummary?.total_reports ?? 0) > 0 ? "Analyze New Report" : "Upload & Analyze"}
            </button>
          </section>

          <section className="stats-row" aria-label="Dashboard summary">
            <article className="stat-card">
              <span>Total Reports</span>
              <strong>{dashboardSummary?.total_reports ?? 0}</strong>
            </article>
            <article className="stat-card">
              <span>Health Alerts</span>
              <strong>{dashboardSummary?.total_alerts ?? 0}</strong>
            </article>
            <article className="stat-card">
              <span>Profiles Tracked</span>
              <strong>{dashboardSummary?.profiles_tracked ?? 0}</strong>
            </article>
          </section>

          {errorMessage && <p className="status-text error-text">{errorMessage}</p>}

          {historyLoading && <p className="muted-copy center-text">Loading your reports…</p>}

          {(dashboardSummary?.profiles?.length ?? 0) > 0 && (
            <section className="profile-groups-section">
              {dashboardSummary?.profiles.map((profile) => (
                <div className="profile-group-block" key={profile.profile_id}>
                  <div className="profile-group-head">
                    <h2>{profile.full_name} - {relationshipLabel(profile.relationship)}</h2>
                  </div>

                  <div className="study-cards-grid">
                    {profile.studies.map((study) => (
                      <article className="study-dashboard-card" key={study.id}>
                        <div className="study-card-top">
                          <h3>{study.name}</h3>
                          <span className={`study-alert-dot ${study.has_alerts ? "warn" : "ok"}`} />
                        </div>
                        <p>{study.report_count} reports • {fmtRange(study.range_start, study.range_end)}</p>
                        {study.consistent_lab_name && (
                          <p className="study-lab-chip">Lab: {study.consistent_lab_name}</p>
                        )}
                        <div className="study-card-actions">
                          <button
                            className="secondary-button"
                            type="button"
                            disabled={viewLoadingStudyId === study.id}
                            onClick={() => { void openStudyCombinedAnalysis(study.id, study.name, study.report_count); }}
                          >
                            {viewLoadingStudyId === study.id ? "Opening..." : "View"}
                          </button>
                          <button
                            className="primary-button compact"
                            type="button"
                            onClick={() => startAddReportsShortcut(profile, study)}
                          >
                            Add Reports
                          </button>
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              ))}
            </section>
          )}

          {!historyLoading && (dashboardSummary?.profiles?.length ?? 0) === 0 && (
            <div className="empty-state">
              <h3>No studies yet</h3>
              <p>Create your first profile study flow by clicking Analyze New Report.</p>
              <div className="feature-row">
                <div className="feature-pill">Profiles</div>
                <div className="feature-pill">Study timelines</div>
                <div className="feature-pill">AI trend tracking</div>
                <div className="feature-pill">Alerts monitoring</div>
              </div>
            </div>
          )}
        </main>
      )}

      {/* ANALYZE */}
      {view === "analyze" && (
        <main className="analyze-shell">
          <div className="analyze-card">
            <div className="analyze-header">
              <button className="back-btn" onClick={() => setView("home")} type="button">Back</button>
              <h2>Upload Medical Reports</h2>
              <p>Select your PDF reports and we'll extract and analyze all the test data.</p>
              {studyContext && (
                <div className="study-upload-summary">
                  {studyContext.mode === "existing"
                    ? `Adding to: ${studyContext.study.name} • ${studyContext.study.report_count} existing reports`
                    : `New Study: ${studyContext.study.name} • For: ${studyContext.profile.full_name}`}
                </div>
              )}
            </div>

            {!studyContext && (
              <div className="mode-tabs">
                <button className={`mode-tab ${uploadMode === "new" ? "active" : ""}`} onClick={() => setUploadMode("new")} type="button">Upload New Reports</button>
                <button className={`mode-tab ${uploadMode === "append" ? "active" : ""}`} onClick={() => setUploadMode("append")} type="button">Add to Existing Data</button>
              </div>
            )}

            <form className="upload-form" onSubmit={(e) => { e.preventDefault(); void submitAnalysis(); }}>
              <label className="upload-zone">
                <input accept="application/pdf" multiple type="file" onChange={(e) => setPdfFiles(Array.from(e.target.files ?? []))} />
                <div className="upload-zone-inner">
                  <span className="upload-icon">Upload</span>
                  <strong>Drag &amp; drop or click to browse</strong>
                  <span>PDF files only • Up to 200MB each</span>
                </div>
              </label>

              {pdfFiles.length > 0 && (
                <div className="file-list">
                  {pdfFiles.map((f) => (
                    <div className="file-row" key={`${f.name}-${f.size}`}>
                      <span className="file-name">{f.name}</span>
                      <span className="file-size">{(f.size / 1024 / 1024).toFixed(1)} MB</span>
                      <button className="file-remove" type="button" onClick={() => setPdfFiles((prev) => prev.filter((x) => x !== f))}>×</button>
                    </div>
                  ))}
                </div>
              )}

              {!studyContext && uploadMode === "append" && (
                <label className="field-block">
                  <span>Upload Previous Excel / CSV (optional)</span>
                  <input accept=".xlsx,.xls,.csv" type="file" onChange={(e) => setExistingDataFile(e.target.files?.[0] ?? null)} />
                  {existingDataFile && <span className="file-chip accent">{existingDataFile.name}</span>}
                </label>
              )}

              {(isAnalyzing || analyzeStep === "error") && (
                <section className={`analysis-progress-panel ${analyzeStep === "error" ? "error" : ""}`}>
                  <div className="analysis-progress-head">
                    <div>
                      <span className="analysis-live-chip">Live Stream</span>
                      <h3>{analyzeStep === "error" ? "Analysis interrupted" : "Processing reports"}</h3>
                      <p className="analysis-progress-subcopy">
                        Real-time progress is streamed from the backend analysis pipeline.
                      </p>
                    </div>
                    <span className="analysis-clock-badge">{elapsedMinutes}m {elapsedSeconds}s elapsed</span>
                  </div>

                  <div className="analysis-progress-layout">
                    <ol className="analysis-phase-list">
                      {analyzeSteps.map((step, index) => {
                        const state = stageStates[step.id];
                        const stateLabel = state === "complete"
                          ? "Complete"
                          : state === "active"
                            ? "In Progress"
                            : state === "error"
                              ? "Error"
                              : "Pending";
                        return (
                          <li
                            key={step.id}
                            className={`analysis-phase-item ${state}`.trim()}
                          >
                            <span className="analysis-phase-node" aria-hidden="true">
                              {state === "complete" ? "OK" : state === "active" ? "..." : index + 1}
                            </span>
                            <div className="analysis-phase-copy">
                              <strong>{step.label}</strong>
                              <p>{step.detail}</p>
                            </div>
                            <span className={`analysis-phase-state ${state}`}>{stateLabel}</span>
                          </li>
                        );
                      })}
                    </ol>

                    <div className="analysis-progress-main">
                      <div className="analysis-overall-progress">
                        <div className="analysis-overall-meta">
                          <strong>{processedFilesCount} of {fileProgress.length} files processed</strong>
                          <span>{overallProgressPercent}% complete</span>
                        </div>
                        <div className="analysis-overall-track" aria-hidden="true">
                          <div
                            className="analysis-overall-fill"
                            style={{ width: `${overallProgressPercent}%` }}
                          />
                        </div>
                        <span className="analysis-eta-copy">
                          {estimatedRemainingSeconds !== null
                            ? `Estimated remaining time: ${etaMinutes}m ${etaSeconds}s`
                            : "Calculating estimated completion time..."}
                        </span>
                      </div>

                      <div className="analysis-file-list" role="list">
                        {fileProgress.map((item) => (
                          <article className="analysis-file-item" role="listitem" key={item.file}>
                            <div className="analysis-file-head">
                              <strong>{item.file}</strong>
                              <span className={`analysis-file-status ${item.status}`}>
                                {{
                                  done: "Completed",
                                  failed: "Failed",
                                  queued: "Queued",
                                  extracting: "Extracting",
                                  parsing: "Parsing",
                                }[item.status]}
                              </span>
                            </div>
                            <div className="analysis-file-track" aria-hidden="true">
                              <div className={`analysis-file-fill ${item.status}`} style={{ width: `${item.percent}%` }} />
                            </div>
                            <div className="analysis-file-meta">
                              <span>{item.percent}%</span>
                              {item.error && <span className="status-text error-text">{item.error}</span>}
                            </div>
                          </article>
                        ))}
                      </div>
                    </div>
                  </div>

                  {stageStates.processing === "active" && currentParsingFile && (
                    <p className="analysis-current-file">Currently parsing: {currentParsingFile}</p>
                  )}

                  <p className="analysis-progress-note">
                    Large report sets can take a few minutes. Keep this tab open until processing completes.
                  </p>
                </section>
              )}

              {errorMessage && <p className="status-text error-text">{errorMessage}</p>}

              <button className="primary-button submit-button" disabled={isAnalyzing} type="submit">
                {isAnalyzing ? "Analyzing…" : "Analyze Reports"}
              </button>
            </form>
          </div>
        </main>
      )}

      {/* RESULT */}
      {view === "result" && analysis && (
        <main className="result-shell">
          <div className="result-topbar">
            <button className="back-btn" onClick={() => setView("home")} type="button">Dashboard</button>
            <button className="secondary-button" onClick={() => { void openStudyFlow(); }} type="button">Analyze More Reports</button>
            {saveStatus === "saved" && <span className="save-badge">✅ Saved</span>}
            {saveStatus === "saving" && <span className="save-badge muted">Saving…</span>}
            {saveStatus === "error" && <span className="save-badge error-badge">Not saved</span>}
          </div>

          <div className="success-banner">
            {studySuccessMessage || `Successfully analyzed ${analysis.total_records} test records.`}
          </div>

          {/* Patient profile */}
          <section className="result-section">
            <h2>Patient Profile</h2>
            <div className="patient-grid">
              {[
                { label: "NAME", icon: "", value: analysis.patient_info.name },
                { label: "AGE", icon: "", value: analysis.patient_info.age },
                { label: "GENDER", icon: "", value: analysis.patient_info.gender },
                { label: "LAB", icon: "", value: analysis.patient_info.lab_name },
              ].map((item) => (
                <div className="patient-card" key={item.label}>
                  <span>{item.icon ? `${item.icon} ` : ""}{item.label}</span>
                  <strong>{item.value || "N/A"}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="result-section">
            <h2>Trend Snapshot</h2>
            <p className="muted-copy">Quick view of stable vs concerning patterns across included reports and report dates.</p>

            <div className="trend-summary-grid">
              <article className="trend-stat good">
                <span>Stable Results</span>
                <strong>{normalCount}</strong>
              </article>
              <article className="trend-stat warn">
                <span>Concerning Results</span>
                <strong>{concerningCount}</strong>
              </article>
              <article className="trend-stat muted">
                <span>Reports Included</span>
                <strong>{reportsIncludedCount}</strong>
              </article>
            </div>

            <p className="muted-copy">Unique report dates in trend view: {trendByDate.length}</p>
            <p className="muted-copy">Reports with parsed test rows: {reportsWithDataCount}/{reportsIncludedCount}</p>

            <div className="trend-mini-chart">
              {reportTrendRows.map((point) => {
                const concerningPct = point.total > 0 ? Math.round((point.concerning / point.total) * 100) : 0;
                const goodPct = point.total > 0 ? Math.round((point.good / point.total) * 100) : 0;
                return (
                  <div className="trend-mini-row" key={point.id}>
                    <span className="trend-mini-date">{point.label}</span>
                    <div className="trend-mini-bars" aria-label={`Concerning ${concerningPct}% and stable ${goodPct}% for ${point.label}`}>
                      <div className="trend-mini-bar bad" style={{ width: `${point.total > 0 ? Math.max(4, concerningPct) : 0}%` }} />
                      <div className="trend-mini-bar good" style={{ width: `${point.total > 0 ? Math.max(4, goodPct) : 0}%` }} />
                    </div>
                    <span className="trend-mini-meta">{point.concerning}/{point.total} alerts</span>
                  </div>
                );
              })}
            </div>
          </section>

          <ClinicalChatPanel
            records={analysis.records}
          />

          <section className="result-section assistant-workbench">
            <div className="assistant-workbench-head">
              <div>
                <span className="assistant-kicker">Clinical Assistant</span>
                <h2>Ask Follow-Up Questions</h2>
                <p className="muted-copy">Get report-aware answers about abnormalities, trends, and clinical priorities. Markdown is supported in responses.</p>
              </div>
            </div>

            <div className="assistant-workbench-grid">
              <aside className="assistant-context-panel">
                <h3>Prompt Ideas</h3>
                <div className={`assistant-quick-grid ${showAssistantPrompts ? "" : "disabled"}`}>
                  {assistantPrompts.map((prompt) => (
                    <button
                      key={prompt}
                      className="assistant-quick-btn"
                      type="button"
                      disabled={isChatPending || !showAssistantPrompts}
                      onClick={() => handleQuickQuestion(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
                <div className="assistant-note">
                  <strong>Tip</strong>
                  <p>Ask about date-wise changes and alert clusters for the most useful clinical summaries.</p>
                </div>
                <button
                  type="button"
                  className="assistant-clear-chat-btn"
                  onClick={resetChatState}
                  disabled={isChatPending || (chatHistory.length === 0 && showAssistantPrompts)}
                >
                  Clear chat
                </button>
              </aside>

              <div className="assistant-chat-panel">
                <div className="assistant-chat-history" role="log" aria-live="polite">
                  {chatHistory.length === 0 && (
                    <div className="assistant-chat-empty">
                      Start with a question like "What changed the most in the latest report?"
                    </div>
                  )}

                  {chatHistory.map((turn) => {
                    if (turn.role === "system") {
                      return (
                        <div key={turn.id} className="assistant-system-error">
                          {turn.content}
                        </div>
                      );
                    }

                    const isUser = turn.role === "user";
                    return (
                      <div key={turn.id} className={`assistant-chat-message-row ${isUser ? "user" : "assistant"}`}>
                        <div className={`assistant-chat-avatar ${isUser ? "user" : "assistant"}`}>
                          {isUser ? userInitials : "AI"}
                        </div>
                        <div className={`assistant-chat-bubble-wrap ${isUser ? "user" : "assistant"}`}>
                          <div className={`assistant-chat-bubble ${isUser ? "user" : "assistant"}${turn.isLoading ? " loading" : ""}`}>
                            {turn.isLoading ? (
                              <div className="chat-typing-dots" aria-label="Assistant is thinking" role="status">
                                <span />
                                <span />
                                <span />
                              </div>
                            ) : isUser ? (
                              <p className="assistant-chat-user-text">{turn.content}</p>
                            ) : (
                              <div className="assistant-chat-markdown">
                                <ReactMarkdown
                                  remarkPlugins={[remarkGfm]}
                                  components={{
                                    p: ({ children }) => (
                                      <p style={{ margin: "0 0 10px 0", lineHeight: 1.7, color: "#d4b483", fontSize: 14 }}>{children}</p>
                                    ),
                                    strong: ({ children }) => (
                                      <strong style={{ color: "#fef3c7", fontWeight: 600 }}>
                                        {children}
                                      </strong>
                                    ),
                                    em: ({ children }) => (
                                      <em style={{ color: "#fbbf24", fontStyle: "italic" }}>
                                        {children}
                                      </em>
                                    ),
                                    ul: ({ children }) => (
                                      <ul style={{ margin: "8px 0", paddingLeft: 20, color: "#d4b483" }}>{children}</ul>
                                    ),
                                    ol: ({ children }) => (
                                      <ol style={{ margin: "8px 0", paddingLeft: 20, color: "#d4b483" }}>{children}</ol>
                                    ),
                                    li: ({ children }) => (
                                      <li style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.6 }}>{children}</li>
                                    ),
                                    h1: ({ children }) => (
                                      <h1 style={{ fontSize: 16, fontWeight: 600, color: "#fef3c7", margin: "12px 0 6px" }}>{children}</h1>
                                    ),
                                    h2: ({ children }) => (
                                      <h2 style={{ fontSize: 15, fontWeight: 600, color: "#fef3c7", margin: "10px 0 6px" }}>{children}</h2>
                                    ),
                                    h3: ({ children }) => (
                                      <h3 style={{ fontSize: 14, fontWeight: 600, color: "#fbbf24", margin: "8px 0 4px" }}>{children}</h3>
                                    ),
                                    code: ({ children, className }) => {
                                      const isInline = !className;
                                      return isInline ? (
                                        <code style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 12, color: "#fbbf24", background: "#2d1f08", padding: "1px 5px", borderRadius: 4 }}>{children}</code>
                                      ) : (
                                        <pre style={{ background: "#1c1917", border: "1px solid #292524", borderRadius: 8, padding: "10px 14px", overflowX: "auto", margin: "8px 0" }}>
                                          <code style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 12, color: "#d4b483" }}>{children}</code>
                                        </pre>
                                      );
                                    },
                                    blockquote: ({ children }) => (
                                      <blockquote style={{ borderLeft: "3px solid #d97706", marginLeft: 0, paddingLeft: 12, color: "#a8a29e", fontStyle: "italic" }}>{children}</blockquote>
                                    ),
                                    table: ({ children }) => (
                                      <div style={{ overflowX: "auto", margin: "8px 0" }}>
                                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>{children}</table>
                                      </div>
                                    ),
                                    th: ({ children }) => (
                                      <th style={{ padding: "6px 10px", textAlign: "left", borderBottom: "1px solid #44403c", color: "#fbbf24", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.06em" }}>{children}</th>
                                    ),
                                    td: ({ children }) => (
                                      <td style={{ padding: "6px 10px", borderBottom: "1px solid #292524", color: "#d4b483" }}>{children}</td>
                                    ),
                                  }}
                                >
                                  {turn.content}
                                </ReactMarkdown>
                              </div>
                            )}
                          </div>
                          {!turn.isLoading && (
                            <div className={`assistant-chat-timestamp ${isUser ? "user" : "assistant"}`}>
                              {formatChatTimestamp(turn.createdAt)}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                  <div ref={assistantMessagesEndRef} />
                </div>

                <form
                  className="assistant-input-row"
                  onSubmit={(e) => {
                    e.preventDefault();
                    startChatTransition(() => {
                      void submitChatQuestion();
                    });
                  }}
                >
                  <input
                    className="text-input"
                    type="text"
                    placeholder="Ask a clinical question about trends, abnormalities, or next actions..."
                    disabled={isChatPending}
                    value={chatQuestion}
                    onChange={(e) => setChatQuestion(e.target.value)}
                  />
                  <button className="primary-button" type="submit" disabled={isChatPending || !chatQuestion.trim()}>
                    Send
                  </button>
                </form>
              </div>
            </div>
          </section>

          <section className="result-section">
            <h2>Test Result Visualizations</h2>
            <div className="viz-layout">
              <div className="viz-controls selector-panel">
                <label className="field-block selector-field">
                  <span className="selector-caption">Body System</span>
                  <select className="select-input" value={selectedBodySystem} onChange={(e) => setSelectedBodySystem(e.target.value)}>
                    <option value="all">All Systems</option>
                    {allBodySystems.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </label>
                <label className="field-block selector-field">
                  <span className="selector-caption">Category</span>
                  <select className="select-input" value={selectedCategory} onChange={(e) => setSelectedCategory(e.target.value)}>
                    <option value="all">All Categories</option>
                    {allCategories.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
                <label className="field-block selector-field">
                  <span className="selector-caption">Test Name</span>
                  <select className="select-input" value={selectedTest} onChange={(e) => setSelectedTest(e.target.value)}>
                    <option value="">-- Select a test --</option>
                    {allTests.map((t) => <option key={t} value={t}>{t}</option>)}
                  </select>
                </label>
              </div>
              <div className="viz-chart-area">
                {selectedTest && selectedTestData.length > 0 ? (
                  <TrendChart records={selectedTestData} testName={selectedTest} />
                ) : (
                  <div className="viz-empty">
                    <h3>Select a Test to Visualize</h3>
                    <p>Choose a test from the dropdown on the left to see charts and trends.</p>
                  </div>
                )}
              </div>
            </div>
          </section>

          <OrganizedDataTree records={analysis.records} />

          {analysis.health_summary.concerns.length > 0 && (
            <AlertsByCategory concerns={analysis.health_summary.concerns} />
          )}

          {/* Downloads */}
          <section className="result-section download-section">
            <h2>Download Health Report</h2>
            <p className="muted-copy">Download a comprehensive report with your test results and AI-powered insights.</p>
            <div className="download-buttons">
              <button className="primary-button" disabled={isExporting !== null} onClick={handleDownloadReport} type="button">
                {isExporting === "pdf" ? "Generating…" : "Download Health Report (PDF)"}
              </button>
              <button className="secondary-button" disabled={isExporting !== null} onClick={handleExportExcel} type="button">
                {isExporting === "excel" ? "Generating…" : "Download Excel with Charts"}
              </button>
            </div>
          </section>
        </main>
      )}

      <StudyFlowModal
        open={studyFlowOpen}
        userName={user.displayName ?? user.email ?? "User"}
        profiles={profiles}
        selectedProfile={selectedProfile}
        studies={profileStudies}
        selectedAction={selectedAction}
        selectedStudy={selectedStudy}
        step={studyFlowStep}
        loading={flowLoading}
        formError={flowError}
        onClose={() => {
          setStudyFlowOpen(false);
          setFlowError(null);
        }}
        onSelectProfile={(profile) => { void handleSelectProfile(profile); }}
        onSelectAction={handleSelectAction}
        onSelectStudy={(study) => {
          setSelectedStudy(study);
          setStudyFlowStep("confirm-add");
        }}
        onConfirmAdd={handleConfirmAddToStudy}
        onBack={handleStudyFlowBack}
        newStudyName={newStudyName}
        newStudyDescription={newStudyDescription}
        onChangeNewStudyName={setNewStudyName}
        onChangeNewStudyDescription={setNewStudyDescription}
        onCreateStudy={() => { void handleCreateStudyAndContinue(); }}
        newMemberName={newMemberName}
        newMemberRelationship={newMemberRelationship}
        newMemberOtherRelationship={newMemberOtherRelationship}
        newMemberDob={newMemberDob}
        onChangeMemberName={setNewMemberName}
        onChangeMemberRelationship={setNewMemberRelationship}
        onChangeMemberOtherRelationship={setNewMemberOtherRelationship}
        onChangeMemberDob={setNewMemberDob}
        onCreateMember={() => { void handleCreateFamilyMemberAndContinue(); }}
        onStartNewMember={() => {
          setFlowError(null);
          setStudyFlowStep("new-member");
        }}
      />
    </div>
  );
}