"use client";

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAuth } from "@/lib/auth-context";
import {
  type AnalysisResponse,
  type AnalysisHistoryItem,
  type DashboardSummary,
  type ProfileItem,
  type ReportJob,
  type StudySummary,
  createProfile,
  createStudy,
  fetchProfiles,
  sendChatMessage,
  fetchStudiesForProfile,
  fetchStudiesDashboard,
  fetchStudyCombinedReport,
  fetchReportHistory,
  fetchReportById,
  startAnalysisJob,
  isAbortError,
  waitForJob,
  exportExcel,
} from "@/lib/api";
import AlertsByCategory from "./AlertsByCategory";
import YourDataPanel from "./YourDataPanel";
import OrganizedDataTree from "./OrganizedDataTree";
import { CANONICAL_CATEGORIES } from "@/lib/categoryMap";
import { parseMedicalDate } from "@/lib/medicalDate";

const TrendChart = dynamic(() => import("./TrendChart"), { ssr: false });
const ClinicalChatPanel = dynamic(() => import("./ClinicalChatPanel"), { ssr: false });


import NavBar from "./NavBar";
import StudyFlowModal from "./StudyFlowModal";
import {
  buildUserInitials,
  chatTarget,
  createChatMessageId,
  dashboardUrl,
  formatChatTimestamp,
  formatFileSize,
  fmtRange,
  isConcerningStatus,
  normalizeSourceFileName,
  relationshipLabel,
  shouldRetryWithFreshToken,
  sleep,
  splitAssistantResponseForSources,
  toApiChatHistory,
} from "./helpers";
import { CITATION_COMPONENTS, MARKDOWN_COMPONENTS } from "./markdown";
import {
  ACTIVE_JOB_KEY,
  MIN_CHAT_LOADING_MS,
  type AnalyzeContext,
  type AnalyzeStep,
  type ChatMessage,
  type FileProgressItem,
  type StageState,
  type StudyAction,
} from "./types";

export default function DashboardPage() {
  const { user, loading, getToken, logout } = useAuth();
  const router = useRouter();

  const hardRedirectToLogin = useCallback(() => {
    if (typeof window !== "undefined") {
      window.localStorage.clear();
      window.sessionStorage.clear();
      window.location.href = "/login";
      return;
    }
    router.replace("/login");
  }, [router]);

  useEffect(() => {
    if (!loading && !user) {
      hardRedirectToLogin();
    }
  }, [user, loading, hardRedirectToLogin]);

  const [view, setView] = useState<"home" | "analyze" | "result">("home");
  const viewRef = useRef(view);
  viewRef.current = view;
  const jobAbortRef = useRef<AbortController | null>(null);
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
  const chatSessionRef = useRef(chatSessionId);
  useEffect(() => {
    chatSessionRef.current = chatSessionId;
  }, [chatSessionId]);
  const [activeAnalysisId, setActiveAnalysisId] = useState<string | null>(null);
  const [showAssistantPrompts, setShowAssistantPrompts] = useState(true);
  const [chatError, setChatError] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const sendingRef = useRef(false);
  const [selectedBodySystem, setSelectedBodySystem] = useState<string>("all");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedTest, setSelectedTest] = useState<string>("");
  const [isExporting, setIsExporting] = useState<"pdf" | "excel" | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [uploadConsent, setUploadConsent] = useState(false);
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
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
      hardRedirectToLogin();
    }
    throw new Error("Your session is no longer valid. Please sign in again.");
  }, [hardRedirectToLogin, logout]);

  const handleSignOut = useCallback(async () => {
    try {
      await logout();
    } finally {
      hardRedirectToLogin();
    }
  }, [hardRedirectToLogin, logout]);

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
      const [items, summary] = await runWithTokenRetry((_token) => Promise.all([
        fetchReportHistory(),
        fetchStudiesDashboard(),
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
      let rows = await runWithTokenRetry((_token) => fetchProfiles());
      if (!rows.some((item) => item.relationship.trim().toLowerCase() === "self")) {
        const fallbackName = (user?.displayName || user?.email?.split("@")[0] || "My Profile").trim() || "My Profile";
        try {
          const createdSelf = await runWithTokenRetry((_token) => createProfile({
            full_name: fallbackName,
            relationship: "self",
          }));
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
      const studies = await runWithTokenRetry((_token) => fetchStudiesForProfile(profile.id));
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
      const created = await runWithTokenRetry((_token) => createStudy(
        {
          profile_id: selectedProfile.id,
          name: newStudyName.trim(),
          description: newStudyDescription.trim() || undefined,
        },
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
      const created = await runWithTokenRetry((_token) => createProfile(
        {
          full_name: newMemberName.trim(),
          relationship,
          date_of_birth: newMemberDob || undefined,
        },
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

  function applyJobProgress(job: ReportJob) {
    const { stage, files, processed, eta_seconds } = job.progress;
    const stageOrder = ["processing", "saving", "done"];
    const reached = stageOrder.indexOf(stage);
    setStageStates({
      validating: "complete",
      uploading: "complete",
      processing: reached >= 1 ? "complete" : "active",
      saving: reached >= 2 ? "complete" : reached === 1 ? "active" : "pending",
    });
    setAnalyzeStep(stage === "saving" ? "saving" : "processing");
    setFileProgress(Object.entries(files).map(([file, item]) => ({ file, percent: item.percent, status: item.step, error: item.error })));
    setProcessedFilesCount(processed);
    setEstimatedRemainingSeconds(typeof eta_seconds === "number" ? eta_seconds : null);
    const active = Object.entries(files).find(([, item]) => item.step === "extracting" || item.step === "parsing");
    setCurrentParsingFile(active ? active[0] : null);
  }

  /** The server stores naive UTC timestamps; without a zone, the browser would read them as local time. */
  function parseServerTime(value: string): number | null {
    const parsed = Date.parse(/[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`);
    return Number.isNaN(parsed) ? null : parsed;
  }

  /**
   * Follows a job to the end and shows its result. The job id alone is enough, so a refresh resumes here.
   * Only one job is followed at a time; starting another, or leaving the page, stops the previous poll.
   */
  async function followJob(jobId: string, context: AnalyzeContext | null) {
    jobAbortRef.current?.abort();
    const controller = new AbortController();
    jobAbortRef.current = controller;
    window.localStorage.setItem(ACTIVE_JOB_KEY, jobId);
    setView("analyze");
    setIsAnalyzing(true);
    setErrorMessage(null);
    const onProgress = (job: ReportJob) => {
      if (controller.signal.aborted) return;
      setAnalysisStartedAt((started) => started ?? parseServerTime(job.created_at));
      applyJobProgress(job);
    };
    try {
      const job = await runWithTokenRetry((_token) => waitForJob(jobId, onProgress, controller.signal));
      if (job.status !== "done") throw new Error(job.error ?? "Analysis failed.");
      setSaveStatus("saved");
      const result = job.study_id
        ? await runWithTokenRetry((_token) => fetchStudyCombinedReport(job.study_id as string))
        : await runWithTokenRetry((_token) => fetchReportById(job.analysis_id as number));
      if (controller.signal.aborted) return;
      await loadDashboardData();
      // The user may have gone back to the dashboard while this ran; refresh it, but do not pull them away.
      if (viewRef.current !== "analyze") return;
      const warnings = job.progress.warnings ?? [];
      const savedCount = job.source_filenames.length - warnings.length;
      const notes = job.progress.notes ?? [];
      const list = (items: string[]) => items.map((item) => item.replace(/\.+$/, "")).join("; ");
      const skipped = (warnings.length ? ` Skipped ${warnings.length}: ${list(warnings)}.` : "")
        + (notes.length ? ` Checked against the reports: ${list(notes)}.` : "");
      if (context?.mode === "existing") {
        setStudySuccessMessage(`${savedCount} new reports added to ${context.study.name}. Dashboard updated with new trends.${skipped}`);
      } else if (context) {
        setStudySuccessMessage(`New study ${context.study.name} created for ${context.profile.full_name}.${skipped}`);
      } else {
        setStudySuccessMessage(`Analysis saved.${skipped}`);
      }
      setResultStudyReportCount(job.study_id ? result.combined_report_file_names?.length ?? null : null);
      setAnalysis(result);
      setActiveAnalysisId(job.study_id ? `study-${job.study_id}` : `history-${job.analysis_id}`);
      resetChatState();
      setSelectedBodySystem("all");
      setSelectedCategory("all");
      setSelectedTest("");
      setView("result");
      setAnalyzeStep("idle");
    } catch (err) {
      if (isAbortError(err) || controller.signal.aborted) return;
      setAnalyzeStep("error");
      setStageStates((prev) => ({ ...prev, processing: prev.processing === "complete" ? "complete" : "error" }));
      setErrorMessage(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      // An aborted poll is a page leaving or a newer job taking over: the saved id must survive for the resume.
      if (!controller.signal.aborted) {
        window.localStorage.removeItem(ACTIVE_JOB_KEY);
        setIsAnalyzing(false);
        setAnalysisStartedAt(null);
      }
      if (jobAbortRef.current === controller) jobAbortRef.current = null;
    }
  }

  useEffect(() => {
    const pending = user ? window.localStorage.getItem(ACTIVE_JOB_KEY) : null;
    if (pending) void followJob(pending, null);
    // Leaving the page, or StrictMode's double mount in dev, must not leave a second poll running.
    return () => jobAbortRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  // The URL names the current view, so a refresh, Back or Forward lands on the same screen.
  const urlRestoredRef = useRef(false);

  function openFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const studyId = params.get("study");
    const analysisId = Number(params.get("analysis"));
    const opened = studyId ? openStudyCombinedAnalysis(studyId) : analysisId ? loadHistoryItem(analysisId) : null;
    if (opened) {
      // A deleted study or someone else's id: show home, and stop the address pointing at it.
      void opened.then((ok) => { if (!ok) { setView("home"); window.history.replaceState(null, "", "/dashboard"); } });
    } else {
      setView(params.get("view") === "upload" ? "analyze" : "home");
    }
  }

  useEffect(() => {
    if (!user) return;
    if (!urlRestoredRef.current) {
      urlRestoredRef.current = true;
      // An upload still running takes precedence; followJob shows it.
      if (!window.localStorage.getItem(ACTIVE_JOB_KEY)) openFromUrl();
    }
    const onPopState = () => openFromUrl();
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  useEffect(() => {
    if (!urlRestoredRef.current) return; // the address must be read before it is overwritten
    const target = dashboardUrl(view, activeAnalysisId);
    if (window.location.pathname + window.location.search !== target) window.history.pushState(null, "", target);
  }, [view, activeAnalysisId]);

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
    setAnalyzeStep("uploading");
    setStageStates({ validating: "complete", uploading: "active", processing: "pending", saving: "pending" });

    const fd = new FormData();
    for (const f of pdfFiles) fd.append("pdf_files", f);
    if (existingDataFile) fd.append("existing_data", existingDataFile);
    if (studyContext) fd.append("study_id", studyContext.study.id);

    try {
      const job = await runWithTokenRetry((_token) => startAnalysisJob(fd));
      await followJob(job.id, studyContext);
    } catch (err) {
      setAnalyzeStep("error");
      setIsAnalyzing(false);
      setAnalysisStartedAt(null);
      setErrorMessage(err instanceof Error ? err.message : "Analysis failed.");
    }
  }

  async function loadHistoryItem(id: number): Promise<boolean> {
    try {
      const result = await runWithTokenRetry((_token) => fetchReportById(id));
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
      return true;
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to load.");
      return false;
    }
  }

  async function openStudyCombinedAnalysis(studyId: string, studyName?: string, reportCount: number | null = null): Promise<boolean> {
    setErrorMessage(null);
    setViewLoadingStudyId(studyId);
    try {
      const result = await runWithTokenRetry((_token) => fetchStudyCombinedReport(studyId));
      setStudyContext(null);
      setStudySuccessMessage(studyName ? `Loaded combined analysis for ${studyName}.` : null);
      setResultStudyReportCount(reportCount);
      setAnalysis(result);
      setActiveAnalysisId(`study-${studyId}`);
      resetChatState();
      setSelectedBodySystem("all");
      setSelectedCategory("all");
      setSelectedTest("");
      setView("result");
      return true;
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to open study analysis.");
      return false;
    } finally {
      setViewLoadingStudyId(null);
    }
  }

  async function sendAssistantQuestion(rawQuestion: string) {
    if (!analysis) return;
    const cleanQuestion = rawQuestion.trim();
    if (!cleanQuestion) return;

    if (sendingRef.current) return; // keep the typed question; it can be sent once the answer arrives
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

    sendingRef.current = true;
    setIsSending(true);
    const sessionAtSend = chatSessionRef.current;
    const isThreadMessage = (message: ChatMessage) =>
      !message.isLoading && (message.role === "user" || message.role === "assistant");
    const existingThread = chatHistory.filter(isThreadMessage);
    setChatHistory((prev) => [...prev.filter(isThreadMessage), userMessage, loadingMessage]);

    const startedAt = Date.now();
    try {
      const target = chatTarget(activeAnalysisId);
      if (!target) throw new Error("Save this analysis to a study or your history before asking about it.");
      const resp = await runWithTokenRetry((_token) => sendChatMessage({
        ...target,
        sessionId: chatSessionId,
        history: toApiChatHistory(existingThread).slice(-20),
        question: cleanQuestion,
      }));

      const elapsed = Date.now() - startedAt;
      if (elapsed < MIN_CHAT_LOADING_MS) {
        await sleep(MIN_CHAT_LOADING_MS - elapsed);
      }

      if (chatSessionRef.current !== sessionAtSend) return;
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
      if (chatSessionRef.current !== sessionAtSend) return;
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
    } finally {
      sendingRef.current = false;
      setIsSending(false);
    }
  }

  async function submitChatQuestion() {
    if (!chatQuestion.trim() || !analysis) return;
    await sendAssistantQuestion(chatQuestion);
  }

  function handleQuickQuestion(q: string) {
    void sendAssistantQuestion(q);
  }

  async function handleDownloadReport() {
    if (!analysis) return;
    setIsExporting("pdf");
    setExportError(null);
    try {
      const { generateClinicalPdfReport } = await import("@/lib/pdf");
      const blob = generateClinicalPdfReport(analysis);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `health-report-${analysis.patient_info.name || "patient"}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      setExportError("Could not create the PDF. Please try again.");
    } finally { setIsExporting(null); }
  }

  async function handleExportExcel() {
    if (!analysis) return;
    setIsExporting("excel");
    setExportError(null);
    try {
      const blob = await runWithTokenRetry((_token) => exportExcel(analysis.records, analysis.patient_info));
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `medical-data-${analysis.patient_info.name || "patient"}.xlsx`;
      document.body.appendChild(a); // Firefox ignores clicks on detached links
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      setExportError(err instanceof Error ? `Could not create the Excel file: ${err.message}` : "Could not create the Excel file.");
    } finally { setIsExporting(null); }
  }

  const derived = useMemo(() => {
    const records = analysis?.records ?? [];
    const categoryOf = (record: AnalysisResponse["records"][number]) => record.Test_Category ?? "Other";
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
    return { records, allBodySystems, filteredBySystem, allCategories, filteredByCategory, allTests, selectedTestData, sourceReportCount, normalCount, concerningCount, trendByDate, reportsIncludedCount, combinedReportFileNames, reportTrendRows, reportsWithDataCount };
  }, [analysis, selectedBodySystem, selectedCategory, selectedTest, resultStudyReportCount]);
  const {
    records,
    allBodySystems,
    filteredBySystem,
    allCategories,
    filteredByCategory,
    allTests,
    selectedTestData,
    sourceReportCount,
    normalCount,
    concerningCount,
    trendByDate,
    reportsIncludedCount,
    combinedReportFileNames,
    reportTrendRows,
    reportsWithDataCount,
  } = derived;

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
      <NavBar user={user} onLogout={() => { void handleSignOut(); }} onHome={() => setView("home")} />

      {/* HOME */}
      {view === "home" && (
        <main className="home-shell">
          <section className="dashboard-header">
            <div>
              <h1>Your Health Reports</h1>
              <p>Track, analyze, and revisit your medical history.</p>
            </div>
            <button className="primary-button" onClick={() => { void openStudyFlow(); }} type="button">
              Upload &amp; Analyze
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

                  {profile.studies.length === 0 && (
                    <p className="muted-copy">No reports yet. Choose Upload &amp; Analyze to add the first one.</p>
                  )}
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
              <p>Choose Upload &amp; Analyze to add your first report.</p>
              <div className="feature-row">
                <div className="feature-pill">Profiles</div>
                <div className="feature-pill">Study timelines</div>
                <div className="feature-pill">AI trend tracking</div>
                <div className="feature-pill">Alerts monitoring</div>
              </div>
            </div>
          )}

          <YourDataPanel runWithTokenRetry={runWithTokenRetry} onAccountDeleted={() => { void handleSignOut(); }} />
        </main>
      )}

      {/* ANALYZE */}
      {view === "analyze" && (
        <main className="analyze-shell">
          <div className="analyze-card">
            <div className="analyze-header">
              <button className="back-btn" onClick={() => setView("home")} type="button">Back</button>
              <h2>Upload Medical Reports</h2>
              <p>Select your PDF reports and we&apos;ll extract and analyze all the test data.</p>
              {studyContext && (
                <div className="study-upload-summary">
                  {studyContext.mode === "existing"
                    ? `Adding to: ${studyContext.study.name} • ${studyContext.study.report_count} ${studyContext.study.report_count === 1 ? "report" : "reports"} so far`
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
              <label
                className={`upload-zone${isDraggingFiles ? " dragging" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setIsDraggingFiles(true); }}
                onDragLeave={() => setIsDraggingFiles(false)}
                onDrop={(e) => {
                  // Without this the browser opens the dropped PDF in the tab and the app is lost.
                  e.preventDefault();
                  setIsDraggingFiles(false);
                  const dropped = Array.from(e.dataTransfer.files).filter((f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));
                  setPdfFiles((prev) => [...prev, ...dropped.filter((f) => !prev.some((p) => p.name === f.name && p.size === f.size))]);
                }}
              >
                <input className="visually-hidden" accept="application/pdf" multiple type="file" onChange={(e) => setPdfFiles(Array.from(e.target.files ?? []))} />
                <div className="upload-zone-inner">
                  <span className="upload-icon">Upload</span>
                  <strong>Drag &amp; drop or click to browse</strong>
                  <span>PDF files only • Up to 20 files, 10 MB each, 25 MB in total</span>
                </div>
              </label>

              {pdfFiles.length > 0 && (
                <div className="file-list">
                  {pdfFiles.map((f) => (
                    <div className="file-row" key={`${f.name}-${f.size}`}>
                      <span className="file-name">{f.name}</span>
                      <span className="file-size">{formatFileSize(f.size)}</span>
                      <button className="file-remove" type="button" aria-label={`Remove ${f.name}`} onClick={() => setPdfFiles((prev) => prev.filter((x) => x !== f))}>×</button>
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
                      <h3>{analyzeStep === "error" ? "Analysis stopped" : "Processing reports"}</h3>
                      <p className="analysis-progress-subcopy">
                        {analyzeStep === "error"
                          ? "Nothing was saved. The reason for each file is listed below."
                          : analyzeStep === "uploading"
                            ? "Uploading your files."
                            : "Your reports are being read on the server. Progress updates every few seconds."}
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
                        {isAnalyzing && (
                          <span className="analysis-eta-copy">
                            {estimatedRemainingSeconds !== null
                              ? `Estimated remaining time: ${etaMinutes}m ${etaSeconds}s`
                              : "Calculating estimated completion time..."}
                          </span>
                        )}
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

                  {isAnalyzing && (
                    <p className="analysis-progress-note">
                      Large report sets can take a few minutes. You can close this tab or refresh; the upload keeps going and the result is saved to your account.
                    </p>
                  )}
                </section>
              )}

              {errorMessage && <p className="status-text error-text">{errorMessage}</p>}

              <label className="consent-check">
                <input type="checkbox" required checked={uploadConsent} onChange={(e) => setUploadConsent(e.target.checked)} />
                <span>
                  I understand the text of these reports is sent to Google&apos;s Gemini AI to be read, that I have permission
                  to upload them, and that the results are not medical advice. <Link href="/privacy" target="_blank">How data is handled</Link>
                </span>
              </label>

              <button className="primary-button submit-button" disabled={isAnalyzing || !uploadConsent} type="submit">
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

          {studySuccessMessage && <div className="success-banner">{studySuccessMessage}</div>}

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
                <p className="muted-copy">Answers come only from your own results and can be wrong. They are not medical advice: check anything important with your doctor.</p>
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
                      disabled={isSending || !showAssistantPrompts}
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
                  disabled={isSending || (chatHistory.length === 0 && showAssistantPrompts)}
                >
                  Clear chat
                </button>
              </aside>

              <div className="assistant-chat-panel">
                <div className="assistant-chat-history" role="log" aria-live="polite">
                  {chatHistory.length === 0 && (
                    <div className="assistant-chat-empty">
                      Start with a question like &quot;What changed the most in the latest report?&quot;
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
                    const assistantParsedContent = !isUser && !turn.isLoading
                      ? splitAssistantResponseForSources(turn.content)
                      : null;
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
                              <>
                                <div className="assistant-chat-markdown">
                                  <ReactMarkdown
                                    remarkPlugins={[remarkGfm]}
                                    components={MARKDOWN_COMPONENTS}
                                  >
                                    {assistantParsedContent ? assistantParsedContent.body : turn.content}
                                  </ReactMarkdown>
                                </div>
                                {assistantParsedContent?.sources && (
                                  <div className="assistant-chat-citations">
                                    <ReactMarkdown
                                      remarkPlugins={[remarkGfm]}
                                      components={CITATION_COMPONENTS}
                                    >
                                      {assistantParsedContent.sources}
                                    </ReactMarkdown>
                                  </div>
                                )}
                              </>
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
                    void submitChatQuestion();
                  }}
                >
                  <input
                    className="text-input"
                    type="text"
                    placeholder="Ask a clinical question about trends, abnormalities, or next actions..."
                    disabled={isSending}
                    value={chatQuestion}
                    onChange={(e) => setChatQuestion(e.target.value)}
                  />
                  <button className="primary-button" type="submit" disabled={isSending || !chatQuestion.trim()}>
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
            {exportError && <p className="error-text" role="alert">{exportError}</p>}
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