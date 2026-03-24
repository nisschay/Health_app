"use client";

import { useState, useTransition, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import {
  type AnalysisResponse,
  type AnalysisHistoryItem,
  type ChatTurn,
  type ProfileItem,
  type StudySummary,
  analyzeReports,
  createProfile,
  createStudy,
  fetchProfiles,
  sendChatMessage,
  fetchStudiesForProfile,
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

type AnalyzeStep = "idle" | "preparing" | "uploading" | "processing" | "saving" | "error";
type StudyAction = "add-existing" | "start-new";

type AnalyzeContext = {
  profile: ProfileItem;
  study: StudySummary;
  mode: "existing" | "new";
};

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

function fmtDate(iso: string) {
  try {
    return new Date(iso).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
  } catch { return iso; }
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

function HistoryCard({ item, onLoad }: { item: AnalysisHistoryItem; onLoad: (id: number) => void }) {
  return (
    <article className="history-card">
      <div className="history-card-info">
        <h3>{item.patient_name ?? "Unknown Patient"}</h3>
        <p>{item.total_records} records &bull; {item.lab_name ?? "Unknown Lab"} &bull; {fmtDate(item.created_at)}</p>
        {item.source_filenames.length > 0 && <p className="history-files">{item.source_filenames.join(", ")}</p>}
      </div>
      <button className="secondary-button" onClick={() => onLoad(item.id)} type="button">View</button>
    </article>
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
              {profiles.map((profile) => {
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
              <input className="input" type="date" value={newMemberDob} onChange={(e) => onChangeMemberDob(e.target.value)} />
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
  const [pdfFiles, setPdfFiles] = useState<File[]>([]);
  const [existingDataFile, setExistingDataFile] = useState<File | null>(null);
  const [uploadMode, setUploadMode] = useState<"new" | "append">("new");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeStep, setAnalyzeStep] = useState<AnalyzeStep>("idle");
  const [analysisStartedAt, setAnalysisStartedAt] = useState<number | null>(null);
  const [analysisElapsedSeconds, setAnalysisElapsedSeconds] = useState(0);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatTurn[]>([]);
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

  const loadHistory = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    setHistoryLoading(true);
    try {
      const items = await fetchReportHistory(token);
      setHistory(items);
    } catch { /* non-critical */ } finally { setHistoryLoading(false); }
  }, [getToken]);

  useEffect(() => { if (user) loadHistory(); }, [user, loadHistory]);

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
      const token = await getToken();
      if (!token) {
        setFlowError("Please sign in again to continue.");
        return;
      }
      const rows = await fetchProfiles(token);
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
      const token = await getToken();
      if (!token) {
        setFlowError("Please sign in again to continue.");
        return;
      }
      const studies = await fetchStudiesForProfile(profile.id, token);
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
      const token = await getToken();
      if (!token) {
        setFlowError("Please sign in again to continue.");
        return;
      }
      const created = await createStudy(
        {
          profile_id: selectedProfile.id,
          name: newStudyName.trim(),
          description: newStudyDescription.trim() || undefined,
        },
        token,
      );
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
      const token = await getToken();
      if (!token) {
        setFlowError("Please sign in again to continue.");
        return;
      }
      const created = await createProfile(
        {
          full_name: newMemberName.trim(),
          relationship,
          date_of_birth: newMemberDob || undefined,
        },
        token,
      );
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

  async function submitAnalysis() {
    setErrorMessage(null);
    setIsAnalyzing(true);
    setAnalyzeStep("preparing");
    setAnalysisStartedAt(Date.now());
    setAnalysisElapsedSeconds(0);
    const token = await getToken();
    if (pdfFiles.length === 0) {
      setErrorMessage("Please select at least one PDF file.");
      setIsAnalyzing(false);
      setAnalyzeStep("error");
      return;
    }
    const fd = new FormData();
    for (const f of pdfFiles) fd.append("pdf_files", f);
    if (existingDataFile) fd.append("existing_data", existingDataFile);
    setAnalyzeStep("uploading");
    try {
      setAnalyzeStep("processing");
      const result = await analyzeReports(fd, token ?? undefined);
      setAnalysis(result);
      setChatHistory([]);
      setSelectedBodySystem("all");
      setSelectedCategory("all");
      setSelectedTest("");
      setView("result");
      if (token) {
        setAnalyzeStep("saving");
        setSaveStatus("saving");
        try {
          if (studyContext) {
            const saved = await saveStudyAnalysis(studyContext.study.id, result, pdfFiles.map((f) => f.name), token);
            setStudySuccessMessage(
              studyContext.mode === "existing"
                ? `${saved.added_reports} new reports added to ${saved.study_name}. Dashboard updated with new trends.`
                : `New study ${saved.study_name} created for ${studyContext.profile.full_name}.`,
            );
          } else {
            await saveAnalysis(result, pdfFiles.map((f) => f.name), token);
            setStudySuccessMessage(null);
          }
          setSaveStatus("saved");
          loadHistory();
        } catch { setSaveStatus("error"); }
      }
      setAnalyzeStep("idle");
    } catch (err) {
      setAnalyzeStep("error");
      setErrorMessage(err instanceof Error ? err.message : "Analysis failed.");
    }
    finally {
      setIsAnalyzing(false);
      setAnalysisStartedAt(null);
    }
  }

  async function loadHistoryItem(id: number) {
    const token = await getToken();
    if (!token) return;
    try {
      const result = await fetchReportById(id, token);
      setStudyContext(null);
      setStudySuccessMessage(null);
      setAnalysis(result); setChatHistory([]); setSelectedBodySystem("all"); setSelectedCategory("all"); setSelectedTest(""); setView("result");
    } catch (err) { setErrorMessage(err instanceof Error ? err.message : "Failed to load."); }
  }

  async function submitChatQuestion() {
    if (!chatQuestion.trim() || !analysis) return;
    setChatError(null);
    const q = chatQuestion; setChatQuestion("");
    const token = await getToken();
    const newHistory = [...chatHistory, { role: "user", content: q }];
    setChatHistory(newHistory);
    try {
      const resp = await sendChatMessage(analysis.records, q, newHistory, token ?? undefined);
      setChatHistory([...newHistory, { role: "assistant", content: resp.answer }]);
    } catch (err) { setChatError(err instanceof Error ? err.message : "Chat failed."); }
  }

  function handleQuickQuestion(q: string) {
    startChatTransition(() => {
      void (async () => {
        if (!analysis) return;
        setChatError(null);
        const token = await getToken();
        const clean = q.replace(/^[^\s]+ /, "");
        const newHistory = [...chatHistory, { role: "user", content: clean }];
        setChatHistory(newHistory);
        try {
          const resp = await sendChatMessage(analysis.records, clean, newHistory, token ?? undefined);
          setChatHistory([...newHistory, { role: "assistant", content: resp.answer }]);
        } catch (err) { setChatError(err instanceof Error ? err.message : "Chat failed."); }
      })();
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
      const token = await getToken();
      const blob = await exportExcel(analysis.records, analysis.patient_info, token ?? undefined);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url;
      a.download = `medical-data-${analysis.patient_info.name || "patient"}.xlsx`; a.click();
      URL.revokeObjectURL(url);
    } catch { /* graceful */ } finally { setIsExporting(null); }
  }

  const records = analysis?.records ?? [];
  const allBodySystems = Array.from(new Set(records.map((r) => r.Test_Category?.split("/")[0]?.trim() ?? "Other").filter(Boolean))).sort();
  const filteredBySystem = selectedBodySystem === "all" ? records : records.filter((r) => (r.Test_Category ?? "").split("/")[0]?.trim() === selectedBodySystem);
  const allCategories = Array.from(new Set(filteredBySystem.map((r) => r.Test_Category ?? "Other").filter(Boolean))).sort();
  const filteredByCategory = selectedCategory === "all" ? filteredBySystem : filteredBySystem.filter((r) => r.Test_Category === selectedCategory);
  const allTests = Array.from(new Set(filteredByCategory.map((r) => r.Test_Name ?? "").filter(Boolean))).sort();
  const selectedTestData = selectedTest ? filteredByCategory.filter((r) => r.Test_Name === selectedTest) : [];
  const latestHistory = history.reduce<AnalysisHistoryItem | null>((latest, item) => {
    if (!latest) return item;
    return new Date(item.created_at).getTime() > new Date(latest.created_at).getTime() ? item : latest;
  }, null);
  const lastAnalysisLabel = latestHistory ? fmtRelativeDate(latestHistory.created_at) : "Not yet";
  const alertsCount = analysis?.health_summary?.concerns?.length ?? 0;
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
    { id: "preparing", label: "Validating files", detail: "Checking selected PDFs and upload mode" },
    { id: "uploading", label: "Uploading reports", detail: "Sending files securely to backend" },
    { id: "processing", label: "Running AI extraction", detail: "Parsing PDFs and building structured data" },
    { id: "saving", label: "Saving to your history", detail: "Storing the analysis in your account" },
  ] as const;

  const activeStepIndex = Math.max(0, analyzeSteps.findIndex((step) => step.id === analyzeStep));
  const elapsedMinutes = Math.floor(analysisElapsedSeconds / 60);
  const elapsedSeconds = analysisElapsedSeconds % 60;

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
              {history.length > 0 ? "Analyze New Report" : "Upload & Analyze"}
            </button>
          </section>

          <section className="stats-row" aria-label="Dashboard summary">
            <article className="stat-card">
              <span>Total Reports</span>
              <strong>{history.length}</strong>
            </article>
            <article className="stat-card">
              <span>Last Analysis</span>
              <strong>{lastAnalysisLabel}</strong>
            </article>
            <article className="stat-card">
              <span>Health Alerts</span>
              <strong>{alertsCount}</strong>
            </article>
          </section>

          {historyLoading && <p className="muted-copy center-text">Loading your reports…</p>}

          {history.length > 0 && (
            <section className="history-section">
              <h2>Past Analyses</h2>
              <div className="history-list">
                {history.map((item) => <HistoryCard key={item.id} item={item} onLoad={loadHistoryItem} />)}
              </div>
            </section>
          )}

          {!historyLoading && history.length === 0 && (
            <div className="empty-state">
              <h3>No reports yet</h3>
              <p>Upload your medical PDF reports and the AI will extract and analyze all test data automatically.</p>
              <div className="feature-row">
                <div className="feature-pill">PDF extraction</div>
                <div className="feature-pill">Health trends</div>
                <div className="feature-pill">AI assistant</div>
                <div className="feature-pill">Excel export</div>
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
                    <h3>{analyzeStep === "error" ? "Analysis stopped" : "Analysis in progress"}</h3>
                    {isAnalyzing && <span>{elapsedMinutes}m {elapsedSeconds}s</span>}
                  </div>

                  <ul className="analysis-progress-list">
                    {analyzeSteps.map((step, idx) => {
                      const isDone = idx < activeStepIndex && analyzeStep !== "error";
                      const isActive = isAnalyzing && idx === activeStepIndex;
                      const isError = analyzeStep === "error" && idx === activeStepIndex;
                      return (
                        <li
                          key={step.id}
                          className={`progress-step ${isDone ? "done" : ""} ${isActive ? "active" : ""} ${isError ? "error" : ""}`.trim()}
                        >
                          <span className="progress-dot" aria-hidden="true" />
                          <div>
                            <strong>{step.label}</strong>
                            <p>{step.detail}</p>
                          </div>
                        </li>
                      );
                    })}
                  </ul>

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
            <p className="muted-copy">Quick view of stable vs concerning patterns across report dates.</p>

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
                <span>Report Dates</span>
                <strong>{trendByDate.length}</strong>
              </article>
            </div>

            <div className="trend-mini-chart">
              {trendByDate.map((point) => {
                const concerningPct = point.total > 0 ? Math.round((point.concerning / point.total) * 100) : 0;
                const goodPct = point.total > 0 ? Math.round((point.good / point.total) * 100) : 0;
                return (
                  <div className="trend-mini-row" key={point.date}>
                    <span className="trend-mini-date">{point.date}</span>
                    <div className="trend-mini-bars" aria-label={`Concerning ${concerningPct}% and stable ${goodPct}% on ${point.date}`}>
                      <div className="trend-mini-bar bad" style={{ width: `${Math.max(4, concerningPct)}%` }} />
                      <div className="trend-mini-bar good" style={{ width: `${Math.max(4, goodPct)}%` }} />
                    </div>
                    <span className="trend-mini-meta">{point.concerning}/{point.total} alerts</span>
                  </div>
                );
              })}
            </div>
          </section>

          <ClinicalChatPanel
            records={analysis.records}
            chatHistory={chatHistory}
            chatError={chatError}
            isChatPending={isChatPending}
            chatQuestion={chatQuestion}
            onChangeQuestion={setChatQuestion}
            onSubmit={() => {
              startChatTransition(() => {
                void submitChatQuestion();
              });
            }}
            onQuickQuestion={(q) => handleQuickQuestion(q)}
          />

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