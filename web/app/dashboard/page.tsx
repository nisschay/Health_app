"use client";

import { useState, useTransition, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import {
  type AnalysisResponse,
  type AnalysisHistoryItem,
  type ChatTurn,
  type MedicalRecord,
  analyzeReports,
  sendChatMessage,
  fetchReportHistory,
  fetchReportById,
  saveAnalysis,
  exportPdf,
  exportExcel,
} from "@/lib/api";
import TrendChart from "./TrendChart";

type AnalyzeStep = "idle" | "preparing" | "uploading" | "processing" | "saving" | "error";

function getStatusTone(status: string | null | undefined): string {
  const n = status?.toLowerCase();
  if (n === "normal" || n === "negative") return "good";
  if (n === "high" || n === "low" || n === "positive") return "warn";
  if (n === "critical" || n === "flagged") return "bad";
  return "muted";
}

function fmtDate(iso: string) {
  try {
    return new Date(iso).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
  } catch { return iso; }
}

function NavBar({ user, onLogout, onHome }: { user: { displayName?: string | null; email?: string | null } | null; onLogout: () => void; onHome: () => void }) {
  return (
    <nav className="top-nav">
      <button className="nav-brand" onClick={onHome} type="button">🏥 Medical Report Analyzer</button>
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
        {item.source_filenames.length > 0 && <p className="history-files">📄 {item.source_filenames.join(", ")}</p>}
      </div>
      <button className="secondary-button" onClick={() => onLoad(item.id)} type="button">View →</button>
    </article>
  );
}

function DataTable({ records }: { records: MedicalRecord[] }) {
  const valid = records.filter((r) => r.Test_Name && r.Test_Name !== "N/A" && r.Result !== null && r.Result !== undefined);
  const dateLabs = Array.from(new Set(valid.map((r) => `${r.Test_Date ?? "Unknown"}__${r.Lab_Name ?? "Unknown Lab"}`))).sort();
  const rowMap = new Map<string, Map<string, string>>();
  for (const r of valid) {
    const key = `${r.Test_Category ?? ""}||${r.Test_Name ?? ""}`;
    const dlKey = `${r.Test_Date ?? "Unknown"}__${r.Lab_Name ?? "Unknown Lab"}`;
    if (!rowMap.has(key)) rowMap.set(key, new Map());
    rowMap.get(key)!.set(dlKey, String(r.Result ?? ""));
  }
  const rows = Array.from(rowMap.entries()).map(([key, values]) => {
    const [category, name] = key.split("||");
    return { category, name, values };
  });
  return (
    <div className="table-shell">
      <table className="records-table">
        <thead>
          <tr>
            <th>Category</th>
            <th>Test</th>
            {dateLabs.map((dl) => { const [date, lab] = dl.split("__"); return <th key={dl}>{date}<br /><small>{lab}</small></th>; })}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td>{row.category}</td>
              <td>{row.name}</td>
              {dateLabs.map((dl) => <td key={dl}>{row.values.get(dl) ?? "—"}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
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
      setSelectedTest("");
      setView("result");
      if (token) {
        setAnalyzeStep("saving");
        setSaveStatus("saving");
        try {
          await saveAnalysis(result, pdfFiles.map((f) => f.name), token);
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
      setAnalysis(result); setChatHistory([]); setSelectedTest(""); setView("result");
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
      const resp = await sendChatMessage(analysis.records, q, chatHistory, token ?? undefined);
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
          const resp = await sendChatMessage(analysis.records, clean, chatHistory, token ?? undefined);
          setChatHistory([...newHistory, { role: "assistant", content: resp.answer }]);
        } catch (err) { setChatError(err instanceof Error ? err.message : "Chat failed."); }
      })();
    });
  }

  async function handleExportPdf() {
    if (!analysis) return;
    setIsExporting("pdf");
    try {
      const token = await getToken();
      const blob = await exportPdf(analysis.records, analysis.patient_info, token ?? undefined);
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

  useEffect(() => { setSelectedCategory("all"); setSelectedTest(""); }, [selectedBodySystem]);
  useEffect(() => { setSelectedTest(""); }, [selectedCategory]);

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
          <section className="home-hero">
            <h1>Your Health Reports</h1>
            <p>{history.length > 0 ? "Review past analyses or upload new reports." : "Welcome! Upload your first medical report PDFs to get started."}</p>
            <button className="primary-button" onClick={() => setView("analyze")} type="button">
              🔬 {history.length > 0 ? "Analyze New Reports" : "Upload & Analyze"}
            </button>
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
              <div className="empty-icon">📋</div>
              <h3>No reports yet</h3>
              <p>Upload your medical PDF reports and the AI will extract and analyze all test data automatically.</p>
              <div className="feature-row">
                <div className="feature-pill">📄 PDF extraction</div>
                <div className="feature-pill">📊 Health trends</div>
                <div className="feature-pill">🤖 AI chat</div>
                <div className="feature-pill">📥 Excel export</div>
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
              <button className="back-btn" onClick={() => setView("home")} type="button">← Back</button>
              <h2>Upload Medical Reports</h2>
              <p>Select your PDF reports and we'll extract and analyze all the test data.</p>
            </div>

            <div className="mode-tabs">
              <button className={`mode-tab ${uploadMode === "new" ? "active" : ""}`} onClick={() => setUploadMode("new")} type="button">📄 Upload New Reports</button>
              <button className={`mode-tab ${uploadMode === "append" ? "active" : ""}`} onClick={() => setUploadMode("append")} type="button">➕ Add to Existing Data</button>
            </div>

            <form className="upload-form" onSubmit={(e) => { e.preventDefault(); void submitAnalysis(); }}>
              <label className="upload-zone">
                <input accept="application/pdf" multiple type="file" onChange={(e) => setPdfFiles(Array.from(e.target.files ?? []))} />
                <div className="upload-zone-inner">
                  <span className="upload-icon">☁️</span>
                  <strong>Drag &amp; drop or click to browse</strong>
                  <span>PDF files only • Up to 200MB each</span>
                </div>
              </label>

              {pdfFiles.length > 0 && (
                <div className="file-list">
                  {pdfFiles.map((f) => (
                    <div className="file-row" key={`${f.name}-${f.size}`}>
                      <span>📄</span>
                      <span className="file-name">{f.name}</span>
                      <span className="file-size">{(f.size / 1024 / 1024).toFixed(1)} MB</span>
                      <button className="file-remove" type="button" onClick={() => setPdfFiles((prev) => prev.filter((x) => x !== f))}>×</button>
                    </div>
                  ))}
                </div>
              )}

              {uploadMode === "append" && (
                <label className="field-block">
                  <span>Upload Previous Excel / CSV (optional)</span>
                  <input accept=".xlsx,.xls,.csv" type="file" onChange={(e) => setExistingDataFile(e.target.files?.[0] ?? null)} />
                  {existingDataFile && <span className="file-chip accent">📊 {existingDataFile.name}</span>}
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
                {isAnalyzing ? "🔬 Analyzing…" : "🔬 Analyze Reports"}
              </button>
            </form>
          </div>
        </main>
      )}

      {/* RESULT */}
      {view === "result" && analysis && (
        <main className="result-shell">
          <div className="result-topbar">
            <button className="back-btn" onClick={() => setView("home")} type="button">← Dashboard</button>
            <button className="secondary-button" onClick={() => setView("analyze")} type="button">+ Analyze More</button>
            {saveStatus === "saved" && <span className="save-badge">✅ Saved</span>}
            {saveStatus === "saving" && <span className="save-badge muted">Saving…</span>}
            {saveStatus === "error" && <span className="save-badge error-badge">⚠️ Not saved</span>}
          </div>

          <div className="success-banner">🎉 Successfully analyzed {analysis.total_records} test records!</div>

          {/* Patient profile */}
          <section className="result-section">
            <h2>👤 Patient Profile</h2>
            <div className="patient-grid">
              {[
                { label: "NAME", icon: "👤", value: analysis.patient_info.name },
                { label: "AGE", icon: "🎂", value: analysis.patient_info.age },
                { label: "GENDER", icon: "⚧", value: analysis.patient_info.gender },
                { label: "LAB", icon: "🏥", value: analysis.patient_info.lab_name },
              ].map((item) => (
                <div className="patient-card" key={item.label}>
                  <span>{item.icon} {item.label}</span>
                  <strong>{item.value || "N/A"}</strong>
                </div>
              ))}
            </div>
          </section>

          {/* AI Health Assistant */}
          <section className="result-section">
            <h2>💬 AI Health Assistant</h2>
            <p className="muted-copy">Ask questions about your medical report and get AI-powered insights.</p>

            {chatHistory.length === 0 && (
              <div className="quick-questions">
                <h3>💡 Quick Questions</h3>
                <div className="quick-grid">
                  {["📊 Show a general overview of my report", "🔬 Explain my blood test results", "⚠️ Are there any concerning findings?", "📈 Show my test trends over time"].map((q) => (
                    <button className="quick-btn" disabled={isChatPending} key={q} onClick={() => handleQuickQuestion(q)} type="button">{q}</button>
                  ))}
                </div>
              </div>
            )}

            <div className="chat-stream">
              {chatHistory.map((turn, i) => (
                <div className={`chat-bubble ${turn.role === "user" ? "user" : "assistant"}`} key={i}>{turn.content}</div>
              ))}
              {isChatPending && <div className="chat-bubble assistant muted">🤔 Thinking…</div>}
            </div>

            {chatError && <p className="status-text error-text">{chatError}</p>}

            <form className="chat-input-row" onSubmit={(e) => { e.preventDefault(); startChatTransition(() => { void submitChatQuestion(); }); }}>
              <input className="text-input" disabled={isChatPending} placeholder="Ask me anything about your health report…" type="text" value={chatQuestion} onChange={(e) => setChatQuestion(e.target.value)} />
              <button className="primary-button" disabled={isChatPending || !chatQuestion.trim()} type="submit">Send</button>
            </form>
          </section>

          {/* Visualizations */}
          <section className="result-section">
            <h2>📈 Test Result Visualizations</h2>
            <div className="viz-layout">
              <div className="viz-controls">
                <label className="field-block">
                  <span>Filter by Body System</span>
                  <select className="select-input" value={selectedBodySystem} onChange={(e) => setSelectedBodySystem(e.target.value)}>
                    <option value="all">🔍 All Systems</option>
                    {allBodySystems.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </label>
                <label className="field-block">
                  <span>Select Category</span>
                  <select className="select-input" value={selectedCategory} onChange={(e) => setSelectedCategory(e.target.value)}>
                    <option value="all">📋 All Categories</option>
                    {allCategories.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
                <label className="field-block">
                  <span>Select Test</span>
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
                    <span>📊</span>
                    <h3>Select a Test to Visualize</h3>
                    <p>Choose a test from the dropdown on the left to see charts and trends.</p>
                  </div>
                )}
              </div>
            </div>
          </section>

          {/* Health Alerts */}
          {analysis.health_summary.concerns.length > 0 && (
            <section className="result-section">
              <h2>⚠️ Health Alerts</h2>
              <p className="muted-copy">{analysis.health_summary.concerns.length} result{analysis.health_summary.concerns.length !== 1 ? "s" : ""} flagged for review</p>
              <div className="concern-list">
                {analysis.health_summary.concerns.map((c, i) => (
                  <article className="concern-row" key={i}>
                    <div>
                      <h3>{c.test_name}</h3>
                      <p>{c.category} &bull; {c.date} &bull; Ref {c.reference}</p>
                    </div>
                    <div className="report-meta">
                      <span className={`status-pill ${getStatusTone(c.status)}`}>{c.status}</span>
                      <strong>{String(c.result)}</strong>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          )}

          {/* Data Table */}
          <section className="result-section">
            <h2>📋 Organized Data by Date</h2>
            <p className="muted-copy">Your medical test results organized by test category, test name, and date.</p>
            <DataTable records={analysis.records} />
          </section>

          {/* Downloads */}
          <section className="result-section download-section">
            <h2>📥 Download Health Report</h2>
            <p className="muted-copy">Download a comprehensive report with your test results and AI-powered insights.</p>
            <div className="download-buttons">
              <button className="primary-button" disabled={isExporting !== null} onClick={handleExportPdf} type="button">
                {isExporting === "pdf" ? "Generating…" : "📥 Download Health Report (PDF)"}
              </button>
              <button className="secondary-button" disabled={isExporting !== null} onClick={handleExportExcel} type="button">
                {isExporting === "excel" ? "Generating…" : "📊 Download Excel with Charts"}
              </button>
            </div>
          </section>
        </main>
      )}
    </div>
  );
}