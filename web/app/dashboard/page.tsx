"use client";

import { useState, useTransition } from "react";

import {
  type AnalysisResponse,
  analyzeReports,
  type ChatTurn,
  getApiBaseUrl,
  sendChatMessage,
  type MedicalRecord
} from "@/lib/api";

const recentReports = [
  {
    name: "Quarterly Metabolic Panel",
    date: "11 Mar 2026",
    status: "Analyzed",
    note: "3 abnormal markers flagged for review"
  },
  {
    name: "CBC Follow-up",
    date: "02 Feb 2026",
    status: "Analyzed",
    note: "Stable across previous test dates"
  },
  {
    name: "Lipid Profile",
    date: "18 Dec 2025",
    status: "Imported",
    note: "Awaiting user-confirmed history merge"
  }
];

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function getStatusTone(status: string | null | undefined): string {
  const normalized = status?.toLowerCase();
  if (normalized === "normal" || normalized === "negative") {
    return "good";
  }
  if (normalized === "high" || normalized === "low" || normalized === "positive") {
    return "warn";
  }
  if (normalized === "critical" || normalized === "flagged") {
    return "bad";
  }
  return "muted";
}

export default function DashboardPage() {
  const [pdfFiles, setPdfFiles] = useState<File[]>([]);
  const [existingDataFile, setExistingDataFile] = useState<File | null>(null);
  const [includeRawTexts, setIncludeRawTexts] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [token, setToken] = useState("");
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatTurn[]>([]);
  const [chatError, setChatError] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const [isChatPending, startChatTransition] = useTransition();

  const dashboardMetrics = analysis
    ? [
        { label: "Health score", value: `${analysis.health_summary.overall_score} / 100` },
        { label: "Tracked systems", value: String(analysis.body_systems.length) },
        { label: "Abnormal markers", value: String(analysis.health_summary.concerns.length) },
        { label: "Records extracted", value: String(analysis.total_records) }
      ]
    : [
        { label: "Health score", value: "Awaiting upload" },
        { label: "Tracked systems", value: "0" },
        { label: "Abnormal markers", value: "0" },
        { label: "Records extracted", value: "0" }
      ];

  async function submitAnalysis() {
    if (pdfFiles.length === 0 && !existingDataFile) {
      setErrorMessage("Select at least one PDF report or a prior Excel or CSV file.");
      return;
    }

    const formData = new FormData();
    pdfFiles.forEach((file) => {
      formData.append("pdf_files", file);
    });

    if (existingDataFile) {
      formData.append("existing_data", existingDataFile);
    }

    formData.append("include_raw_texts", String(includeRawTexts));

    try {
      setErrorMessage(null);
      const authToken = token.trim() || undefined;
      const result = await analyzeReports(formData, authToken);
      setAnalysis(result);
      setChatHistory([]);
      setChatError(null);
    } catch (error) {
      setAnalysis(null);
      setErrorMessage(error instanceof Error ? error.message : "Unable to analyze the uploaded files.");
    }
  }

  async function submitChatQuestion() {
    if (!analysis) {
      setChatError("Analyze at least one report before using chat.");
      return;
    }
    if (!chatQuestion.trim()) {
      setChatError("Enter a question for the assistant.");
      return;
    }

    const records: MedicalRecord[] = analysis.records;
    const userTurn: ChatTurn = { role: "user", content: chatQuestion.trim() };
    const nextHistory = [...chatHistory, userTurn];

    try {
      setChatError(null);
      const authToken = token.trim() || undefined;
      const response = await sendChatMessage(records, userTurn.content, chatHistory, authToken);
      setChatHistory([...nextHistory, { role: "assistant", content: response.answer }]);
      setChatQuestion("");
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "Unable to get a chatbot response.");
    }
  }

  return (
    <main className="dashboard-shell">
      <section className="dashboard-hero">
        <div>
          <p className="eyebrow">Dashboard upload flow</p>
          <h1>Patient overview and report workflow</h1>
          <p className="lede">
            This view now posts real multipart uploads to the Python API and renders the returned
            analysis instead of staying as a static prototype.
          </p>
        </div>
        <div className="upload-card">
          <span className="panel-label">Connected backend route</span>
          <strong>POST {getApiBaseUrl()}/api/v1/reports/analyze</strong>
          <form
            className="upload-form"
            onSubmit={(event) => {
              event.preventDefault();
              startTransition(() => {
                void submitAnalysis();
              });
            }}
          >
            <label className="field-block">
              <span>Medical report PDFs</span>
              <input
                type="file"
                accept="application/pdf"
                multiple
                onChange={(event) => {
                  setPdfFiles(Array.from(event.target.files ?? []));
                }}
              />
            </label>

            <label className="field-block">
              <span>Existing Excel or CSV history</span>
              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={(event) => {
                  setExistingDataFile(event.target.files?.[0] ?? null);
                }}
              />
            </label>

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={includeRawTexts}
                onChange={(event) => {
                  setIncludeRawTexts(event.target.checked);
                }}
              />
              <span>Include extracted text previews in the response</span>
            </label>

            <label className="field-block">
              <span>Firebase bearer token (optional for secured API mode)</span>
              <input
                className="text-input"
                type="password"
                placeholder="Paste ID token when API_REQUIRE_AUTH=true"
                value={token}
                onChange={(event) => {
                  setToken(event.target.value);
                }}
              />
            </label>

            <div className="selected-files">
              {pdfFiles.length > 0 ? (
                pdfFiles.map((file) => (
                  <span className="file-chip" key={`${file.name}-${file.size}`}>
                    {file.name}
                  </span>
                ))
              ) : (
                <span className="muted-copy">No PDF files selected yet.</span>
              )}
              {existingDataFile ? (
                <span className="file-chip accent">History: {existingDataFile.name}</span>
              ) : null}
            </div>

            <button className="primary-button submit-button" disabled={isPending} type="submit">
              {isPending ? "Analyzing reports..." : "Analyze reports"}
            </button>

            {errorMessage ? <p className="status-text error-text">{errorMessage}</p> : null}
            {analysis ? (
              <p className="status-text success-text">
                Analysis complete for {analysis.patient_info.name || "patient"}.
              </p>
            ) : null}
          </form>
        </div>
      </section>

      <section className="metric-grid">
        {dashboardMetrics.map((metric) => (
          <article className="metric-card" key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
          </article>
        ))}
      </section>

      <section className="dashboard-columns">
        <div className="report-list-card">
          <div className="section-heading">
            <h2>{analysis ? "Analysis summary" : "Recent reports"}</h2>
            <p>
              {analysis
                ? "The backend response is rendered below from the uploaded files."
                : "Firebase persistence is still pending, so these remain illustrative entries."}
            </p>
          </div>

          {analysis ? (
            <div className="analysis-stack">
              <section className="detail-card">
                <div className="section-heading compact">
                  <h2>Patient profile</h2>
                </div>
                <div className="info-grid">
                  <div>
                    <span>Name</span>
                    <strong>{analysis.patient_info.name || "N/A"}</strong>
                  </div>
                  <div>
                    <span>Age</span>
                    <strong>{analysis.patient_info.age || "N/A"}</strong>
                  </div>
                  <div>
                    <span>Gender</span>
                    <strong>{analysis.patient_info.gender || "N/A"}</strong>
                  </div>
                  <div>
                    <span>Patient ID</span>
                    <strong>{analysis.patient_info.patient_id || "N/A"}</strong>
                  </div>
                  <div>
                    <span>Lab</span>
                    <strong>{analysis.patient_info.lab_name || "N/A"}</strong>
                  </div>
                  <div>
                    <span>Report date</span>
                    <strong>{analysis.patient_info.date || "N/A"}</strong>
                  </div>
                </div>
              </section>

              <section className="detail-card">
                <div className="section-heading compact">
                  <h2>Flagged concerns</h2>
                  <p>{analysis.health_summary.concerns.length} markers need attention.</p>
                </div>
                <div className="concern-list">
                  {analysis.health_summary.concerns.length > 0 ? (
                    analysis.health_summary.concerns.slice(0, 6).map((concern) => (
                      <article className="concern-row" key={`${concern.test_name}-${concern.date}`}>
                        <div>
                          <h3>{concern.test_name}</h3>
                          <p>
                            {concern.category} • {concern.date} • Ref {concern.reference}
                          </p>
                        </div>
                        <div className="report-meta">
                          <span className={`status-pill ${getStatusTone(concern.status)}`}>
                            {concern.status}
                          </span>
                          <strong>{concern.result}</strong>
                        </div>
                      </article>
                    ))
                  ) : (
                    <p className="muted-copy">No abnormal results were flagged in the analyzed data.</p>
                  )}
                </div>
              </section>

              <section className="detail-card">
                <div className="section-heading compact">
                  <h2>Extracted records</h2>
                  <p>Showing the first 8 normalized test rows returned by the API.</p>
                </div>
                <div className="table-shell">
                  <table className="records-table">
                    <thead>
                      <tr>
                        <th>Test</th>
                        <th>Category</th>
                        <th>Result</th>
                        <th>Status</th>
                        <th>Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.records.slice(0, 8).map((record, index) => (
                        <tr key={`${record.Test_Name}-${record.Test_Date}-${index}`}>
                          <td>{record.Test_Name ?? "N/A"}</td>
                          <td>{record.Test_Category ?? "N/A"}</td>
                          <td>
                            {record.Result ?? "N/A"}
                            {record.Unit ? ` ${record.Unit}` : ""}
                          </td>
                          <td>
                            <span className={`status-pill ${getStatusTone(record.Status)}`}>
                              {record.Status ?? "N/A"}
                            </span>
                          </td>
                          <td>{record.Test_Date ?? "N/A"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
          ) : (
            <div className="report-list">
              {recentReports.map((report) => (
                <article className="report-row" key={`${report.name}-${report.date}`}>
                  <div>
                    <h3>{report.name}</h3>
                    <p>{report.note}</p>
                  </div>
                  <div className="report-meta">
                    <span>{report.date}</span>
                    <strong>{report.status}</strong>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>

        <div className="chat-card">
          <div className="section-heading">
            <h2>{analysis ? "Assistant panel" : "Assistant panel"}</h2>
            <p>
              {analysis
                ? "Ask follow-up questions about the analyzed records using /api/v1/reports/chat."
                : "Run report analysis first to unlock contextual chatbot responses."}
            </p>
          </div>

          {analysis ? (
            <div className="chat-live-shell">
              <div className="chat-stream">
                {chatHistory.length === 0 ? (
                  <div className="chat-bubble assistant muted">
                    Ask a report question like: Why are my liver markers flagged?
                  </div>
                ) : (
                  chatHistory.map((turn, index) => (
                    <div
                      className={`chat-bubble ${turn.role === "assistant" ? "assistant" : "user"}`}
                      key={`${turn.role}-${index}-${turn.content.slice(0, 20)}`}
                    >
                      {turn.content}
                    </div>
                  ))
                )}
              </div>

              <form
                className="chat-input-row"
                onSubmit={(event) => {
                  event.preventDefault();
                  startChatTransition(() => {
                    void submitChatQuestion();
                  });
                }}
              >
                <input
                  className="text-input"
                  type="text"
                  placeholder="Ask about results, trends, or abnormalities"
                  value={chatQuestion}
                  onChange={(event) => {
                    setChatQuestion(event.target.value);
                  }}
                />
                <button className="secondary-button" disabled={isChatPending} type="submit">
                  {isChatPending ? "Sending..." : "Ask"}
                </button>
              </form>

              {chatError ? <p className="status-text error-text">{chatError}</p> : null}
            </div>
          ) : (
            <div className="chat-placeholder">
              <div className="chat-bubble assistant">
                I can explain abnormal findings, summarize trends, and generate patient-friendly
                answers after the uploaded report is analyzed.
              </div>
              <div className="chat-bubble user">Why is my LDL still high compared with last quarter?</div>
              <div className="chat-bubble assistant muted">
                Pending API wiring. This panel will render responses from the FastAPI backend.
              </div>
            </div>
          )}
        </div>
      </section>

      {analysis ? (
        <section className="report-list-card raw-text-section">
          <div className="section-heading">
            <h2>Body systems overview</h2>
            <p>Generated by the backend insights layer for quick triage by system.</p>
          </div>
          <div className="system-list">
            {analysis.body_systems.map((system) => (
              <article className="system-card" key={system.system}>
                <div className="system-heading">
                  <strong>
                    {system.emoji} {system.system}
                  </strong>
                  <span className={`status-pill ${getStatusTone(system.concern_level)}`}>
                    {system.concern_level}
                  </span>
                </div>
                <p>
                  {system.abnormal_count} of {system.total_count} tests abnormal
                  ({formatPercent(system.abnormal_ratio)})
                </p>
                <div className="chip-row">
                  {system.categories.map((category) => (
                    <span className="file-chip" key={`${system.system}-${category}`}>
                      {category}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {analysis?.raw_texts.length ? (
        <section className="report-list-card raw-text-section">
          <div className="section-heading">
            <h2>Raw extraction previews</h2>
            <p>Useful for comparing the uploaded report text against the structured output.</p>
          </div>
          <div className="raw-text-grid">
            {analysis.raw_texts.map((item) => (
              <article className="raw-text-card" key={item.name}>
                <strong>{item.name}</strong>
                <pre>{item.text}</pre>
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </main>
  );
}