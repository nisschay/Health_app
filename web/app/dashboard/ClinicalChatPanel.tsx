"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChatTurn, MedicalRecord } from "@/lib/api";
import { parseMedicalDate } from "@/lib/clinical";

type StructuredAssistant = {
  summary: string;
  keyFindings: string[];
  metrics: Array<{ label: string; value: string; status: string; testName: string }>;
  recommendations: string[];
};

function parseRefRange(ref: string | null | undefined): { low: number | null; high: number | null } {
  if (!ref) return { low: null, high: null };
  const rangeMatch = ref.match(/([0-9.]+)\s*[-–]\s*([0-9.]+)/);
  if (rangeMatch) return { low: parseFloat(rangeMatch[1]), high: parseFloat(rangeMatch[2]) };
  const ltMatch = ref.match(/[<≤]\s*([0-9.]+)/);
  if (ltMatch) return { low: null, high: parseFloat(ltMatch[1]) };
  const gtMatch = ref.match(/[>≥]\s*([0-9.]+)/);
  if (gtMatch) return { low: parseFloat(gtMatch[1]), high: null };
  return { low: null, high: null };
}

function parseAssistantResponse(content: string, records: MedicalRecord[]): StructuredAssistant {
  const lines = content
    .split("\n")
    .map((line) => line.replace(/^[-*]\s*/, "").trim())
    .filter(Boolean);

  const summary = lines.slice(0, 2).join(" ") || "Clinical review generated based on available report data.";

  const findingLines = lines
    .filter((line) => /high|low|critical|positive|abnormal|concern|risk/i.test(line))
    .slice(0, 4);

  const recommendations = lines
    .filter((line) => /recommend|follow|monitor|discuss|repeat|consult/i.test(line))
    .slice(0, 3);

  const metrics: Array<{ label: string; value: string; status: string; testName: string }> = [];
  const seenMetricTests = new Set<string>();
  for (const record of records) {
    if (!record.Test_Name || record.Test_Name === "N/A") continue;
    if (seenMetricTests.has(record.Test_Name)) continue;
    const status = String(record.Status ?? "");
    if (!status || status.toLowerCase() === "normal" || status.toLowerCase() === "negative") continue;
    seenMetricTests.add(record.Test_Name);
    metrics.push({
      label: record.Test_Name,
      testName: record.Test_Name,
      value: `${String(record.Result ?? "N/A")}${record.Unit ? ` ${record.Unit}` : ""}`,
      status,
    });
    if (metrics.length >= 5) break;
  }

  return {
    summary,
    keyFindings: findingLines.length > 0 ? findingLines : ["No specific critical findings were highlighted in this response."],
    metrics,
    recommendations:
      recommendations.length > 0 ? recommendations : ["Discuss flagged trends with your clinician for interpretation."],
  };
}

function numericTrendForMetric(records: MedicalRecord[], metric: string | null) {
  if (!metric) return [] as Array<{ date: string; value: number }>;
  return records
    .filter((r) => r.Test_Name === metric)
    .map((r) => {
      const numeric = typeof r.Result_Numeric === "number" ? r.Result_Numeric : Number.parseFloat(String(r.Result ?? ""));
      return {
        date: r.Test_Date ?? "Unknown",
        value: Number.isNaN(numeric) ? NaN : numeric,
      };
    })
    .filter((row) => Number.isFinite(row.value))
    .sort((a, b) => parseMedicalDate(a.date) - parseMedicalDate(b.date));
}

function categoryComparison(records: MedicalRecord[]) {
  const map = new Map<string, number>();
  for (const record of records) {
    const status = String(record.Status ?? "").toLowerCase();
    if (status === "normal" || status === "negative" || !status) continue;
    const category = record.Test_Category?.trim() || "Uncategorized";
    map.set(category, (map.get(category) ?? 0) + 1);
  }
  return Array.from(map.entries())
    .map(([category, alerts]) => ({ category, alerts }))
    .sort((a, b) => b.alerts - a.alerts)
    .slice(0, 6);
}

function badgeTone(status: string) {
  const s = status.toLowerCase();
  if (s === "critical") return "bad";
  if (s === "high" || s === "positive" || s === "flagged") return "warn";
  if (s === "low") return "muted";
  return "muted";
}

export default function ClinicalChatPanel({
  records,
  chatHistory,
  chatError,
  isChatPending,
  chatQuestion,
  onChangeQuestion,
  onSubmit,
  onQuickQuestion,
}: {
  records: MedicalRecord[];
  chatHistory: ChatTurn[];
  chatError: string | null;
  isChatPending: boolean;
  chatQuestion: string;
  onChangeQuestion: (value: string) => void;
  onSubmit: () => void;
  onQuickQuestion: (question: string) => void;
}) {
  const assistantTurn = [...chatHistory].reverse().find((turn) => turn.role !== "user");
  const structured = useMemo(
    () => parseAssistantResponse(assistantTurn?.content ?? "", records),
    [assistantTurn?.content, records]
  );

  const [activeMetric, setActiveMetric] = useState<string | null>(structured.metrics[0]?.testName ?? null);
  const [hoveredMetric, setHoveredMetric] = useState<string | null>(null);

  const trendData = useMemo(() => numericTrendForMetric(records, activeMetric), [records, activeMetric]);
  const categoryData = useMemo(() => categoryComparison(records), [records]);
  const activeRecord = useMemo(
    () => records.find((row) => row.Test_Name === activeMetric),
    [records, activeMetric]
  );
  const activeRange = parseRefRange(activeRecord?.Reference_Range ?? null);

  const quickSuggestions = [
    "Show trends over time",
    "Explain abnormal values",
    "Compare categories with highest alerts",
  ];

  return (
    <section className="result-section">
      <h2>Clinical Intelligence Panel</h2>
      <div className="clinical-panel-layout">
        <div className="clinical-chat-column">
          {chatHistory.length === 0 && (
            <div className="clinical-empty-state">
              <p>Ask a question to generate structured clinical insights.</p>
              <div className="quick-grid">
                {quickSuggestions.map((q) => (
                  <button key={q} className="quick-btn" type="button" disabled={isChatPending} onClick={() => onQuickQuestion(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="chat-stream">
            {chatHistory.map((turn, index) => {
              const isAssistant = turn.role !== "user";
              const parsed = isAssistant ? parseAssistantResponse(turn.content, records) : null;

              return (
                <article key={index} className={`clinical-chat-item ${isAssistant ? "assistant" : "user"}`}>
                  <div className={`chat-bubble ${isAssistant ? "assistant" : "user"}`}>{turn.content}</div>

                  {isAssistant && parsed && (
                    <div className="clinical-structured-cards">
                      <div className="clinical-structured-card">
                        <h4>Summary</h4>
                        <p>{parsed.summary}</p>
                      </div>

                      <div className="clinical-structured-card">
                        <h4>Key Findings</h4>
                        <ul>
                          {parsed.keyFindings.map((finding, i) => (
                            <li key={`${index}-finding-${i}`}>{finding}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="clinical-structured-card">
                        <h4>Key Metrics</h4>
                        <div className="metric-chip-row">
                          {parsed.metrics.length === 0 && <span className="muted-copy">No highlighted metrics.</span>}
                          {parsed.metrics.map((metric, metricIndex) => (
                            <button
                              type="button"
                              key={`${index}-${metric.testName}-${metric.status}-${metric.value}-${metricIndex}`}
                              className={`metric-chip ${activeMetric === metric.testName ? "active" : ""} ${hoveredMetric === metric.testName ? "linked" : ""}`}
                              onClick={() => setActiveMetric(metric.testName)}
                            >
                              <span>{metric.label}</span>
                              <strong>{metric.value}</strong>
                              <em className={`status-pill ${badgeTone(metric.status)}`}>{metric.status}</em>
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="clinical-structured-card">
                        <h4>Recommendations</h4>
                        <ul>
                          {parsed.recommendations.map((rec, i) => (
                            <li key={`${index}-rec-${i}`}>{rec}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </article>
              );
            })}

            {isChatPending && <div className="chat-bubble assistant muted">Analyzing clinical context...</div>}
          </div>

          {chatError && <p className="status-text error-text">{chatError}</p>}

          <form
            className="chat-input-row"
            onSubmit={(e) => {
              e.preventDefault();
              onSubmit();
            }}
          >
            <input
              className="text-input"
              type="text"
              placeholder="Ask about trends, abnormalities, and clinical context..."
              disabled={isChatPending}
              value={chatQuestion}
              onChange={(e) => onChangeQuestion(e.target.value)}
            />
            <button className="primary-button" type="submit" disabled={isChatPending || !chatQuestion.trim()}>
              Send
            </button>
          </form>
        </div>

        <aside className="clinical-viz-column">
          <div className="clinical-viz-sticky">
            <div className="clinical-viz-card">
              <h3>Trends</h3>
              {trendData.length > 0 ? (
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart
                    data={trendData}
                    onMouseMove={() => {
                      if (activeMetric) setHoveredMetric(activeMetric);
                    }}
                    onMouseLeave={() => setHoveredMetric(null)}
                  >
                    <CartesianGrid stroke="#3f3a34" strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fill: "#a8a29e", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#a8a29e", fontSize: 11 }} />
                    <Tooltip />
                    <Line dataKey="value" stroke="#d97706" strokeWidth={2} dot={{ r: 3, fill: "#d97706" }} />
                    {activeRange.low !== null && <ReferenceLine y={activeRange.low} stroke="#78716c" strokeDasharray="4 4" />}
                    {activeRange.high !== null && <ReferenceLine y={activeRange.high} stroke="#78716c" strokeDasharray="4 4" />}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="muted-copy">Select a metric with numeric history to view trend.</p>
              )}
            </div>

            <div className="clinical-viz-card">
              <h3>Category Comparison</h3>
              {categoryData.length > 0 ? (
                <ResponsiveContainer width="100%" height={190}>
                  <BarChart data={categoryData}>
                    <CartesianGrid stroke="#3f3a34" strokeDasharray="3 3" />
                    <XAxis dataKey="category" tick={{ fill: "#a8a29e", fontSize: 10 }} interval={0} angle={-22} textAnchor="end" height={65} />
                    <YAxis tick={{ fill: "#a8a29e", fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="alerts" fill="#d97706" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="muted-copy">No abnormal category comparisons available.</p>
              )}
            </div>

            <div className="clinical-viz-card">
              <h3>Range Indicator</h3>
              {activeMetric ? (
                <div className="range-indicator-block">
                  <div className="range-indicator-head">
                    <strong>{activeMetric}</strong>
                    <span>{String(activeRecord?.Result ?? "N/A")}{activeRecord?.Unit ? ` ${activeRecord.Unit}` : ""}</span>
                  </div>
                  <div className="range-bar-track">
                    <div className="range-bar-fill" />
                  </div>
                  <p className="muted-copy">
                    Reference: {activeRecord?.Reference_Range ?? "N/A"}
                  </p>
                </div>
              ) : (
                <p className="muted-copy">Pick a metric from chat cards to view reference range context.</p>
              )}
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}
