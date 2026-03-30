"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MedicalRecord } from "@/lib/api";
import { parseMedicalDate } from "@/lib/clinical";

type StructuredAssistant = {
  summary: string;
  keyFindings: Array<{
    testName: string;
    status: string;
    date: string;
    reference: string;
    value: string;
  }>;
  metrics: Array<{ label: string; value: string; status: string; testName: string }>;
  trends: string[];
  recommendations: string[];
  totalAlerts: number;
  mostAffectedCategory: string;
};

function isAbnormalStatus(status: string | null | undefined): boolean {
  const normalized = String(status ?? "").toLowerCase();
  return normalized === "high" || normalized === "low" || normalized === "critical" || normalized === "positive" || normalized === "flagged" || normalized === "insufficient";
}

function parseAssistantResponse(content: string, records: MedicalRecord[]): StructuredAssistant {
  const lines = content
    .split("\n")
    .map((line) => line.replace(/^[-*]\s*/, "").trim())
    .filter(Boolean);

  const summary = lines.slice(0, 2).join(" ") || "Clinical review generated based on available report data.";

  const abnormalRecords = records
    .filter((record) => isAbnormalStatus(record.Status))
    .sort((a, b) => {
      const rank = (status: string | null | undefined) => {
        const s = (status ?? "").toLowerCase();
        if (s === "critical") return 0;
        if (s === "high" || s === "positive" || s === "flagged") return 1;
        if (s === "low" || s === "insufficient") return 2;
        return 3;
      };
      return rank(a.Status) - rank(b.Status);
    });

  const keyFindings = abnormalRecords.slice(0, 8).map((record) => ({
    testName: record.Test_Name ?? "Unknown Test",
    status: record.Status ?? "N/A",
    date: record.Test_Date ?? "Unknown",
    reference: record.Reference_Range ?? "N/A",
    value: `${String(record.Result ?? "N/A")}${record.Unit ? ` ${record.Unit}` : ""}`,
  }));

  const categoryMap = new Map<string, number>();
  for (const record of abnormalRecords) {
    const category = record.Test_Category?.trim() || "Uncategorized";
    categoryMap.set(category, (categoryMap.get(category) ?? 0) + 1);
  }

  const mostAffectedCategory = Array.from(categoryMap.entries())
    .sort((a, b) => b[1] - a[1])[0]?.[0] ?? "No category alerts";

  const recommendations = lines
    .filter((line) => /recommend|follow|monitor|discuss|repeat|consult/i.test(line))
    .slice(0, 3);

  const trends = lines
    .filter((line) => /trend|increase|decrease|stable|over time/i.test(line))
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
    keyFindings,
    metrics,
    trends,
    recommendations:
      recommendations.length > 0 ? recommendations : ["Discuss flagged trends with your clinician for interpretation."],
    totalAlerts: abnormalRecords.length,
    mostAffectedCategory,
  };
}

function alertTrendByReportDate(records: MedicalRecord[]) {
  const byDate = new Map<string, { date: string; alertCount: number; totalCount: number }>();

  for (const record of records) {
    const date = String(record.Test_Date ?? "Unknown").trim() || "Unknown";
    if (!byDate.has(date)) {
      byDate.set(date, { date, alertCount: 0, totalCount: 0 });
    }
    const bucket = byDate.get(date)!;
    bucket.totalCount += 1;
    if (isAbnormalStatus(record.Status)) {
      bucket.alertCount += 1;
    }
  }

  return Array.from(byDate.values()).sort((a, b) => parseMedicalDate(a.date) - parseMedicalDate(b.date));
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
  if (s === "normal" || s === "negative") return "good";
  return "muted";
}

function findingTone(status: string) {
  const s = status.toLowerCase();
  if (s === "critical" || s === "high" || s === "positive" || s === "flagged") return "critical";
  if (s === "low" || s === "insufficient") return "low";
  return "normal";
}

export default function ClinicalChatPanel({
  records,
}: {
  records: MedicalRecord[];
}) {
  const analysisData = useMemo(
    () => parseAssistantResponse("", records),
    [records]
  );

  const [activeMetric, setActiveMetric] = useState<string | null>(analysisData.metrics[0]?.testName ?? null);
  const [hoveredMetric, setHoveredMetric] = useState<string | null>(null);

  const trendData = useMemo(() => alertTrendByReportDate(records), [records]);
  const categoryData = useMemo(() => categoryComparison(records), [records]);
  const activeRecord = useMemo(
    () => {
      if (!activeMetric) return undefined;
      return [...records]
        .filter((row) => row.Test_Name === activeMetric)
        .sort((a, b) => parseMedicalDate(b.Test_Date) - parseMedicalDate(a.Test_Date))[0];
    },
    [records, activeMetric]
  );
  const criticalCount = useMemo(
    () => records.filter((row) => String(row.Status ?? "").toLowerCase() === "critical").length,
    [records],
  );
  const stableCount = useMemo(
    () => records.filter((row) => {
      const normalized = String(row.Status ?? "").toLowerCase();
      return normalized === "normal" || normalized === "negative";
    }).length,
    [records],
  );
  const alertRate = records.length > 0 ? Math.round((analysisData.totalAlerts / records.length) * 100) : 0;
  const clinicalState = useMemo(() => {
    if (criticalCount > 0) {
      return {
        tone: "critical",
        label: "Attention Required",
        note: `${criticalCount} critical findings need immediate review.`,
      };
    }
    if (alertRate >= 30) {
      return {
        tone: "watch",
        label: "Monitor Closely",
        note: `${alertRate}% of results are outside reference status.`,
      };
    }
    return {
      tone: "stable",
      label: "Generally Stable",
      note: "No critical spikes detected in the current records.",
    };
  }, [alertRate, criticalCount]);
  const trendNarrative = useMemo(() => {
    if (trendData.length === 0) {
      return "No report-date trend data is available yet.";
    }
    if (trendData.length === 1) {
      return "Only one report date is available. Add more report dates to compare alert movement over time.";
    }
    const first = trendData[0]!.alertCount;
    const last = trendData[trendData.length - 1]!.alertCount;
    const delta = last - first;
    const pct = first === 0 ? (last > 0 ? 100 : 0) : Math.round((delta / Math.abs(first)) * 100);

    if (delta === 0) {
      return "Alert count is stable across report dates.";
    }
    if (delta > 0) {
      return `Alert burden is increasing over time (${pct}% change from the earliest report date).`;
    }
    return `Alert burden is decreasing over time (${Math.abs(pct)}% change from the earliest report date).`;
  }, [trendData]);

  return (
    <section className="result-section intelligence-section">
      <div className="intelligence-header">
        <div className="intelligence-title-wrap">
          <span className="intelligence-kicker">Clinical Intelligence</span>
          <h2>Clinical Intelligence Panel</h2>
          <p>{analysisData.summary}</p>
        </div>
        <article className={`intelligence-state-card ${clinicalState.tone}`}>
          <span className="intelligence-state-label">Current Assessment</span>
          <strong>{clinicalState.label}</strong>
          <p>{clinicalState.note}</p>
        </article>
      </div>

      <div className="intelligence-stat-strip">
        <article className="intelligence-stat-card">
          <span>Total Alerts</span>
          <strong>{analysisData.totalAlerts}</strong>
        </article>
        <article className="intelligence-stat-card">
          <span>Most Affected Category</span>
          <strong>{analysisData.mostAffectedCategory}</strong>
        </article>
        <article className="intelligence-stat-card">
          <span>Stable Results</span>
          <strong>{stableCount}</strong>
        </article>
        <article className="intelligence-stat-card">
          <span>Alert Rate</span>
          <strong>{alertRate}%</strong>
        </article>
      </div>

      <div className="intelligence-grid">
        <div className="intelligence-main-column">
          <section className="intelligence-block">
            <div className="intelligence-block-head">
              <h3>Priority Findings</h3>
              <span>{analysisData.keyFindings.length} flagged observations</span>
            </div>

            <div className="intelligence-findings-grid" role="list">
              {analysisData.keyFindings.length === 0 && (
                <div className="intelligence-finding-card" role="listitem">
                  <strong>No abnormal findings were detected in the current record set.</strong>
                </div>
              )}

              {analysisData.keyFindings.map((finding, idx) => (
                <div className={`intelligence-finding-card ${findingTone(finding.status)}`} role="listitem" key={`${finding.testName}-${finding.date}-${idx}`}>
                  <div className="intelligence-finding-main">
                    <strong>{finding.testName}</strong>
                    <p>{finding.value}</p>
                    <small>{finding.date} | Ref: {finding.reference}</small>
                  </div>
                  <span className={`status-pill ${badgeTone(finding.status)}`}>{finding.status}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="intelligence-block">
            <div className="intelligence-block-head">
              <h3>Priority Metrics</h3>
              <span>Select a metric to update charts</span>
            </div>
            <div className="metric-chip-row">
              {analysisData.metrics.length === 0 && <span className="muted-copy">No highlighted metrics.</span>}
              {analysisData.metrics.map((metric, metricIndex) => (
                <button
                  type="button"
                  key={`${metric.testName}-${metric.status}-${metric.value}-${metricIndex}`}
                  className={`metric-chip ${activeMetric === metric.testName ? "active" : ""} ${hoveredMetric === metric.testName ? "linked" : ""}`}
                  onClick={() => setActiveMetric(metric.testName)}
                  onMouseEnter={() => setHoveredMetric(metric.testName)}
                  onMouseLeave={() => setHoveredMetric(null)}
                >
                  <span>{metric.label}</span>
                  <strong>{metric.value}</strong>
                  <em className={`status-pill ${badgeTone(metric.status)}`}>{metric.status}</em>
                </button>
              ))}
            </div>
          </section>

          <section className="intelligence-block intelligence-advice-grid">
            <article>
              <h3>Recommendations</h3>
              <ul>
                {analysisData.recommendations.map((rec, i) => (
                  <li key={`rec-${i}`}>{rec}</li>
                ))}
              </ul>
            </article>
            <article>
              <h3>Trend Note</h3>
              <p>{trendNarrative}</p>
              {analysisData.trends.length > 0 && (
                <ul>
                  {analysisData.trends.map((trend, i) => (
                    <li key={`trend-${i}`}>{trend}</li>
                  ))}
                </ul>
              )}
            </article>
          </section>
        </div>

        <aside className="intelligence-viz-column">
          <div className="intelligence-viz-card">
            <h3>Metric Trend</h3>
            {trendData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart
                  data={trendData}
                  onMouseMove={() => {
                    if (activeMetric) setHoveredMetric(activeMetric);
                  }}
                  onMouseLeave={() => setHoveredMetric(null)}
                >
                  <CartesianGrid stroke="#3f3a34" strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fill: "#a8a29e", fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fill: "#a8a29e", fontSize: 11 }} />
                  <Tooltip formatter={(value) => [`${String(value ?? 0)} alerts`, "Alert Count"]} />
                  <Line dataKey="alertCount" stroke="#d97706" strokeWidth={2.5} dot={{ r: 3, fill: "#d97706" }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="muted-copy">No report-date alert trend is available yet.</p>
            )}
          </div>

          <div className="intelligence-viz-card">
            <h3>Category Comparison</h3>
            {categoryData.length > 0 ? (
              <ResponsiveContainer width="100%" height={210}>
                <BarChart data={categoryData}>
                  <CartesianGrid stroke="#3f3a34" strokeDasharray="3 3" />
                  <XAxis dataKey="category" tick={{ fill: "#a8a29e", fontSize: 10 }} interval={0} angle={-22} textAnchor="end" height={65} />
                  <YAxis tick={{ fill: "#a8a29e", fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="alerts" fill="#d97706" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="muted-copy">No abnormal category comparisons available.</p>
            )}
          </div>

          <div className="intelligence-viz-card">
            <h3>Reference Context</h3>
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
              <p className="muted-copy">Pick a metric from Priority Metrics to view range context.</p>
            )}
          </div>
        </aside>
      </div>
    </section>
  );
}
