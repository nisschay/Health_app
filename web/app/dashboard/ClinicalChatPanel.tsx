"use client";

import { useEffect, useMemo, useState } from "react";
import { Minus, TrendingDown, TrendingUp } from "lucide-react";
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
import { canonicalizeCategory } from "@/lib/categoryMap";
import { normalizeTestName } from "@/lib/testNameMap";

type FindingSeverity = "HIGH" | "BORDERLINE" | "NORMAL";
type MetricTrend = "improving" | "worsening" | "stable" | "insufficient_data";

type FindingEntry = {
  testName: string;
  category: string;
  valueText: string;
  valueNumeric: number | null;
  unit: string;
  referenceRange: string;
  status: string;
  severity: FindingSeverity;
  deviationPercent: number;
  reportDate: string;
  timestamp: number;
  source: MedicalRecord;
};

type PriorityFinding = {
  name: string;
  displayValue: string;
  unit: string;
  referenceRange: string;
  severity: FindingSeverity;
  statusLabel: string;
  tone: "critical" | "low" | "normal";
  worstDate: string;
  latestValue: string;
  latestDate: string;
  latestSeverity: FindingSeverity;
  totalReports: number;
  abnormalCount: number;
  streak: number;
  score: number;
};

type PriorityMetric = {
  name: string;
  latestValue: string;
  latestValueNumeric: number | null;
  latestUnit: string;
  latestDate: string;
  latestStatus: string;
  latestSeverity: FindingSeverity;
  trend: MetricTrend;
  trendDelta: number | null;
  reportCount: number;
  lowerIsBetter: boolean;
  latestEntry: FindingEntry;
};

type StructuredAssistant = {
  summary: string;
  keyFindings: PriorityFinding[];
  metrics: PriorityMetric[];
  trends: string[];
  recommendations: string[];
  totalAlerts: number;
  mostAffectedCategory: string;
};

const LOWER_IS_BETTER_KEYS = [
  "ldl",
  "triglycerides",
  "glucose",
  "hba1c",
  "sgot",
  "sgpt",
  "creatinine",
  "urea",
  "fastingbloodglucose",
  "glycatedhaemoglobinhba1c",
  "aspartateaminotransferasesgotast",
  "alanineaminotransferasesgptalt",
  "serumcreatinine",
  "bloodureanitrogenbun",
];

function normalizeStatus(status: string | null | undefined): string {
  const normalized = String(status ?? "").trim().toLowerCase();
  if (!normalized || normalized === "n/a" || normalized === "na") return "N/A";
  if (normalized.includes("critical")) return "Critical";
  if (normalized.includes("insufficient")) return "Insufficient";
  if (normalized.includes("borderline")) return "Borderline";
  if (normalized.includes("positive")) return "Positive";
  if (normalized.includes("negative")) return "Negative";
  if (normalized.includes("flag")) return "Flagged";
  if (normalized.includes("high")) return "High";
  if (normalized.includes("low")) return "Low";
  if (normalized.includes("normal") || normalized.includes("within")) return "Normal";
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function toSeverity(status: string): FindingSeverity {
  const normalized = normalizeStatus(status).toLowerCase();
  if (normalized === "critical" || normalized === "high" || normalized === "positive" || normalized === "flagged") {
    return "HIGH";
  }
  if (normalized === "low" || normalized === "insufficient" || normalized === "borderline") {
    return "BORDERLINE";
  }
  return "NORMAL";
}

function isAbnormalStatus(status: string | null | undefined): boolean {
  return toSeverity(String(status ?? "")) !== "NORMAL";
}

function toTimestamp(dateText: string | null | undefined): number {
  const parsed = parseMedicalDate(dateText ?? "");
  return parsed === Number.MAX_SAFE_INTEGER ? -1 : parsed;
}

function parseNumeric(value: string | number | null | undefined): number | null {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (value === null || value === undefined) return null;
  const parsed = Number.parseFloat(String(value).replace(/,/g, "").trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function parseRefRange(ref: string | null | undefined): { low: number | null; high: number | null } {
  if (!ref) return { low: null, high: null };
  const value = String(ref);
  const rangeMatch = value.match(/([0-9.]+)\s*[-–]\s*([0-9.]+)/);
  if (rangeMatch) {
    return { low: Number.parseFloat(rangeMatch[1]), high: Number.parseFloat(rangeMatch[2]) };
  }
  const ltMatch = value.match(/[<≤]\s*([0-9.]+)/);
  if (ltMatch) {
    return { low: null, high: Number.parseFloat(ltMatch[1]) };
  }
  const gtMatch = value.match(/[>≥]\s*([0-9.]+)/);
  if (gtMatch) {
    return { low: Number.parseFloat(gtMatch[1]), high: null };
  }
  return { low: null, high: null };
}

function calculateDeviationPercent(
  valueNumeric: number | null,
  referenceRange: string | null | undefined,
  severity: FindingSeverity,
): number {
  if (valueNumeric === null) {
    if (severity === "HIGH") return 20;
    if (severity === "BORDERLINE") return 8;
    return 0;
  }

  const ref = parseRefRange(referenceRange);
  if (ref.low !== null && ref.high !== null) {
    if (valueNumeric < ref.low && ref.low !== 0) return Math.round(((ref.low - valueNumeric) / Math.abs(ref.low)) * 100);
    if (valueNumeric > ref.high && ref.high !== 0) return Math.round(((valueNumeric - ref.high) / Math.abs(ref.high)) * 100);
    return 0;
  }
  if (ref.high !== null && valueNumeric > ref.high && ref.high !== 0) {
    return Math.round(((valueNumeric - ref.high) / Math.abs(ref.high)) * 100);
  }
  if (ref.low !== null && valueNumeric < ref.low && ref.low !== 0) {
    return Math.round(((ref.low - valueNumeric) / Math.abs(ref.low)) * 100);
  }

  if (severity === "HIGH") return 20;
  if (severity === "BORDERLINE") return 8;
  return 0;
}

function formatValue(value: string | number | null | undefined, unit: string | null | undefined): string {
  const text = value === null || value === undefined || String(value).trim() === "" ? "N/A" : String(value);
  const unitText = unit ? ` ${unit}` : "";
  return `${text}${unitText}`.trim();
}

function matchKey(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function isLowerBetterMetric(testName: string): boolean {
  const key = matchKey(testName);
  return LOWER_IS_BETTER_KEYS.some((candidate) => key.includes(candidate));
}

function buildFindingEntries(records: MedicalRecord[]): FindingEntry[] {
  return records
    .map((record) => {
      const testName = normalizeTestName(record.Test_Name ?? record.Original_Test_Name ?? "Unknown Test");
      const status = normalizeStatus(record.Status);
      const severity = toSeverity(status);
      const valueNumeric = parseNumeric(record.Result_Numeric ?? record.Result);
      const deviationPercent = calculateDeviationPercent(valueNumeric, record.Reference_Range, severity);

      return {
        testName,
        category: canonicalizeCategory(record.Test_Category),
        valueText: formatValue(record.Result, record.Unit),
        valueNumeric,
        unit: record.Unit ?? "",
        referenceRange: record.Reference_Range ?? "N/A",
        status,
        severity,
        deviationPercent,
        reportDate: record.Test_Date ?? "Unknown",
        timestamp: toTimestamp(record.Test_Date),
        source: record,
      };
    })
    .filter((entry) => entry.testName && entry.testName !== "Unknown Test");
}

function severityRank(severity: FindingSeverity): number {
  if (severity === "HIGH") return 3;
  if (severity === "BORDERLINE") return 2;
  return 1;
}

function toneForSeverity(severity: FindingSeverity): "critical" | "low" | "normal" {
  if (severity === "HIGH") return "critical";
  if (severity === "BORDERLINE") return "low";
  return "normal";
}

function buildPriorityFindings(entries: FindingEntry[]): PriorityFinding[] {
  const grouped = new Map<string, FindingEntry[]>();
  for (const entry of entries) {
    if (!grouped.has(entry.testName)) {
      grouped.set(entry.testName, []);
    }
    grouped.get(entry.testName)!.push(entry);
  }

  const scored = Array.from(grouped.entries()).map(([name, groupEntries]) => {
    const sortedByDate = [...groupEntries].sort((a, b) => b.timestamp - a.timestamp);
    const abnormalEntries = groupEntries.filter((entry) => entry.severity !== "NORMAL");
    const latestEntry = sortedByDate[0]!;

    const mostDeviant = [...groupEntries].sort((a, b) => {
      if (b.deviationPercent !== a.deviationPercent) return b.deviationPercent - a.deviationPercent;
      return severityRank(b.severity) - severityRank(a.severity);
    })[0]!;

    let streak = 0;
    for (const entry of sortedByDate) {
      if (entry.severity !== "NORMAL") streak += 1;
      else break;
    }

    const severityScore = mostDeviant.severity === "HIGH" ? 100 : mostDeviant.severity === "BORDERLINE" ? 50 : 10;
    const score = severityScore + streak * 25 + mostDeviant.deviationPercent;

    return {
      name,
      displayValue: mostDeviant.valueText,
      unit: mostDeviant.unit,
      referenceRange: mostDeviant.referenceRange,
      severity: mostDeviant.severity,
      statusLabel: mostDeviant.status,
      tone: toneForSeverity(mostDeviant.severity),
      worstDate: mostDeviant.reportDate,
      latestValue: latestEntry.valueText,
      latestDate: latestEntry.reportDate,
      latestSeverity: latestEntry.severity,
      totalReports: groupEntries.length,
      abnormalCount: abnormalEntries.length,
      streak,
      score,
    } satisfies PriorityFinding;
  });

  return scored
    .filter((finding) => finding.abnormalCount > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 8);
}

function trendThreshold(referenceRange: string, baseline: number): number {
  const ref = parseRefRange(referenceRange);
  if (ref.low !== null && ref.high !== null) {
    const width = Math.abs(ref.high - ref.low);
    if (width > 0) return width * 0.05;
  }
  if (ref.high !== null && ref.high !== 0) return Math.abs(ref.high) * 0.05;
  if (ref.low !== null && ref.low !== 0) return Math.abs(ref.low) * 0.05;
  return Math.max(Math.abs(baseline) * 0.05, 0.5);
}

function buildPriorityMetrics(entries: FindingEntry[]): PriorityMetric[] {
  const grouped = new Map<string, FindingEntry[]>();
  for (const entry of entries) {
    if (!grouped.has(entry.testName)) {
      grouped.set(entry.testName, []);
    }
    grouped.get(entry.testName)!.push(entry);
  }

  const metrics = Array.from(grouped.entries()).map(([name, groupEntries]) => {
    const sortedByDate = [...groupEntries].sort((a, b) => b.timestamp - a.timestamp);
    const latestEntry = sortedByDate[0]!;
    const previousEntry = sortedByDate[1];
    const lowerIsBetter = isLowerBetterMetric(name);

    let trend: MetricTrend = "insufficient_data";
    let trendDelta: number | null = null;

    if (latestEntry.valueNumeric !== null && previousEntry && previousEntry.valueNumeric !== null) {
      trendDelta = Number((latestEntry.valueNumeric - previousEntry.valueNumeric).toFixed(2));
      const threshold = trendThreshold(latestEntry.referenceRange, previousEntry.valueNumeric);
      if (Math.abs(trendDelta) <= threshold) {
        trend = "stable";
      } else if (lowerIsBetter) {
        trend = trendDelta < 0 ? "improving" : "worsening";
      } else {
        trend = trendDelta > 0 ? "improving" : "worsening";
      }
    }

    return {
      name,
      latestValue: latestEntry.valueText,
      latestValueNumeric: latestEntry.valueNumeric,
      latestUnit: latestEntry.unit,
      latestDate: latestEntry.reportDate,
      latestStatus: latestEntry.status,
      latestSeverity: latestEntry.severity,
      trend,
      trendDelta,
      reportCount: groupEntries.length,
      lowerIsBetter,
      latestEntry,
    } satisfies PriorityMetric;
  });

  return metrics
    .sort((a, b) => {
      const abnormalA = a.latestSeverity !== "NORMAL" ? 1 : 0;
      const abnormalB = b.latestSeverity !== "NORMAL" ? 1 : 0;
      if (abnormalA !== abnormalB) return abnormalB - abnormalA;

      const worseningA = a.trend === "worsening" ? 1 : 0;
      const worseningB = b.trend === "worsening" ? 1 : 0;
      if (worseningA !== worseningB) return worseningB - worseningA;

      return b.reportCount - a.reportCount;
    })
    .slice(0, 6);
}

function buildStructuredAssistant(records: MedicalRecord[]): StructuredAssistant {
  const entries = buildFindingEntries(records);
  const findings = buildPriorityFindings(entries);
  const metrics = buildPriorityMetrics(entries);
  const totalAlerts = entries.filter((entry) => entry.severity !== "NORMAL").length;

  const categoryAlerts = new Map<string, number>();
  for (const entry of entries) {
    if (entry.severity === "NORMAL") continue;
    categoryAlerts.set(entry.category, (categoryAlerts.get(entry.category) ?? 0) + 1);
  }

  const mostAffectedCategory = Array.from(categoryAlerts.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "No category alerts";
  const uniqueDates = new Set(entries.map((entry) => entry.reportDate)).size;

  const recommendations = [
    findings.some((finding) => finding.streak >= 2)
      ? "Persistent abnormalities detected across consecutive reports - prioritize clinician review."
      : "No sustained multi-report abnormal streaks detected in top findings.",
    metrics.some((metric) => metric.trend === "worsening")
      ? "Some key metrics are worsening compared with the previous reading."
      : "Most tracked metrics are stable or improving versus previous readings.",
    "Use the latest and worst values together to separate current state from historical risk.",
  ];

  const trends = [
    findings.some((finding) => finding.streak >= 2)
      ? "At least one high-priority marker remains abnormal across consecutive reports."
      : "No major persistent abnormal streak among the top-ranked findings.",
    metrics.some((metric) => metric.trend === "worsening")
      ? "Worsening movement is present in priority metrics."
      : "Priority metrics show stable or improving movement.",
  ];

  return {
    summary: `Cross-report review generated from ${records.length} records across ${uniqueDates} report dates.`,
    keyFindings: findings,
    metrics,
    trends,
    recommendations,
    totalAlerts,
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
    if (!isAbnormalStatus(record.Status)) continue;
    const category = canonicalizeCategory(record.Test_Category);
    map.set(category, (map.get(category) ?? 0) + 1);
  }
  return Array.from(map.entries())
    .map(([category, alerts]) => ({ category, alerts }))
    .sort((a, b) => b.alerts - a.alerts);
}

function badgeTone(status: string) {
  const s = normalizeStatus(status).toLowerCase();
  if (s === "critical") return "bad";
  if (s === "high" || s === "positive" || s === "flagged") return "warn";
  if (s === "low" || s === "insufficient" || s === "borderline") return "muted";
  if (s === "normal" || s === "negative") return "good";
  return "muted";
}

function deltaClass(metric: PriorityMetric): string {
  if (metric.trend === "improving") return "improving";
  if (metric.trend === "worsening") return "worsening";
  return "stable";
}

function deltaText(metric: PriorityMetric): string {
  if (metric.trend === "insufficient_data" || metric.trendDelta === null) return "insufficient data";
  const sign = metric.trendDelta >= 0 ? "+" : "";
  return `${sign}${metric.trendDelta.toFixed(1)} from last`;
}

function CategoryTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "#292524",
        border: "1px solid #44403c",
        borderRadius: "8px",
        padding: "8px 12px",
        fontFamily: "DM Sans",
      }}
    >
      <p style={{ fontSize: 12, color: "#78716c", margin: 0, marginBottom: 4 }}>{label}</p>
      <p
        style={{
          margin: 0,
          fontSize: 14,
          fontFamily: "JetBrains Mono",
          color: "#fbbf24",
          fontWeight: 600,
        }}
      >
        {payload[0]?.value ?? 0} alerts
      </p>
    </div>
  );
}

export default function ClinicalChatPanel({
  records,
}: {
  records: MedicalRecord[];
}) {
  const analysisData = useMemo(() => buildStructuredAssistant(records), [records]);

  const [activeMetric, setActiveMetric] = useState<string | null>(analysisData.metrics[0]?.name ?? null);

  const trendData = useMemo(() => alertTrendByReportDate(records), [records]);
  const categoryData = useMemo(() => categoryComparison(records), [records]);
  const activeMetricData = useMemo(() => analysisData.metrics.find((metric) => metric.name === activeMetric), [analysisData.metrics, activeMetric]);
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

  useEffect(() => {
    if (analysisData.metrics.length === 0) {
      setActiveMetric(null);
      return;
    }
    if (!activeMetric || !analysisData.metrics.some((metric) => metric.name === activeMetric)) {
      setActiveMetric(analysisData.metrics[0]!.name);
    }
  }, [analysisData.metrics, activeMetric]);

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

  const categoryChartMinWidth = Math.max(420, categoryData.length * 72);

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
                <div className={`intelligence-finding-card ${finding.tone}`} role="listitem" key={`${finding.name}-${finding.worstDate}-${idx}`}>
                  <div className="intelligence-finding-main">
                    <strong>{finding.name}</strong>
                    <span className="finding-caption">WORST RECORDED</span>
                    <p>{finding.displayValue} · {finding.worstDate}</p>
                    {finding.latestDate !== finding.worstDate && (
                      <small className={`finding-latest ${finding.latestSeverity === "NORMAL" ? "good" : "warn"}`}>
                        Latest: {finding.latestValue} · {finding.latestDate}
                      </small>
                    )}
                    <small>Ref: {finding.referenceRange}</small>
                    <div className="finding-footer">
                      {finding.streak >= 2 ? (
                        <span className="finding-streak streak-amber">
                          <span className="streak-dot" />
                          {finding.streak}x consecutive
                        </span>
                      ) : finding.streak === 1 ? (
                        <span className="finding-streak">{finding.abnormalCount} of {finding.totalReports} reports flagged</span>
                      ) : (
                        <span className="finding-streak streak-good">Currently normal</span>
                      )}
                    </div>
                  </div>
                  <span className={`status-pill ${badgeTone(finding.statusLabel)}`}>{finding.statusLabel}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="intelligence-block">
            <div className="intelligence-block-head">
              <h3>Priority Metrics</h3>
              <span>Latest values ranked by risk and trend</span>
            </div>
            <div className="metric-chip-row">
              {analysisData.metrics.length === 0 && <span className="muted-copy">No highlighted metrics.</span>}
              {analysisData.metrics.map((metric, metricIndex) => (
                <button
                  type="button"
                  key={`${metric.name}-${metric.latestDate}-${metricIndex}`}
                  className={`metric-chip ${activeMetric === metric.name ? "active" : ""}`}
                  onClick={() => setActiveMetric(metric.name)}
                >
                  <div className="metric-chip-head">
                    <span className="metric-chip-name">{metric.name}</span>
                    {metric.trend !== "insufficient_data" && metric.trend === "stable" && <Minus size={14} className="metric-trend-icon stable" />}
                    {metric.trend === "improving" && metric.lowerIsBetter && <TrendingDown size={14} className="metric-trend-icon improving" />}
                    {metric.trend === "improving" && !metric.lowerIsBetter && <TrendingUp size={14} className="metric-trend-icon improving" />}
                    {metric.trend === "worsening" && metric.lowerIsBetter && <TrendingUp size={14} className="metric-trend-icon worsening" />}
                    {metric.trend === "worsening" && !metric.lowerIsBetter && <TrendingDown size={14} className="metric-trend-icon worsening" />}
                  </div>
                  <strong>{metric.latestValue}</strong>
                  <div className="metric-chip-status-row">
                    <em className={`status-pill ${badgeTone(metric.latestStatus)}`}>{metric.latestStatus}</em>
                    <span className={`metric-chip-delta ${deltaClass(metric)}`}>{deltaText(metric)}</span>
                  </div>
                  <small className="metric-chip-readings">{metric.reportCount} readings</small>
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
              <div className="category-chart-scroll" style={{ overflowX: categoryData.length > 6 ? "auto" : "visible" }}>
                <div style={{ minWidth: categoryData.length > 6 ? `${categoryChartMinWidth}px` : "100%", height: "210px" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={categoryData}>
                      <CartesianGrid stroke="#3f3a34" strokeDasharray="3 3" />
                      <XAxis
                        dataKey="category"
                        tick={{ fill: "#78716c", fontSize: 11, fontFamily: "DM Sans" }}
                        tickLine={false}
                        axisLine={false}
                        interval={0}
                        angle={-35}
                        textAnchor="end"
                        height={60}
                        tickFormatter={(value: string) => value.length > 10 ? `${value.slice(0, 10)}...` : value}
                      />
                      <YAxis tick={{ fill: "#a8a29e", fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip content={<CategoryTooltip />} cursor={{ fill: "#ffffff08" }} />
                      <Bar dataKey="alerts" fill="#d97706" radius={[4, 4, 0, 0]} activeBar={{ fill: "#fbbf24" }} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <p className="muted-copy">No abnormal category comparisons available.</p>
            )}
          </div>

          <div className="intelligence-viz-card">
            <h3>Reference Context</h3>
            {activeMetricData ? (
              <div className="range-indicator-block">
                <div className="range-indicator-head">
                  <strong>{activeMetricData.name}</strong>
                  <span>{activeMetricData.latestValue}</span>
                </div>
                <div className="range-bar-track">
                  <div className="range-bar-fill" />
                </div>
                <p className="muted-copy">
                  Reference: {activeMetricData.latestEntry.referenceRange || "N/A"}
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
