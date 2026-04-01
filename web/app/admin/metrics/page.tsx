"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useAuth } from "@/lib/auth-context";
import {
  fetchMetricsDashboard,
  metricStatus,
  type MetricsCard,
  type MetricsDashboard,
} from "@/lib/metrics";

const CARD_ORDER = [
  "json_validity",
  "pdf_processing_success",
  "hallucination_detection",
  "api_reliability",
  "context_retention",
  "extraction_f1",
] as const;

const CARD_TITLES: Record<(typeof CARD_ORDER)[number], string> = {
  json_validity: "JSON Validity Rate",
  pdf_processing_success: "PDF Processing Success",
  hallucination_detection: "Hallucinations Detected",
  api_reliability: "API Reliability",
  context_retention: "Context Retention",
  extraction_f1: "Extraction F1 Score",
};

function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

function formatCardValue(metricKey: string, value: number): string {
  if (metricKey === "hallucination_detection") {
    return String(Math.round(value));
  }
  return formatPercent(value);
}

function asNumber(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function renderCardSubtext(metricKey: string, card: MetricsCard): string {
  const details = card.details ?? {};

  if (metricKey === "json_validity") {
    const display = typeof details.display === "string" ? details.display : null;
    if (display) return display;
    const valid = asNumber(details.valid_calls);
    const total = asNumber(details.total_calls);
    return `${formatPercent(card.current_value)} (${valid}/${total} calls valid)`;
  }

  if (metricKey === "pdf_processing_success") {
    const success = asNumber(details.success_count);
    const total = asNumber(details.total_count);
    const failed = asNumber(details.failed_count);
    const avgFindings = asNumber(details.avg_findings_per_pdf);
    return `${success}/${total} successful, ${failed} failed, avg findings ${avgFindings.toFixed(1)}`;
  }

  if (metricKey === "hallucination_detection") {
    const last100 = asNumber(details.last_100_extractions);
    return `${Math.round(card.current_value)} hallucinations in last ${last100} extractions`;
  }

  if (metricKey === "api_reliability") {
    const p95 = asNumber(details.p95_latency_ms);
    const avg = asNumber(details.avg_latency_ms);
    return `p95 ${Math.round(p95)} ms, avg ${Math.round(avg)} ms`;
  }

  if (metricKey === "context_retention") {
    const runs = asNumber(details.runs);
    const average = asNumber(details.seven_day_average);
    return `${runs} runs, 7-day avg ${formatPercent(average)}`;
  }

  const precision = asNumber(details.precision);
  const recall = asNumber(details.recall);
  return `Precision ${(precision * 100).toFixed(1)}%, Recall ${(recall * 100).toFixed(1)}%`;
}

export default function AdminMetricsPage() {
  const router = useRouter();
  const { user, loading, getToken, isAdmin } = useAuth();

  const [days, setDays] = useState(7);
  const [
    metrics, setMetrics] = useState<MetricsDashboard | null>(null);
  const [isFetching, setIsFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (loading) return;

    if (!user) {
      router.replace("/login");
      return;
    }

    if (!isAdmin) {
      router.replace("/dashboard");
    }
  }, [loading, user, isAdmin, router]);

  useEffect(() => {
    async function loadMetrics(): Promise<void> {
      if (!user || !isAdmin) return;

      setIsFetching(true);
      setError(null);
      try {
        const token = await getToken();
        if (!token) {
          throw new Error("Authentication token unavailable.");
        }
        const data = await fetchMetricsDashboard(token, days);
        setMetrics(data);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Failed to load metrics.";
        setError(message);
      } finally {
        setIsFetching(false);
      }
    }

    void loadMetrics();
  }, [days, user, isAdmin, getToken]);

  const hallucinationCount = useMemo(() => {
    if (!metrics) return 0;
    return asNumber(metrics.cards.hallucination_detection?.current_value);
  }, [metrics]);

  const apiP95LatencyMs = useMemo(() => {
    if (!metrics) return 0;
    return asNumber(metrics.cards.api_reliability?.details?.p95_latency_ms);
  }, [metrics]);

  if (loading || (user && isAdmin && !metrics && isFetching)) {
    return (
      <main className="admin-metrics-shell">
        <div className="auth-loading">Loading metrics...</div>
      </main>
    );
  }

  if (!user || !isAdmin) {
    return (
      <main className="admin-metrics-shell">
        <div className="auth-loading">Redirecting...</div>
      </main>
    );
  }

  return (
    <main className="admin-metrics-shell">
      <header className="admin-metrics-header">
        <div>
          <span className="panel-label">Admin Only</span>
          <h1>Validation Metrics</h1>
          <p>Live quality telemetry for extraction, reliability, and safety.</p>
        </div>
        <div className="admin-metrics-actions">
          <label className="admin-metrics-days">
            <span>Window</span>
            <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
              <option value={7}>7 days</option>
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
            </select>
          </label>
          <button className="secondary-button" type="button" onClick={() => router.push("/dashboard")}>Back to Dashboard</button>
        </div>
      </header>

      {error && <p className="status-text error-text">{error}</p>}

      {hallucinationCount > 0 && (
        <section className="admin-alert-banner">
          Warning: {Math.round(hallucinationCount)} hallucinations were detected in recent extractions.
        </section>
      )}

      {apiP95LatencyMs > 10_000 && (
        <section className="admin-alert-banner">
          Warning: API p95 latency is above 10 seconds ({Math.round(apiP95LatencyMs)} ms).
        </section>
      )}

      <section className="admin-metrics-grid">
        {CARD_ORDER.map((metricKey) => {
          const card = metrics?.cards?.[metricKey];
          if (!card) {
            return null;
          }

          const status = metricStatus(metricKey, card.current_value);
          return (
            <article key={metricKey} className="admin-metric-card">
              <span className="admin-metric-eyebrow">{CARD_TITLES[metricKey]}</span>
              <div className="admin-metric-main-row">
                <p className="admin-metric-value metrics-mono">{formatCardValue(metricKey, card.current_value)}</p>
                <span className={`admin-metric-status admin-metric-status-${status}`} />
              </div>
              <p className="admin-metric-target">Target: {card.target_value}</p>
              <p className="admin-metric-subtext">{renderCardSubtext(metricKey, card)}</p>

              <div className="admin-metric-sparkline">
                <ResponsiveContainer width="100%" height={72}>
                  <LineChart data={card.series}>
                    <XAxis dataKey="date" hide />
                    <YAxis hide domain={["auto", "auto"]} />
                    <Tooltip
                      labelFormatter={(label) => `Date: ${label}`}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="#d97706"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </article>
          );
        })}
      </section>

      <section className="admin-metrics-table-card">
        <h2>Failed PDF Processing Attempts</h2>
        {metrics && metrics.failed_pdfs.length > 0 ? (
          <div className="admin-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Size (KB)</th>
                  <th>Findings</th>
                  <th>Reason</th>
                  <th>Processing Time</th>
                  <th>When</th>
                </tr>
              </thead>
              <tbody>
                {metrics.failed_pdfs.map((row, index) => (
                  <tr key={`${row.filename}-${row.createdAt}-${index}`}>
                    <td>{row.filename}</td>
                    <td className="metrics-mono">{(asNumber(row.fileSize) / 1024).toFixed(1)}</td>
                    <td className="metrics-mono">{Math.max(asNumber(row.findingsCount), 0)}</td>
                    <td>{row.failureReason}</td>
                    <td className="metrics-mono">{Math.round(asNumber(row.processingTimeMs))} ms</td>
                    <td>{new Date(row.createdAt).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted-copy">No failed PDFs recorded in the selected window.</p>
        )}
      </section>

      <section className="admin-metrics-table-card">
        <h2>Token Usage Per Day</h2>
        {metrics && metrics.token_usage_per_day.length > 0 ? (
          <div className="admin-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Input Tokens</th>
                  <th>Output Tokens</th>
                </tr>
              </thead>
              <tbody>
                {metrics.token_usage_per_day.map((point) => (
                  <tr key={point.date}>
                    <td>{point.date}</td>
                    <td className="metrics-mono">{point.input_tokens}</td>
                    <td className="metrics-mono">{point.output_tokens}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted-copy">No token usage recorded yet.</p>
        )}
      </section>

      {process.env.NODE_ENV === "development" && (
        <button
          onClick={async () => {
            const { getAuth } = await import("firebase/auth");
            const token = await getAuth().currentUser?.getIdToken(true);
            if (token) {
              await navigator.clipboard.writeText(token);
              alert('Token copied to clipboard. Paste into terminal as:\nexport TOKEN="<paste>"');
              console.log("Firebase ID Token:", token);
            } else {
              alert("No user signed in");
            }
          }}
          style={{
            position: "fixed",
            bottom: 16,
            right: 16,
            zIndex: 9999,
            background: "#d97706",
            color: "#0c0a09",
            border: "none",
            padding: "8px 14px",
            borderRadius: 8,
            fontSize: 12,
            cursor: "pointer",
            fontWeight: 600,
          }}
          type="button"
        >
          Copy Token
        </button>
      )}
    </main>
  );
}
