import { normalizeTestName } from "./testNameMap";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ??
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "/backend";

export type MetricName =
  | "json_validity"
  | "pdf_processing_success"
  | "hallucination_detection"
  | "api_reliability"
  | "context_retention"
  | "extraction_f1";

export interface MetricEvent {
  metricName: MetricName;
  value: number;
  metadata: Record<string, unknown>;
  timestamp: Date;
  userId?: string;
}

export type MetricSeriesPoint = {
  date: string;
  value: number;
};

export type MetricsCard = {
  metric_name: string;
  current_value: number;
  target_value: string;
  series: MetricSeriesPoint[];
  details: Record<string, unknown>;
};

export type FailedPdfEntry = {
  filename: string;
  fileSize: number;
  findingsCount?: number;
  processingTimeMs: number;
  failureReason: string;
  createdAt: string;
};

export type TokenUsagePoint = {
  date: string;
  input_tokens: number;
  output_tokens: number;
};

export type MetricsDashboard = {
  generated_at: string;
  cards: Record<string, MetricsCard>;
  failed_pdfs: FailedPdfEntry[];
  token_usage_per_day: TokenUsagePoint[];
};

export type MetricUiStatus = "ok" | "warn" | "fail";

export interface LogMetricOptions {
  token: string;
  throwOnError?: boolean;
  signal?: AbortSignal;
}

export interface MetricLogResult {
  ok: boolean;
  error?: string;
}

export interface Finding {
  testName: string;
  value: number | null;
  unit?: string;
}

export interface HallucinationReport {
  hasHallucination: boolean;
  count: number;
  details: string[];
}

export interface ContextRetentionTurn {
  userMsg: string;
  mustContainOneOf: string[];
}

export interface ContextRetentionMessage {
  role: "user" | "assistant";
  content: string;
}

export type ClinicalAssistantCaller = (args: {
  message: string;
  history: ContextRetentionMessage[];
  analysisId: string;
}) => Promise<string>;

export interface ContextRetentionResult {
  score: number;
  retentionScore: number;
  totalTurns: number;
  transcript: ContextRetentionMessage[];
}

export interface GroundTruthFinding {
  testName: string;
  value: number;
  unit: string;
}

export interface GroundTruthEntry {
  pdfFile: string;
  expectedFindings: GroundTruthFinding[];
}

export interface ExtractionFinding {
  testName: string;
  value: number;
  unit?: string;
}

export interface ExtractionF1Result {
  precision: number;
  recall: number;
  f1: number;
  truePositives: number;
  falsePositives: number;
  falseNegatives: number;
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = "Request failed.";
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function detectHallucination(
  originalText: string,
  extractedFindings: Finding[],
): HallucinationReport {
  const hallucinated: string[] = [];

  for (const finding of extractedFindings) {
    if (finding.value === null || !Number.isFinite(Number(finding.value))) {
      continue;
    }

    const numericValue = Number(finding.value);
    const valueStr = String(finding.value);
    const valuePattern = new RegExp(
      `${escapeRegex(valueStr)}|${escapeRegex(numericValue.toFixed(1))}|${escapeRegex(numericValue.toFixed(2))}`,
    );

    if (!valuePattern.test(originalText)) {
      const unit = finding.unit ? ` ${finding.unit}` : "";
      hallucinated.push(`${finding.testName}: ${finding.value}${unit} NOT FOUND in source text`);
    }
  }

  return {
    hasHallucination: hallucinated.length > 0,
    count: hallucinated.length,
    details: hallucinated,
  };
}

export async function logMetric(
  event: Omit<MetricEvent, "timestamp"> & { timestamp?: Date },
  options: LogMetricOptions,
): Promise<MetricLogResult> {
  const payload = {
    metric_name: event.metricName,
    value: event.value,
    metadata: {
      ...event.metadata,
      timestamp: (event.timestamp ?? new Date()).toISOString(),
      userId: event.userId ?? null,
    },
  };

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/metrics/log`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${options.token}`,
      },
      body: JSON.stringify(payload),
      signal: options.signal,
    });

    await parseJson<Record<string, unknown>>(response);
    return { ok: true };
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Metric logging failed.";
    console.warn(`[Metrics] Failed to log ${event.metricName}: ${message}`);
    if (options.throwOnError) {
      throw new Error(message);
    }
    return { ok: false, error: message };
  }
}

export async function fetchMetricsDashboard(
  token: string,
  days = 7,
): Promise<MetricsDashboard> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/admin/metrics?days=${encodeURIComponent(String(days))}`,
    {
      headers: {
        Authorization: `Bearer ${token}`,
      },
      cache: "no-store",
    },
  );

  return parseJson<MetricsDashboard>(response);
}

export function metricStatus(metricKey: string, value: number): MetricUiStatus {
  if (metricKey === "json_validity") {
    if (value >= 100) return "ok";
    if (value >= 95) return "warn";
    return "fail";
  }
  if (metricKey === "pdf_processing_success") {
    if (value >= 95) return "ok";
    if (value >= 90) return "warn";
    return "fail";
  }
  if (metricKey === "hallucination_detection") {
    if (value <= 0) return "ok";
    if (value <= 1) return "warn";
    return "fail";
  }
  if (metricKey === "api_reliability") {
    if (value >= 99) return "ok";
    if (value >= 97) return "warn";
    return "fail";
  }
  if (metricKey === "context_retention") {
    if (value >= 90) return "ok";
    if (value >= 80) return "warn";
    return "fail";
  }
  if (metricKey === "extraction_f1") {
    if (value >= 95) return "ok";
    if (value >= 90) return "warn";
    return "fail";
  }
  return "warn";
}

export async function evaluateContextRetentionScore(args: {
  conversation: ContextRetentionTurn[];
  analysisId: string;
  callClinicalAssistant: ClinicalAssistantCaller;
}): Promise<ContextRetentionResult> {
  const { conversation, analysisId, callClinicalAssistant } = args;
  const transcript: ContextRetentionMessage[] = [];
  let retentionScore = 0;

  for (const turn of conversation) {
    const response = await callClinicalAssistant({
      message: turn.userMsg,
      history: transcript,
      analysisId,
    });

    const passed = turn.mustContainOneOf.some((keyword) =>
      response.toLowerCase().includes(keyword.toLowerCase()),
    );
    if (passed) {
      retentionScore += 1;
    }

    transcript.push({ role: "user", content: turn.userMsg });
    transcript.push({ role: "assistant", content: response });
  }

  const totalTurns = conversation.length;
  const score = totalTurns > 0 ? (retentionScore / totalTurns) * 100 : 0;

  return {
    score,
    retentionScore,
    totalTurns,
    transcript,
  };
}

function normalizedName(testName: string): string {
  return normalizeTestName(testName).toLowerCase().trim();
}

function isMatch(extracted: ExtractionFinding, expected: GroundTruthFinding): boolean {
  return (
    normalizedName(extracted.testName) === normalizedName(expected.testName)
    && Math.abs(extracted.value - expected.value) < 0.01
  );
}

export function calculateF1(
  extracted: ExtractionFinding[],
  groundTruth: GroundTruthFinding[],
): ExtractionF1Result {
  if (extracted.length === 0 && groundTruth.length === 0) {
    return {
      precision: 1,
      recall: 1,
      f1: 1,
      truePositives: 0,
      falsePositives: 0,
      falseNegatives: 0,
    };
  }

  const usedTruthIndexes = new Set<number>();
  let truePositives = 0;

  for (const finding of extracted) {
    const matchIndex = groundTruth.findIndex((expected, index) => {
      if (usedTruthIndexes.has(index)) {
        return false;
      }
      return isMatch(finding, expected);
    });

    if (matchIndex >= 0) {
      usedTruthIndexes.add(matchIndex);
      truePositives += 1;
    }
  }

  const falsePositives = Math.max(extracted.length - truePositives, 0);
  const falseNegatives = Math.max(groundTruth.length - truePositives, 0);

  const precision = extracted.length > 0 ? truePositives / extracted.length : 0;
  const recall = groundTruth.length > 0 ? truePositives / groundTruth.length : 0;
  const f1 = precision + recall > 0 ? (2 * (precision * recall)) / (precision + recall) : 0;

  return {
    precision,
    recall,
    f1,
    truePositives,
    falsePositives,
    falseNegatives,
  };
}

export function createF1MetricMetadata(result: ExtractionF1Result): Record<string, unknown> {
  return {
    precision: result.precision,
    recall: result.recall,
    truePositives: result.truePositives,
    falsePositives: result.falsePositives,
    falseNegatives: result.falseNegatives,
  };
}
