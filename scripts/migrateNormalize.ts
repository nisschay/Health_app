import { normalizeAnalysisPayload, type NormalizableMedicalRecord } from "../web/lib/normalizeTest";

declare function require(name: string): any;
declare const process: {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
};

type PgQueryResult<Row> = { rows: Row[] };

type PgClient = {
  connect(): Promise<void>;
  query<Row = Record<string, unknown>>(text: string, values?: unknown[]): Promise<PgQueryResult<Row>>;
  end(): Promise<void>;
};

type PgClientCtor = new (config: { connectionString: string }) => PgClient;

const { Client } = require("pg") as { Client: PgClientCtor };

type AnalysisPayload = {
  records?: NormalizableMedicalRecord[];
  total_records?: number;
  health_summary?: {
    overall_score?: number;
    category_scores?: Record<string, { score: number; total_tests: number; abnormal_count: number }>;
    concerns?: Array<{
      test_name: string;
      result: string | number;
      status: string;
      reference: string;
      category: string;
      date: string;
    }>;
    [key: string]: unknown;
  };
  body_systems?: unknown[];
  patient_info?: {
    name?: string | null;
    age?: string | null;
    gender?: string | null;
    patient_id?: string | null;
    date?: string | null;
    lab_name?: string | null;
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }

  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }

  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${stableStringify(obj[key])}`).join(",")}}`;
}

function toPayload(value: unknown): AnalysisPayload | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as AnalysisPayload;
  }

  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as AnalysisPayload;
      }
    } catch {
      return null;
    }
  }

  return null;
}

function toNullableText(value: unknown): string | null {
  const text = String(value ?? "").trim();
  return text ? text : null;
}

function normalizeStatus(value: unknown): string {
  const status = String(value ?? "").trim();
  if (!status) return "N/A";
  const key = status.toLowerCase();
  if (key.includes("critical")) return "Critical";
  if (key.includes("insufficient")) return "Insufficient";
  if (key.includes("borderline")) return "Borderline";
  if (key.includes("positive")) return "Positive";
  if (key.includes("negative")) return "Negative";
  if (key.includes("flag")) return "Flagged";
  if (key.includes("high")) return "High";
  if (key.includes("low")) return "Low";
  if (key.includes("normal") || key.includes("within")) return "Normal";
  return status;
}

function isAbnormalStatus(status: string): boolean {
  const key = status.toLowerCase();
  return key === "critical" || key === "high" || key === "low" || key === "positive" || key === "flagged" || key === "borderline" || key === "insufficient";
}

function buildHealthSummary(records: NormalizableMedicalRecord[]): NonNullable<AnalysisPayload["health_summary"]> {
  const concerns: NonNullable<AnalysisPayload["health_summary"]>["concerns"] = [];
  const categoryStats = new Map<string, { total: number; abnormal: number }>();

  for (const row of records) {
    const category = String(row.Test_Category ?? "Other").trim() || "Other";
    const status = normalizeStatus(row.Status);

    if (!categoryStats.has(category)) {
      categoryStats.set(category, { total: 0, abnormal: 0 });
    }

    const bucket = categoryStats.get(category)!;
    bucket.total += 1;

    if (isAbnormalStatus(status)) {
      bucket.abnormal += 1;
      concerns.push({
        test_name: String(row.Test_Name ?? row.Original_Test_Name ?? "Unknown Test"),
        result: (row.Result as string | number | null) ?? "N/A",
        status,
        reference: String(row.Reference_Range ?? "N/A"),
        category,
        date: String(row.Test_Date ?? "N/A"),
      });
    }
  }

  const category_scores: Record<string, { score: number; total_tests: number; abnormal_count: number }> = {};
  let scoreSum = 0;
  let scoreCount = 0;

  for (const [category, stat] of categoryStats.entries()) {
    const score = stat.total > 0 ? Math.max(0, Math.round(((stat.total - stat.abnormal) / stat.total) * 100)) : 0;
    category_scores[category] = {
      score,
      total_tests: stat.total,
      abnormal_count: stat.abnormal,
    };
    scoreSum += score;
    scoreCount += 1;
  }

  const overall_score = scoreCount > 0 ? Math.round(scoreSum / scoreCount) : 0;

  return {
    overall_score,
    category_scores,
    concerns,
  };
}

function normalizePayload(payload: AnalysisPayload): AnalysisPayload {
  const normalized = normalizeAnalysisPayload(payload);
  const records = Array.isArray(normalized.records) ? normalized.records : [];
  return {
    ...normalized,
    health_summary: buildHealthSummary(records),
    body_systems: Array.isArray(payload.body_systems) ? payload.body_systems : [],
  };
}

async function migrateReports(client: PgClient): Promise<{ updated: number; skipped: number }> {
  const result = await client.query<{ id: string; analysis_data: unknown }>(
    "SELECT id, analysis_data FROM reports",
  );

  let updated = 0;
  let skipped = 0;

  for (const row of result.rows) {
    const payload = toPayload(row.analysis_data);
    if (!payload) {
      skipped += 1;
      continue;
    }

    const before = stableStringify(payload);
    const nextPayload = normalizePayload(payload);
    const after = stableStringify(nextPayload);

    if (before === after) {
      skipped += 1;
      continue;
    }

    await client.query(
      "UPDATE reports SET analysis_data = $2::jsonb WHERE id = $1",
      [row.id, JSON.stringify(nextPayload)],
    );
    updated += 1;
  }

  return { updated, skipped };
}

async function migrateHistory(client: PgClient): Promise<{ updated: number; skipped: number }> {
  const result = await client.query<{
    id: number;
    analysis_json: string;
  }>("SELECT id, analysis_json FROM report_analyses");

  let updated = 0;
  let skipped = 0;

  for (const row of result.rows) {
    const payload = toPayload(row.analysis_json);
    if (!payload) {
      skipped += 1;
      continue;
    }

    const before = stableStringify(payload);
    const nextPayload = normalizePayload(payload);
    const after = stableStringify(nextPayload);

    if (before === after) {
      skipped += 1;
      continue;
    }

    const patientInfo = (nextPayload.patient_info ?? {}) as Record<string, unknown>;

    await client.query(
      `UPDATE report_analyses
       SET analysis_json = $2,
           total_records = $3,
           patient_name = $4,
           patient_age = $5,
           patient_gender = $6,
           patient_id = $7,
           lab_name = $8,
           report_date = $9
       WHERE id = $1`,
      [
        row.id,
        JSON.stringify(nextPayload),
        Number(nextPayload.total_records ?? 0) || 0,
        toNullableText(patientInfo.name),
        toNullableText(patientInfo.age),
        toNullableText(patientInfo.gender),
        toNullableText(patientInfo.patient_id),
        toNullableText(patientInfo.lab_name),
        toNullableText(patientInfo.date),
      ],
    );

    updated += 1;
  }

  return { updated, skipped };
}

async function main(): Promise<void> {
  const databaseUrl =
    process.env.DATABASE_URL ?? "postgresql://medical_user:medical_pass@localhost:5432/medical_project";

  const client = new Client({ connectionString: databaseUrl });
  await client.connect();

  try {
    await client.query("BEGIN");

    const reportStats = await migrateReports(client);
    const historyStats = await migrateHistory(client);

    await client.query("COMMIT");

    console.log("Normalization migration complete");
    console.log(`- Study report payloads updated: ${reportStats.updated}`);
    console.log(`- Study report payloads unchanged/skipped: ${reportStats.skipped}`);
    console.log(`- History payloads updated: ${historyStats.updated}`);
    console.log(`- History payloads unchanged/skipped: ${historyStats.skipped}`);
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    await client.end();
  }
}

main().catch((error) => {
  console.error("Normalization migration failed:", error);
  process.exit(1);
});
