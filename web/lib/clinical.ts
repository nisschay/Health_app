import type { AnalysisConcern, MedicalRecord } from "./api";

export type SeverityLevel = "critical" | "high" | "low" | "normal";

export type GroupedCategoryAlert = {
  category: string;
  totalAlerts: number;
  severity: SeverityLevel;
  criticalCount: number;
  highCount: number;
  lowCount: number;
  normalCount: number;
  tests: Array<{
    testName: string;
    result: string | number;
    status: string;
    reference: string;
    date: string;
  }>;
};

const severityRank: Record<SeverityLevel, number> = {
  critical: 0,
  high: 1,
  low: 2,
  normal: 3,
};

function normalizeStatus(status: string | null | undefined): string {
  return (status ?? "").trim().toLowerCase();
}

function severityFromCounts(critical: number, high: number, low: number): SeverityLevel {
  if (critical > 0) return "critical";
  if (high > 0) return "high";
  if (low > 0) return "low";
  return "normal";
}

export function groupByCategory(concerns: AnalysisConcern[]): GroupedCategoryAlert[] {
  const grouped = new Map<string, GroupedCategoryAlert>();

  for (const concern of concerns) {
    const category = concern.category?.trim() || "Uncategorized";
    if (!grouped.has(category)) {
      grouped.set(category, {
        category,
        totalAlerts: 0,
        severity: "normal",
        criticalCount: 0,
        highCount: 0,
        lowCount: 0,
        normalCount: 0,
        tests: [],
      });
    }

    const row = grouped.get(category)!;
    row.totalAlerts += 1;
    const status = normalizeStatus(concern.status);
    if (status === "critical") row.criticalCount += 1;
    else if (status === "high" || status === "positive" || status === "flagged") row.highCount += 1;
    else if (status === "low") row.lowCount += 1;
    else row.normalCount += 1;

    row.tests.push({
      testName: concern.test_name,
      result: concern.result,
      status: concern.status,
      reference: concern.reference,
      date: concern.date,
    });
  }

  const list = Array.from(grouped.values()).map((entry) => {
    entry.severity = severityFromCounts(entry.criticalCount, entry.highCount, entry.lowCount);
    entry.tests.sort((a, b) => a.testName.localeCompare(b.testName));
    return entry;
  });

  list.sort((a, b) => {
    const bySeverity = severityRank[a.severity] - severityRank[b.severity];
    if (bySeverity !== 0) return bySeverity;
    return b.totalAlerts - a.totalAlerts;
  });

  return list;
}

export type DateCategoryGroup = {
  date: string;
  timestamp: number;
  categories: Array<{
    category: string;
    tests: MedicalRecord[];
  }>;
};

export type TestTimelineEntry = {
  date: string;
  timestamp: number;
  value: string | number;
  referenceRange: string;
  status: string;
  unit: string;
  category: string;
};

export type GroupedTestTimeline = {
  testName: string;
  latest: TestTimelineEntry;
  timeline: TestTimelineEntry[];
};

export function parseMedicalDate(value: string | null | undefined): number {
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

export function groupRecordsByDateAndCategory(records: MedicalRecord[]): DateCategoryGroup[] {
  const valid = records.filter(
    (r) => r.Test_Name && r.Test_Name !== "N/A" && r.Result !== null && r.Result !== undefined
  );

  const byDate = new Map<string, MedicalRecord[]>();
  for (const record of valid) {
    const dateKey = record.Test_Date ?? "Unknown";
    if (!byDate.has(dateKey)) byDate.set(dateKey, []);
    byDate.get(dateKey)!.push(record);
  }

  return Array.from(byDate.entries())
    .map(([date, dateRecords]) => {
      const byCategory = new Map<string, MedicalRecord[]>();
      for (const row of dateRecords) {
        const category = row.Test_Category?.trim() || "Uncategorized";
        if (!byCategory.has(category)) byCategory.set(category, []);
        byCategory.get(category)!.push(row);
      }

      const categories = Array.from(byCategory.entries())
        .map(([category, tests]) => ({
          category,
          tests: [...tests].sort((a, b) => String(a.Test_Name ?? "").localeCompare(String(b.Test_Name ?? ""))),
        }))
        .sort((a, b) => a.category.localeCompare(b.category));

      return {
        date,
        timestamp: parseMedicalDate(date),
        categories,
      };
    })
    .sort((a, b) => a.timestamp - b.timestamp);
}

export function groupByTestName(records: MedicalRecord[]): GroupedTestTimeline[] {
  const grouped = new Map<string, TestTimelineEntry[]>();

  for (const record of records) {
    const testName = record.Test_Name?.trim();
    if (!testName || testName === "N/A") continue;
    if (record.Result === null || record.Result === undefined || String(record.Result).trim() === "") continue;

    const entry: TestTimelineEntry = {
      date: record.Test_Date ?? "Unknown",
      timestamp: parseMedicalDate(record.Test_Date),
      value: record.Result,
      referenceRange: record.Reference_Range ?? "N/A",
      status: record.Status ?? "N/A",
      unit: record.Unit ?? "",
      category: record.Test_Category?.trim() || "Uncategorized",
    };

    if (!grouped.has(testName)) grouped.set(testName, []);
    grouped.get(testName)!.push(entry);
  }

  return Array.from(grouped.entries())
    .map(([testName, timeline]) => {
      const sorted = [...timeline].sort((a, b) => a.timestamp - b.timestamp);
      return {
        testName,
        latest: sorted[sorted.length - 1]!,
        timeline: sorted,
      };
    })
    .sort((a, b) => a.testName.localeCompare(b.testName));
}

export function formatForPDF(records: MedicalRecord[], concerns: AnalysisConcern[]) {
  const groupedAlerts = groupByCategory(concerns);
  const groupedRecords = groupRecordsByDateAndCategory(records);
  return {
    groupedAlerts,
    groupedRecords,
  };
}
