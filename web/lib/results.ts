import type { MedicalRecord } from "./api";
import { parseMedicalDate } from "./medicalDate";
import {
  formatValue,
  formatRange,
  formatUnit,
  formatAmount,
  formatDate,
} from "./format";
import { CANONICAL_CATEGORIES } from "./categoryMap";
export type StatusKind =
  "critical" | "high" | "low" | "borderline" | "detected" | "normal" | "none";
export function normalizeStatus(s?: string | null): StatusKind {
  const v = s?.trim().toLowerCase();
  if (v === "critical" || v === "high" || v === "low" || v === "borderline")
    return v;
  if (v === "positive" || v === "flagged") return "detected";
  if (v === "insufficient") return "low";
  if (v === "normal" || v === "negative" || v === "in range") return "normal";
  return "none";
}
export function parseNumeric(v: unknown): number | null {
  const s = String(v ?? "")
    .replaceAll(",", "")
    .trim();
  return /^[+-]?(?:\d+\.?\d*|\.\d+)$/.test(s) && Number.isFinite(Number(s))
    ? Number(s)
    : null;
}
export function parseRefRange(text?: string | null): {
  low: number | null;
  high: number | null;
} {
  const s = (text ?? "").replaceAll(",", "").trim();
  const pair = s.match(/^(-?\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(-?\d+(?:\.\d+)?)/i);
  if (pair) return { low: Number(pair[1]), high: Number(pair[2]) };
  const single = s.match(
    /^(<=?|>=?|≤|≥|under|over|at least|at most)\s*(\d+(?:\.\d+)?)/i,
  );
  if (single)
    return /^(>|≥|over|at least)/.test(single[1])
      ? { low: Number(single[2]), high: null }
      : { low: null, high: Number(single[2]) };
  return { low: null, high: null };
}
export function toReading(r: MedicalRecord) {
  return {
    date: r.Test_Date ?? null,
    time: parseMedicalDate(r.Test_Date),
    valueText: formatValue(r.Result, r.Result_Numeric),
    value: r.Result_Numeric ?? parseNumeric(r.Result),
    unit: formatUnit(r.Unit),
    rangeText: formatRange(r.Reference_Range),
    ...parseRefRange(r.Reference_Range),
    status: normalizeStatus(r.Status),
    negative: r.Status?.toLowerCase() === "negative",
    lab: r.Lab_Name ?? null,
    file: r.Source_Filename ?? null,
  };
}
export type Reading = ReturnType<typeof toReading>;
export type Verdict =
  "better" | "worse" | "same" | "back-in-range" | "first" | null;
const outside = (r: Reading) => !["normal", "none"].includes(r.status);
const distance = (r: Reading) =>
  r.value === null
    ? 0
    : Math.max(
        0,
        r.low === null ? 0 : r.low - r.value,
        r.high === null ? 0 : r.value - r.high,
      );
export function verdict(latest: Reading, previous: Reading | null): Verdict {
  if (!previous) return "first";
  if (
    latest.value === null ||
    previous.value === null ||
    latest.unit !== previous.unit ||
    (latest.low === null && latest.high === null) ||
    (previous.low === null && previous.high === null)
  )
    return null;
  const a = distance(previous),
    b = distance(latest);
  if (a > 0 && b === 0) return "back-in-range";
  const threshold =
    0.05 *
    (latest.low !== null && latest.high !== null
      ? latest.high - latest.low
      : Math.abs(latest.high ?? latest.low ?? 0));
  return a - b > threshold ? "better" : b - a > threshold ? "worse" : "same";
}
export function summariseTests(records: MedicalRecord[]) {
  const groups = new Map<string, MedicalRecord[]>();
  records.forEach((r) => {
    const name = r.Test_Name || r.Original_Test_Name || "Unnamed test";
    groups.set(name, [...(groups.get(name) ?? []), r]);
  });
  return Array.from(groups, ([name, rows]) => {
    const days = new Map<number, Reading>();
    rows.forEach((r) => {
      const reading = toReading(r);
      days.set(reading.time, reading);
    });
    const readings = Array.from(days.values()).sort(
      (a, b) =>
        (a.time === Number.MAX_SAFE_INTEGER ? -Infinity : a.time) -
        (b.time === Number.MAX_SAFE_INTEGER ? -Infinity : b.time),
    );
    const latest = readings[readings.length - 1]!,
      previous = readings.at(-2) ?? null;
    const delta =
      previous?.value !== null &&
      previous &&
      latest.value !== null &&
      latest.unit === previous.unit
        ? latest.value - previous.value
        : null;
    let streak = 0;
    for (let i = readings.length - 1; i >= 0 && outside(readings[i]!); i--)
      streak++;
    return {
      name,
      category: rows[0]?.Test_Category || "Other",
      printedNames: Array.from(
        new Set(
          rows
            .flatMap((r) => [r.Original_Test_Name, ...(r.Aliases ?? [])])
            .filter((n): n is string => !!n && n !== name),
        ),
      ),
      readings,
      latest,
      previous,
      needsAttention: outside(latest),
      verdict: verdict(latest, previous),
      change:
        delta === null
          ? null
          : `${delta === 0 ? "No change" : `${delta > 0 ? "Up" : "Down"} ${formatAmount(delta, Math.max(...[latest, previous!].map((r) => r.valueText.split(".")[1]?.length ?? 0)))}`} since ${formatDate(previous?.date, { year: false })}`,
      firstTimeOut:
        !!previous &&
        outside(latest) &&
        !outside(previous) &&
        !readings.slice(0, -1).some(outside),
      streak,
      deviation:
        distance(latest) /
        Math.max(1, Math.abs(latest.high ?? latest.low ?? 1)),
    };
  });
}
export type TestSummary = ReturnType<typeof summariseTests>[number];
export function summariseStudy(tests: TestSummary[]) {
  const dates = Array.from(
    new Set(
      tests
        .flatMap((t) => t.readings.map((r) => r.date))
        .filter(
          (d): d is string =>
            !!d && parseMedicalDate(d) !== Number.MAX_SAFE_INTEGER,
        ),
    ),
  ).sort((a, b) => parseMedicalDate(b) - parseMedicalDate(a));
  const count = (s: StatusKind) =>
    tests.filter((t) => t.latest.status === s).length;
  return {
    total: tests.length,
    attention: tests.filter((t) => t.needsAttention).length,
    critical: count("critical"),
    high: count("high"),
    low: count("low"),
    borderline: count("borderline"),
    detected: count("detected"),
    inRange: count("normal"),
    unknown: count("none"),
    latestReportDate: dates[0],
    earliestReportDate: dates[dates.length - 1],
    previousReportDate: dates[1],
    better: tests.filter((t) => t.verdict === "better").length,
    worse: tests.filter((t) => t.verdict === "worse").length,
    backInRange: tests.filter((t) => t.verdict === "back-in-range").length,
    criticalTests: tests.filter((t) => t.latest.status === "critical"),
  };
}
export function orderForAttention(tests: TestSummary[]) {
  return [...tests].sort(
    (a, b) =>
      Number(b.latest.status === "critical") -
        Number(a.latest.status === "critical") ||
      Number(b.firstTimeOut) - Number(a.firstTimeOut) ||
      Number(b.verdict === "worse") - Number(a.verdict === "worse") ||
      b.deviation - a.deviation,
  );
}
export function doctorList(tests: TestSummary[]) {
  return orderForAttention(
    tests.filter(
      (t) =>
        t.needsAttention &&
        (t.latest.status === "critical" ||
          t.firstTimeOut ||
          t.streak >= 3 ||
          t.verdict === "worse"),
    ),
  )
    .slice(0, 6)
    .map(
      (t) =>
        `${t.name} ${t.latest.status === "critical" ? "is marked critical" : t.firstTimeOut ? "is outside the range for the first time" : t.streak >= 3 ? `has been outside the range in ${t.streak} reports in a row` : "is getting worse"}: ${t.latest.valueText} ${t.latest.unit} on ${formatDate(t.latest.date)} (range ${t.latest.rangeText}).`,
    );
}
export function groupByCategory(tests: TestSummary[]) {
  const groups = Array.from(new Set(tests.map((t) => t.category))).map(
    (category) => ({
      category,
      tests: tests.filter((t) => t.category === category),
      attention: tests.filter(
        (t) => t.category === category && t.needsAttention,
      ).length,
    }),
  );
  return groups.sort(
    (a, b) =>
      Number(b.attention > 0) - Number(a.attention > 0) ||
      (CANONICAL_CATEGORIES as readonly string[]).indexOf(a.category) -
        (CANONICAL_CATEGORIES as readonly string[]).indexOf(b.category),
  );
}
