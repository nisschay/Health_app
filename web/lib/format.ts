import { parseMedicalDate } from "./medicalDate";
export function formatValue(value: unknown, numeric?: number | null): string {
  const raw = String(value ?? numeric ?? "").trim();
  if (!raw) return "Not reported";
  if (/^[+-]?[\d,]+(?:\.\d+)?$/.test(raw)) {
    const decimals = raw.split(".")[1]?.length ?? 0;
    return Number(raw.replaceAll(",", "")).toLocaleString("en-IN", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }
  return raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase();
}
export function formatAmount(n: number, decimals = 2): string {
  return Math.abs(n).toLocaleString("en-IN", {
    maximumFractionDigits: decimals,
  });
}
export function formatUnit(unit?: string | null): string {
  return (
    (
      {
        "mg/dl": "mg/dL",
        "cells/cumm": "/cumm",
        "/cu mm": "/cumm",
        "uIU/mL": "µIU/mL",
      } as Record<string, string>
    )[unit ?? ""] ??
    unit ??
    ""
  );
}
export function formatRange(text?: string | null): string {
  if (!text?.trim()) return "No range given";
  return text
    .replace(/(\d[\d,.]*)\s*[-–]\s*(\d[\d,.]*)/g, "$1 to $2")
    .replace(/<=|≤/g, "at most ")
    .replace(/>=|≥/g, "at least ")
    .replace(/<\s*/g, "under ")
    .replace(/>\s*/g, "over ")
    .replace(/\d[\d,]*(?:\.\d+)?/g, (n) => formatValue(n));
}
export function formatDate(
  text?: string | null,
  options = { year: true },
): string {
  const time = parseMedicalDate(text);
  return time === Number.MAX_SAFE_INTEGER
    ? "Date not found"
    : new Intl.DateTimeFormat("en-IN", {
        day: "numeric",
        month: "short",
        ...(options.year ? { year: "numeric" as const } : {}),
        timeZone: "UTC",
      }).format(time);
}
export function formatMonthRange(
  start: string | null,
  end: string | null,
): string {
  const month = (s: string) =>
    parseMedicalDate(s) === Number.MAX_SAFE_INTEGER
      ? "Date not found"
      : new Intl.DateTimeFormat("en-IN", {
          month: "short",
          year: "numeric",
          timeZone: "UTC",
        }).format(parseMedicalDate(s));
  return start && end
    ? month(start) === month(end)
      ? month(start)
      : `${month(start)} to ${month(end)}`
    : start || end
      ? month((start || end)!)
      : "No reports yet";
}
export const plural = (n: number, one: string, many = `${one}s`) =>
  `${n} ${n === 1 ? one : many}`;
