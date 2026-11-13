import { describe, it, expect } from "vitest";
import {
  formatValue,
  formatAmount,
  formatRange,
  formatDate,
  formatUnit,
} from "../lib/format";
import { summariseTests, summariseStudy, parseRefRange } from "../lib/results";
import type { MedicalRecord } from "../lib/api";
const row = (
  value: number,
  date: string,
  status: string,
  range = "10 - 20",
  unit = "mg/dl",
): MedicalRecord => ({
  Test_Name: "Example",
  Result: value,
  Test_Date: date,
  Status: status,
  Reference_Range: range,
  Unit: unit,
});
describe("report formatting", () => {
  it("preserves printed decimals and Indian grouping", () => {
    expect(formatValue("132000")).toBe("1,32,000");
    expect(formatValue("7.90")).toBe("7.90");
    expect(formatAmount(-129000)).toBe("1,29,000");
    expect(formatAmount(0.8000000001, 1)).toBe("0.8");
  });
  it("formats units, one-sided ranges and unknown dates", () => {
    expect(formatUnit("cells/cumm")).toBe("/cumm");
    expect(formatRange("< 100")).toBe("under 100");
    expect(formatRange("100000 - 200000")).toBe("1,00,000 to 2,00,000");
    expect(formatDate(null)).toBe("Date not found");
    expect(formatDate("15-03-2026")).toBe("15 Mar 2026");
    expect(parseRefRange("> 40")).toEqual({ low: 40, high: null });
  });
});
describe("latest readings and verdicts", () => {
  it("counts only the latest reading", () => {
    const t = summariseTests([
      row(30, "2025-01-01", "High"),
      row(15, "2025-02-01", "Normal"),
    ]);
    expect(t[0]?.verdict).toBe("back-in-range");
    expect(summariseStudy(t).attention).toBe(0);
  });
  it("uses distance to range in both directions", () => {
    expect(
      summariseTests([
        row(2, "2025-01-01", "Low"),
        row(7, "2025-02-01", "Low"),
      ])[0]?.verdict,
    ).toBe("better");
    expect(
      summariseTests([
        row(7, "2025-01-01", "Low"),
        row(2, "2025-02-01", "Low"),
      ])[0]?.verdict,
    ).toBe("worse");
    expect(
      summariseTests([
        row(30, "2025-01-01", "High"),
        row(22, "2025-02-01", "High"),
      ])[0]?.verdict,
    ).toBe("better");
  });
  it("supports one-sided ranges and the five percent band", () => {
    expect(
      summariseTests([
        row(130, "2025-01-01", "High", "< 100"),
        row(128, "2025-02-01", "High", "< 100"),
      ])[0]?.verdict,
    ).toBe("same");
  });
  it("ignores duplicate days and does not use unknown dates as latest", () => {
    const t = summariseTests([
      row(30, "2025-01-01", "High"),
      row(15, "2025-01-01", "Normal"),
      row(100, "unknown", "Critical"),
    ])[0]!;
    expect(t.latest.value).toBe(15);
    expect(t.readings).toHaveLength(2);
  });
  it("handles one reading, text results and mixed units", () => {
    expect(summariseTests([row(30, "2025-01-01", "High")])[0]?.verdict).toBe(
      "first",
    );
    expect(
      summariseTests([
        row(30, "2025-01-01", "High"),
        { ...row(20, "2025-02-01", "High"), Result: "Trace" },
      ])[0]?.verdict,
    ).toBeNull();
    expect(
      summariseTests([
        row(30, "2025-01-01", "High"),
        row(20, "2025-02-01", "High", "10 - 20", "mmol/L"),
      ])[0]?.change,
    ).toBeNull();
  });
  it("distinguishes first time out, recurrence and unknown status", () => {
    expect(
      summariseTests([
        row(15, "2025-01-01", "Normal"),
        row(25, "2025-02-01", "High"),
      ])[0]?.firstTimeOut,
    ).toBe(true);
    expect(
      summariseTests([
        row(25, "2024-01-01", "High"),
        row(15, "2025-01-01", "Normal"),
        row(25, "2025-02-01", "High"),
      ])[0]?.firstTimeOut,
    ).toBe(false);
    expect(
      summariseStudy(summariseTests([row(15, "2025-01-01", "Unknown")]))
        .inRange,
    ).toBe(0);
  });
});
