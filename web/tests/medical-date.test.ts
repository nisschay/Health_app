import { describe, expect, it } from "vitest";
import { parseMedicalDate } from "../lib/medicalDate";

describe("parseMedicalDate", () => {
  it("parses ISO dates, which the chart used to turn into NaN", () => {
    expect(parseMedicalDate("2024-01-15")).toBe(new Date("2024-01-15").getTime());
    expect(Number.isNaN(parseMedicalDate("2024-01-15"))).toBe(false);
  });

  it("parses day-first dates", () => {
    expect(parseMedicalDate("15-01-2024")).toBe(new Date("2024-01-15").getTime());
    expect(parseMedicalDate("5/3/2024")).toBe(new Date("2024-03-05").getTime());
  });

  it("expands two-digit years", () => {
    expect(parseMedicalDate("15-01-24")).toBe(new Date("2024-01-15").getTime());
  });

  it("sorts unparseable values last instead of scrambling the series", () => {
    const sorted = ["2024-03-05", "not a date", "2024-01-15"].sort(
      (a, b) => parseMedicalDate(a) - parseMedicalDate(b)
    );
    expect(sorted).toEqual(["2024-01-15", "2024-03-05", "not a date"]);
  });

  it("treats empty input as unknown", () => {
    expect(parseMedicalDate("")).toBe(Number.MAX_SAFE_INTEGER);
    expect(parseMedicalDate(null)).toBe(Number.MAX_SAFE_INTEGER);
    expect(parseMedicalDate(undefined)).toBe(Number.MAX_SAFE_INTEGER);
  });

  it("orders a mixed-format series chronologically", () => {
    const sorted = ["2024-06-01", "15-01-2024", "2024-03-05"].sort(
      (a, b) => parseMedicalDate(a) - parseMedicalDate(b)
    );
    expect(sorted).toEqual(["15-01-2024", "2024-03-05", "2024-06-01"]);
  });
});
