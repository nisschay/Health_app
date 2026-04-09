import { describe, expect, it } from "vitest";
import { areSameTest, normalizeRecordsForClient, type NormalizableMedicalRecord } from "@/lib/normalizeTest";
import { normalizeTestName } from "@/lib/testNameMap";

describe("Normalization Golden Set Regression", () => {
  it("maps known test aliases to stable canonical names", () => {
    const goldenCases: Array<{ input: string; expected: string }> = [
      { input: "HbA1C", expected: "Glycated Haemoglobin (HbA1C)" },
      { input: "haemoglobin a1c", expected: "Glycated Haemoglobin (HbA1C)" },
      { input: "TSH", expected: "Thyroid Stimulating Hormone (TSH)" },
      { input: "thyroid stimulating hormone", expected: "Thyroid Stimulating Hormone (TSH)" },
      { input: "high sensitivity c-reactive protein (hs-crp)", expected: "hs-CRP" },
      { input: "LDL", expected: "LDL Cholesterol" },
      { input: "albumin - serum", expected: "Albumin" },
      { input: "vitamin b-12", expected: "Vitamin B12 (Cobalamin)" },
    ];

    for (const { input, expected } of goldenCases) {
      expect(normalizeTestName(input)).toBe(expected);
    }
  });

  it("does not merge known must-not-merge pairs", () => {
    expect(areSameTest("Albumin", "Albumin/Globulin Ratio (A/G)"))
      .toBe(false);
    expect(areSameTest("Bilirubin Total", "Bilirubin Indirect"))
      .toBe(false);
    expect(areSameTest("TSH", "T3"))
      .toBe(false);
    expect(areSameTest("TSH", "T3H"))
      .toBe(false);
  });

  it("deduplicates canonical synonyms while preserving aliases", () => {
    const rows: NormalizableMedicalRecord[] = [
      {
        Test_Name: "HbA1C",
        Test_Category: "Diabetes",
        Test_Date: "2024-01-01",
        Result: "6.8",
        Unit: "%",
      },
      {
        Test_Name: "haemoglobin a1c",
        Test_Category: "Other",
        Test_Date: "2024-01-01",
        Result: "6.8",
        Unit: "%",
      },
    ];

    const normalized = normalizeRecordsForClient(rows);
    const first = normalized[0] as NormalizableMedicalRecord | undefined;

    expect(normalized).toHaveLength(1);
    expect(first?.Test_Name).toBe("Glycated Haemoglobin (HbA1C)");
    expect(first?.Aliases).toEqual(
      expect.arrayContaining(["HbA1C", "haemoglobin a1c", "Glycated Haemoglobin (HbA1C)"]),
    );
  });

  it("keeps albumin and A/G ratio as separate tests in normalization output", () => {
    const rows: NormalizableMedicalRecord[] = [
      {
        Test_Name: "Albumin",
        Test_Category: "Liver Function",
        Test_Date: "2024-02-01",
        Result: "4.2",
        Unit: "g/dL",
      },
      {
        Test_Name: "Albumin/Globulin Ratio (A/G)",
        Test_Category: "Liver Function",
        Test_Date: "2024-02-01",
        Result: "1.3",
        Unit: "ratio",
      },
    ];

    const normalized = normalizeRecordsForClient(rows);
    const names = normalized.map((row) => row.Test_Name).sort();

    expect(normalized).toHaveLength(2);
    expect(names).toEqual([
      "Albumin",
      "Albumin/Globulin Ratio (A/G)",
    ]);
  });
});
