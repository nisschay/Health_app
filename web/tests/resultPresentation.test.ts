import { describe, expect, it } from "vitest";
import { summariseTests } from "../lib/results";
import { recentAttention } from "../lib/resultPresentation";

describe("v2 attention order", () => {
  it("shows critical first, then newest outside readings, with stable name order", () => {
    const tests = summariseTests([
      {
        Test_Name: "Older high",
        Result: 20,
        Status: "High",
        Test_Date: "2024-01-01",
      },
      {
        Test_Name: "Latest normal",
        Result: 10,
        Status: "Normal",
        Test_Date: "2026-01-01",
      },
      {
        Test_Name: "Zinc",
        Result: 20,
        Status: "High",
        Test_Date: "2025-01-01",
      },
      {
        Test_Name: "Critical",
        Result: 20,
        Status: "Critical",
        Test_Date: "2023-01-01",
      },
      {
        Test_Name: "Albumin",
        Result: 20,
        Status: "High",
        Test_Date: "2025-01-01",
      },
      { Test_Name: "Unknown date", Result: 20, Status: "High" },
    ]);
    expect(recentAttention(tests).map((t) => t.name)).toEqual([
      "Critical",
      "Albumin",
      "Zinc",
      "Older high",
      "Unknown date",
    ]);
    expect(tests).toHaveLength(6);
  });
});
