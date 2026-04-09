import { beforeEach, describe, expect, it, vi } from "vitest";

const payload = {
  user: {
    user_id: "u-1",
    email: "test@example.com",
    authenticated: true,
  },
  patient_info: {
    name: "Patient",
    age: "40",
    gender: "Male",
    patient_id: "p-1",
    date: "2024-01-01",
    lab_name: "Lab",
  },
  total_records: 2,
  records: [
    {
      Test_Name: "HbA1C",
      Original_Test_Name: "HbA1C",
      Test_Category: "Other",
      Test_Date: "2024-01-01",
      Result: "6.8",
      Unit: "%",
      Status: "High",
    },
    {
      Test_Name: "haemoglobin a1c",
      Original_Test_Name: "haemoglobin a1c",
      Test_Category: "Diabetes",
      Test_Date: "2024-01-01",
      Result: "6.8",
      Unit: "%",
      Status: "High",
    },
  ],
  health_summary: {
    overall_score: 70,
    category_scores: {},
    concerns: [],
  },
  body_systems: [],
  raw_texts: [],
};

function mockJsonResponse(data: unknown): Response {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("Combined Report Client Normalization Gate", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
    delete process.env.NEXT_PUBLIC_CLIENT_NORMALIZATION_FALLBACK;
  });

  it("returns backend combined payload as-is when fallback is disabled", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(mockJsonResponse(payload));
    const api = await import("@/lib/api");

    api.setAuthTokenProvider(async () => "token-123");
    const result = await api.fetchStudyCombinedReport("study-1");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.total_records).toBe(2);
    expect(result.records).toHaveLength(2);
    expect(result.records[0]?.Test_Name).toBe("HbA1C");
  });

  it("applies client normalization only when fallback flag is enabled", async () => {
    process.env.NEXT_PUBLIC_CLIENT_NORMALIZATION_FALLBACK = "true";

    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(mockJsonResponse(payload));
    const api = await import("@/lib/api");

    api.setAuthTokenProvider(async () => "token-456");
    const result = await api.fetchStudyCombinedReport("study-1");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.total_records).toBe(1);
    expect(result.records).toHaveLength(1);
    expect(result.records[0]?.Test_Name).toBe("Glycated Haemoglobin (HbA1C)");
  });
});
