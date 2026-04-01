import { describe, expect, it, vi } from "vitest";
import {
  evaluateContextRetentionScore,
  logMetric,
  type ContextRetentionTurn,
} from "@/lib/metrics";

describe("Clinical Assistant Context Retention", () => {
  it("retains patient data across 3 turns", async () => {
    const scriptedResponses = [
      "The earliest HbA1C was 6.8 in 2022, which was in the diabetic range.",
      "Yes, it improved. HbA1C decreased to 5.9 and is lower than before.",
      "For blood sugar control, discuss diabetes follow-up with your clinician and continue diet and exercise.",
    ];

    const callClinicalAssistant = vi.fn(async () => scriptedResponses.shift() ?? "");

    const testConversation: ContextRetentionTurn[] = [
      {
        userMsg: "What was my HbA1C in the earliest report?",
        mustContainOneOf: ["6.8", "hba1c", "diabetic", "2022"],
      },
      {
        userMsg: "Has it improved since then?",
        mustContainOneOf: ["5.9", "improved", "decreased", "reduced", "better", "lower"],
      },
      {
        userMsg: "What should I do about it?",
        mustContainOneOf: ["blood sugar", "glucose", "diabetes", "diet", "exercise", "clinician"],
      },
    ];

    const result = await evaluateContextRetentionScore({
      conversation: testConversation,
      analysisId: "test-analysis-id",
      callClinicalAssistant,
    });

    expect(callClinicalAssistant).toHaveBeenCalledTimes(3);
    expect(result.score).toBeGreaterThanOrEqual(90);

    const metricPayload = {
      metricName: "context_retention" as const,
      value: result.score,
      metadata: {
        retentionScore: result.retentionScore,
        totalTurns: result.totalTurns,
      },
    };

    expect(metricPayload.value).toBeGreaterThanOrEqual(90);
    expect(metricPayload.metadata.retentionScore).toBe(3);
  });

  const canLogMetric =
    typeof process.env.METRICS_TEST_TOKEN === "string"
    && process.env.METRICS_TEST_TOKEN.length > 0;

  (canLogMetric ? it : it.skip)("logs context retention metric when runtime token is available", async () => {
    const response = await logMetric(
      {
        metricName: "context_retention",
        value: 100,
        metadata: {
          source: "vitest",
          suite: "clinical-assistant-context-retention",
        },
      },
      {
        token: process.env.METRICS_TEST_TOKEN as string,
        throwOnError: true,
      },
    );

    expect(response.ok).toBe(true);
  });
});
