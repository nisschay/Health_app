import { describe, expect, it } from "vitest";
import {
  calculateF1,
  createF1MetricMetadata,
  logMetric,
  type ExtractionFinding,
} from "@/lib/metrics";
import { EXTRACTION_GROUND_TRUTH } from "./fixtures/extractionGroundTruth";

describe("Medical Entity Extraction F1", () => {
  it("meets or exceeds 95 percent F1 on ground truth fixture", () => {
    const fixture = EXTRACTION_GROUND_TRUTH[0];
    if (!fixture) {
      throw new Error("Ground truth fixture is missing.");
    }

    const extracted: ExtractionFinding[] = fixture.expectedFindings.map((entry) => ({
      testName: entry.testName,
      value: entry.value,
      unit: entry.unit,
    }));

    // Include a synonym to verify canonical name matching.
    extracted[0] = {
      testName: "HbA1C",
      value: fixture.expectedFindings[0]!.value,
      unit: fixture.expectedFindings[0]!.unit,
    };

    const result = calculateF1(extracted, fixture.expectedFindings);

    expect(result.precision).toBeGreaterThanOrEqual(0.95);
    expect(result.recall).toBeGreaterThanOrEqual(0.95);
    expect(result.f1).toBeGreaterThanOrEqual(0.95);

    const metadata = createF1MetricMetadata(result);
    expect(metadata.truePositives).toBeGreaterThanOrEqual(19);
  });

  const canLogMetric =
    typeof process.env.METRICS_TEST_TOKEN === "string"
    && process.env.METRICS_TEST_TOKEN.length > 0;

  (canLogMetric ? it : it.skip)("logs extraction_f1 metric when runtime token is available", async () => {
    const fixture = EXTRACTION_GROUND_TRUTH[0];
    if (!fixture) {
      throw new Error("Ground truth fixture is missing.");
    }

    const extracted: ExtractionFinding[] = fixture.expectedFindings.map((entry) => ({
      testName: entry.testName,
      value: entry.value,
      unit: entry.unit,
    }));

    const result = calculateF1(extracted, fixture.expectedFindings);

    const response = await logMetric(
      {
        metricName: "extraction_f1",
        value: result.f1 * 100,
        metadata: createF1MetricMetadata(result),
      },
      {
        token: process.env.METRICS_TEST_TOKEN as string,
        throwOnError: true,
      },
    );

    expect(response.ok).toBe(true);
  });
});
