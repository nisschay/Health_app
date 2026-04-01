import type { GroundTruthEntry } from "@/lib/metrics";

export const EXTRACTION_GROUND_TRUTH: GroundTruthEntry[] = [
  {
    pdfFile: "ravi_panel_a.pdf",
    expectedFindings: [
      { testName: "Glycated Haemoglobin (HbA1C)", value: 6.8, unit: "%" },
      { testName: "Fasting Blood Glucose", value: 112, unit: "mg/dL" },
      { testName: "Serum Creatinine", value: 1.1, unit: "mg/dL" },
      { testName: "Blood Urea Nitrogen (BUN)", value: 18, unit: "mg/dL" },
      { testName: "Thyroid Stimulating Hormone (TSH)", value: 3.2, unit: "uIU/mL" },
      { testName: "25-OH Vitamin D (Total)", value: 24.5, unit: "ng/mL" },
      { testName: "Vitamin B12 (Cobalamin)", value: 401, unit: "pg/mL" },
      { testName: "LDL Cholesterol", value: 132, unit: "mg/dL" },
      { testName: "HDL Cholesterol", value: 43, unit: "mg/dL" },
      { testName: "Triglycerides", value: 168, unit: "mg/dL" },
      { testName: "Haemoglobin (Hb)", value: 13.6, unit: "g/dL" },
      { testName: "Aspartate Aminotransferase (SGOT/AST)", value: 32, unit: "U/L" },
      { testName: "Alanine Aminotransferase (SGPT/ALT)", value: 37, unit: "U/L" },
      { testName: "Serum Ferritin", value: 86.2, unit: "ng/mL" },
      { testName: "Fasting Blood Glucose", value: 106, unit: "mg/dL" },
      { testName: "Glycated Haemoglobin (HbA1C)", value: 6.1, unit: "%" },
      { testName: "Serum Creatinine", value: 1.0, unit: "mg/dL" },
      { testName: "LDL Cholesterol", value: 118, unit: "mg/dL" },
      { testName: "Triglycerides", value: 149, unit: "mg/dL" },
      { testName: "Thyroid Stimulating Hormone (TSH)", value: 2.8, unit: "uIU/mL" },
    ],
  },
];
