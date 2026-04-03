export const CANONICAL_CATEGORIES = [
  "Haematology",
  "Lipid Profile",
  "Liver Function",
  "Kidney Function",
  "Diabetes & Glucose",
  "Thyroid Function",
  "Vitamins & Minerals",
  "Hormones",
  "Cardiac Markers",
  "Immunology",
  "Urinalysis",
  "Inflammation",
  "Proteins",
  "Other",
] as const;

export type CanonicalCategory = (typeof CANONICAL_CATEGORIES)[number];

export const CATEGORY_MAP: Record<string, CanonicalCategory> = {
  "vitamin b-12": "Vitamins & Minerals",
  "vitamin b12": "Vitamins & Minerals",
  "vitamin d": "Vitamins & Minerals",
  "vitamin levels": "Vitamins & Minerals",
  "vitamin profile": "Vitamins & Minerals",
  vitamins: "Vitamins & Minerals",
  minerals: "Vitamins & Minerals",
  "iron profile": "Vitamins & Minerals",
  "iron studies": "Vitamins & Minerals",

  diabetes: "Diabetes & Glucose",
  "diabetes panel": "Diabetes & Glucose",
  "glucose metabolism": "Diabetes & Glucose",
  "blood sugar": "Diabetes & Glucose",
  "diabetes / glucose metabolism": "Diabetes & Glucose",
  "diabetes markers": "Diabetes & Glucose",
  hba1c: "Diabetes & Glucose",

  "liver function test": "Liver Function",
  "liver function": "Liver Function",
  lft: "Liver Function",
  hepatic: "Liver Function",

  "kidney function test": "Kidney Function",
  "kidney function": "Kidney Function",
  "renal function": "Kidney Function",
  kft: "Kidney Function",
  renal: "Kidney Function",

  "thyroid function test": "Thyroid Function",
  "thyroid profile": "Thyroid Function",
  thyroid: "Thyroid Function",
  tft: "Thyroid Function",

  "lipid profile": "Lipid Profile",
  lipids: "Lipid Profile",
  cholesterol: "Lipid Profile",

  haematology: "Haematology",
  hematology: "Haematology",
  cbc: "Haematology",
  "complete blood count": "Haematology",
  "blood count": "Haematology",

  hormone: "Hormones",
  hormones: "Hormones",
  "hormone profile": "Hormones",
  "sex hormones": "Hormones",
  "reproductive hormones": "Hormones",

  "cardiac risk markers": "Cardiac Markers",
  "cardiac markers": "Cardiac Markers",
  cardiac: "Cardiac Markers",
  heart: "Cardiac Markers",

  "inflammation marker": "Inflammation",
  "inflammation markers": "Inflammation",
  inflammatory: "Inflammation",
  crp: "Inflammation",

  proteins: "Proteins",
  "protein profile": "Proteins",

  immunology: "Immunology",
  immune: "Immunology",
  serology: "Immunology",

  urinalysis: "Urinalysis",
  urine: "Urinalysis",
  "urine routine": "Urinalysis",
  "urine analysis": "Urinalysis",

  electrolytes: "Kidney Function",
  "blood glucose": "Diabetes & Glucose",
  biochemistry: "Other",
  "bone health": "Vitamins & Minerals",
  other: "Other",
};

const warnedCategories = new Set<string>();

function normalizeCategoryLookupKey(raw: string): string {
  return raw
    .toLowerCase()
    .trim()
    .replace(/[()]/g, "")
    .replace(/[_]+/g, " ")
    .replace(/\s*\/\s*/g, " / ")
    .replace(/[^a-z0-9\-&/+\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function canonicalizeCategory(raw: string | null | undefined): CanonicalCategory {
  if (!raw || !String(raw).trim()) return "Other";

  const original = String(raw).trim();
  const key = normalizeCategoryLookupKey(original);
  const exact = CATEGORY_MAP[key];
  if (exact) return exact;

  for (const [mapKey, canonical] of Object.entries(CATEGORY_MAP)) {
    if (key.includes(mapKey) || mapKey.includes(key)) {
      return canonical;
    }
  }

  if (!warnedCategories.has(original)) {
    warnedCategories.add(original);
    console.warn(`[Category] Unmapped category: "${original}" - add to CATEGORY_MAP`);
  }

  return "Other";
}
