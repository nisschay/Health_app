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
