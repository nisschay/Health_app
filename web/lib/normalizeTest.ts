import { canonicalizeCategory } from "./categoryMap";
import { normalizeTestName } from "./testNameMap";

export type NormalizableMedicalRecord = {
  Test_Name?: string | null;
  Original_Test_Name?: string | null;
  Test_Category?: string | null;
  Test_Date?: string | null;
  Status?: string | null;
  Result?: string | number | null;
  Result_Numeric?: number | null;
  Unit?: string | null;
  Reference_Range?: string | null;
  Aliases?: string[] | null;
  [key: string]: unknown;
};

const CATEGORY_PRIORITY: Record<string, number> = {
  "Haematology": 10,
  "Lipid Profile": 10,
  "Liver Function": 10,
  "Kidney Function": 10,
  "Diabetes & Glucose": 10,
  "Thyroid Function": 10,
  "Vitamins & Minerals": 10,
  Hormones: 10,
  "Cardiac Markers": 10,
  Immunology: 10,
  Urinalysis: 10,
  Inflammation: 10,
  Proteins: 5,
  Other: 1,
};

const TEST_CATEGORY_HINTS: Record<string, string> = {
  "aspartate aminotransferase": "Liver Function",
  "sgot/ast": "Liver Function",
  "ldl cholesterol": "Lipid Profile",
  "total leucocyte count": "Haematology",
  albumin: "Liver Function",
  "albumin/globulin ratio": "Liver Function",
  globulin: "Liver Function",
  triglycerides: "Lipid Profile",
  calcium: "Vitamins & Minerals",
  "malarial parasite": "Haematology",
  "rheumatoid factor": "Immunology",
};

const MUST_NOT_MERGE_PAIRS: Array<[string, string]> = [
  ["globulin", "albumin/globulin ratio"],
  ["albumin", "albumin/globulin ratio"],
  ["albumin", "alb creat ratio"],
  ["creatinine", "alb creat ratio"],
  ["iron", "total iron binding capacity"],
  ["iron", "transferrin saturation"],
  ["iron", "unsat iron-binding capacity"],
  ["apolipoprotein-a1", "apolipoprotein-b"],
  ["apo-a1", "apo-b"],
  ["apolipoprotein a", "apolipoprotein b"],
  ["bilirubin total", "bilirubin direct"],
  ["bilirubin total", "bilirubin indirect"],
  ["bilirubin total", "bilirubin conjugated"],
  ["bilirubin direct", "bilirubin indirect"],
  ["bilirubin conjugated", "bilirubin unconjugated"],
  ["neutrophils", "neut/lympho ratio"],
  ["neutrophil", "neut/lympho ratio"],
  ["prostate specific antigen", "free psa"],
  ["t3", "t4"],
  ["free t3", "free t4"],
  ["t3", "tsh"],
  ["t4", "tsh"],
];

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    const text = String(value ?? "").trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    result.push(text);
  }
  return result;
}

function coerceAliases(value: unknown): string[] {
  if (Array.isArray(value)) {
    return uniqueStrings(value.map((item) => String(item)));
  }
  if (typeof value === "string" && value.trim()) {
    return [value.trim()];
  }
  return [];
}

function tokenize(value: string): string[] {
  return value.split(/[\s\-/,.]+/).map((token) => token.trim()).filter(Boolean);
}

function initials(value: string): string {
  return tokenize(value.toLowerCase())
    .map((token) => token[0])
    .filter(Boolean)
    .join("");
}

function compact(value: string): string {
  return value.toLowerCase().replace(/[\s\-/,.]+/g, "");
}

export function structuralClean(raw: string | null | undefined): string {
  if (!raw || !String(raw).trim()) return "unknown test";

  let cleaned = String(raw).toLowerCase().trim();
  cleaned = cleaned.replace(
    /\s*\(\s*(urine|serum|plasma|blood|routine|total|electrical\s+impedance|rbc\s+histogram|appearance)\s*\)/gi,
    "",
  );

  const source = cleaned;
  cleaned = source.replace(/\(\s*([a-z]{2,6})\s*\)/gi, (full, abbrRaw: string, offset: number) => {
    const abbr = abbrRaw.toLowerCase().trim();
    const baseSegment = source.slice(0, offset).trim();
    const baseTokens = baseSegment.split(/\s+/).filter(Boolean);
    const baseInitials = baseTokens.map((token) => token[0]).join("");
    const compactBase = baseTokens.join("");
    if ((baseInitials && abbr && baseInitials.includes(abbr)) || (compactBase && compactBase.includes(abbr))) {
      return "";
    }
    return full;
  });

  cleaned = cleaned
    .replace(/[()]/g, " ")
    .replace(/\s*\/\s*/g, "/")
    .replace(/\s*-\s*/g, "-")
    .replace(/[^a-z0-9%\-/,.\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  return cleaned || "unknown test";
}

function normalizedBlocklistKey(value: string): string {
  return structuralClean(value)
    .replace(/[\-/,.]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function isBlocklisted(a: string, b: string): boolean {
  const aNorm = normalizedBlocklistKey(a);
  const bNorm = normalizedBlocklistKey(b);
  const [shorter, longer] = aNorm.length <= bNorm.length ? [aNorm, bNorm] : [bNorm, aNorm];

  return MUST_NOT_MERGE_PAIRS.some(([p1, p2]) => {
    const p1Norm = normalizedBlocklistKey(p1);
    const p2Norm = normalizedBlocklistKey(p2);
    return (shorter.includes(p1Norm) && longer.includes(p2Norm))
      || (shorter.includes(p2Norm) && longer.includes(p1Norm));
  });
}

export function areSameTest(a: string, b: string): boolean {
  const aKey = structuralClean(a);
  const bKey = structuralClean(b);

  if (aKey === bKey) {
    return true;
  }

  if (isBlocklisted(aKey, bKey)) {
    return false;
  }

  const aIsRatio = aKey.includes("ratio") || aKey.includes("/");
  const bIsRatio = bKey.includes("ratio") || bKey.includes("/");
  if (aIsRatio !== bIsRatio) {
    return false;
  }

  const aIsPercent = aKey.includes("percent") || aKey.endsWith("%") || aKey.includes("(%)");
  const bIsPercent = bKey.includes("percent") || bKey.endsWith("%") || bKey.includes("(%)");
  const aIsAbsolute = /(?:absolute|count|(?:\u00d7|x)10|10\^)/.test(aKey);
  const bIsAbsolute = /(?:absolute|count|(?:\u00d7|x)10|10\^)/.test(bKey);
  if ((aIsPercent && bIsAbsolute) || (bIsPercent && aIsAbsolute)) {
    return false;
  }

  if (aKey.includes("bilirubin") && bKey.includes("bilirubin")) {
    const qualifiers = ["direct", "indirect", "total", "conjugated", "unconjugated"] as const;
    const aQualifier = qualifiers.find((q) => aKey.includes(q));
    const bQualifier = qualifiers.find((q) => bKey.includes(q));
    if (aQualifier && bQualifier && aQualifier !== bQualifier) {
      return false;
    }
    if ((aQualifier && !bQualifier) || (bQualifier && !aQualifier)) {
      return false;
    }
  }

  const compactA = compact(aKey);
  const compactB = compact(bKey);
  const initialsA = initials(aKey);
  const initialsB = initials(bKey);
  if (compactA === initialsB || compactB === initialsA) {
    return true;
  }

  const tokensA = tokenize(aKey).filter((token) => token.length > 2);
  const tokensB = tokenize(bKey).filter((token) => token.length > 2);

  if (tokensA.length >= 2 && tokensB.length >= 2) {
    const setA = new Set(tokensA);
    const setB = new Set(tokensB);
    const shorter = setA.size <= setB.size ? setA : setB;
    const longer = setA.size <= setB.size ? setB : setA;
    if (shorter.size > 0) {
      let overlap = 0;
      for (const token of shorter) {
        if (longer.has(token)) overlap += 1;
      }
      if (overlap / shorter.size >= 0.8) {
        return true;
      }
    }
  }

  const strippedA = compact(aKey);
  const strippedB = compact(bKey);
  if (strippedA === strippedB) {
    return true;
  }

  const [shortText, longText] = aKey.length <= bKey.length ? [aKey, bKey] : [bKey, aKey];
  if (shortText.length >= 8 && longText.startsWith(shortText)) {
    return true;
  }

  return false;
}

function normalizeStatus(raw: string | null | undefined): string {
  if (!raw) return "N/A";
  const normalized = String(raw).trim().toLowerCase();
  if (!normalized || normalized === "n/a" || normalized === "na" || normalized === "not applicable") return "N/A";
  if (normalized.includes("critical")) return "Critical";
  if (normalized.includes("insufficient")) return "Insufficient";
  if (normalized.includes("borderline")) return "Borderline";
  if (normalized.includes("positive")) return "Positive";
  if (normalized.includes("negative")) return "Negative";
  if (normalized.includes("flag")) return "Flagged";
  if (normalized.includes("high")) return "High";
  if (normalized.includes("low")) return "Low";
  if (normalized.includes("normal") || normalized.includes("within")) return "Normal";
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function parseNumeric(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  const parsed = Number.parseFloat(String(value ?? "").replace(/,/g, "").trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function parseMedicalDate(value: string | null | undefined): number {
  if (!value) return Number.MAX_SAFE_INTEGER;
  const raw = value.trim();
  const ddmmyyyy = raw.match(/^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})$/);
  if (ddmmyyyy) {
    const day = ddmmyyyy[1]!.padStart(2, "0");
    const month = ddmmyyyy[2]!.padStart(2, "0");
    const year = ddmmyyyy[3]!.length === 2 ? `20${ddmmyyyy[3]}` : ddmmyyyy[3]!;
    return new Date(`${year}-${month}-${day}`).getTime();
  }
  const parsed = new Date(raw).getTime();
  return Number.isNaN(parsed) ? Number.MAX_SAFE_INTEGER : parsed;
}

function rowCompletenessScore(row: NormalizableMedicalRecord): number {
  let score = 0;
  if (parseNumeric(row.Result) !== null) score += 2;
  if (String(row.Reference_Range ?? "").trim()) score += 2;
  if (String(row.Unit ?? "").trim()) score += 1;
  const status = String(row.Status ?? "").trim().toLowerCase();
  if (status && status !== "n/a") score += 1;
  return score;
}

function mergeReadingsByDate(rows: NormalizableMedicalRecord[]): NormalizableMedicalRecord[] {
  const byDate = new Map<string, NormalizableMedicalRecord[]>();

  for (const row of rows) {
    const dateKey = String(row.Test_Date ?? "N/A").trim() || "N/A";
    if (!byDate.has(dateKey)) byDate.set(dateKey, []);
    byDate.get(dateKey)!.push(row);
  }

  const merged: NormalizableMedicalRecord[] = [];
  for (const [, entries] of byDate) {
    if (entries.length === 1) {
      merged.push({ ...entries[0] });
      continue;
    }

    const withNumeric = entries.filter((entry) => parseNumeric(entry.Result) !== null);
    const pool = withNumeric.length > 0 ? withNumeric : entries;

    const best = [...pool].sort((a, b) => {
      const scoreDiff = rowCompletenessScore(b) - rowCompletenessScore(a);
      if (scoreDiff !== 0) return scoreDiff;
      const refDiff = String(b.Reference_Range ?? "").length - String(a.Reference_Range ?? "").length;
      if (refDiff !== 0) return refDiff;
      return String(b.Unit ?? "").length - String(a.Unit ?? "").length;
    })[0]!;

    merged.push({ ...best });
  }

  return merged.sort((a, b) => parseMedicalDate(String(a.Test_Date ?? "N/A")) - parseMedicalDate(String(b.Test_Date ?? "N/A")));
}

function categoryFromTestHint(name: string | null | undefined): string | null {
  if (!name) return null;
  const cleaned = structuralClean(name);
  for (const [hint, category] of Object.entries(TEST_CATEGORY_HINTS)) {
    if (areSameTest(cleaned, hint) || cleaned.includes(hint) || hint.includes(cleaned)) {
      return category;
    }
  }
  return null;
}

function resolveCategory(categories: string[], canonicalName: string): string {
  const hinted = categoryFromTestHint(canonicalName);
  if (hinted) return hinted;

  const normalized = categories.map((category) => canonicalizeCategory(category || "Other"));
  const nonOther = normalized.filter((category) => category !== "Other");
  if (nonOther.length === 0) return "Other";

  const frequency = new Map<string, number>();
  for (const category of nonOther) {
    frequency.set(category, (frequency.get(category) ?? 0) + 1);
  }

  return [...frequency.entries()]
    .sort((a, b) => {
      const priorityDiff = (CATEGORY_PRIORITY[b[0]] ?? 0) - (CATEGORY_PRIORITY[a[0]] ?? 0);
      if (priorityDiff !== 0) return priorityDiff;
      return b[1] - a[1];
    })[0]![0];
}

function pickCanonicalName(names: string[]): string {
  const unique = uniqueStrings(names);
  if (unique.length === 0) return "Unknown Test";

  const ranked = unique
    .map((name) => {
      const words = name.split(/\s+/).filter(Boolean);
      const hasBrackets = /\(.*\)/.test(name);
      const hasFullWords = words.some((word) => word.length > 4);
      const lettersOnly = name.replace(/[^A-Za-z]/g, "");
      const isAllCaps = Boolean(lettersOnly) && lettersOnly === lettersOnly.toUpperCase();
      const isPureAbbr = words.length > 0 && words.every((word) => word.replace(/[^A-Za-z]/g, "").length <= 4);

      let score = name.length;
      if (hasBrackets && hasFullWords) score += 20;
      if (isAllCaps) score -= 15;
      if (isPureAbbr) score -= 20;
      return { name: name.trim(), score };
    })
    .sort((a, b) => b.score - a.score);

  return ranked[0]!.name;
}

function normalizeSingleRecord<T extends NormalizableMedicalRecord>(row: T): T {
  const normalized = { ...row };
  const rawName = String(row.Original_Test_Name ?? row.Test_Name ?? "").trim();

  normalized.Original_Test_Name = rawName || normalized.Original_Test_Name || null;
  normalized.Test_Name = normalizeTestName(rawName || String(row.Test_Name ?? "Unknown Test"));
  normalized.Test_Category = canonicalizeCategory(String(row.Test_Category ?? "Other"));
  normalized.Status = normalizeStatus(String(row.Status ?? "N/A"));

  const aliases = uniqueStrings([
    ...coerceAliases(row.Aliases),
    rawName,
    String(normalized.Test_Name ?? "").trim(),
  ]);
  normalized.Aliases = aliases;

  if (normalized.Result_Numeric === undefined || normalized.Result_Numeric === null) {
    const parsed = parseNumeric(normalized.Result);
    normalized.Result_Numeric = parsed;
  }

  return normalized;
}

export function normalizeRecordsForClient<T extends NormalizableMedicalRecord>(records: T[]): T[] {
  const normalizedRows = records.filter((record): record is T => Boolean(record)).map((record) => normalizeSingleRecord(record));
  if (normalizedRows.length === 0) return [];

  const keyBuckets = new Map<string, T[]>();
  for (const row of normalizedRows) {
    const key = structuralClean(String(row.Test_Name ?? row.Original_Test_Name ?? "Unknown Test"));
    if (!keyBuckets.has(key)) keyBuckets.set(key, []);
    keyBuckets.get(key)!.push(row);
  }

  const keys = [...keyBuckets.keys()];
  const adjacency = new Map<string, Set<string>>();
  for (const key of keys) {
    adjacency.set(key, new Set<string>());
  }

  for (let i = 0; i < keys.length; i += 1) {
    const left = keys[i]!;
    for (let j = i + 1; j < keys.length; j += 1) {
      const right = keys[j]!;
      if (areSameTest(left, right)) {
        adjacency.get(left)!.add(right);
        adjacency.get(right)!.add(left);
      }
    }
  }

  const clusters = new Map<string, string[]>();
  const visited = new Set<string>();

  for (const key of keys) {
    if (visited.has(key)) continue;

    const stack = [key];
    const cluster: string[] = [];

    while (stack.length > 0) {
      const current = stack.pop()!;
      if (visited.has(current)) continue;
      visited.add(current);
      cluster.push(current);

      for (const neighbor of adjacency.get(current) ?? []) {
        if (!visited.has(neighbor)) stack.push(neighbor);
      }
    }

    const canonicalKey = [...cluster].sort((a, b) => b.length - a.length)[0]!;
    clusters.set(canonicalKey, cluster);
  }

  const deduped: T[] = [];

  for (const [, clusterKeys] of clusters) {
    const rows = clusterKeys.flatMap((clusterKey) => keyBuckets.get(clusterKey) ?? []);
    const rawNames = uniqueStrings(
      rows.map((row) => String(row.Original_Test_Name ?? row.Test_Name ?? "").trim()),
    );

    const aliases = uniqueStrings([
      ...rows.flatMap((row) => coerceAliases(row.Aliases)),
      ...rawNames,
    ]);

    const canonicalName = normalizeTestName(pickCanonicalName(rawNames));
    const canonicalCategory = resolveCategory(
      rows.map((row) => String(row.Test_Category ?? "Other")),
      canonicalName,
    );

    const mergedRows = mergeReadingsByDate(rows);
    for (const entry of mergedRows) {
      const next = { ...entry } as T;
      const previousName = String(next.Test_Name ?? next.Original_Test_Name ?? canonicalName).trim();
      next.Test_Name = canonicalName;
      next.Test_Category = canonicalCategory;
      next.Aliases = uniqueStrings([...aliases, canonicalName]);
      if (!next.Original_Test_Name) {
        next.Original_Test_Name = previousName || canonicalName;
      }
      if (next.Result_Numeric === undefined || next.Result_Numeric === null) {
        next.Result_Numeric = parseNumeric(next.Result);
      }
      deduped.push(next);
    }
  }

  return deduped.sort((a, b) => {
    const byDate = parseMedicalDate(String(a.Test_Date ?? "N/A")) - parseMedicalDate(String(b.Test_Date ?? "N/A"));
    if (byDate !== 0) return byDate;
    return String(a.Test_Name ?? "").localeCompare(String(b.Test_Name ?? ""));
  });
}

export function normalizeAnalysisPayload<T extends { records?: NormalizableMedicalRecord[]; total_records?: number }>(payload: T): T {
  const rows = Array.isArray(payload.records) ? payload.records : [];
  const normalizedRecords = normalizeRecordsForClient(rows);
  return {
    ...payload,
    records: normalizedRecords,
    total_records: normalizedRecords.length,
  };
}
