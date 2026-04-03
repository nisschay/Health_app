import { MEDICAL_KNOWLEDGE_BASE, type KnowledgeEntry } from "@/lib/medicalKnowledge";

type RetrievalOptions = {
  conversationHistory?: string[];
  maxResults?: number;
  minScore?: number;
};

export type RetrievedGuideline = {
  entry: KnowledgeEntry;
  score: number;
  matchedTerms: string[];
  rationale: string[];
};

const STOP_WORDS = new Set([
  "the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "with", "my", "me", "about", "is", "are", "was", "were", "be", "been", "it", "this", "that", "how", "what", "why", "when", "should", "could", "would", "do", "does", "did", "please", "explain",
]);

const INTENT_TERMS = {
  trend: ["trend", "change", "changed", "improved", "worse", "increased", "decreased", "over time", "history"],
  urgency: ["danger", "dangerous", "urgent", "critical", "worry", "risk", "serious"],
  action: ["what should i do", "next step", "treatment", "diet", "exercise", "lifestyle", "plan"],
};

const CONCEPT_EXPANSIONS: Record<string, string[]> = {
  sugar: ["glucose", "glycemic", "hba1c", "diabetes", "prediabetes"],
  cholesterol: ["ldl", "hdl", "triglycerides", "lipid", "non-hdl"],
  liver: ["ast", "alt", "sgot", "sgpt", "bilirubin", "alp"],
  kidney: ["creatinine", "egfr", "renal", "urea", "urate"],
  thyroid: ["tsh", "free t4", "ft4", "thyroxine"],
  blood: ["hemoglobin", "platelet", "ferritin", "b12", "cbc"],
  inflammation: ["crp", "hs-crp", "inflammatory"],
};

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9%+\-\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(text: string): string[] {
  return normalizeText(text)
    .split(" ")
    .filter((token) => token.length > 1 && !STOP_WORDS.has(token));
}

function parseRecencyScore(lastUpdated: string): number {
  const match = /^(\d{4})(?:-(\d{2}))?$/.exec(lastUpdated.trim());
  if (!match) return 0;

  const year = Number(match[1]);
  const month = Number(match[2] ?? "1");
  const updatedAt = new Date(Date.UTC(year, Math.max(0, month - 1), 1));
  const ageMs = Date.now() - updatedAt.getTime();
  const ageMonths = ageMs / (1000 * 60 * 60 * 24 * 30);

  if (ageMonths <= 12) return 3;
  if (ageMonths <= 24) return 2;
  if (ageMonths <= 36) return 1;
  return 0;
}

function addMatchTerm(set: Set<string>, term: string): void {
  const clean = normalizeText(term);
  if (clean) {
    set.add(clean);
  }
}

function hasAnyIntent(query: string, terms: string[]): boolean {
  return terms.some((term) => query.includes(term));
}

export function retrieveRelevantGuidelines(
  userMessage: string,
  activeFindings: string[],
  options: RetrievalOptions = {},
): RetrievedGuideline[] {
  const maxResults = options.maxResults ?? 3;
  const minScore = options.minScore ?? 15;

  const historyText = (options.conversationHistory ?? []).slice(-6).join(" ");
  const combinedQuery = normalizeText(`${userMessage} ${historyText}`);
  const queryTokens = tokenize(combinedQuery);

  const expandedTokens = new Set<string>(queryTokens);
  for (const token of queryTokens) {
    const expansions = CONCEPT_EXPANSIONS[token];
    if (expansions) {
      for (const expansion of expansions) {
        expandedTokens.add(normalizeText(expansion));
      }
    }
  }

  const normalizedFindings = activeFindings.map((finding) => normalizeText(finding));

  const candidates: RetrievedGuideline[] = MEDICAL_KNOWLEDGE_BASE.map((entry) => {
    let score = 0;
    const matchedTerms = new Set<string>();
    const rationale: string[] = [];

    const entryNames = [...entry.testNames, ...entry.aliases].map((name) => normalizeText(name));
    const entryKeywords = entry.keywords.map((keyword) => normalizeText(keyword));
    const entryRelated = entry.relatedTests.map((test) => normalizeText(test));

    for (const finding of normalizedFindings) {
      for (const name of entryNames) {
        if (!finding || !name) continue;
        if (finding === name) {
          score += 28;
          addMatchTerm(matchedTerms, name);
          rationale.push("exact active finding match");
        } else if (finding.includes(name) || name.includes(finding)) {
          score += 18;
          addMatchTerm(matchedTerms, name);
          rationale.push("partial active finding match");
        }
      }

      for (const related of entryRelated) {
        if (!finding || !related) continue;
        if (finding === related || finding.includes(related) || related.includes(finding)) {
          score += 9;
          addMatchTerm(matchedTerms, related);
          rationale.push("related test proximity");
        }
      }
    }

    for (const name of entryNames) {
      if (!name) continue;
      if (combinedQuery.includes(name)) {
        score += 20;
        addMatchTerm(matchedTerms, name);
        rationale.push("query includes canonical test alias");
      }
    }

    for (const keyword of entryKeywords) {
      if (!keyword) continue;
      if (combinedQuery.includes(keyword)) {
        score += 8;
        addMatchTerm(matchedTerms, keyword);
        rationale.push("query keyword match");
      }
    }

    for (const token of expandedTokens) {
      if (!token || token.length < 2) continue;
      if (entryNames.some((name) => name.includes(token)) || entryKeywords.some((keyword) => keyword.includes(token))) {
        score += 3;
        addMatchTerm(matchedTerms, token);
      }
    }

    const queryHasTrendIntent = hasAnyIntent(combinedQuery, INTENT_TERMS.trend);
    const queryHasUrgencyIntent = hasAnyIntent(combinedQuery, INTENT_TERMS.urgency);
    const queryHasActionIntent = hasAnyIntent(combinedQuery, INTENT_TERMS.action);

    if (queryHasTrendIntent && entry.trendSignals.length > 0) {
      score += 5;
      rationale.push("trend intent alignment");
    }
    if (queryHasUrgencyIntent && entry.escalationTriggers.length > 0) {
      score += 5;
      rationale.push("urgency intent alignment");
    }
    if (queryHasActionIntent && entry.patientFriendlyActions.length > 0) {
      score += 4;
      rationale.push("action intent alignment");
    }

    if (entry.evidenceLevel === "A") {
      score += 2;
    } else if (entry.evidenceLevel === "B") {
      score += 1;
    }

    score += parseRecencyScore(entry.lastUpdated);

    return {
      entry,
      score,
      matchedTerms: [...matchedTerms],
      rationale,
    };
  });

  const sorted = candidates
    .filter((candidate) => candidate.score >= minScore)
    .sort((a, b) => b.score - a.score);

  // Keep retrieval diverse across categories unless there is a strong direct-match cluster.
  const categoryUsage = new Map<string, number>();
  const selected: RetrievedGuideline[] = [];

  for (const candidate of sorted) {
    if (selected.length >= maxResults) break;

    const category = candidate.entry.category;
    const used = categoryUsage.get(category) ?? 0;
    const hasStrongSignal = candidate.score >= 30;

    if (used >= 1 && !hasStrongSignal) {
      continue;
    }

    selected.push(candidate);
    categoryUsage.set(category, used + 1);
  }

  return selected;
}
