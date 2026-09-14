const UNPARSEABLE_SORT_VALUE = Number.MAX_SAFE_INTEGER;
const DAY_FIRST_PATTERN = /^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})$/;

/** Sortable timestamp for a lab date, accepting dd-mm-yyyy as well as ISO. */
export function parseMedicalDate(value: string | null | undefined): number {
  if (!value) return UNPARSEABLE_SORT_VALUE;
  const raw = value.trim();

  const dayFirst = raw.match(DAY_FIRST_PATTERN);
  if (dayFirst) {
    const day = dayFirst[1]!.padStart(2, "0");
    const month = dayFirst[2]!.padStart(2, "0");
    const year = dayFirst[3]!.length === 2 ? `20${dayFirst[3]}` : dayFirst[3]!;
    const parsed = new Date(`${year}-${month}-${day}`).getTime();
    return Number.isNaN(parsed) ? UNPARSEABLE_SORT_VALUE : parsed;
  }

  const parsed = new Date(raw).getTime();
  return Number.isNaN(parsed) ? UNPARSEABLE_SORT_VALUE : parsed;
}
