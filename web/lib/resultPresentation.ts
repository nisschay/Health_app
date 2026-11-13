import type { TestSummary } from "./results";
/** Latest results first, with critical results always first, as in the v2 study view. */
export function recentAttention(tests: TestSummary[]) {
  const time = (t: TestSummary) =>
    t.latest.time === Number.MAX_SAFE_INTEGER ? -Infinity : t.latest.time;
  return tests
    .filter((t) => t.needsAttention)
    .sort(
      (a, b) =>
        Number(b.latest.status === "critical") -
          Number(a.latest.status === "critical") ||
        time(b) - time(a) ||
        a.name.localeCompare(b.name),
    );
}
