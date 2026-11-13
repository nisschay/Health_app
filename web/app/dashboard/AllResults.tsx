"use client";
import { useState } from "react";
import { groupByCategory, type TestSummary } from "@/lib/results";
import ResultRow from "./ResultRow";
export default function AllResults({
  tests,
  base,
}: {
  tests: TestSummary[];
  base: string;
}) {
  const [query, setQuery] = useState("");
  const groups = groupByCategory(
    tests.filter((t) =>
      [t.name, t.category, ...t.printedNames].some((n) =>
        n.toLowerCase().includes(query.toLowerCase()),
      ),
    ),
  );
  groups.sort(
    (a, b) => b.attention - a.attention || a.category.localeCompare(b.category),
  );
  return (
    <section className="section">
      <div className="section-head">
        <h2>
          All results <span>{tests.length}</span>
        </h2>
        <input
          aria-label="Search tests"
          className="search"
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Find a test"
        />
      </div>
      <div className="card">
        {groups.map((g, i) => (
          <details
            className="cat"
            key={`${g.category}-${query}`}
            open={!!query || i === 0}
          >
            <summary>
              <span className="chev">›</span>
              <strong className="cat-name">{g.category}</strong>
              <span className="cat-n">{g.tests.length} tests</span>
              <span className={`cat-state ${g.attention ? "high" : ""}`}>
                {g.attention
                  ? `${g.attention} outside range`
                  : "Nothing outside range"}
              </span>
            </summary>
            <ul className="result-list">
              {[...g.tests]
                .sort(
                  (a, b) =>
                    Number(b.latest.status === "critical") -
                      Number(a.latest.status === "critical") ||
                    Number(b.needsAttention) - Number(a.needsAttention) ||
                    a.name.localeCompare(b.name),
                )
                .map((t) => (
                  <ResultRow test={t} key={t.name} base={base} compact />
                ))}
            </ul>
          </details>
        ))}
        {!groups.length && (
          <p className="card-pad">
            No test matches &quot;{query}&quot;. Try a shorter word.
          </p>
        )}
      </div>
    </section>
  );
}
