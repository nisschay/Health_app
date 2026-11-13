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
  const [filter, setFilter] = useState("all");
  const groups = groupByCategory(
    tests.filter(
      (t) =>
        (filter === "all" ||
          (filter === "attention"
            ? t.needsAttention
            : t.latest.status === "normal")) &&
        [t.name, t.category, ...t.printedNames].some((n) =>
          n.toLowerCase().includes(query.toLowerCase()),
        ),
    ),
  );
  return (
    <section className="card">
      <h2 className="card-head h2">All results</h2>
      <div className="toolbar">
        <label className="field grow">
          Search tests
          <input
            className="input"
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Test name or category"
          />
        </label>
        <div className="choices" aria-label="Filter results">
          {[
            ["all", `All ${tests.length}`],
            [
              "attention",
              `Needs attention ${tests.filter((t) => t.needsAttention).length}`,
            ],
            [
              "normal",
              `In range ${tests.filter((t) => t.latest.status === "normal").length}`,
            ],
          ].map(([v, label]) => (
            <button
              className="choice"
              aria-pressed={filter === v}
              key={v}
              onClick={() => setFilter(v!)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      {groups.map((g, i) => (
        <details
          className="group"
          key={`${g.category}-${query}-${filter}`}
          open={!!query || i === 0}
        >
          <summary>
            <span>›</span>
            <strong className="grow">{g.category}</strong>
            <span className="group-count">{g.tests.length} tests</span>
            <span className={`chip ${g.attention ? "high" : ""}`}>
              {g.attention
                ? `${g.attention} need attention`
                : "No results needing attention"}
            </span>
          </summary>
          <ul className="result-list">
            {g.tests.map((t) => (
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
    </section>
  );
}
