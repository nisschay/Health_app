"use client";
import { useState } from "react";
import type { AnalysisResponse } from "@/lib/api";
import { type TestSummary } from "@/lib/results";
import { recentAttention } from "@/lib/resultPresentation";
import ResultRow from "./ResultRow";
import AllResults from "./AllResults";
import StudySummary from "./StudySummary";
import ReportsList from "./ReportsList";
export default function ResultsScreen({
  analysis,
  tests,
  base,
  ask,
}: {
  analysis: AnalysisResponse;
  tests: TestSummary[];
  base: string;
  ask: React.ReactNode;
}) {
  const [filter, setFilter] = useState("all");
  const attention = recentAttention(tests);
  const filtered = attention.filter(
    (t) => filter === "all" || t.latest.status === filter,
  );
  return (
    <div>
      <StudySummary tests={tests} base={base} />
      {ask}
      <section className="section">
        <div className="section-head">
          <h2>
            Needs attention <span>{attention.length}</span>
          </h2>
          <div className="pills" aria-label="Filter results needing attention">
            {["all", "critical", "high", "low"].map((f) => (
              <button
                key={f}
                className={`pill ${filter === f ? "on" : ""}`}
                aria-pressed={filter === f}
                onClick={() => setFilter(f)}
              >
                {f[0].toUpperCase() + f.slice(1)}{" "}
                <b>
                  {f === "all"
                    ? attention.length
                    : attention.filter((t) => t.latest.status === f).length}
                </b>
              </button>
            ))}
          </div>
        </div>
        <ul className="card rows">
          {filtered.map((t) => (
            <ResultRow key={t.name} test={t} base={base} />
          ))}
        </ul>
        {!filtered.length && (
          <p className="card card-pad">
            No{" "}
            {filter === "all" ? "results outside range" : filter + " results"}.
          </p>
        )}
      </section>
      <AllResults tests={tests} base={base} />
      <ReportsList analysis={analysis} />
    </div>
  );
}
