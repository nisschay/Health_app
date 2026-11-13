"use client";
import { useState } from "react";
import type { AnalysisResponse } from "@/lib/api";
import {
  summariseStudy,
  orderForAttention,
  doctorList,
  type TestSummary,
} from "@/lib/results";
import { formatDate, plural } from "@/lib/format";
import { StatusChip } from "./ui";
import ResultRow from "./ResultRow";
import AllResults from "./AllResults";
export default function ResultsScreen({
  analysis,
  tests,
  base,
  person,
  title,
  download,
}: {
  analysis: AnalysisResponse;
  tests: TestSummary[];
  base: string;
  person: string;
  title: string;
  download: () => void;
}) {
  const s = summariseStudy(tests);
  const notes = doctorList(tests);
  const [copied, setCopied] = useState("");
  const files = Array.from(
    new Set([
      ...(analysis.combined_report_file_names ?? []),
      ...analysis.records.map((r) => r.Source_Filename || "Report"),
    ]),
  );
  return (
    <div className="stack sections">
      {s.critical > 0 && (
        <section className="banner banner-critical" role="alert">
          <div>
            <strong>{plural(s.critical, "result")} marked critical</strong>
            {s.criticalTests.map((t) => (
              <p key={t.name}>
                {t.name} was {t.latest.valueText} {t.latest.unit} on{" "}
                {formatDate(t.latest.date)} (lab range {t.latest.rangeText}).
                Please contact a doctor promptly.
              </p>
            ))}
          </div>
        </section>
      )}
      <section className="card summary">
        <div className="summary-main">
          <h2 className="label">Right now</h2>
          <p className="summary-headline">
            {!s.total ? (
              "No results were read from these reports"
            ) : s.attention ? (
              <>
                <b>{s.attention}</b> of {s.total} tests need attention
              </>
            ) : s.inRange === s.total ? (
              `All ${s.total} tests are in range`
            ) : (
              `${s.inRange} of ${s.total} tests are in range`
            )}
          </p>
          <div className="tally">
            {(
              [
                "critical",
                "high",
                "low",
                "borderline",
                "detected",
                "normal",
                "none",
              ] as const
            ).map((status) => {
              const n = tests.filter((t) => t.latest.status === status).length;
              return (
                n > 0 && (
                  <span className="row" key={status}>
                    {n}
                    <StatusChip status={status} />
                  </span>
                )
              );
            })}
          </div>
          <p className="small muted">
            Counted from the latest reading of each test. Most recent report:{" "}
            {formatDate(s.latestReportDate)}.
          </p>
        </div>
        <div className="summary-side">
          {s.previousReportDate ? (
            <>
              <h3 className="h3">
                Since the report of {formatDate(s.previousReportDate)}
              </h3>
              <p>{s.better} getting better</p>
              <p>{s.worse} getting worse</p>
              <p>{s.backInRange} back in range</p>
            </>
          ) : (
            <p>Add an older or newer report to see what is changing.</p>
          )}
        </div>
      </section>
      <section className="card">
        <h2 className="card-head h2">Needs attention</h2>
        <ul className="result-list">
          {orderForAttention(tests.filter((t) => t.needsAttention)).map((t) => (
            <ResultRow key={t.name} test={t} base={base} />
          ))}
        </ul>
        {!s.attention && (
          <p className="card-pad">
            {s.unknown
              ? "No results are marked outside the range. Some reports did not include a range."
              : "Nothing needs attention. Every test is inside the lab’s range in the latest reports."}
          </p>
        )}
      </section>
      <section className="card card-pad stack">
        <h2 className="h2">To discuss with the doctor</h2>
        <p className="small muted">
          A starting list from these results, not advice.
        </p>
        <ul className="stack">
          {(notes.length
            ? notes
            : [
                s.attention
                  ? "Ask your doctor about the results listed above."
                  : "Nothing is marked outside the lab’s ranges in the latest reports.",
              ]
          ).map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
        <div className="row row-wrap">
          <button
            className="btn btn-secondary"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(
                  `${person}\n${title}\n${formatDate(s.latestReportDate)}\n${notes.join("\n")}`,
                );
                setCopied("Copied");
              } catch {
                setCopied("Could not copy. Select the text and copy it.");
              }
            }}
          >
            Copy this list
          </button>
          <button className="btn btn-secondary" onClick={download}>
            PDF for the doctor
          </button>
          <span role="status">{copied}</span>
        </div>
      </section>
      <AllResults tests={tests} base={base} />
      <section className="card">
        <h2 className="card-head h2">Reports in this study</h2>
        {files.map((file) => {
          const rows = analysis.records.filter(
            (r) => (r.Source_Filename || "Report") === file,
          );
          return (
            <div className="report-row" key={file}>
              <span>▤</span>
              <div>
                <strong>{formatDate(rows[0]?.Test_Date)}</strong> ·{" "}
                {rows[0]?.Lab_Name}
                <span className="report-file" title={file}>
                  {file}
                </span>
              </div>
              <span className="small muted">
                {plural(rows.length, "result")},{" "}
                {
                  rows.filter(
                    (r) =>
                      !["normal", "negative", ""].includes(
                        r.Status?.toLowerCase() ?? "",
                      ),
                  ).length
                }{" "}
                outside range
              </span>
            </div>
          );
        })}
      </section>
    </div>
  );
}
