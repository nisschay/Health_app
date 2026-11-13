"use client";
import { useState } from "react";
import type { AnalysisResponse } from "@/lib/api";
import { normalizeStatus } from "@/lib/results";
import { formatDate } from "@/lib/format";
import { parseMedicalDate } from "@/lib/medicalDate";
export default function ReportsList({
  analysis,
}: {
  analysis: AnalysisResponse;
}) {
  const [all, setAll] = useState(false);
  const files = Array.from(
    new Set([
      ...(analysis.combined_report_file_names ?? []),
      ...analysis.records.map((r) => r.Source_Filename || "Report"),
    ]),
  )
    .map((file) => ({
      file,
      rows: analysis.records.filter(
        (r) => (r.Source_Filename || "Report") === file,
      ),
    }))
    .sort(
      (a, b) =>
        parseMedicalDate(b.rows[0]?.Test_Date) -
        parseMedicalDate(a.rows[0]?.Test_Date),
    );
  const mixed =
    new Set(analysis.records.map((r) => r.Lab_Name).filter(Boolean)).size > 1;
  return (
    <section className="section">
      <div className="section-head">
        <h2>
          Reports <span>{files.length}</span>
        </h2>
        <span className="hint">Newest first</span>
      </div>
      <div className="card">
        {(all ? files : files.slice(0, 4)).map(({ file, rows }) => (
          <div className="rep" key={file}>
            <span className="rep-date">{formatDate(rows[0]?.Test_Date)}</span>
            <span className="rep-what">
              {mixed && rows[0]?.Lab_Name}
              <small title={file}>{file}</small>
            </span>
            <span className="rep-count">
              {rows.length} results ·{" "}
              {
                rows.filter((r) =>
                  ["critical", "high", "low"].includes(
                    normalizeStatus(r.Status),
                  ),
                ).length
              }{" "}
              outside range
            </span>
          </div>
        ))}
        {files.length > 4 && (
          <button className="rep-more" onClick={() => setAll(!all)}>
            {all ? "Show latest 4 reports" : `Show all ${files.length} reports`}
            ⌄
          </button>
        )}
      </div>
    </section>
  );
}
