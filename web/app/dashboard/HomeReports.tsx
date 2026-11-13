import type { AnalysisResponse } from "@/lib/api";
import { formatDate } from "@/lib/format";
import { parseMedicalDate } from "@/lib/medicalDate";
import { normalizeStatus } from "@/lib/results";
export default function HomeReports({
  reports,
}: {
  reports: Record<string, AnalysisResponse>;
}) {
  const recent = Object.entries(reports)
    .flatMap(([id, a]) =>
      Array.from(
        new Set(a.records.map((r) => r.Source_Filename || "Report")),
      ).map((file) => ({
        id,
        file,
        rows: a.records.filter((r) => (r.Source_Filename || "Report") === file),
      })),
    )
    .sort(
      (a, b) =>
        parseMedicalDate(b.rows[0]?.Test_Date) -
        parseMedicalDate(a.rows[0]?.Test_Date),
    );
  return (
    <section className="home-reports">
      <h2 className="h2">Recent reports</h2>
      {recent.slice(0, 5).map(({ id, file, rows }) => (
        <a className="rep" key={id + file} href={`?study=${id}`}>
          <span className="rep-date">{formatDate(rows[0]?.Test_Date)}</span>
          <span className="rep-what">{rows[0]?.Lab_Name || file}</span>
          <span className="rep-count">
            {rows.length} results ·{" "}
            {
              rows.filter((r) =>
                ["high", "low", "critical"].includes(normalizeStatus(r.Status)),
              ).length
            }{" "}
            outside range
          </span>
        </a>
      ))}
      {!recent.length && (
        <p className="card-pad muted">Your reports will appear here.</p>
      )}
    </section>
  );
}
