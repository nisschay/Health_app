"use client";
import { useEffect, useState } from "react";
import {
  fetchStudyCombinedReport,
  deleteProfile,
  type DashboardSummary,
  type AnalysisHistoryItem,
  type AnalysisResponse,
} from "@/lib/api";
import { summariseTests } from "@/lib/results";
import { formatDate, plural } from "@/lib/format";
import { buildUserInitials, relationshipLabel } from "./helpers";
import { Disclaimer, Skeleton } from "./ui";
import { ChevronRight, Plus } from "lucide-react";
import { recentAttention } from "@/lib/resultPresentation";
import StudySummary from "./StudySummary";
import ResultRow from "./ResultRow";
import HomeReports from "./HomeReports";
export default function HomeScreen({
  name,
  summary,
  history,
  reload,
  run,
}: {
  name: string;
  summary: DashboardSummary | null;
  history: AnalysisHistoryItem[];
  reload: () => void;
  run: <T>(f: () => Promise<T>) => Promise<T>;
}) {
  const [reports, setReports] = useState<Record<string, AnalysisResponse>>({});
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const queue = [...(summary?.profiles.flatMap((p) => p.studies) ?? [])].sort(
      (a, b) => b.last_updated.localeCompare(a.last_updated),
    );
    const worker = async () => {
      while (active && queue.length) {
        const study = queue.shift()!;
        try {
          const r = await run(() => fetchStudyCombinedReport(study.id));
          if (active) setReports((prev) => ({ ...prev, [study.id]: r }));
        } catch {
          /* The study row remains available for retry. */
        }
      }
    };
    void Promise.all([worker(), worker()]);
    return () => {
      active = false;
    };
  }, [summary, run]);
  const first = name.split(" ")[0];
  const hour = new Date().getHours();
  return (
    <main className="page home-page stack">
      {/* A first visit gets the welcome card below instead of a second greeting. */}
      {!(summary && !summary.total_reports && !history.length) && (
        <>
          <h1 className="display">
            Good {hour < 12 ? "morning" : hour < 17 ? "afternoon" : "evening"},{" "}
            {first}
          </h1>
          <p className="soft">
            {summary
              ? `${summary.profiles.length} ${summary.profiles.length === 1 ? "person" : "people"}, ${summary.total_reports} reports.`
              : "Your family’s latest reports."}
          </p>
        </>
      )}
      {error && <p role="alert">{error}</p>}
      {!summary ? (
        <Skeleton />
      ) : (
        <>
          {!summary.total_reports && !history.length && (
            <section className="card card-pad stack welcome">
              <h2 className="display">Welcome, {first}</h2>
              <p>
                Add a lab report PDF for yourself or someone in your family. We
                will read every result and show you what needs attention.
              </p>
              <a className="btn btn-primary" href="/dashboard?view=upload">
                Add your first report
              </a>
              <ol className="how">
                <li>Add the PDFs</li>
                <li>See what matters</li>
                <li>Watch the trend</li>
              </ol>
            </section>
          )}
          <div className="v2-home">
            <div className="col">
              {summary.profiles.map((p) => (
                <div className="col" key={p.profile_id}>
                  <section className="card">
                    <div className="person-head">
                      <span className="avatar lg">
                        {buildUserInitials(p.full_name)}
                      </span>
                      <div className="grow">
                        <h2>
                          {p.full_name}{" "}
                          <span className="tag">
                            {p.relationship === "self"
                              ? "You"
                              : relationshipLabel(p.relationship)}
                          </span>
                        </h2>
                        {p.relationship !== "self" && (
                          <details className="family-actions">
                            <summary>Manage family member</summary>
                            <button
                              className="btn btn-quiet"
                              onClick={async () => {
                                if (
                                  !confirm(
                                    `Remove ${p.full_name} and all their studies and reports? This cannot be undone.`,
                                  )
                                )
                                  return;
                                try {
                                  await run(() => deleteProfile(p.profile_id));
                                  reload();
                                } catch (e) {
                                  setError(
                                    e instanceof Error
                                      ? e.message
                                      : "Could not remove this family member.",
                                  );
                                }
                              }}
                            >
                              Remove this family member
                            </button>
                          </details>
                        )}
                        <p>
                          {p.studies[0]?.name} ·{" "}
                          {p.studies.reduce((n, s) => n + s.report_count, 0)}{" "}
                          reports{" "}
                          {p.studies[0]?.consistent_lab_name &&
                            ` · ${p.studies[0].consistent_lab_name}`}
                        </p>
                      </div>
                      <a
                        className="btn btn-secondary"
                        href={`?view=upload&profile=${p.profile_id}`}
                      >
                        <Plus size={16} />
                        Add reports
                      </a>
                    </div>
                    {p.studies.map((study) => {
                      const tests = reports[study.id]
                        ? summariseTests(reports[study.id].records)
                        : null;
                      return (
                        <div key={study.id}>
                          {tests ? (
                            <StudySummary
                              tests={tests}
                              base={`?study=${study.id}`}
                              home
                            />
                          ) : (
                            <div className="card-pad">
                              <p>{study.name}</p>
                              <p className="muted">Loading latest results...</p>
                            </div>
                          )}
                          <div className="study-foot card-pad">
                            <span className="eyebrow">
                              {formatDate(study.range_start)} to{" "}
                              {formatDate(study.range_end)}
                            </span>
                            <a
                              className="btn btn-primary"
                              href={`?study=${study.id}`}
                            >
                              See all results <ChevronRight size={16} />
                            </a>
                          </div>
                        </div>
                      );
                    })}
                  </section>
                  {p.studies.map((study) => {
                    const tests = reports[study.id]
                      ? recentAttention(
                          summariseTests(reports[study.id].records).filter(
                            (t) => t.needsAttention,
                          ),
                        )
                      : [];
                    return (
                      tests.length > 0 && (
                        <section className="card latest" key={study.id}>
                          <div className="mini-title">
                            <h2>Most recent results outside range</h2>
                            <a className="link" href={`?study=${study.id}`}>
                              All {tests.length} ›
                            </a>
                          </div>
                          <ul className="rows">
                            {tests.slice(0, 3).map((t) => (
                              <ResultRow
                                key={t.name}
                                test={t}
                                base={`?study=${study.id}`}
                              />
                            ))}
                          </ul>
                        </section>
                      )
                    );
                  })}
                </div>
              ))}
            </div>
            <div className="col">
              <HomeReports reports={reports} />
              <section className="invite">
                <Plus size={24} />
                <div>
                  <h2>Add a family member</h2>
                  <p>
                    Keep a parent&apos;s, partner&apos;s or child&apos;s reports
                    here too. Each person gets their own results and history.
                  </p>
                  <a
                    className="btn btn-secondary"
                    href="?view=upload&newPerson=1"
                  >
                    <Plus size={16} />
                    Add a family member
                  </a>
                </div>
              </section>
            </div>
          </div>
          {history.length > 0 && (
            <section className="stack">
              <h2 className="h2">Earlier one-off analyses</h2>
              <div className="card">
                {history.map((h) => (
                  <a
                    className="study-row"
                    key={h.id}
                    href={`?analysis=${h.id}`}
                  >
                    <div>
                      <strong>Report of {formatDate(h.report_date)}</strong>
                      <p className="small muted">
                        {h.patient_name} · {h.lab_name} ·{" "}
                        {plural(h.total_records, "result")}
                      </p>
                    </div>
                    <ChevronRight />
                  </a>
                ))}
              </div>
            </section>
          )}
        </>
      )}
      <Disclaimer />
    </main>
  );
}
