"use client";
import { useEffect, useState } from "react";
import {
  fetchStudyCombinedReport,
  type DashboardSummary,
  type AnalysisHistoryItem,
  type AnalysisResponse,
  deleteProfile,
} from "@/lib/api";
import { summariseTests, summariseStudy } from "@/lib/results";
import { formatDate, formatMonthRange, plural } from "@/lib/format";
import { buildUserInitials, relationshipLabel } from "./helpers";
import { Disclaimer, Skeleton } from "./ui";
import { ChevronRight, Plus, TriangleAlert, Check } from "lucide-react";
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
      <h1 className="display">
        Good {hour < 12 ? "morning" : hour < 17 ? "afternoon" : "evening"},{" "}
        {first}
      </h1>
      <p className="soft">Here is how everyone&apos;s latest reports look.</p>
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
          {[...summary.profiles]
            .sort(
              (a, b) =>
                Number(b.relationship === "self") -
                  Number(a.relationship === "self") ||
                (b.studies[0]?.last_updated ?? "").localeCompare(
                  a.studies[0]?.last_updated ?? "",
                ),
            )
            .map((p) => {
              const critical = p.studies.flatMap((s) =>
                reports[s.id]
                  ? summariseStudy(
                      summariseTests(reports[s.id]!.records),
                    ).criticalTests.map((t) => ({ t, id: s.id }))
                  : [],
              );
              return (
                <div className="stack" key={p.profile_id}>
                  {critical.length > 0 && (
                    <div className="banner banner-critical" role="alert">
                      <TriangleAlert />
                      <div>
                        <strong>
                          {p.full_name} has{" "}
                          {plural(critical.length, "critical result")}
                        </strong>
                        {critical.map(({ t, id }) => (
                          <p key={id + t.name}>
                            {t.name} was {t.latest.valueText} {t.latest.unit} on{" "}
                            {formatDate(t.latest.date)}. The lab marked it
                            critical. <a href={`?study=${id}`}>See results</a>
                          </p>
                        ))}
                        <p>Please contact a doctor promptly.</p>
                      </div>
                    </div>
                  )}
                  <section className="card">
                    <div className="person-head">
                      <span className="avatar">
                        {buildUserInitials(p.full_name)}
                      </span>
                      <div className="grow">
                        <div className="row row-wrap">
                          <h2 className="person-name">{p.full_name}</h2>
                          <span className="rel">
                            {p.relationship === "self"
                              ? "You"
                              : relationshipLabel(p.relationship)}
                          </span>
                        </div>
                        <p className="person-status">
                          {p.studies.length
                            ? `Latest report ${formatDate([...p.studies].sort((a, b) => (b.range_end ?? "").localeCompare(a.range_end ?? ""))[0]?.range_end)}`
                            : "No reports yet"}
                        </p>
                      </div>
                      <a
                        className="btn btn-secondary btn-sm"
                        href={`?view=upload&profile=${p.profile_id}`}
                      >
                        <Plus size={16} />
                        {p.studies.length ? "Add reports" : "Add first report"}
                      </a>
                    </div>
                    {p.studies.map((s) => {
                      const c = reports[s.id]
                        ? summariseStudy(summariseTests(reports[s.id]!.records))
                        : null;
                      return (
                        <a
                          className="study-row"
                          key={s.id}
                          href={`?study=${s.id}`}
                        >
                          <div>
                            <span className="study-name">{s.name}</span>
                            <p className="study-meta">
                              {plural(s.report_count, "report")} ·{" "}
                              {formatMonthRange(s.range_start, s.range_end)}
                              {s.consistent_lab_name &&
                                ` · ${s.consistent_lab_name}`}
                            </p>
                          </div>
                          <span
                            className={`chip ${c?.attention ? "high" : c?.total && c.inRange === c.total ? "normal" : ""}`}
                          >
                            {c?.attention ? (
                              <TriangleAlert size={16} />
                            ) : (
                              <Check size={16} />
                            )}{" "}
                            {c
                              ? c.attention
                                ? `${c.attention} need attention${c.critical ? ` · ${c.critical} critical` : ""}`
                                : c.total && c.inRange === c.total
                                  ? "All in range"
                                  : "No ranges to compare"
                              : s.has_alerts
                                ? "Has results to review"
                                : "Checking latest results"}
                          </span>
                          <ChevronRight size={20} />
                        </a>
                      );
                    })}
                    {p.relationship !== "self" && (
                      <div className="card-pad">
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
                      </div>
                    )}
                  </section>
                </div>
              );
            })}
          <a className="add-person" href="?view=upload&newPerson=1">
            <Plus />
            Add a family member
          </a>
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
