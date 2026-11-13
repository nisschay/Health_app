"use client";
import { useState, useEffect, useCallback, useMemo } from "react";
import { useAuth } from "@/lib/auth-context";
import {
  fetchProfiles,
  createProfile,
  fetchStudiesDashboard,
  fetchReportHistory,
  fetchStudyCombinedReport,
  fetchReportById,
  deleteStudy,
  exportExcel,
  type DashboardSummary,
  type AnalysisHistoryItem,
  type AnalysisResponse,
  type ReportJob,
} from "@/lib/api";
import { summariseTests, summariseStudy } from "@/lib/results";
import { formatDate } from "@/lib/format";
import TopBar from "./TopBar";
import HomeScreen from "./HomeScreen";
import UploadScreen from "./UploadScreen";
import ResultsScreen from "./ResultsScreen";
import TestScreen from "./TestScreen";
import AssistantPanel from "./AssistantPanel";
import AccountScreen from "./AccountScreen";
import VerifyEmailPanel from "./VerifyEmailPanel";
import { Skeleton, Disclaimer } from "./ui";
import { useTokenRetry } from "./useTokenRetry";
import { ACTIVE_JOB_KEY } from "./types";
import { useUploadJob } from "./useUploadJob";
export default function DashboardPage() {
  const auth = useAuth();
  const run = useTokenRetry();
  const [params, setParams] = useState(() => new URLSearchParams());
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loadingResult, setLoadingResult] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState("");
  const navigate = useCallback((url: string) => {
    window.history.pushState(null, "", url);
    setParams(new URLSearchParams(window.location.search));
    window.scrollTo(0, 0);
  }, []);
  useEffect(() => {
    const click = (event: MouseEvent) => {
      if (
        event.defaultPrevented ||
        event.button !== 0 ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        event.altKey
      )
        return;
      const link = (event.target as Element).closest<HTMLAnchorElement>(
        "a[href]",
      );
      if (!link || link.target || link.hasAttribute("download")) return;
      const url = new URL(link.href, location.href);
      if (url.origin !== location.origin || url.pathname !== "/dashboard")
        return;
      event.preventDefault();
      navigate(url.pathname + url.search);
    };
    document.addEventListener("click", click);
    return () => document.removeEventListener("click", click);
  }, [navigate]);
  useEffect(() => {
    if (localStorage.getItem(ACTIVE_JOB_KEY))
      window.history.replaceState(null, "", "/dashboard?view=upload");
    setParams(new URLSearchParams(window.location.search));
    const pop = () => setParams(new URLSearchParams(window.location.search));
    window.addEventListener("popstate", pop);
    return () => window.removeEventListener("popstate", pop);
  }, []);
  useEffect(() => {
    if (!auth.loading && !auth.user) window.location.replace("/login");
  }, [auth.loading, auth.user]);
  const reload = useCallback(async () => {
    setError("");
    try {
      const profiles = await run(fetchProfiles);
      if (!profiles.some((p) => p.relationship === "self"))
        await run(() =>
          createProfile({
            full_name:
              auth.user?.displayName || auth.user?.email?.split("@")[0] || "Me",
            relationship: "self",
          }),
        );
      const [s, h] = await run(() =>
        Promise.all([fetchStudiesDashboard(), fetchReportHistory()]),
      );
      setSummary(s);
      setHistory(h);
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "We could not load your reports. Check your connection and try again.",
      );
    }
  }, [run, auth.user]);
  useEffect(() => {
    if (auth.user && auth.emailVerified) void reload();
  }, [auth.user, auth.emailVerified, reload]);
  const study = params.get("study"),
    saved = params.get("analysis"),
    view = params.get("view");
  const target = study ? `study-${study}` : saved ? `history-${saved}` : "";
  const base = study
    ? `?study=${encodeURIComponent(study)}`
    : `?analysis=${saved}`;
  useEffect(() => {
    // Wait for Firebase to restore the sign-in: asking before it has would read as an expired session.
    if (!auth.user || !auth.emailVerified || view === "upload" || (!study && !saved)) return;
    let active = true;
    setLoadingResult(true);
    setAnalysis(null);
    void run(() =>
      study ? fetchStudyCombinedReport(study) : fetchReportById(Number(saved)),
    )
      .then((a) => {
        if (active) setAnalysis(a);
      })
      .catch(() => {
        if (active) {
          setNotice("That study could not be opened.");
          navigate("/dashboard");
        }
      })
      .finally(() => {
        if (active) setLoadingResult(false);
      });
    return () => {
      active = false;
    };
  }, [auth.user, study, saved, view, run, navigate, auth.emailVerified]);
  const completed = useCallback(
    (j: ReportJob) => {
      void reload();
      if (!(j.progress.warnings?.length || j.progress.notes?.length)) {
        setNotice(`${j.source_filenames.length} reports added`);
        if (
          new URLSearchParams(window.location.search).get("view") === "upload"
        )
          navigate(
            j.study_id
              ? `/dashboard?study=${j.study_id}`
              : `/dashboard?analysis=${j.analysis_id}`,
          );
      }
    },
    [reload, navigate],
  );
  const upload = useUploadJob(
    !!auth.user && auth.emailVerified,
    run,
    completed,
  );
  const tests = useMemo(
    () => summariseTests(analysis?.records ?? []),
    [analysis],
  );
  const selectedTest = tests.find((t) => t.name === params.get("test"));
  const personGroup = summary?.profiles.find((p) =>
    p.studies.some((s) => s.id === study),
  );
  const studyInfo = personGroup?.studies.find((s) => s.id === study);
  const person =
    personGroup?.full_name || analysis?.patient_info.name || "Your";
  const title =
    studyInfo?.name || `Report of ${formatDate(analysis?.patient_info.date)}`;
  const stats = summariseStudy(tests);
  useEffect(() => {
    if (analysis && params.has("test") && !selectedTest) {
      setNotice("That test is not in this study.");
      navigate(`/dashboard${base}`);
    }
  }, [analysis, params, selectedTest, base, navigate]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 5000);
    return () => clearTimeout(timer);
  }, [notice]);
  async function download(kind: "pdf" | "excel") {
    if (!analysis || exporting) return;
    setExporting(true);
    setExportError("");
    try {
      const blob =
        kind === "excel"
          ? await run(() =>
              exportExcel(analysis.records, analysis.patient_info),
            )
          : (await import("@/lib/pdf")).generateClinicalPdfReport(analysis);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `LabTrail-results.${kind === "pdf" ? "pdf" : "xlsx"}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setExportError(
        e instanceof Error
          ? e.message
          : "Could not prepare the download. Please try again.",
      );
    } finally {
      setExporting(false);
    }
  }
  const logout = async () => {
    await auth.logout();
    window.location.assign("/login");
  };
  if (auth.loading || !auth.user)
    return (
      <main className="page">
        <Skeleton />
      </main>
    );
  if (!auth.emailVerified)
    return (
      <VerifyEmailPanel
        email={auth.user.email}
        onContinue={auth.refreshEmailVerified}
        onResend={auth.resendVerificationEmail}
        onSignOut={() => void logout()}
      />
    );
  return (
    <>
      <TopBar
        name={auth.user.displayName || auth.user.email?.split("@")[0] || "You"}
        email={auth.user.email || ""}
        logout={() => void logout()}
      />
      {notice && (
        <p className="toast" role="status">
          {notice}
        </p>
      )}
      {error && (
        <div className="page">
          <div className="banner banner-critical" role="alert">
            {error}
            <button className="btn btn-secondary" onClick={() => void reload()}>
              Try again
            </button>
          </div>
        </div>
      )}
      {upload.busy && view !== "upload" && (
        <div className="page">
          <div className="banner banner-info">
            Reading reports. {upload.job?.progress.processed ?? 0} finished.
            <a href="?view=upload">View progress</a>
          </div>
        </div>
      )}
      {view === "upload" ? (
        <UploadScreen
          params={params}
          upload={upload}
          run={run}
          reload={() => void reload()}
        />
      ) : view === "account" ? (
        <main className="page page-narrow stack">
          <a href="/dashboard">← Family</a>
          <AccountScreen
            runWithTokenRetry={run}
            onAccountDeleted={() => window.location.assign("/login")}
          />
        </main>
      ) : target ? (
        <main className="page has-fab">
          <nav className="crumbs">
            <a href={selectedTest ? `/dashboard${base}` : "/dashboard"}>
              ← {selectedTest ? title : "Family"}
            </a>
            <span>
              {person}
              {personGroup &&
                ` · ${personGroup.relationship === "self" ? "You" : personGroup.relationship}`}
            </span>
          </nav>
          {loadingResult || !analysis ? (
            <Skeleton />
          ) : (
            <>
              <div className="page-heading">
                <div className="grow">
                  <h1 className="display">{selectedTest?.name || title}</h1>
                  <p className="soft">
                    {selectedTest
                      ? `${selectedTest.category}${selectedTest.printedNames.length ? ` · Printed as ${selectedTest.printedNames.join(", ")}` : ""}`
                      : `${analysis.combined_report_file_names?.length || new Set(analysis.records.map((r) => r.Source_Filename)).size} reports · ${stats.earliestReportDate && stats.earliestReportDate !== stats.latestReportDate ? `${formatDate(stats.earliestReportDate)} to ` : ""}${formatDate(stats.latestReportDate)}`}
                  </p>
                </div>
                <div className="row row-wrap">
                  {study && (
                    <a
                      className="btn btn-primary"
                      href={`?view=upload&study=${study}&profile=${personGroup?.profile_id || ""}`}
                    >
                      Add reports
                    </a>
                  )}
                  <details>
                    <summary className="btn btn-secondary">
                      {exporting ? "Preparing..." : "Download"}
                    </summary>
                    <div className="stack">
                      <button
                        className="btn btn-secondary"
                        disabled={exporting}
                        onClick={() => void download("pdf")}
                      >
                        PDF summary for the doctor
                      </button>
                      <button
                        className="btn btn-secondary"
                        disabled={exporting}
                        onClick={() => void download("excel")}
                      >
                        Excel sheet of every result
                      </button>
                    </div>
                  </details>
                </div>
              </div>
              {exportError && <p role="alert">{exportError}</p>}
              <div className="with-rail">
                <div>
                  {selectedTest ? (
                    <TestScreen test={selectedTest} />
                  ) : (
                    <ResultsScreen
                      analysis={analysis}
                      tests={tests}
                      base={base}
                      person={person}
                      title={title}
                      download={() => void download("pdf")}
                    />
                  )}
                  <Disclaimer />
                  {study && (
                    <button
                      className="btn btn-danger"
                      onClick={async () => {
                        if (
                          !confirm(
                            `Delete ${title} and all its reports? This cannot be undone.`,
                          )
                        )
                          return;
                        try {
                          await run(() => deleteStudy(study));
                          await reload();
                          navigate("/dashboard");
                        } catch (e) {
                          setError(
                            e instanceof Error
                              ? e.message
                              : "Could not delete study.",
                          );
                        }
                      }}
                    >
                      Delete this study
                    </button>
                  )}
                </div>
                <AssistantPanel
                  key={target}
                  person={person}
                  target={target}
                  test={selectedTest?.name}
                  run={run}
                />
              </div>
            </>
          )}
        </main>
      ) : (
        <HomeScreen
          name={
            auth.user.displayName || auth.user.email?.split("@")[0] || "You"
          }
          summary={summary}
          history={history}
          reload={() => void reload()}
          run={run}
        />
      )}
    </>
  );
}
