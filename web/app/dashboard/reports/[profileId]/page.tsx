"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import {
  fetchStudiesDashboard,
  fetchStudyCombinedReport,
  type AnalysisResponse,
  type DashboardProfileGroup,
} from "@/lib/api";

function shouldRetryWithFreshToken(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  return /invalid firebase token|firebase token expired|firebase token revoked|firebase token project mismatch|authentication required|missing bearer token|not authenticated|unauthorized|\b401\b/i.test(
    error.message,
  );
}

export default function ProfileReportsPage() {
  const params = useParams<{ profileId: string }>();
  const profileId = params?.profileId;
  const router = useRouter();
  const { user, loading, getToken, logout } = useAuth();

  const [profile, setProfile] = useState<DashboardProfileGroup | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeStudyId, setActiveStudyId] = useState<string | null>(null);
  const [combinedReport, setCombinedReport] = useState<AnalysisResponse | null>(null);
  const [combinedLoading, setCombinedLoading] = useState(false);
  const [combinedError, setCombinedError] = useState<string | null>(null);
  const [combinedReloadNonce, setCombinedReloadNonce] = useState(0);
  const reauthRedirectedRef = useRef(false);

  const hardRedirectToLogin = useCallback(() => {
    if (typeof window !== "undefined") {
      window.localStorage.clear();
      window.sessionStorage.clear();
      window.location.href = "/login";
      return;
    }
    router.replace("/login");
  }, [router]);

  const forceReauth = useCallback(async (): Promise<never> => {
    if (!reauthRedirectedRef.current) {
      reauthRedirectedRef.current = true;
      try {
        await logout();
      } catch {
        // Ignore logout failures and still redirect to login.
      }
      hardRedirectToLogin();
    }
    throw new Error("Your session is no longer valid. Please sign in again.");
  }, [hardRedirectToLogin, logout]);

  const runWithTokenRetry = useCallback(async <T,>(operation: (token: string) => Promise<T>): Promise<T> => {
    const token = await getToken();
    if (!token) {
      return forceReauth();
    }

    try {
      return await operation(token);
    } catch (error) {
      if (!shouldRetryWithFreshToken(error)) {
        throw error;
      }
      const refreshedToken = await getToken(true);
      if (!refreshedToken) {
        return forceReauth();
      }

      try {
        return await operation(refreshedToken);
      } catch (retryError) {
        if (shouldRetryWithFreshToken(retryError)) {
          return forceReauth();
        }
        throw retryError;
      }
    }
  }, [forceReauth, getToken]);

  useEffect(() => {
    if (!loading && !user) {
      hardRedirectToLogin();
    }
  }, [hardRedirectToLogin, loading, user]);

  useEffect(() => {
    async function load() {
      if (!profileId) return;
      setLoadingProfile(true);
      setError(null);
      try {
        const summary = await runWithTokenRetry((_token) => fetchStudiesDashboard());
        const selected = summary.profiles.find((item) => item.profile_id === profileId);
        if (!selected) {
          throw new Error("Profile not found or you do not have access.");
        }
        setProfile(selected);
        if (selected.studies.length > 0) {
          setActiveStudyId(selected.studies[0].id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load profile reports.");
      } finally {
        setLoadingProfile(false);
      }
    }

    void load();
  }, [profileId, runWithTokenRetry]);

  const totalReports = useMemo(
    () => profile?.studies.reduce((acc, study) => acc + study.report_count, 0) ?? 0,
    [profile],
  );

  useEffect(() => {
    async function loadCombined() {
      if (!activeStudyId) {
        setCombinedReport(null);
        return;
      }
      setCombinedLoading(true);
      setCombinedError(null);
      setCombinedReport(null);
      try {
        const report = await runWithTokenRetry((_token) => fetchStudyCombinedReport(activeStudyId));
        setCombinedReport(report);
      } catch (err) {
        setCombinedError(err instanceof Error ? err.message : "Failed to load combined report.");
      } finally {
        setCombinedLoading(false);
      }
    }

    void loadCombined();
  }, [activeStudyId, combinedReloadNonce, runWithTokenRetry]);

  function handleViewCombinedReport(studyId: string): void {
    setActiveStudyId(studyId);
    setCombinedReloadNonce((value) => value + 1);
  }

  if (loading || loadingProfile) {
    return (
      <main className="result-shell">
        <section className="result-section">
          <h2>Loading profile reports...</h2>
        </section>
      </main>
    );
  }

  return (
    <main className="result-shell">
      <section className="result-section">
        <div className="result-topbar">
          <button className="back-btn" type="button" onClick={() => router.push("/dashboard")}>Dashboard</button>
        </div>

        {error && <p className="status-text error-text">{error}</p>}

        {!error && profile && (
          <>
            <h2>{profile.full_name} - {profile.relationship}</h2>
            <p className="muted-copy">Total reports: {totalReports}</p>

            <div className="study-cards-grid">
              {profile.studies.map((study) => (
                <article className="study-dashboard-card" key={study.id}>
                  <div className="study-card-top">
                    <h3>{study.name}</h3>
                    <span className={`study-alert-dot ${study.has_alerts ? "warn" : "ok"}`} />
                  </div>
                  <p>{study.report_count} reports</p>
                  {study.range_start && study.range_end && (
                    <p className="muted-copy">{study.range_start} to {study.range_end}</p>
                  )}
                  {study.consistent_lab_name && <p className="study-lab-chip">Lab: {study.consistent_lab_name}</p>}
                  <div className="study-card-actions">
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => handleViewCombinedReport(study.id)}
                      disabled={combinedLoading && activeStudyId === study.id}
                    >
                      {combinedLoading && activeStudyId === study.id ? "Loading..." : "View Combined Report"}
                    </button>
                  </div>
                </article>
              ))}
            </div>

            <section className="result-section" style={{ marginTop: "1rem" }}>
              <h3>Combined Report</h3>
              {combinedLoading && <p className="muted-copy">Loading combined report...</p>}
              {combinedError && <p className="status-text error-text">{combinedError}</p>}

              {!combinedLoading && !combinedError && combinedReport && (
                <div className="study-dashboard-card">
                  <p>
                    Patient: {combinedReport.patient_info.name || "N/A"} • Total Records: {combinedReport.total_records}
                  </p>
                  <p>
                    Alerts: {combinedReport.health_summary.concerns.length} • Overall Score: {combinedReport.health_summary.overall_score}
                  </p>
                  {combinedReport.health_summary.concerns.length > 0 && (
                    <div style={{ marginTop: "0.6rem" }}>
                      <p className="muted-copy">Top Findings</p>
                      <ul>
                        {combinedReport.health_summary.concerns.slice(0, 8).map((item, index) => (
                          <li key={`${item.test_name}-${index}`}>
                            {item.test_name}: {String(item.result)} ({item.status})
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </section>
          </>
        )}
      </section>
    </main>
  );
}
