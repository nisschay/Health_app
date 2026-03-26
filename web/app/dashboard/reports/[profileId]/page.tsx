"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { fetchStudiesDashboard, type DashboardProfileGroup } from "@/lib/api";

export default function ProfileReportsPage() {
  const params = useParams<{ profileId: string }>();
  const profileId = params?.profileId;
  const router = useRouter();
  const { user, loading, getToken } = useAuth();

  const [profile, setProfile] = useState<DashboardProfileGroup | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!loading && !user) {
      router.replace("/login");
    }
  }, [loading, user, router]);

  useEffect(() => {
    async function load() {
      if (!profileId) return;
      setLoadingProfile(true);
      setError(null);
      try {
        const token = await getToken();
        if (!token) throw new Error("Authentication required.");
        const summary = await fetchStudiesDashboard(token);
        const selected = summary.profiles.find((item) => item.profile_id === profileId);
        if (!selected) {
          throw new Error("Profile not found or you do not have access.");
        }
        setProfile(selected);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load profile reports.");
      } finally {
        setLoadingProfile(false);
      }
    }

    void load();
  }, [getToken, profileId]);

  const totalReports = useMemo(
    () => profile?.studies.reduce((acc, study) => acc + study.report_count, 0) ?? 0,
    [profile],
  );

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
                </article>
              ))}
            </div>
          </>
        )}
      </section>
    </main>
  );
}
