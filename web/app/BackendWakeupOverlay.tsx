"use client";

import { useEffect, useMemo, useState } from "react";

const HEALTH_PATH = "/health";
const POLL_INTERVAL_MS = 5_000;
const MAX_POLL_DURATION_MS = 120_000;
const REQUEST_TIMEOUT_MS = 4_000;

function normalizePublicApiUrl(raw?: string): string | null {
  if (!raw) return null;

  const trimmed = raw.trim();
  if (!trimmed) return null;

  const unquoted = (
    (trimmed.startsWith("\"") && trimmed.endsWith("\""))
    || (trimmed.startsWith("'") && trimmed.endsWith("'"))
  )
    ? trimmed.slice(1, -1).trim()
    : trimmed;

  if (!unquoted) return null;
  return unquoted.replace(/\/$/, "");
}

async function pingHealthEndpoint(url: string): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
      headers: {
        "Cache-Control": "no-cache",
      },
    });

    return response.ok;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export default function BackendWakeupOverlay() {
  const [showOverlay, setShowOverlay] = useState(false);
  const [didTimeOut, setDidTimeOut] = useState(false);

  const healthEndpoint = useMemo(() => {
    const apiBase = normalizePublicApiUrl(process.env.NEXT_PUBLIC_API_URL);
    if (!apiBase) return null;
    return `${apiBase}${HEALTH_PATH}`;
  }, []);

  useEffect(() => {
    if (!healthEndpoint) return;

    let isMounted = true;
    let isChecking = false;
    let pollIntervalId: number | null = null;
    let maxWaitTimeoutId: number | null = null;

    const clearTimers = () => {
      if (pollIntervalId !== null) {
        window.clearInterval(pollIntervalId);
        pollIntervalId = null;
      }
      if (maxWaitTimeoutId !== null) {
        window.clearTimeout(maxWaitTimeoutId);
        maxWaitTimeoutId = null;
      }
    };

    const checkHealth = async () => {
      if (isChecking) return;
      isChecking = true;

      const isHealthy = await pingHealthEndpoint(healthEndpoint);
      isChecking = false;

      if (!isMounted) return;

      if (isHealthy) {
        setShowOverlay(false);
        setDidTimeOut(false);
        clearTimers();
        return;
      }

      setShowOverlay(true);
    };

    pollIntervalId = window.setInterval(() => {
      void checkHealth();
    }, POLL_INTERVAL_MS);

    maxWaitTimeoutId = window.setTimeout(() => {
      if (!isMounted) return;
      clearTimers();
      setDidTimeOut(true);
    }, MAX_POLL_DURATION_MS);

    void checkHealth();

    return () => {
      isMounted = false;
      clearTimers();
    };
  }, [healthEndpoint]);

  if (!showOverlay) return null;

  return (
    <div
      className="backend-wakeup-overlay"
      role="status"
      aria-live="polite"
      aria-label="Checking backend availability"
    >
      <div className="backend-wakeup-card">
        <div className="backend-wakeup-spinner" aria-hidden="true" />
        <p className="backend-wakeup-title">Backend is waking up, please wait...</p>
        {didTimeOut && (
          <p className="backend-wakeup-note">This is taking longer than expected. You can keep this tab open while the backend starts.</p>
        )}
      </div>
    </div>
  );
}
