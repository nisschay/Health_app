"use client";

import { useEffect, useState } from "react";
import { getDirectApiBaseUrl, getPublicApiBaseUrl } from "@/lib/apiBaseUrl";

const POLL_INTERVAL_MS = 5_000;
const REQUEST_TIMEOUT_MS = 4_000;

type Health = { status?: string; database?: string };
type Waking = "backend" | "database" | null;

function healthUrl(): string | null {
  const publicBase = getPublicApiBaseUrl();
  const base = (publicBase.startsWith("/") ? getDirectApiBaseUrl() : publicBase).replace(/\/$/, "");
  return base ? `${base}/health` : null;
}

async function probe(url: string): Promise<Waking> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, { cache: "no-store", signal: controller.signal });
    if (!response.ok) return "backend";
    const body = (await response.json()) as Health;
    return body.database === "ok" ? null : "database";
  } catch {
    return "backend";
  } finally {
    window.clearTimeout(timeout);
  }
}

/** A small, dismissible pill. It never blocks the page; the backend decides when it is ready. */
export default function BackendWakeupOverlay() {
  const [waking, setWaking] = useState<Waking>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    const url = healthUrl();
    if (!url) return;

    let active = true;
    let timer: number | null = null;

    const tick = async () => {
      const result = await probe(url);
      if (!active) return;
      setWaking(result);
      if (result) timer = window.setTimeout(tick, POLL_INTERVAL_MS);
    };

    void tick();
    return () => {
      active = false;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, []);

  if (!waking || dismissed) return null;

  return (
    <div className="backend-wakeup-pill" role="status" aria-live="polite">
      <span className="backend-wakeup-spinner" aria-hidden="true" />
      <span>{waking === "database" ? "Database is waking up" : "Backend is waking up"}</span>
      <button type="button" className="backend-wakeup-dismiss" aria-label="Dismiss" onClick={() => setDismissed(true)}>
        ×
      </button>
    </div>
  );
}
