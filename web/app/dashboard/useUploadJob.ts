"use client";
import { useState, useRef, useEffect, useCallback } from "react";
import {
  waitForJob,
  startAnalysisJob,
  isAbortError,
  JobNotFoundError,
  type ReportJob,
} from "@/lib/api";
import { ACTIVE_JOB_KEY } from "./types";
export function useUploadJob(
  enabled: boolean,
  run: <T>(f: () => Promise<T>) => Promise<T>,
  onDone: (j: ReportJob) => void,
) {
  const [job, setJob] = useState<ReportJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  const done = useRef(onDone);
  done.current = onDone;
  const follow = useCallback(
    async (id: string) => {
      controller.current?.abort();
      const c = new AbortController();
      controller.current = c;
      localStorage.setItem(ACTIVE_JOB_KEY, id);
      setBusy(true);
      setError("");
      try {
        const settled = await run(() =>
          waitForJob(
            id,
            (j) => {
              if (!c.signal.aborted) setJob(j);
            },
            c.signal,
          ),
        );
        if (c.signal.aborted) return;
        localStorage.removeItem(ACTIVE_JOB_KEY);
        if (settled.status === "done") done.current(settled);
        else
          setError(
            settled.error || "We could not read these reports. Try again.",
          );
      } catch (e) {
        if (e instanceof JobNotFoundError)
          localStorage.removeItem(ACTIVE_JOB_KEY);
        if (!isAbortError(e) && !c.signal.aborted)
          setError(
            e instanceof Error
              ? e.message
              : "Lost contact with the server. Return to this page to resume.",
          );
      } finally {
        if (!c.signal.aborted) setBusy(false);
      }
    },
    [run],
  );
  useEffect(() => {
    if (enabled) {
      const id = localStorage.getItem(ACTIVE_JOB_KEY);
      if (id) void follow(id);
    }
    return () => controller.current?.abort();
  }, [enabled, follow]);
  const start = async (
    files: File[],
    study: string,
    existingDataFile: File | null = null,
  ) => {
    setBusy(true);
    setError("");
    setJob(null);
    const form = new FormData();
    files.forEach((f) => form.append("pdf_files", f));
    form.append("study_id", study);
    if (existingDataFile) form.append("existing_data", existingDataFile);
    try {
      const j = await run(() => startAnalysisJob(form));
      await follow(j.id);
    } catch (e) {
      setBusy(false);
      setError(
        e instanceof Error ? e.message : "Could not upload. Please try again.",
      );
    }
  };
  return {
    job,
    busy,
    error,
    start,
    clear: () => {
      setJob(null);
      setError("");
    },
  };
}
export type UploadJobState = ReturnType<typeof useUploadJob>;
