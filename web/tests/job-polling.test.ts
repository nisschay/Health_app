import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AuthError, JobNotFoundError, isAbortError, setAuthTokenProvider, startAnalysisJob, waitForJob } from "../lib/api";

const originalFetch = globalThis.fetch;

function job(status: string) {
  return {
    id: "j1",
    status,
    progress: { stage: "processing", files: {}, processed: 0, total: 1 },
    error: null,
    study_id: null,
    analysis_id: status === "done" ? 7 : null,
    source_filenames: ["r.pdf"],
    created_at: "2026-11-12T10:00:00",
    finished_at: null,
  };
}

const ok = (status: string) => new Response(JSON.stringify(job(status)), { status: 200 });

/** Runs the poller with fake timers, advancing time until it settles. */
async function settle<T>(promise: Promise<T>): Promise<T> {
  let done = false;
  const tracked = promise.finally(() => {
    done = true;
  });
  tracked.catch(() => undefined);
  for (let i = 0; i < 200 && !done; i++) await vi.advanceTimersByTimeAsync(1_000);
  return tracked;
}

describe("waitForJob", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setAuthTokenProvider(async () => "token");
  });

  afterEach(() => {
    vi.useRealTimers();
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("keeps polling through a network drop and a waking server, because the job keeps running", async () => {
    const responses = [
      () => Promise.resolve(ok("running")),
      () => Promise.reject(new TypeError("Failed to fetch")),
      () => Promise.resolve(new Response("", { status: 502 })),
      () => Promise.resolve(new Response("", { status: 503 })),
      () => Promise.resolve(ok("done")),
    ];
    globalThis.fetch = vi.fn(() => responses.shift()!()) as typeof fetch;
    const seen: string[] = [];
    const result = await settle(waitForJob("j1", (j) => seen.push(j.status)));
    expect(result.status).toBe("done");
    expect(seen).toEqual(["running", "done"]);
  });

  it("stops at once when the job is gone", async () => {
    globalThis.fetch = vi.fn(async () => new Response("", { status: 404 })) as typeof fetch;
    await expect(settle(waitForJob("j1", () => undefined))).rejects.toBeInstanceOf(JobNotFoundError);
    const calls = vi.mocked(globalThis.fetch).mock.calls.length;
    await vi.advanceTimersByTimeAsync(30_000);
    expect(vi.mocked(globalThis.fetch).mock.calls.length).toBe(calls);
  });

  it("lets an expired session reach the caller, which refreshes the token", async () => {
    globalThis.fetch = vi.fn(async () => new Response("", { status: 401 })) as typeof fetch;
    await expect(settle(waitForJob("j1", () => undefined))).rejects.toBeInstanceOf(AuthError);
  });

  it("stops polling when aborted, so a left page or a newer job leaves no loop behind", async () => {
    globalThis.fetch = vi.fn(async () => ok("running")) as typeof fetch;
    const controller = new AbortController();
    const polling = waitForJob("j1", () => undefined, controller.signal);
    polling.catch(() => undefined);
    await vi.advanceTimersByTimeAsync(5_000);
    controller.abort();
    const error = await polling.catch((e) => e);
    expect(isAbortError(error)).toBe(true);
    const calls = vi.mocked(globalThis.fetch).mock.calls.length;
    await vi.advanceTimersByTimeAsync(30_000);
    expect(vi.mocked(globalThis.fetch).mock.calls.length).toBe(calls);
  });

  it("gives up after its deadline instead of polling a dead server forever", async () => {
    globalThis.fetch = vi.fn(async () => new Response("", { status: 502 })) as typeof fetch;
    const polling = waitForJob("j1", () => undefined);
    polling.catch(() => undefined);
    await vi.advanceTimersByTimeAsync(41 * 60_000);
    await expect(polling).rejects.toThrow(/Lost contact/);
  });
});

describe("startAnalysisJob", () => {
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("is sent once even when the server errors, so an upload can never start two jobs", async () => {
    setAuthTokenProvider(async () => "token");
    globalThis.fetch = vi.fn(async () => new Response("{}", { status: 502 })) as typeof fetch;
    await expect(startAnalysisJob(new FormData())).rejects.toThrow();
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });
});
