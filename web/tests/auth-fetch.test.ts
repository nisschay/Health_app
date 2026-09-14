import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AuthError, fetchProfiles, setAuthTokenProvider } from "../lib/api";

const originalFetch = globalThis.fetch;

describe("authFetch on 401", () => {
  beforeEach(() => {
    setAuthTokenProvider(async () => "token");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("throws a typed AuthError so the caller can refresh once, instead of navigating away", async () => {
    globalThis.fetch = vi.fn(async () => new Response("{}", { status: 401 })) as typeof fetch;
    await expect(fetchProfiles()).rejects.toBeInstanceOf(AuthError);
  });

  it("throws AuthError when no token is available", async () => {
    setAuthTokenProvider(async () => null);
    globalThis.fetch = vi.fn() as typeof fetch;
    await expect(fetchProfiles()).rejects.toBeInstanceOf(AuthError);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("does not turn other failures into AuthError", async () => {
    globalThis.fetch = vi.fn(async () => new Response("nope", { status: 500 })) as typeof fetch;
    await expect(fetchProfiles()).rejects.not.toBeInstanceOf(AuthError);
  });
});
