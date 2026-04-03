function normalizeEnvBaseUrl(raw: string | undefined): string | null {
  if (!raw) return null;

  const trimmed = raw.trim();
  if (!trimmed) return null;

  const unquoted = (
    (trimmed.startsWith('"') && trimmed.endsWith('"'))
    || (trimmed.startsWith("'") && trimmed.endsWith("'"))
  )
    ? trimmed.slice(1, -1).trim()
    : trimmed;

  if (!unquoted) return null;
  return unquoted.replace(/\/$/, "");
}

function isAbsoluteHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function fallbackDirectBaseUrl(): string {
  return "http://localhost:8000";
}

export function getPublicApiBaseUrl(): string {
  const publicBase = (
    normalizeEnvBaseUrl(process.env.NEXT_PUBLIC_API_URL)
    ?? normalizeEnvBaseUrl(process.env.NEXT_PUBLIC_API_BASE_URL)
  );

  if (publicBase) {
    return publicBase;
  }

  if (process.env.NODE_ENV === "production") {
    return getDirectApiBaseUrl();
  }

  return "/backend";
}

export function getDirectApiBaseUrl(): string {
  return (
    normalizeEnvBaseUrl(process.env.NEXT_PUBLIC_DIRECT_API_URL)
    ?? normalizeEnvBaseUrl(process.env.NEXT_PUBLIC_API_URL)
    ?? normalizeEnvBaseUrl(process.env.NEXT_PUBLIC_API_BASE_URL)
    ?? fallbackDirectBaseUrl()
  );
}

export function getServerBackendBaseUrl(): string {
  const raw = (
    normalizeEnvBaseUrl(process.env.NEXT_PUBLIC_DIRECT_API_URL)
    ?? normalizeEnvBaseUrl(process.env.NEXT_PUBLIC_API_BASE_URL)
    ?? normalizeEnvBaseUrl(process.env.NEXT_PUBLIC_API_URL)
    ?? fallbackDirectBaseUrl()
  );

  if (raw.startsWith("/")) {
    return fallbackDirectBaseUrl();
  }

  return raw;
}

export function buildApiUrl(
  baseUrl: string,
  path: string,
  query?: Record<string, string | null | undefined>,
): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  if (isAbsoluteHttpUrl(baseUrl)) {
    const url = new URL(normalizedPath, `${baseUrl}/`);
    if (query) {
      for (const [key, value] of Object.entries(query)) {
        if (value === null || value === undefined) continue;
        if (!value.trim()) continue;
        url.searchParams.set(key, value);
      }
    }
    return url.toString();
  }

  const normalizedBase = baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;
  const params = new URLSearchParams();
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === null || value === undefined) continue;
      if (!value.trim()) continue;
      params.set(key, value);
    }
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return `${normalizedBase}${normalizedPath}${suffix}`;
}
