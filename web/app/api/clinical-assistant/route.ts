import { NextRequest, NextResponse } from "next/server";

type ChatTurnPayload = {
  role: string;
  content: string;
};

type ClinicalAssistantPayload = {
  analysisId?: string;
  reportContext?: {
    records?: unknown[];
    [key: string]: unknown;
  };
  history?: unknown;
  question?: string;
};

export const runtime = "nodejs";

function resolveBackendBaseUrl(): string {
  const raw =
    process.env.NEXT_PUBLIC_DIRECT_API_URL?.replace(/\/$/, "") ||
    process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
    "http://127.0.0.1:8000";

  if (raw.startsWith("/")) {
    return "http://127.0.0.1:8000";
  }
  return raw;
}

function sanitizeHistory(value: unknown): ChatTurnPayload[] {
  if (!Array.isArray(value)) return [];

  return value
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const role = "role" in item ? String((item as { role?: unknown }).role ?? "").trim() : "";
      const content = "content" in item ? String((item as { content?: unknown }).content ?? "").trim() : "";
      if (!role || !content) return null;
      return { role, content };
    })
    .filter((item): item is ChatTurnPayload => item !== null);
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) return payload.detail;
  } catch {
    // Ignore JSON parsing issues and use fallback below.
  }
  return response.statusText || "Request failed.";
}

export async function POST(request: NextRequest) {
  const authHeader = request.headers.get("authorization");
  if (!authHeader) {
    return NextResponse.json(
      { detail: "Missing authentication token. Please sign in again." },
      { status: 401 },
    );
  }

  let payload: ClinicalAssistantPayload;
  try {
    payload = (await request.json()) as ClinicalAssistantPayload;
  } catch {
    return NextResponse.json({ detail: "Invalid JSON payload." }, { status: 400 });
  }

  const question = typeof payload.question === "string" ? payload.question.trim() : "";
  if (!question) {
    return NextResponse.json({ detail: "Question is required." }, { status: 400 });
  }

  const reportContext = payload.reportContext && typeof payload.reportContext === "object"
    ? payload.reportContext
    : {};
  const records = Array.isArray(reportContext.records) ? reportContext.records : [];
  const history = sanitizeHistory(payload.history);

  const backendPayload = {
    records,
    question,
    history,
    analysis_id: payload.analysisId ?? null,
    report_context: reportContext,
  };

  const backendBaseUrl = resolveBackendBaseUrl();

  try {
    const response = await fetch(`${backendBaseUrl}/api/v1/reports/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: authHeader,
      },
      body: JSON.stringify(backendPayload),
      cache: "no-store",
    });

    if (response.status === 401 || response.status === 403) {
      return NextResponse.json(
        { detail: "Session expired or token invalid. Please sign in again." },
        { status: 401 },
      );
    }

    if (!response.ok) {
      const detail = await readErrorDetail(response);
      if (response.status >= 500) {
        return NextResponse.json(
          { detail: `Clinical assistant model error: ${detail}` },
          { status: 500 },
        );
      }
      return NextResponse.json({ detail }, { status: response.status });
    }

    const data = (await response.json()) as { answer?: unknown };
    if (typeof data.answer !== "string") {
      return NextResponse.json(
        { detail: "Clinical assistant returned an invalid response." },
        { status: 500 },
      );
    }

    return NextResponse.json({ answer: data.answer });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown assistant service error.";
    return NextResponse.json(
      { detail: `Clinical assistant model error: ${detail}` },
      { status: 500 },
    );
  }
}
