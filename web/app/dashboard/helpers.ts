/** Small pure helpers for the dashboard: formatting, chat history, ids. */
import { AuthError, type ChatTurn } from "@/lib/api";
import type { ChatMessage } from "./types";

export function createChatMessageId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export function toApiChatHistory(messages: ChatMessage[]): ChatTurn[] {
  return messages
    .filter((message) => !message.isLoading && (message.role === "user" || message.role === "assistant"))
    .map((message) => ({
      role: message.role === "assistant" ? "assistant" : "user",
      content: message.content,
    }));
}

export function isConcerningStatus(status: string | null | undefined): boolean {
  const normalized = status?.toLowerCase();
  return normalized === "high" || normalized === "low" || normalized === "critical" || normalized === "positive" || normalized === "flagged";
}

export function fmtRelativeDate(iso: string) {
  try {
    const value = new Date(iso).getTime();
    if (Number.isNaN(value)) return "Unknown";
    const deltaMs = Date.now() - value;
    const days = Math.floor(deltaMs / 86400000);
    if (days <= 0) return "Today";
    if (days === 1) return "1 day ago";
    if (days < 30) return `${days} days ago`;
    const months = Math.floor(days / 30);
    if (months === 1) return "1 month ago";
    return `${months} months ago`;
  } catch {
    return "Unknown";
  }
}

export function fmtRange(start: string | null, end: string | null): string {
  if (!start && !end) return "No reports yet";
  const toLabel = (v: string) => {
    try {
      return new Date(v).toLocaleDateString("en-IN", { month: "short", year: "numeric" });
    } catch {
      return v;
    }
  };
  if (start && end) return `${toLabel(start)} - ${toLabel(end)}`;
  return toLabel(start ?? end ?? "");
}

export function relationshipLabel(value: string): string {
  const clean = (value || "").trim();
  if (!clean) return "Family";
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

export function normalizeSourceFileName(value: string | null | undefined): string {
  if (!value) return "";
  const normalized = value.replace(/\\/g, "/").split("/").pop() ?? value;
  return normalized.trim().toLowerCase();
}

/** "study-<uuid>" or "history-<id>": what the assistant answers about. */
export function chatTarget(activeId: string | null): { studyId: string } | { analysisId: number } | null {
  if (activeId?.startsWith("study-")) return { studyId: activeId.slice("study-".length) };
  if (activeId?.startsWith("history-")) {
    const id = Number(activeId.slice("history-".length));
    return Number.isInteger(id) ? { analysisId: id } : null;
  }
  return null;
}

export function shouldRetryWithFreshToken(error: unknown): boolean {
  return error instanceof AuthError;
}

export function formatChatTimestamp(createdAt: number): string {
  if (Date.now() - createdAt < 60_000) {
    return "Just now";
  }
  return new Date(createdAt).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
  });
}

export function splitAssistantResponseForSources(content: string): { body: string; sources: string | null } {
  const normalized = content.replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return { body: "", sources: null };
  }

  const sourcesHeaderMatch = /^(?:\*\*)?Sources:(?:\*\*)?/im.exec(normalized);
  if (!sourcesHeaderMatch || sourcesHeaderMatch.index < 0) {
    return { body: normalized, sources: null };
  }

  const body = normalized.slice(0, sourcesHeaderMatch.index).trim();
  const sources = normalized.slice(sourcesHeaderMatch.index).trim();
  return {
    body,
    sources: sources || null,
  };
}

export function buildUserInitials(displayName?: string | null, email?: string | null): string {
  const fallback = (displayName || email || "User").trim();
  if (!fallback) return "U";

  const fromWords = fallback
    .replace(/@.*$/, "")
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");

  const initials = fromWords.slice(0, 2);
  if (initials) {
    return initials;
  }
  return fallback.slice(0, 2).toUpperCase();
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
