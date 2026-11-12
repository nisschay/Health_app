/** State shapes shared by the dashboard page and its parts. */
import type { ProfileItem, StudySummary } from "@/lib/api";

export type AnalyzeStep = "idle" | "preparing" | "uploading" | "processing" | "saving" | "error";
export const ACTIVE_JOB_KEY = "health-app.active-job";
export type StudyAction = "add-existing" | "start-new";

export type StageState = "pending" | "active" | "complete" | "error";

export type FileStatus = "queued" | "extracting" | "parsing" | "done" | "failed";

export type FileProgressItem = {
  file: string;
  percent: number;
  status: FileStatus;
  error?: string;
};

export type AnalyzeContext = {
  profile: ProfileItem;
  study: StudySummary;
  mode: "existing" | "new";
};

export type ChatRole = "user" | "assistant" | "system";

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  isLoading?: boolean;
};

export const MIN_CHAT_LOADING_MS = 400;
