import { describe, expect, it } from "vitest";

import { chatTarget, dashboardUrl, splitAssistantResponseForSources, toApiChatHistory } from "../app/dashboard/helpers";

describe("chatTarget", () => {
  it("names a study or a saved analysis, the only things the assistant may read", () => {
    expect(chatTarget("study-7f3c")).toEqual({ studyId: "7f3c" });
    expect(chatTarget("history-42")).toEqual({ analysisId: 42 });
  });

  it("refuses anything else rather than guessing", () => {
    expect(chatTarget(null)).toBeNull();
    expect(chatTarget("history-abc")).toBeNull();
    expect(chatTarget("analysis-5")).toBeNull();
  });
});

describe("splitAssistantResponseForSources", () => {
  it("separates the cited sources from the answer", () => {
    const { body, sources } = splitAssistantResponseForSources("Your HbA1c is 6.1%.\n\n**Sources:** [ADA](https://example.org)");
    expect(body).toBe("Your HbA1c is 6.1%.");
    expect(sources).toBe("**Sources:** [ADA](https://example.org)");
  });

  it("leaves an answer without sources whole", () => {
    expect(splitAssistantResponseForSources("No sources here.")).toEqual({ body: "No sources here.", sources: null });
  });
});

describe("toApiChatHistory", () => {
  it("sends only finished user and assistant turns", () => {
    const history = toApiChatHistory([
      { id: "1", role: "user", content: "q", createdAt: 0 },
      { id: "2", role: "assistant", content: "", createdAt: 0, isLoading: true },
      { id: "3", role: "system", content: "note", createdAt: 0 },
      { id: "4", role: "assistant", content: "a", createdAt: 0 },
    ]);
    expect(history).toEqual([{ role: "user", content: "q" }, { role: "assistant", content: "a" }]);
  });
});

describe("dashboardUrl", () => {
  it("gives each view its own address so refresh and Back return to it", () => {
    expect(dashboardUrl("home", null)).toBe("/dashboard");
    expect(dashboardUrl("analyze", "study-abc")).toBe("/dashboard?view=upload");
    expect(dashboardUrl("result", "study-1b2c")).toBe("/dashboard?study=1b2c");
    expect(dashboardUrl("result", "history-42")).toBe("/dashboard?analysis=42");
    expect(dashboardUrl("result", null)).toBe("/dashboard");
  });
});
