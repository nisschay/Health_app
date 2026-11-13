"use client";
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { sendChatMessage, type ChatTurn } from "@/lib/api";
import {
  chatTarget,
  createChatMessageId,
  splitAssistantResponseForSources,
} from "./helpers";
import { MessageCircle, Send, X, TriangleAlert } from "lucide-react";
export default function AssistantPanel({
  person,
  target,
  test,
  run,
}: {
  person: string;
  target: string;
  test?: string;
  run: <T>(f: () => Promise<T>) => Promise<T>;
}) {
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  const session = useRef(createChatMessageId("session"));
  const lock = useRef(false);
  const end = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" });
  }, [history, busy]);
  useEffect(() => {
    if (open) dialog.current?.showModal();
    else dialog.current?.close();
  }, [open]);
  async function send(q: string) {
    const to = chatTarget(target);
    if (!q.trim() || !to || lock.current) return;
    lock.current = true;
    setBusy(true);
    setError("");
    setQuestion("");
    const previous = history;
    setHistory([...previous, { role: "user", content: q }]);
    try {
      const r = await run(() =>
        sendChatMessage({
          ...to,
          question: q,
          history: previous,
          sessionId: session.current,
        }),
      );
      setHistory([
        ...previous,
        { role: "user", content: q },
        { role: "assistant", content: r.answer },
      ]);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not answer. Please try again.",
      );
      setQuestion(q);
      setHistory(previous);
    } finally {
      setBusy(false);
      lock.current = false;
    }
  }
  const panel = (
    <div className="card assistant">
      <div className="assistant-head">
        <MessageCircle size={22} />
        <div className="grow">
          <h2 className="h3">
            Ask about {person.split(" ")[0]}&apos;s results
          </h2>
          <p className="assistant-note">
            Answers use only these reports and can be wrong. Not medical advice.
          </p>
        </div>
        {open && (
          <button
            className="icon-btn"
            aria-label="Close assistant"
            onClick={() => setOpen(false)}
          >
            <X />
          </button>
        )}
      </div>
      <div className="assistant-log" role="log" aria-live="polite">
        {!history.length && (
          <div className="suggest">
            {[
              test
                ? `What changed in ${test}?`
                : "Which results need attention?",
              "What changed since the last report?",
              "What should I ask the doctor?",
            ].map((q) => (
              <button key={q} disabled={busy} onClick={() => void send(q)}>
                {q}
              </button>
            ))}
          </div>
        )}
        {history.map((m, i) => {
          const match =
            m.role === "assistant"
              ? /\n*_*Not found in your results:([\s\S]*)/i.exec(m.content)
              : null;
          const parsed = splitAssistantResponseForSources(
            match ? m.content.slice(0, match.index) : m.content,
          );
          return (
            <div
              key={i}
              className={`msg ${m.role === "user" ? "msg-user" : "msg-bot"}`}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {parsed.body}
              </ReactMarkdown>
              {parsed.sources && (
                <div className="msg-sources">
                  <ReactMarkdown>{parsed.sources}</ReactMarkdown>
                </div>
              )}
              {match && (
                <div className="msg-check">
                  <TriangleAlert size={18} />
                  <div>
                    <strong>Check these numbers</strong>
                    <p>{match[1]?.replace(/_$/, "").trim()}</p>
                  </div>
                </div>
              )}
            </div>
          );
        })}
        {busy && <p role="status">Reading your question...</p>}
        {error && (
          <div className="msg-error" role="alert">
            {error}
            <button
              className="btn btn-quiet"
              onClick={() => void send(question)}
            >
              Try again
            </button>
          </div>
        )}
        <div ref={end} />
      </div>
      <form
        className="assistant-form"
        onSubmit={(e) => {
          e.preventDefault();
          void send(question);
        }}
      >
        <label
          className="sr-only"
          htmlFor={open ? "mobile-question" : "question"}
        >
          Your question
        </label>
        <textarea
          id={open ? "mobile-question" : "question"}
          className="textarea"
          rows={2}
          placeholder="Type your question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send(question);
            }
          }}
        />
        <button
          className="btn btn-primary"
          aria-label="Send"
          disabled={busy || !question.trim()}
        >
          <Send size={20} />
        </button>
      </form>
      <button
        className="btn btn-quiet"
        disabled={busy}
        onClick={() => {
          setHistory([]);
          setError("");
          session.current = createChatMessageId("session");
        }}
      >
        Clear conversation
      </button>
    </div>
  );
  return (
    <>
      <aside className="rail">{!open && panel}</aside>
      <button className="btn btn-primary ask-fab" onClick={() => setOpen(true)}>
        <MessageCircle size={20} />
        Ask about these results
      </button>
      <dialog
        className="assistant-dialog"
        ref={dialog}
        onCancel={() => setOpen(false)}
      >
        {open && panel}
      </dialog>
    </>
  );
}
