"use client";

import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { MedicalRecord, sendChatMessageStream, ChatTurn } from "@/lib/api";

type ChatMessage = ChatTurn | { role: "system"; content: string };

interface ClinicalAssistantChatProps {
  records: MedicalRecord[];
  sessionId?: string;
  runWithTokenRetry: <T>(operation: (token: string) => Promise<T>) => Promise<T>;
}

const DEFAULT_SUGGESTIONS = [
  "What does my HbA1C mean?",
  "Is my Vitamin D level dangerous?",
  "Explain my liver enzyme results",
  "What should I do next?",
];

export default function ClinicalAssistantChat({
  records,
  sessionId,
  runWithTokenRetry,
}: ClinicalAssistantChatProps) {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [assistantDraft, setAssistantDraft] = useState("");
  const [showSuggestions, setShowSuggestions] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const sessionRef = useRef<string>(sessionId ?? `session-${Date.now()}`);

  useEffect(() => {
    if (chatHistory.length > 0 && showSuggestions) {
      const hasUserMessage = chatHistory.some((m) => m.role === "user");
      if (hasUserMessage) {
        setShowSuggestions(false);
      }
    }
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, showSuggestions]);

  const handleInputInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
  };

  const submitQuestion = (question: string) => {
    if (!question.trim() || isStreaming) return;

    const newQuestion = question.trim();
    setInputValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    const priorThread: ChatTurn[] = chatHistory.filter(
      (turn): turn is ChatTurn => turn.role === "user" || turn.role === "assistant",
    );
    const updatedHistory: ChatTurn[] = [...priorThread, { role: "user", content: newQuestion }];
    setChatHistory(updatedHistory);
    setAssistantDraft("");
    setIsStreaming(true);

    void (async () => {
      const requestStartedAt = Date.now();
      let firstResponseLogged = false;
      try {
        const analysisId = sessionRef.current;
        const resp = await runWithTokenRetry((_token) =>
          sendChatMessageStream(
            {
              analysisId,
              sessionId: analysisId,
              reportContext: {
                records,
              },
              history: updatedHistory.slice(-8),
              message: newQuestion,
            },
            (event) => {
              if (!firstResponseLogged && (event.type === "started" || event.type === "delta" || event.type === "done")) {
                firstResponseLogged = true;
                console.info("[CHAT_METRIC] ttft_ms", Date.now() - requestStartedAt);
              }
              if (event.type === "delta") {
                setAssistantDraft((prev) => prev + event.text);
              }
              if (event.type === "done") {
                setAssistantDraft(event.answer);
                console.info("[CHAT_METRIC] total_ms", Date.now() - requestStartedAt, {
                  backendLatencyMs: event.backendLatencyMs,
                  totalLatencyMs: event.totalLatencyMs,
                });
              }
            },
          )
        );
        setAssistantDraft("");
        setChatHistory((prev) => [...prev, { role: "assistant", content: resp.answer }]);
      } catch (err) {
        setAssistantDraft("");
        const msg = err instanceof Error ? err.message : "Unknown error connecting to chat service.";
        setChatHistory((prev) => [
          ...prev,
          { role: "system", content: `**Error:** ${msg}` },
        ]);
      } finally {
        setIsStreaming(false);
      }
    })();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuestion(inputValue);
    }
  };

  const reportCount = new Set(records.map(r => r.Source_Filename)).size || 1;

  const renderMessageContent = (turn: ChatMessage) => {
    if (turn.role === "system") {
      return (
        <div className="chat-system-error">
          <ReactMarkdown>{turn.content}</ReactMarkdown>
        </div>
      );
    }
    
    return <ReactMarkdown components={{
      code(props) {
        const {children, className, node, ...rest} = props
        return (
          <code {...rest} className={className}>
            {children}
          </code>
        )
      }
    }}>{turn.content}</ReactMarkdown>;
  };

  return (
    <section className="clinical-chat-section">
      <header className="clinical-chat-header">
        <div className="clinical-chat-title-group">
          <div className="pulse-dot"></div>
          <div className="clinical-chat-title-text">
            <h2>Clinical Assistant</h2>
            <p>Ask about trends, abnormalities, or clinical context · Markdown supported</p>
          </div>
        </div>
        <div className="clinical-chat-context-chip">
          <span>📋 Context: </span>
          <strong>{reportCount} reports loaded</strong>
        </div>
      </header>

      <div className="clinical-chat-messages-area">
        {chatHistory.length === 0 && (
          <div className="clinical-chat-empty">
            <p>Ready to help analyze your clinical data. Ask me anything.</p>
          </div>
        )}

        {chatHistory.map((turn, idx) => {
          const isAI = turn.role === "assistant";
          const isUser = turn.role === "user";
          const isSystem = turn.role === "system";

          return (
            <div key={idx} className={`clinical-chat-row ${turn.role}`}>
              <div className="clinical-chat-avatar">
                {isAI || isSystem ? "AI" : "U"}
              </div>
              <div className="clinical-chat-bubble">
                {renderMessageContent(turn)}
                {isUser && <div className="clinical-chat-timestamp">Just now</div>}
              </div>
            </div>
          );
        })}

        {isStreaming && (
          <div className="clinical-chat-row assistant">
            <div className="clinical-chat-avatar">AI</div>
            {assistantDraft ? (
              <div className="clinical-chat-bubble">
                <ReactMarkdown>{assistantDraft}</ReactMarkdown>
              </div>
            ) : (
              <div className="clinical-chat-bubble typing">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="clinical-chat-bottom-area">
        {showSuggestions && (
          <div className="clinical-chat-suggestions">
            {DEFAULT_SUGGESTIONS.map((sug, i) => (
              <button
                key={i}
                type="button"
                className="clinical-chat-chip"
                onClick={() => submitQuestion(sug)}
              >
                {sug}
              </button>
            ))}
          </div>
        )}

        <form
          className="clinical-chat-input-row"
          onSubmit={(e) => {
            e.preventDefault();
            submitQuestion(inputValue);
          }}
        >
          <textarea
            ref={textareaRef}
            className="clinical-chat-textarea"
            placeholder="Ask a clinical question..."
            value={inputValue}
            onChange={handleInputInput}
            onKeyDown={handleKeyDown}
            disabled={isStreaming}
            rows={1}
          />
          <button
            type="submit"
            className="clinical-chat-send-btn"
            disabled={isStreaming || !inputValue.trim()}
          >
            ↑ Send
          </button>
        </form>
      </div>
    </section>
  );
}
