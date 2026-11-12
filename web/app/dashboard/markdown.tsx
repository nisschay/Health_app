"use client";

import type { Components } from "react-markdown";

export const MARKDOWN_COMPONENTS: Components = {
  p: ({ children }) => (
    <p style={{ margin: "0 0 10px 0", lineHeight: 1.7, color: "#d4b483", fontSize: 14 }}>{children}</p>
  ),
  strong: ({ children }) => (
    <strong style={{ color: "#fef3c7", fontWeight: 600 }}>
      {children}
    </strong>
  ),
  em: ({ children }) => (
    <em style={{ color: "#fbbf24", fontStyle: "italic" }}>
      {children}
    </em>
  ),
  ul: ({ children }) => (
    <ul style={{ margin: "8px 0", paddingLeft: 20, color: "#d4b483" }}>{children}</ul>
  ),
  ol: ({ children }) => (
    <ol style={{ margin: "8px 0", paddingLeft: 20, color: "#d4b483" }}>{children}</ol>
  ),
  li: ({ children }) => (
    <li style={{ margin: "4px 0", fontSize: 14, lineHeight: 1.6 }}>{children}</li>
  ),
  h1: ({ children }) => (
    <h1 style={{ fontSize: 16, fontWeight: 600, color: "#fef3c7", margin: "12px 0 6px" }}>{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 style={{ fontSize: 15, fontWeight: 600, color: "#fef3c7", margin: "10px 0 6px" }}>{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 style={{ fontSize: 14, fontWeight: 600, color: "#fbbf24", margin: "8px 0 4px" }}>{children}</h3>
  ),
  code: ({ children, className }) => {
    const isInline = !className;
    return isInline ? (
      <code style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 12, color: "#fbbf24", background: "#2d1f08", padding: "1px 5px", borderRadius: 4 }}>{children}</code>
    ) : (
      <pre style={{ background: "#1c1917", border: "1px solid #292524", borderRadius: 8, padding: "10px 14px", overflowX: "auto", margin: "8px 0" }}>
        <code style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 12, color: "#d4b483" }}>{children}</code>
      </pre>
    );
  },
  blockquote: ({ children }) => (
    <blockquote style={{ borderLeft: "3px solid #d97706", marginLeft: 0, paddingLeft: 12, color: "#a8a29e", fontStyle: "italic" }}>{children}</blockquote>
  ),
  table: ({ children }) => (
    <div style={{ overflowX: "auto", margin: "8px 0" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th style={{ padding: "6px 10px", textAlign: "left", borderBottom: "1px solid #44403c", color: "#fbbf24", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.06em" }}>{children}</th>
  ),
  td: ({ children }) => (
    <td style={{ padding: "6px 10px", borderBottom: "1px solid #292524", color: "#d4b483" }}>{children}</td>
  ),
};

export const CITATION_COMPONENTS: Components = {
  p: ({ children }) => <p>{children}</p>,
  a: ({ children, href }) => (
    <a href={href || "#"} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
};
