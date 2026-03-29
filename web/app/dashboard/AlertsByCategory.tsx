"use client";

import { useMemo, useState } from "react";
import type { AnalysisConcern } from "@/lib/api";
import { groupByCategory } from "@/lib/clinical";

function badgeTone(status: string) {
  const normalized = status.toLowerCase();
  if (normalized === "critical") return "bad";
  if (normalized === "high" || normalized === "positive" || normalized === "flagged") return "warn";
  if (normalized === "low") return "muted";
  return "muted";
}

export default function AlertsByCategory({ concerns }: { concerns: AnalysisConcern[] }) {
  const grouped = useMemo(() => groupByCategory(concerns), [concerns]);
  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({});

  const criticalTotal = grouped.reduce((sum, c) => sum + c.criticalCount, 0);

  return (
    <section className="result-section">
      <h2>Health Alerts</h2>
      <div className="alerts-summary-bar">
        <span className="status-pill bad">Total Alerts: {concerns.length}</span>
        <span className="status-pill warn">Categories: {grouped.length}</span>
        <span className="status-pill bad">Critical: {criticalTotal}</span>
      </div>

      <div className="alert-category-list">
        {grouped.map((category) => {
          const isOpen = Boolean(openCategories[category.category]);
          return (
            <article className="alert-category-card" key={category.category}>
              <button
                className="alert-category-head"
                type="button"
                onClick={() => setOpenCategories((prev) => ({ ...prev, [category.category]: !isOpen }))}
              >
                <div>
                  <h3>{category.category}</h3>
                  <p>{category.totalAlerts} flagged tests</p>
                </div>
                <div className="alert-category-meta">
                  <span className="status-pill bad">Critical {category.criticalCount}</span>
                  <span className="status-pill warn">High {category.highCount}</span>
                  <span className="status-pill muted">Low {category.lowCount}</span>
                </div>
              </button>

              {isOpen && (
                <div className="alert-category-body">
                  {category.tests.map((test, index) => (
                    <div className="alert-test-row" key={`${category.category}-${test.testName}-${test.date}-${index}`}>
                      <div>
                        <strong>{test.testName}</strong>
                        <span>{test.date} • Ref: {test.reference || "N/A"}</span>
                      </div>
                      <div className="alert-test-meta">
                        <span className={`status-pill ${badgeTone(test.status)}`}>{test.status || "N/A"}</span>
                        <strong>{String(test.result)}</strong>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
