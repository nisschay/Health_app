"use client";

import { useMemo, useState } from "react";
import type { MedicalRecord } from "@/lib/api";
import { groupByTestName } from "@/lib/clinical";

function badgeTone(status: string | null | undefined) {
  const normalized = (status ?? "").toLowerCase();
  if (normalized === "critical") return "bad";
  if (normalized === "high" || normalized === "positive" || normalized === "flagged") return "warn";
  if (normalized === "low") return "muted";
  if (normalized === "normal" || normalized === "negative") return "good";
  return "muted";
}

export default function OrganizedDataTree({ records }: { records: MedicalRecord[] }) {
  const grouped = useMemo(() => groupByTestName(records), [records]);
  const [openTests, setOpenTests] = useState<Record<string, boolean>>({});

  return (
    <section className="result-section">
      <h2>Organized Data by Tests</h2>
      <p className="muted-copy">Expand a test to view its timeline from earliest to latest.</p>

      <div className="data-tree">
        {grouped.map((testGroup) => {
          const testOpen = Boolean(openTests[testGroup.testName]);
          return (
            <div className="data-date-block" key={testGroup.testName}>
              <button
                className="data-date-head"
                type="button"
                onClick={() => setOpenTests((prev) => ({ ...prev, [testGroup.testName]: !testOpen }))}
              >
                <div>
                  <strong>{testGroup.testName}</strong>
                  <span>{testGroup.latest.category}</span>
                </div>
                <div className="data-test-side">
                  <strong>{String(testGroup.latest.value)}{testGroup.latest.unit ? ` ${testGroup.latest.unit}` : ""}</strong>
                  <span className={`status-pill ${badgeTone(testGroup.latest.status)}`}>{testGroup.latest.status}</span>
                </div>
              </button>

              {testOpen && (
                <div className="data-category-list">
                  <div className="data-test-list">
                    {testGroup.timeline.map((entry, index) => (
                      <div className="data-test-row" key={`${testGroup.testName}-${entry.date}-${index}`}>
                        <div className="data-test-main">
                          <span>{entry.date}</span>
                          <small>{entry.referenceRange}</small>
                        </div>
                        <div className="data-test-side">
                          <strong>{String(entry.value)}{entry.unit ? ` ${entry.unit}` : ""}</strong>
                          <span className={`status-pill ${badgeTone(entry.status)}`}>{entry.status}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
