"use client";

import { useMemo, useState } from "react";
import type { MedicalRecord } from "@/lib/api";
import { groupRecordsByDateAndCategory } from "@/lib/clinical";

function badgeTone(status: string | null | undefined) {
  const normalized = (status ?? "").toLowerCase();
  if (normalized === "critical") return "bad";
  if (normalized === "high" || normalized === "positive" || normalized === "flagged") return "warn";
  if (normalized === "low") return "muted";
  if (normalized === "normal" || normalized === "negative") return "good";
  return "muted";
}

export default function OrganizedDataTree({ records }: { records: MedicalRecord[] }) {
  const grouped = useMemo(() => groupRecordsByDateAndCategory(records), [records]);
  const [openDates, setOpenDates] = useState<Record<string, boolean>>({});
  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({});

  return (
    <section className="result-section">
      <h2>Organized Data by Date</h2>
      <p className="muted-copy">Earliest to latest. Expand date, then category, then review tests.</p>

      <div className="data-tree">
        {grouped.map((dateGroup) => {
          const dateOpen = Boolean(openDates[dateGroup.date]);
          return (
            <div className="data-date-block" key={dateGroup.date}>
              <button
                className="data-date-head"
                type="button"
                onClick={() => setOpenDates((prev) => ({ ...prev, [dateGroup.date]: !dateOpen }))}
              >
                <strong>{dateGroup.date}</strong>
                <span>{dateGroup.categories.length} categories</span>
              </button>

              {dateOpen && (
                <div className="data-category-list">
                  {dateGroup.categories.map((categoryGroup) => {
                    const key = `${dateGroup.date}::${categoryGroup.category}`;
                    const categoryOpen = Boolean(openCategories[key]);
                    return (
                      <div className="data-category-block" key={key}>
                        <button
                          className="data-category-head"
                          type="button"
                          onClick={() => setOpenCategories((prev) => ({ ...prev, [key]: !categoryOpen }))}
                        >
                          <strong>{categoryGroup.category}</strong>
                          <span>{categoryGroup.tests.length} tests</span>
                        </button>

                        {categoryOpen && (
                          <div className="data-test-list">
                            {categoryGroup.tests.map((test, index) => (
                              <div className="data-test-row" key={`${key}-${test.Test_Name}-${index}`}>
                                <div className="data-test-main">
                                  <span>{test.Test_Name ?? "N/A"}</span>
                                  <small>{test.Reference_Range ?? "N/A"}</small>
                                </div>
                                <div className="data-test-side">
                                  <strong>{String(test.Result ?? "N/A")}</strong>
                                  <span className={`status-pill ${badgeTone(test.Status)}`}>{test.Status ?? "N/A"}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
