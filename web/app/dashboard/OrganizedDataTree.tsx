"use client";

import { useEffect, useMemo, useState } from "react";
import type { MedicalRecord } from "@/lib/api";
import { parseMedicalDate } from "@/lib/clinical";

function badgeTone(status: string | null | undefined) {
  const normalized = (status ?? "").toLowerCase();
  if (normalized === "critical") return "bad";
  if (normalized === "high" || normalized === "positive" || normalized === "flagged") return "warn";
  if (normalized === "low" || normalized === "insufficient") return "low";
  if (normalized === "normal" || normalized === "negative") return "good";
  return "na";
}

function isAbnormalStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? "").toLowerCase();
  return normalized === "critical" || normalized === "high" || normalized === "low" || normalized === "positive" || normalized === "flagged" || normalized === "insufficient";
}

function severityRank(status: string | null | undefined): number {
  const normalized = (status ?? "").toLowerCase();
  if (normalized === "critical") return 0;
  if (normalized === "high" || normalized === "positive" || normalized === "flagged") return 1;
  if (normalized === "low" || normalized === "insufficient") return 2;
  if (normalized === "normal" || normalized === "negative") return 3;
  return 4;
}

export default function OrganizedDataTree({ records }: { records: MedicalRecord[] }) {
  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({});
  const [search, setSearch] = useState("");

  const grouped = useMemo(() => {
    const valid = records.filter((record) => record.Test_Name && record.Test_Name !== "N/A");
    const byCategory = new Map<string, MedicalRecord[]>();

    for (const record of valid) {
      const category = record.Test_Category?.trim() || "Uncategorized";
      if (!byCategory.has(category)) byCategory.set(category, []);
      byCategory.get(category)!.push(record);
    }

    return Array.from(byCategory.entries())
      .map(([category, tests]) => {
        const sortedTests = [...tests].sort((a, b) => {
          const byAbnormal = Number(isAbnormalStatus(b.Status)) - Number(isAbnormalStatus(a.Status));
          if (byAbnormal !== 0) return byAbnormal;
          const bySeverity = severityRank(a.Status) - severityRank(b.Status);
          if (bySeverity !== 0) return bySeverity;
          const byDate = parseMedicalDate(a.Test_Date) - parseMedicalDate(b.Test_Date);
          if (byDate !== 0) return byDate;
          return String(a.Test_Name ?? "").localeCompare(String(b.Test_Name ?? ""));
        });

        const alertCount = sortedTests.filter((test) => isAbnormalStatus(test.Status)).length;

        return {
          category,
          tests: sortedTests,
          alertCount,
        };
      })
      .sort((a, b) => {
        const byAlerts = Number(b.alertCount > 0) - Number(a.alertCount > 0);
        if (byAlerts !== 0) return byAlerts;
        if (a.alertCount !== b.alertCount) return b.alertCount - a.alertCount;
        return a.category.localeCompare(b.category);
      });
  }, [records]);

  const filteredGroups = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return grouped;

    return grouped
      .map((group) => {
        const categoryMatch = group.category.toLowerCase().includes(q);
        if (categoryMatch) return group;
        const tests = group.tests.filter((test) => String(test.Test_Name ?? "").toLowerCase().includes(q));
        return {
          ...group,
          tests,
        };
      })
      .filter((group) => group.tests.length > 0);
  }, [grouped, search]);

  useEffect(() => {
    if (!search.trim()) return;
    const nextOpen: Record<string, boolean> = {};
    for (const group of filteredGroups) {
      nextOpen[group.category] = true;
    }
    setOpenCategories(nextOpen);
  }, [search, filteredGroups]);

  return (
    <section className="result-section">
      <h2>Organized Data by Tests</h2>
      <p className="muted-copy">Grouped by category with alerts prioritized at the top.</p>

      <div className="organized-search-wrap">
        <input
          className="text-input"
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search tests or categories..."
        />
      </div>

      <div className="data-tree">
        {filteredGroups.map((group) => {
          const isOpen = Boolean(openCategories[group.category]);
          return (
            <div className="data-date-block organized-category-block" key={group.category}>
              <button
                className="data-date-head"
                type="button"
                onClick={() => setOpenCategories((prev) => ({ ...prev, [group.category]: !isOpen }))}
              >
                <div className="organized-category-left">
                  <span className={`organized-chevron ${isOpen ? "open" : ""}`}>▸</span>
                  <div>
                    <strong>{group.category}</strong>
                    <span>{group.tests.length} tests</span>
                  </div>
                </div>
                {group.alertCount > 0 ? (
                  <span className="status-pill warn">{group.alertCount} Alerts</span>
                ) : (
                  <span className="status-pill good">All Normal</span>
                )}
              </button>

              {isOpen && (
                <div className="data-category-list organized-expanded-list">
                  <div className="data-test-list">
                    {group.tests.map((test, index) => (
                      <div className="data-test-row" key={`${group.category}-${test.Test_Name}-${index}`}>
                        <div className="data-test-main">
                          <span>{test.Test_Name ?? "N/A"}</span>
                          <small>{test.Test_Date ?? "Unknown"} • Ref: {test.Reference_Range ?? "N/A"}</small>
                        </div>
                        <div className="data-test-side">
                          <strong>{String(test.Result ?? "N/A")}{test.Unit ? ` ${test.Unit}` : ""}</strong>
                          <span className={`status-pill ${badgeTone(test.Status)}`}>{test.Status ?? "N/A"}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {filteredGroups.length === 0 && (
          <p className="muted-copy">No categories or tests matched your search.</p>
        )}
      </div>
    </section>
  );
}
