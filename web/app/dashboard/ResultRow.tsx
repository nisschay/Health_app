import type { TestSummary } from "@/lib/results";
import { ChevronRight } from "lucide-react";
import { StatusChip, statusIcons, verdictLabels } from "./ui";
export default function ResultRow({
  test: t,
  base,
  compact = false,
}: {
  test: TestSummary;
  base: string;
  compact?: boolean;
}) {
  const Icon = statusIcons[t.latest.status];
  const values = t.readings
    .filter((r) => r.value !== null && r.unit === t.latest.unit)
    .map((r) => r.value!);
  const min = Math.min(...values),
    max = Math.max(...values);
  return (
    <li>
      <a
        className={`result-row ${compact ? "compact" : ""}`}
        href={`${base}&test=${encodeURIComponent(t.name)}`}
      >
        {!compact && (
          <span className={`status-tile ${t.latest.status}`}>
            <Icon />
          </span>
        )}
        <span className="rr-name">
          <span className="h3">{t.name}</span>
          <span className="rr-sub">
            {compact ? `${t.readings.length} readings` : t.category} · range{" "}
            {t.latest.rangeText}
          </span>
        </span>
        <span className="rr-value">
          <span className="val">
            {t.latest.valueText} <span className="unit">{t.latest.unit}</span>
          </span>
          {!compact && (
            <StatusChip status={t.latest.status} negative={t.latest.negative} />
          )}
        </span>
        {compact ? (
          <span className="rr-chip">
            <StatusChip status={t.latest.status} negative={t.latest.negative} />
          </span>
        ) : (
          <>
            <svg
              className="spark rr-spark"
              viewBox="0 0 100 36"
              aria-hidden="true"
            >
              <polyline
                className="line"
                points={values
                  .map(
                    (v, i) =>
                      `${5 + (i * 90) / Math.max(1, values.length - 1)},${30 - ((v - min) * 24) / (max - min || 1)}`,
                  )
                  .join(" ")}
              />
            </svg>
            <span className="rr-change">
              <strong>{t.verdict && verdictLabels[t.verdict]}</strong>
              <br />
              {t.firstTimeOut ? "Out of range for the first time" : t.change}
            </span>
          </>
        )}
        <ChevronRight className="chev" size={18} />
      </a>
    </li>
  );
}
