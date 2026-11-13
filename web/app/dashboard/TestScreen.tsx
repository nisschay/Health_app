import dynamic from "next/dynamic";
import type { TestSummary } from "@/lib/results";
import { formatDate, formatAmount } from "@/lib/format";
import { StatusChip, verdictLabels } from "./ui";
const TrendChart = dynamic(() => import("./TrendChart"), { ssr: false });
export default function TestScreen({ test: t }: { test: TestSummary }) {
  const l = t.latest;
  const values = t.readings.filter(
    (r) => r.value !== null && r.unit === l.unit,
  );
  const first = values[0];
  const hasLab = t.readings.some((r) => r.lab);
  return (
    <div className="stack sections">
      <section className="card card-pad stack">
        <h2 className="h2">Latest result · {formatDate(l.date)}</h2>
        <p className="hero-value">
          {l.valueText} <span className="unit">{l.unit}</span>
        </p>
        <div>
          <StatusChip status={l.status} negative={l.negative} />
        </div>
        {l.value !== null && (l.low !== null || l.high !== null) ? (
          <RangeBar test={t} />
        ) : (
          <p>Lab&apos;s range: {l.rangeText}.</p>
        )}
      </section>
      <section className="card card-pad stack">
        <h2 className="h2">Over time</h2>
        <strong>{t.verdict && verdictLabels[t.verdict]}</strong>
        {values.length > 1 && first && l.value !== null ? (
          <>
            <p>
              {l.value >= first.value! ? "Up" : "Down"}{" "}
              {formatAmount(l.value - first.value!)} {l.unit} since{" "}
              {formatDate(first.date)}, across {values.length} reports.
            </p>
            <TrendChart test={t} />
          </>
        ) : (
          <p>
            {t.readings.length === 1
              ? "Only one reading so far. Add an older or newer report to see the trend."
              : "These results are shown as text. See every reading below."}
          </p>
        )}
      </section>
      <section className="card">
        <h2 className="card-head h2">Every reading</h2>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Result</th>
                <th>Status</th>
                <th>Change</th>
                <th className="hide-sm">Lab&apos;s range</th>
                {hasLab && <th className="hide-sm">Lab</th>}
              </tr>
            </thead>
            <tbody>
              {[...t.readings].reverse().map((r, i) => {
                const prev = t.readings[t.readings.length - i - 2];
                return (
                  <tr key={r.time}>
                    <td>{formatDate(r.date)}</td>
                    <td>
                      {r.valueText} {r.unit}
                    </td>
                    <td>
                      <StatusChip status={r.status} negative={r.negative} />
                    </td>
                    <td>
                      {prev &&
                      prev.value !== null &&
                      r.value !== null &&
                      prev.unit === r.unit
                        ? `${r.value >= prev.value ? "Up" : "Down"} ${formatAmount(r.value - prev.value)}`
                        : prev
                          ? "Not comparable"
                          : "First reading"}
                    </td>
                    <td className="hide-sm">{r.rangeText}</td>
                    {hasLab && <td className="hide-sm">{r.lab || ""}</td>}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
function RangeBar({ test }: { test: TestSummary }) {
  const r = test.latest;
  const width =
    r.low !== null && r.high !== null
      ? r.high - r.low
      : Math.abs(r.high ?? r.low ?? 1);
  const min = Math.min(r.value!, r.low !== null ? r.low - width / 2 : 0),
    max = Math.max(
      r.value!,
      r.high !== null ? r.high + width / 2 : 2 * (r.low ?? 1),
    );
  const pct = (n: number) => ((n - min) / (max - min || 1)) * 100;
  return (
    <div
      role="img"
      aria-label={`${r.valueText} ${r.unit}; lab's range ${r.rangeText}`}
      className="rangebar"
    >
      <span
        className="band"
        style={{
          left: `${pct(r.low ?? min)}%`,
          width: `${pct(r.high ?? max) - pct(r.low ?? min)}%`,
        }}
      />
      <span
        className="mark"
        style={{
          left: `${pct(r.value!)}%`,
          background: `var(--${r.status === "none" ? "neutral" : r.status}-solid)`,
        }}
      />
      <span className="tick band-label" style={{ left: "50%" }}>
        Lab&apos;s range: {r.rangeText}
      </span>
    </div>
  );
}
