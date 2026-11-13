import type { TestSummary } from "@/lib/results";
import { formatDate } from "@/lib/format";
import { ChevronRight } from "lucide-react";
import { StatusChip } from "./ui";
export default function ResultRow({
  test: t,
  base,
  compact = false,
}: {
  test: TestSummary;
  base: string;
  compact?: boolean;
}) {
  const r = t.latest;
  const readings = t.readings.filter(
    (x) => x.value !== null && x.unit === r.unit,
  );
  const hasRange = r.low !== null || r.high !== null;
  const numeric = r.value !== null;
  const tone =
    r.status === "critical" ? "critical" : t.needsAttention ? "outside" : "ok";
  const lo = r.low ?? 0,
    hi = r.high ?? Math.max(lo * 1.5, lo + 1);
  const width = Math.max(hi - lo, 0.001);
  const bandLeft = r.low === null ? 0 : r.high === null ? 38 : 32;
  const bandRight = r.low === null ? 62 : r.high === null ? 100 : 68;
  const squash = (d: number, k = 1.2) => 1 - Math.exp(-k * d);
  const position = (v: number) => {
    if (r.low === null)
      return Math.max(
        4,
        v <= hi
          ? (v / Math.max(hi, 0.001)) * 62
          : 62 + 34 * squash((v - hi) / Math.max(Math.abs(hi), 0.001)),
      );
    if (r.high === null)
      return v >= lo
        ? 38 + Math.min(1, (v - lo) / Math.max(Math.abs(lo), 0.001)) * 58
        : 38 - 34 * squash((lo - v) / Math.max(Math.abs(lo), 0.001), 3);
    return v < lo
      ? 32 - 28 * squash((lo - v) / width)
      : v > hi
        ? 68 + 28 * squash((v - hi) / width)
        : 32 + (36 * (v - lo)) / width;
  };
  let min = Math.min(
      ...readings.map((x) => x.value!),
      ...(r.low !== null ? [lo] : []),
    ),
    max = Math.max(
      ...readings.map((x) => x.value!),
      ...(r.high !== null ? [hi] : []),
    );
  const spread = max - min || Math.abs(max) || 1;
  if (r.low === null) min = Math.min(min, hi - 0.5 * spread);
  if (r.high === null) max = Math.max(max, lo + 0.5 * spread);
  const y = (v: number) => 21 - ((v - min) * 16) / (max - min || 1);
  const firstTime = readings[0]?.time ?? 0,
    lastTime = readings.at(-1)?.time ?? 0;
  const points = readings.map((v, i) => ({
    x:
      lastTime > firstTime
        ? 5 + (198 * (v.time - firstTime)) / (lastTime - firstTime)
        : 5 + (198 * i) / Math.max(1, readings.length - 1),
    y: y(v.value!),
  }));
  const caption = !numeric
    ? `Lab expects ${r.rangeText}`
    : hasRange
      ? `Range ${r.rangeText}`
      : "No range on the report";
  return (
    <li>
      <a className="v2-row" href={`${base}&test=${encodeURIComponent(t.name)}`}>
        <span className="v2-row-id">
          <span className="v2-row-name">{t.name}</span>
          {!compact && <span className="v2-row-sub">{t.category}</span>}
        </span>
        <span className={`v2-row-val ${numeric ? "" : "text"}`}>
          {r.valueText}
          <small>{r.unit}</small>
        </span>
        <span className="v2-row-chip">
          <StatusChip
            status={r.status}
            negative={r.negative || (!numeric && r.status === "normal")}
          />
        </span>
        <span className="v2-row-viz">
          {numeric &&
            hasRange &&
            (readings.length >= 3 ? (
              <svg
                className={`spark ${tone}`}
                viewBox="0 0 208 26"
                role="img"
                aria-label={`Trend of ${readings.length} readings against the lab range`}
              >
                <rect
                  className="band"
                  x="0"
                  y={r.high === null ? 0 : y(hi)}
                  width="208"
                  height={Math.max(
                    1,
                    (r.low === null ? 26 : y(lo)) -
                      (r.high === null ? 0 : y(hi)),
                  )}
                  rx="2"
                />
                <polyline
                  className="line"
                  points={points.map((p) => `${p.x},${p.y}`).join(" ")}
                />
                {points.map((p, i) => (
                  <circle
                    key={i}
                    className={i === points.length - 1 ? "last" : "pt"}
                    cx={p.x}
                    cy={p.y}
                    r={i === points.length - 1 ? 4 : 2}
                  />
                ))}
              </svg>
            ) : (
              <span
                className={`bar ${tone}`}
                role="img"
                aria-label={`${r.valueText} against ${caption}`}
              >
                <i
                  className="band"
                  style={{
                    left: `${bandLeft}%`,
                    width: `${bandRight - bandLeft}%`,
                  }}
                />
                <i
                  className="mark"
                  style={{ left: `${position(r.value!)}%` }}
                />
              </span>
            ))}
          <span className={`cap ${!numeric ? "alone" : ""}`}>
            {caption}
            <span className="m-only"> · {formatDate(r.date)}</span>
          </span>
        </span>
        <span className="v2-row-when">
          {formatDate(r.date)}
          {t.readings.length > 2 ? (
            <span>
              {t.readings.length} readings since{" "}
              {formatDate(t.readings[0]?.date)}
            </span>
          ) : (
            t.previous && (
              <span>
                Was {t.previous.valueText} on{" "}
                {formatDate(t.previous.date, { year: false })}
              </span>
            )
          )}
        </span>
        <ChevronRight className="v2-row-go" size={16} />
      </a>
    </li>
  );
}
