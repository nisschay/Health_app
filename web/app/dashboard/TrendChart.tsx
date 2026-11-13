"use client";
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { TestSummary } from "@/lib/results";
import { formatDate } from "@/lib/format";
import { statusLabels } from "./ui";
export default function TrendChart({ test }: { test: TestSummary }) {
  const readings = test.readings.filter(
    (r) =>
      r.value !== null &&
      r.unit === test.latest.unit &&
      r.time !== Number.MAX_SAFE_INTEGER,
  );
  const { low, high } = test.latest;
  const nums = [
    ...readings.map((r) => r.value!),
    ...[low, high].filter((n): n is number => n !== null),
  ];
  const min = Math.min(...nums),
    max = Math.max(...nums),
    pad = (max - min || Math.abs(max) || 1) * 0.1;
  const domain = [min - pad, max + pad];
  return (
    <>
      <div
        className="trend-chart"
        role="img"
        aria-label={`${test.name} over time. Every reading is listed in the table below.`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={readings}
            margin={{ top: 20, right: 20, bottom: 15, left: 0 }}
          >
            <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
            <XAxis
              dataKey="time"
              type="number"
              scale="time"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v) =>
                formatDate(new Date(v).toISOString(), { year: false })
              }
              stroke="var(--ink-3)"
            />
            <YAxis domain={domain} width={58} stroke="var(--ink-3)" />
            {(low !== null || high !== null) && (
              <ReferenceArea
                y1={low ?? domain[0]}
                y2={high ?? domain[1]}
                fill="var(--chart-band)"
                fillOpacity={1}
              />
            )}
            {[low, high].map(
              (n, i) =>
                n !== null && (
                  <ReferenceLine
                    key={i}
                    y={n}
                    stroke="var(--chart-band-edge)"
                  />
                ),
            )}
            <Tooltip
              content={({ active, payload }) => {
                const r = payload?.[0]?.payload;
                return active && r ? (
                  <div className="card card-pad">
                    <p>{formatDate(r.date)}</p>
                    <p>
                      {r.valueText} {r.unit} ·{" "}
                      {statusLabels[r.status as keyof typeof statusLabels]}
                    </p>
                    <p>Lab&apos;s range: {r.rangeText}</p>
                  </div>
                ) : null;
              }}
            />
            <Line
              type="linear"
              dataKey="value"
              stroke="var(--chart-line)"
              strokeWidth={2}
              isAnimationActive={false}
              connectNulls={false}
              dot={({ cx = 0, cy = 0, payload }) => {
                const color = `var(--${["none", "detected"].includes(payload.status) ? "high" : payload.status}-solid)`;
                const props = {
                  fill:
                    payload.status === "borderline" ? "var(--surface)" : color,
                  stroke: "var(--surface)",
                  strokeWidth: 2,
                };
                return (
                  <g
                    key={payload.time}
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (!["ArrowLeft", "ArrowRight"].includes(event.key))
                        return;
                      event.preventDefault();
                      const points = Array.from(
                        event.currentTarget
                          .closest("svg")
                          ?.querySelectorAll<SVGElement>('g[tabindex="0"]') ??
                          [],
                      );
                      const index = points.indexOf(event.currentTarget);
                      points[
                        (index +
                          (event.key === "ArrowRight" ? 1 : -1) +
                          points.length) %
                          points.length
                      ]?.focus();
                    }}
                    aria-label={`${formatDate(payload.date)}, ${payload.valueText} ${payload.unit}, ${payload.status}`}
                  >
                    <title>
                      {formatDate(payload.date)}: {payload.valueText}{" "}
                      {payload.unit}
                    </title>
                    {payload.status === "critical" ? (
                      <path
                        d={`M${cx},${cy - 8} l8,8 -8,8 -8,-8z`}
                        {...props}
                      />
                    ) : ["high", "low"].includes(payload.status) ? (
                      <path
                        d={
                          payload.status === "high"
                            ? `M${cx},${cy - 8} l8,14 -16,0z`
                            : `M${cx},${cy + 8} l8,-14 -16,0z`
                        }
                        {...props}
                      />
                    ) : (
                      <circle cx={cx} cy={cy} r={6} {...props} />
                    )}
                  </g>
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="small muted">
        Lab&apos;s range: {test.latest.rangeText}.{" "}
        {test.readings.some((r) => r.rangeText !== test.latest.rangeText) &&
          "The lab’s range changed between reports. The band shows the latest."}{" "}
        {test.readings.some((r) => r.unit !== test.latest.unit) &&
          "Only readings in the latest unit appear in the chart. Other units are listed below."}
      </p>
    </>
  );
}
