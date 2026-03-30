"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Area,
  AreaChart,
} from "recharts";
import type { MedicalRecord } from "@/lib/api";

type Props = {
  records: MedicalRecord[];
  testName: string;
};

function parseRefRange(ref: string | null | undefined): { low: number | null; high: number | null } {
  if (!ref) return { low: null, high: null };
  // "3.5 - 5.0" or "< 5.0" or "> 3.5"
  const rangeMatch = ref.match(/([0-9.]+)\s*[-–]\s*([0-9.]+)/);
  if (rangeMatch) return { low: parseFloat(rangeMatch[1]), high: parseFloat(rangeMatch[2]) };
  const ltMatch = ref.match(/[<≤]\s*([0-9.]+)/);
  if (ltMatch) return { low: null, high: parseFloat(ltMatch[1]) };
  const gtMatch = ref.match(/[>≥]\s*([0-9.]+)/);
  if (gtMatch) return { low: parseFloat(gtMatch[1]), high: null };
  return { low: null, high: null };
}

export default function TrendChart({ records, testName }: Props) {
  // Sort by date
  const sorted = [...records].sort((a, b) => {
    const da = a.Test_Date ?? "";
    const db = b.Test_Date ?? "";
    // Try to parse dd-mm-yyyy
    const parse = (d: string) => {
      const parts = d.split(/[-/]/);
      if (parts.length === 3) {
        const [d1, m1, y1] = parts;
        return new Date(`${y1}-${m1!.padStart(2, "0")}-${d1!.padStart(2, "0")}`).getTime();
      }
      return new Date(d).getTime();
    };
    return parse(da) - parse(db);
  });

  const chartData = sorted
    .map((r) => {
      const val = typeof r.Result_Numeric === "number" ? r.Result_Numeric : parseFloat(String(r.Result ?? ""));
      if (isNaN(val)) return null;
      return {
        date: r.Test_Date ?? "Unknown",
        value: val,
        unit: r.Unit ?? "",
        status: r.Status ?? "",
        reference: r.Reference_Range ?? "",
      };
    })
    .filter(Boolean) as { date: string; value: number; unit: string; status: string; reference: string }[];

  if (chartData.length === 0) {
    return (
      <div className="chart-no-data">
        <p>No numeric data available for {testName}</p>
        <p className="muted-copy">Results may be non-numeric or missing values.</p>
      </div>
    );
  }

  const { low, high } = parseRefRange(chartData[0]?.reference);
  const unit = chartData[0]?.unit ?? "";
  const isSinglePoint = chartData.length === 1;

  return (
    <div className="trend-chart-wrapper">
      <div className="chart-header">
        <h3>{testName}</h3>
        {unit && <span className="chart-unit">{unit}</span>}
        {(low !== null || high !== null) && (
          <span className="chart-ref">
            Reference: {low !== null ? low : ""}
            {low !== null && high !== null ? " – " : ""}
            {high !== null ? high : ""} {unit}
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={280}>
        {isSinglePoint ? (
          /* Bar-style for single data points */
          <AreaChart data={chartData} margin={{ top: 16, right: 24, bottom: 16, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#3f3a34" />
            <XAxis dataKey="date" tick={{ fill: "#a8a29e", fontSize: 12 }} />
            <YAxis tick={{ fill: "#a8a29e", fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: "#1c1917", border: "1px solid #d97706", borderRadius: 8, color: "#fef3c7" }}
              formatter={(val) => [`${val} ${unit}`, testName]}
            />
            {low !== null && <ReferenceLine y={low} stroke="#78716c" strokeDasharray="5 5" label={{ value: "Low", fill: "#a8a29e", fontSize: 11 }} />}
            {high !== null && <ReferenceLine y={high} stroke="#78716c" strokeDasharray="5 5" label={{ value: "High", fill: "#a8a29e", fontSize: 11 }} />}
            <Area type="monotone" dataKey="value" stroke="#d97706" fill="#d9770638" strokeWidth={2} dot={{ fill: "#d97706", r: 6 }} activeDot={{ r: 8 }} />
          </AreaChart>
        ) : (
          <LineChart data={chartData} margin={{ top: 16, right: 24, bottom: 16, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#3f3a34" />
            <XAxis dataKey="date" tick={{ fill: "#a8a29e", fontSize: 12 }} />
            <YAxis tick={{ fill: "#a8a29e", fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: "#1c1917", border: "1px solid #d97706", borderRadius: 8, color: "#fef3c7" }}
              formatter={(val) => [`${val} ${unit}`, testName]}
            />
            {low !== null && <ReferenceLine y={low} stroke="#78716c" strokeDasharray="5 5" label={{ value: "Normal Low", fill: "#a8a29e", fontSize: 11 }} />}
            {high !== null && <ReferenceLine y={high} stroke="#78716c" strokeDasharray="5 5" label={{ value: "Normal High", fill: "#a8a29e", fontSize: 11 }} />}
            <Line
              type="monotone"
              dataKey="value"
              stroke="#d97706"
              strokeWidth={2.5}
              dot={(props) => {
                const d = chartData[props.index];
                const color = d?.status?.toLowerCase() === "high" || d?.status?.toLowerCase() === "low" || d?.status?.toLowerCase() === "critical" ? "#92400e" : "#d97706";
                return <circle key={props.index} cx={props.cx} cy={props.cy} r={6} fill={color} stroke="#0a0a0a" strokeWidth={2} />;
              }}
              activeDot={{ r: 8, fill: "#d97706" }}
            />
          </LineChart>
        )}
      </ResponsiveContainer>

      {/* Data points table below chart */}
      <div className="chart-data-points">
        {chartData.map((d, i) => (
          <div className="data-point-row" key={i}>
            <span className="dp-date">{d.date}</span>
            <span className="dp-value">{d.value} {unit}</span>
            <span className={`status-pill ${
              d.status?.toLowerCase() === "normal" ? "good" :
              d.status?.toLowerCase() === "high" || d.status?.toLowerCase() === "low" ? "warn" :
              d.status?.toLowerCase() === "critical" ? "bad" : "muted"
            }`}>{d.status || "—"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
