import {
  ArrowUp,
  ArrowDown,
  Check,
  Minus,
  OctagonAlert,
  CircleDot,
  CirclePlus,
} from "lucide-react";
import type { StatusKind } from "@/lib/results";
export const statusLabels = {
  critical: "Critical",
  high: "High",
  low: "Low",
  normal: "In range",
  borderline: "Borderline",
  detected: "Detected",
  none: "No range",
};
export const statusIcons = {
  critical: OctagonAlert,
  high: ArrowUp,
  low: ArrowDown,
  normal: Check,
  borderline: CircleDot,
  detected: CirclePlus,
  none: Minus,
};
export function StatusChip({
  status,
  negative = false,
}: {
  status: StatusKind;
  negative?: boolean;
}) {
  const Icon = statusIcons[status];
  return (
    <span className={`chip ${status}`}>
      <Icon size={16} />
      {negative ? "As expected" : statusLabels[status]}
    </span>
  );
}
export function Disclaimer() {
  return (
    <p className="disclaimer">
      <span>
        &quot;High&quot;, &quot;Low&quot; and &quot;Critical&quot; come from the
        ranges printed on each lab report. This is not medical advice and can
        contain mistakes. Check every result with your doctor.{" "}
        <a href="/privacy">How your data is handled</a>
      </span>
    </p>
  );
}
export function Skeleton() {
  return (
    <div className="stack" aria-busy="true" aria-label="Loading reports">
      {[1, 2].map((n) => (
        <div key={n} className="card card-pad stack">
          <div className="skel" />
          <div className="skel" />
          <div className="skel" />
        </div>
      ))}
    </div>
  );
}
export const verdictLabels = {
  better: "Getting better",
  worse: "Getting worse",
  same: "About the same",
  "back-in-range": "Back in range",
  first: "First reading",
};
