import { summariseStudy, type TestSummary } from "@/lib/results";
import { formatDate } from "@/lib/format";
import { CircleAlert } from "lucide-react";
import { statusIcons } from "./ui";
export default function StudySummary({
  tests,
  base,
  home = false,
}: {
  tests: TestSummary[];
  base: string;
  home?: boolean;
}) {
  const s = summariseStudy(tests);
  return (
    <section className={home ? "study" : "card v2-hero"}>
      <p className="eyebrow">Latest reading of each test</p>
      <h2>
        {s.total
          ? `${s.attention} of ${s.total} tests are outside the lab's range`
          : "No results were read from these reports"}
      </h2>
      <div className="dist" aria-hidden="true">
        {[
          [s.critical, "critical"],
          [s.attention - s.critical, "outside"],
          [s.inRange, "ok"],
          [s.unknown, "none"],
        ].map(
          ([n, c]) =>
            Number(n) > 0 && (
              <i key={c} className={String(c)} style={{ flex: Number(n) }} />
            ),
        )}
      </div>
      <ul className="legend">
        {(
          [
            "critical",
            "high",
            "low",
            "borderline",
            "detected",
            "normal",
            "none",
          ] as const
        ).map((status) => {
          const n = tests.filter((t) => t.latest.status === status).length;
          const Icon = statusIcons[status];
          return (
            n > 0 && (
              <li key={status}>
                <Icon
                  size={14}
                  className={`t-${status === "critical" ? "critical" : status === "normal" ? "ok" : status === "none" ? "none" : "outside"}`}
                />
                <b>{n}</b>{" "}
                {status === "normal"
                  ? "in range"
                  : status === "none"
                    ? "no range on the report"
                    : status}
              </li>
            )
          );
        })}
      </ul>
      {s.critical > 0 && (
        <div className="notice" role="alert">
          <div className="notice-ic">
            <CircleAlert size={18} />
          </div>
          <div className="notice-body">
            {s.criticalTests.map((t) => (
              <div key={t.name}>
                <strong>{t.name} is marked critical</strong>
                <p>
                  {t.latest.valueText} {t.latest.unit} on{" "}
                  {formatDate(t.latest.date)}. The lab&apos;s range is{" "}
                  {t.latest.rangeText}. Please contact a doctor promptly.
                </p>
              </div>
            ))}
          </div>
          <a
            className="link"
            href={`${base}&test=${encodeURIComponent(s.criticalTests[0]!.name)}`}
          >
            See this result ›
          </a>
        </div>
      )}
    </section>
  );
}
