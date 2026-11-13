import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import type { AnalysisResponse } from "./api";
import {
  summariseTests,
  summariseStudy,
  doctorList,
  orderForAttention,
} from "./results";
import { formatDate } from "./format";
export function generateClinicalPdfReport(analysis: AnalysisResponse): Blob {
  const doc = new jsPDF();
  const tests = summariseTests(analysis.records);
  const summary = summariseStudy(tests);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(20);
  doc.text("LabTrail", 14, 20);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  doc.text(
    doc.splitTextToSize(
      `${analysis.patient_info.name || "Your results"}\nMost recent report: ${formatDate(summary.latestReportDate)}\n${summary.attention} of ${summary.total} tests need attention, counted from each test's latest reading.`,
      180,
    ),
    14,
    30,
  );
  doc.setFontSize(9);
  doc.text(
    doc.splitTextToSize(
      "Not medical advice. Results can contain mistakes. Check every result with your doctor. Ranges come from the lab reports.",
      180,
    ),
    14,
    52,
  );
  autoTable(doc, {
    startY: 68,
    head: [
      [
        "Latest results needing attention",
        "Result",
        "Status",
        "Lab's range",
        "Date",
      ],
    ],
    body: orderForAttention(tests.filter((t) => t.needsAttention)).map((t) => [
      t.name,
      `${t.latest.valueText} ${t.latest.unit}`,
      t.latest.status,
      t.latest.rangeText,
      formatDate(t.latest.date),
    ]),
    styles: { fontSize: 9 },
    headStyles: { fillColor: [15, 88, 78] },
  });
  const notes = doctorList(tests);
  if (notes.length)
    autoTable(doc, {
      head: [["To discuss with the doctor"]],
      body: notes.map((n) => [n]),
      styles: { fontSize: 10 },
      headStyles: { fillColor: [15, 88, 78] },
    });
  doc.addPage();
  doc.setFontSize(16);
  doc.text("Every reading", 14, 20);
  autoTable(doc, {
    startY: 28,
    head: [["Test", "Date", "Result", "Status", "Lab's range", "Lab"]],
    body: tests.flatMap((t) =>
      [...t.readings]
        .reverse()
        .map((r) => [
          t.name,
          formatDate(r.date),
          `${r.valueText} ${r.unit}`,
          r.status === "normal"
            ? "In range"
            : r.status === "none"
              ? "No range given"
              : r.status,
          r.rangeText,
          r.lab || "Lab not named",
        ]),
    ),
    styles: { fontSize: 8 },
    headStyles: { fillColor: [15, 88, 78] },
  });
  return doc.output("blob");
}
