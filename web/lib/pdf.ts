import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import type { AnalysisResponse } from "./api";
import { formatForPDF } from "./clinical";

function text(doc: jsPDF, value: string, x: number, y: number, size = 11, weight: "normal" | "bold" = "normal") {
  doc.setFont("helvetica", weight);
  doc.setFontSize(size);
  doc.text(value, x, y);
}

function addSectionTitle(doc: jsPDF, title: string, y: number): number {
  text(doc, title, 14, y, 14, "bold");
  doc.setDrawColor(220, 220, 220);
  doc.line(14, y + 2, 196, y + 2);
  return y + 10;
}

export function generateClinicalPdfReport(analysis: AnalysisResponse): Blob {
  const { groupedAlerts, groupedRecords } = formatForPDF(analysis.records, analysis.health_summary.concerns);

  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  let y = 20;

  text(doc, "Medical Report Analyzer", 14, y, 20, "bold");
  y += 10;
  text(doc, `Patient: ${analysis.patient_info.name || "N/A"}`, 14, y);
  y += 6;
  text(doc, `Date: ${analysis.patient_info.date || "N/A"}`, 14, y);
  y += 6;
  text(doc, `Lab: ${analysis.patient_info.lab_name || "N/A"}`, 14, y);
  y += 6;
  text(doc, `Summary: ${analysis.total_records} tests, ${analysis.health_summary.concerns.length} alerts`, 14, y);

  y += 14;
  y = addSectionTitle(doc, "Overall Summary", y);
  text(doc, `Overall Health Score: ${analysis.health_summary.overall_score}`, 14, y);
  y += 6;
  text(doc, `Alerts: ${analysis.health_summary.concerns.length}`, 14, y);
  y += 6;
  text(doc, `Categories Affected: ${groupedAlerts.length}`, 14, y);

  y += 12;
  y = addSectionTitle(doc, "Alerts by Category", y);

  autoTable(doc, {
    startY: y,
    head: [["Category", "Alerts", "Critical", "High", "Low", "Severity"]],
    body: groupedAlerts.map((row) => [
      row.category,
      String(row.totalAlerts),
      String(row.criticalCount),
      String(row.highCount),
      String(row.lowCount),
      row.severity.toUpperCase(),
    ]),
    styles: { fontSize: 9, cellPadding: 2 },
    headStyles: { fillColor: [245, 245, 245], textColor: 20 },
    alternateRowStyles: { fillColor: [252, 252, 252] },
  });

  y = (doc as jsPDF & { lastAutoTable?: { finalY: number } }).lastAutoTable?.finalY ?? y;

  for (const category of groupedAlerts) {
    if (y > 240) {
      doc.addPage();
      y = 20;
    }

    y += 10;
    y = addSectionTitle(doc, `Category: ${category.category}`, y);

    autoTable(doc, {
      startY: y,
      head: [["Test", "Value", "Reference", "Status", "Date"]],
      body: category.tests.map((test) => [
        test.testName,
        String(test.result ?? "N/A"),
        test.reference || "N/A",
        test.status || "N/A",
        test.date || "N/A",
      ]),
      styles: { fontSize: 8.5, cellPadding: 2 },
      headStyles: { fillColor: [245, 245, 245], textColor: 20 },
      alternateRowStyles: { fillColor: [252, 252, 252] },
    });

    y = (doc as jsPDF & { lastAutoTable?: { finalY: number } }).lastAutoTable?.finalY ?? y;
  }

  if (groupedRecords.length > 0) {
    if (y > 230) {
      doc.addPage();
      y = 20;
    }

    y += 10;
    y = addSectionTitle(doc, "Trends", y);

    autoTable(doc, {
      startY: y,
      head: [["Date", "Total Tests", "Categories", "Alerts"]],
      body: groupedRecords.map((group) => {
        let alerts = 0;
        let total = 0;
        for (const category of group.categories) {
          total += category.tests.length;
          alerts += category.tests.filter((t) => {
            const status = (t.Status ?? "").toLowerCase();
            return status === "high" || status === "low" || status === "critical" || status === "positive";
          }).length;
        }
        return [group.date, String(total), String(group.categories.length), String(alerts)];
      }),
      styles: { fontSize: 9, cellPadding: 2 },
      headStyles: { fillColor: [245, 245, 245], textColor: 20 },
      alternateRowStyles: { fillColor: [252, 252, 252] },
    });
  }

  return doc.output("blob");
}
