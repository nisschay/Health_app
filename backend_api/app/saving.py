"""Persist an analysis: one report per uploaded file inside a study, or one history row."""
from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from .database import Report, ReportAnalysis, ReportFinding, Study, create_report, replace_findings, save_analysis
from .findings import findings_from_records
from .normalization import NORMALIZATION_VERSION, normalize_records

EMPTY_INSIGHTS: dict[str, Any] = {
    "health_summary": {"overall_score": 0, "category_scores": {}, "concerns": []},
    "body_systems": [],
}


def normalize_analysis_payload(payload: dict[str, Any], service) -> dict[str, Any]:
    normalized = dict(payload)
    records = normalize_records([row for row in payload.get("records") or [] if isinstance(row, dict)])
    normalized["records"] = records
    normalized["total_records"] = len(records)
    normalized.update(service.get_health_insights(records) if records else EMPTY_INSIGHTS)
    return normalized


def parse_report_date(value: str | None) -> date | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        pass
    for sep in ("-", "/", "."):
        parts = raw.split(sep)
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            continue
        if len(parts[0]) == 4:
            year, month, day = parts
        else:
            day, month, year = parts
            if len(year) == 2:
                year = f"20{year}"
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            continue
    return None


def _filename(value: Any) -> str:
    return Path(str(value)).name.strip().lower() if value else ""


def rows_for_file(rows: list[Any], file_name: str, *, fallback_to_all: bool) -> list[dict[str, Any]]:
    """The records extracted from one file, tagged with its name."""
    valid = [row for row in rows if isinstance(row, dict)]
    matched = [row for row in valid if _filename(row.get("Source_Filename")) == _filename(file_name)]
    chosen = matched or (valid if fallback_to_all else [])
    return [{**row, "Source_Filename": row.get("Source_Filename") or file_name} for row in chosen]


def _resolve_report_date(rows: list[dict[str, Any]], fallback: date | None, file_name: str) -> date:
    """Pick a report date, or refuse. Inventing today's date silently corrupted timelines."""
    for row in rows:
        parsed = parse_report_date(str(row.get("Test_Date") or ""))
        if parsed:
            return parsed
    if fallback is None:
        raise ValueError(f"Could not read a report date for '{file_name}'. Set the report date and try again.")
    return fallback


def save_to_study(db: Session, study: Study, analysis: dict[str, Any], source_filenames: list[str]) -> int:
    """Every report and finding of one upload lands in one transaction."""
    patient_info = analysis.get("patient_info") or {}
    batch_date = parse_report_date(patient_info.get("date"))
    all_rows = analysis.get("records") or []
    tagged = any(row.get("Source_Filename") for row in all_rows if isinstance(row, dict))

    for name in source_filenames:
        rows = rows_for_file(all_rows, name, fallback_to_all=not tagged or len(source_filenames) == 1)
        normalized = normalize_records(rows)
        create_report(
            db,
            study_id=study.id,
            file_name=name,
            file_url=f"uploaded://{name}",
            report_date=_resolve_report_date(normalized, batch_date, name),
            lab_name=patient_info.get("lab_name"),
            analysis_data={**analysis, "records": normalized, "total_records": len(normalized)},
            normalized_records=normalized,
            normalization_version=NORMALIZATION_VERSION,
            findings=findings_from_records(study.profile_id, normalized, name),
        )
    db.commit()
    return len(source_filenames)


def save_to_history(db: Session, firebase_uid: str, analysis: dict[str, Any], source_filenames: list[str]) -> ReportAnalysis:
    row = save_analysis(db, firebase_uid, analysis.get("patient_info") or {}, analysis, source_filenames)
    db.commit()
    return row


def rebuild_report(db: Session, report: Report, profile_id: uuid.UUID) -> list[ReportFinding]:
    """Re-normalise a stored report and rewrite its findings; the caller commits."""
    payload = report.analysis_data if isinstance(report.analysis_data, dict) else {}
    normalized = normalize_records(rows_for_file(payload.get("records") or [], report.file_name, fallback_to_all=True))
    report.normalized_records = normalized
    report.is_normalized = True
    report.normalization_version = NORMALIZATION_VERSION
    findings = findings_from_records(profile_id, normalized, report.file_name)
    replace_findings(db, report.id, findings)
    return findings


def findings_for_report(db: Session, report: Report, profile_id: uuid.UUID) -> list[ReportFinding]:
    """Reports saved before report_findings existed get their rows on first read."""
    if report.normalization_version == NORMALIZATION_VERSION and isinstance(report.normalized_records, list):
        findings = findings_from_records(profile_id, report.normalized_records, report.file_name)
        replace_findings(db, report.id, findings)
        return findings
    return rebuild_report(db, report, profile_id)
