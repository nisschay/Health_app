"""Persist an analysis: one report per uploaded file inside a study, or one history row."""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import exists, text
from sqlalchemy.orm import Session

from .database import (
    Report,
    ReportAnalysis,
    ReportFinding,
    SessionLocal,
    Study,
    create_report,
    replace_findings,
    save_analysis,
)
from .findings import findings_from_records
from .normalization import NORMALIZATION_VERSION, normalize_records

logger = logging.getLogger(__name__)

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


class NothingSaved(ValueError):
    """Every file in the upload failed; the job fails with the per-file reasons."""


def save_to_study(
    db: Session,
    study: Study,
    analysis: dict[str, Any],
    source_filenames: list[str],
    failed: dict[str, str] | None = None,
) -> list[str]:
    """Add one report per readable file to the session and return a warning per skipped file.

    A file that failed extraction, or has no report date, is skipped rather than failing the
    batch: the other files' results (and the model quota spent on them) are kept.
    The caller commits, so the reports and the job status land in one transaction.
    """
    failed = dict(failed or {})
    patient_info = analysis.get("patient_info") or {}
    batch_date = parse_report_date(patient_info.get("date"))
    all_rows = analysis.get("records") or []
    tagged = any(row.get("Source_Filename") for row in all_rows if isinstance(row, dict))
    readable = [name for name in source_filenames if name not in failed]

    saved = 0
    for name in readable:
        rows = rows_for_file(all_rows, name, fallback_to_all=not tagged or len(readable) == 1)
        normalized = normalize_records(rows)
        if not normalized:
            failed[name] = "No test results were found in this file."
            continue
        try:
            report_date = _resolve_report_date(normalized, batch_date, name)
        except ValueError as exc:
            failed[name] = str(exc)
            continue
        create_report(
            db,
            study_id=study.id,
            file_name=name,
            file_url=f"uploaded://{name}",
            report_date=report_date,
            lab_name=patient_info.get("lab_name"),
            analysis_data={**analysis, "records": normalized, "total_records": len(normalized)},
            normalized_records=normalized,
            normalization_version=NORMALIZATION_VERSION,
            findings=findings_from_records(study.profile_id, normalized, name),
        )
        saved += 1

    warnings = [f"{name}: {reason}" for name, reason in failed.items()]
    if not saved:
        raise NothingSaved("None of the uploaded files could be saved. " + "; ".join(warnings))
    return warnings


def save_to_history(
    db: Session,
    firebase_uid: str,
    analysis: dict[str, Any],
    source_filenames: list[str],
    failed: dict[str, str] | None = None,
) -> tuple[ReportAnalysis, list[str]]:
    """Add one history row to the session; the caller commits."""
    failed = failed or {}
    names = [name for name in source_filenames if name not in failed] or source_filenames
    row = save_analysis(db, firebase_uid, analysis.get("patient_info") or {}, analysis, names)
    return row, [f"{name}: {reason}" for name, reason in failed.items()]


def _stored_rows(report: Report) -> list[dict[str, Any]]:
    """The report's normalised rows, re-normalising if they predate the current normaliser."""
    if report.normalization_version == NORMALIZATION_VERSION and isinstance(report.normalized_records, list):
        return report.normalized_records
    payload = report.analysis_data if isinstance(report.analysis_data, dict) else {}
    return normalize_records(rows_for_file(payload.get("records") or [], report.file_name, fallback_to_all=True))


def findings_in_memory(report: Report, profile_id: uuid.UUID) -> list[ReportFinding]:
    """Findings for a report that has no rows yet, built without writing anything.

    Reads never write: two concurrent reads used to insert the same findings twice.
    """
    findings = findings_from_records(profile_id, _stored_rows(report), report.file_name)
    for finding in findings:
        finding.report_id = report.id
    return findings


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


# Any constant works; it only has to be the same for every process that backfills.
BACKFILL_LOCK_KEY = 4_172_026


def _take_backfill_lock(db: Session, wait: bool) -> bool:
    """One backfill at a time across processes; released when the transaction ends."""
    if db.get_bind().dialect.name != "postgresql":
        return True
    if wait:
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": BACKFILL_LOCK_KEY})
        return True
    return bool(db.execute(text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": BACKFILL_LOCK_KEY}).scalar())


def backfill_findings(db: Session, *, rebuild_all: bool = False, wait: bool = False) -> tuple[int, int]:
    """Give reports their findings rows in one transaction. Returns (reports, findings).

    By default only reports with no findings yet are touched, so it is cheap to run at every start.
    rebuild_all re-normalises every report, for after a normaliser change.
    """
    if not _take_backfill_lock(db, wait):
        return 0, 0
    query = db.query(Report, Study.profile_id).join(Study, Study.id == Report.study_id)
    if not rebuild_all:
        query = query.filter(~exists().where(ReportFinding.report_id == Report.id))
    reports = findings = 0
    for report, profile_id in query.all():
        if rebuild_all:
            rows = rebuild_report(db, report, profile_id)
        else:
            rows = findings_from_records(profile_id, _stored_rows(report), report.file_name)
            replace_findings(db, report.id, rows)
        reports += 1
        findings += len(rows)
    db.commit()
    return reports, findings


def start_backfill_in_background() -> None:
    """Reports saved before report_findings existed get their rows once, without delaying startup."""

    def _run() -> None:
        try:
            with SessionLocal() as db:
                reports, findings = backfill_findings(db)
            if reports:
                logger.info("Backfilled %d findings across %d reports", findings, reports)
        except Exception:
            logger.exception("Findings backfill failed; it will be retried at the next start")

    threading.Thread(target=_run, name="findings-backfill", daemon=True).start()
