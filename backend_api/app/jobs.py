"""Analyse uploads off the request thread. The job row is the contract with the browser."""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from .auth import RequestUser
from .config import settings
from .database import (
    JOB_DONE,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    ReportJob,
    SessionLocal,
    User,
    get_study_by_id,
    mark_active_jobs_interrupted,
)
from .saving import normalize_analysis_payload, save_to_history, save_to_study
from .security import internal_error

logger = logging.getLogger(__name__)

UPSTREAM_RATE_LIMIT_PREFIX = "RATE_LIMIT_EXCEEDED:"
INTERRUPTED_MESSAGE = "The server restarted while this upload was being processed. Please upload the files again."
_pool = ThreadPoolExecutor(max_workers=settings.job_workers, thread_name_prefix="report-job")


def submit(
    db: Session,
    owner: User,
    service,
    pdf_files: list[tuple[str, bytes]],
    existing_data: tuple[str, bytes] | None,
    include_raw_texts: bool,
    study_id: uuid.UUID | None,
) -> ReportJob:
    names = [name for name, _ in pdf_files]
    job = ReportJob(
        owner_id=owner.id,
        study_id=study_id,
        status=JOB_QUEUED,
        source_filenames=names,
        progress={
            "stage": "processing",
            "files": {name: {"step": "queued", "percent": 0} for name in names},
            "processed": 0,
            "total": len(names),
            "eta_seconds": None,
        },
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    requester = RequestUser(user_id=owner.firebase_uid, email=owner.email)
    _pool.submit(_run, job.id, requester, service, pdf_files, existing_data, include_raw_texts)
    return job


def _run(job_id, requester, service, pdf_files, existing_data, include_raw_texts) -> None:
    _set(job_id, status=JOB_RUNNING, started_at=datetime.utcnow())
    try:
        result = service.analyze_reports(
            pdf_files=pdf_files,
            existing_data_file=existing_data,
            include_raw_texts=include_raw_texts,
            user=requester,
            progress_callback=lambda event: _record(job_id, event),
        )
        result = normalize_analysis_payload(result, service)
        _set(job_id, progress_stage="saving")
        with SessionLocal() as db:
            job = db.get(ReportJob, job_id)
            names = list(job.source_filenames) or ["uploaded-report.pdf"]
            if job.study_id:
                save_to_study(db, get_study_by_id(db, job.study_id), result, names)
            else:
                job.analysis_id = save_to_history(db, requester.user_id, result, names).id
            job.status = JOB_DONE
            job.finished_at = datetime.utcnow()
            job.progress = {**job.progress, "stage": "done"}
            db.commit()
    except Exception as exc:
        logger.exception("Report job %s failed", job_id)
        _set(job_id, status=JOB_FAILED, error=failure_message(exc), finished_at=datetime.utcnow())


def failure_message(exc: Exception) -> str:
    if isinstance(exc, RuntimeError) and str(exc).startswith(UPSTREAM_RATE_LIMIT_PREFIX):
        return "The report reader is rate limited right now. Please retry shortly."
    if isinstance(exc, ValueError):
        return str(exc) or "Could not read the uploaded reports."
    return internal_error(exc, "Analysis").detail


def _set(job_id, progress_stage: str | None = None, **fields: Any) -> None:
    with SessionLocal() as db:
        job = db.get(ReportJob, job_id)
        for key, value in fields.items():
            setattr(job, key, value)
        if progress_stage:
            job.progress = {**job.progress, "stage": progress_stage}
        db.commit()


def _record(job_id, event: dict[str, Any]) -> None:
    """Per-file events from the extractor become the progress the browser polls."""
    if event.get("type") != "file":
        return
    with SessionLocal() as db:
        job = db.get(ReportJob, job_id)
        files = {**job.progress["files"], event["file"]: {k: event[k] for k in ("step", "percent", "error") if k in event}}
        job.progress = {
            **job.progress,
            "files": files,
            "processed": event.get("processed", job.progress["processed"]),
            "eta_seconds": event.get("eta_seconds", job.progress.get("eta_seconds")),
        }
        db.commit()


def mark_interrupted() -> None:
    with SessionLocal() as db:
        count = mark_active_jobs_interrupted(db, INTERRUPTED_MESSAGE)
    if count:
        logger.warning("%d report job(s) were in flight at shutdown and are marked interrupted", count)
