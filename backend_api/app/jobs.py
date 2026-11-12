"""Analyse uploads off the request thread. The job row is the contract with the browser.

A job's files are stored with it until it finishes, and a running job sends a heartbeat.
When the worker disappears (a deploy restarts the server), the job is queued again and
resumed, here at startup or by whichever process sees the stale heartbeat when the
browser polls. A job runs only after an atomic claim, so two processes never both run it.
"""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, update
from sqlalchemy.orm import Session

from .auth import RequestUser
from .config import settings
from .database import (
    JOB_ACTIVE_STATUSES,
    JOB_DONE,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    ReportJob,
    ReportJobFile,
    SessionLocal,
    User,
    get_study_by_id,
)
from .saving import normalize_analysis_payload, save_to_history, save_to_study
from .security import internal_error

logger = logging.getLogger(__name__)

UPSTREAM_RATE_LIMIT_PREFIX = "RATE_LIMIT_EXCEEDED:"
STALE_MESSAGE = "This upload stopped making progress. Please upload the files again."
TOO_MANY_ATTEMPTS_MESSAGE = "This upload failed to finish after several tries. Please upload the files again."
HEARTBEAT_SECONDS = 20
HEARTBEAT_STALE_AFTER = timedelta(seconds=120)
MAX_ATTEMPTS = 3

_pool = ThreadPoolExecutor(max_workers=settings.job_workers, thread_name_prefix="report-job")
_pending = 0  # jobs dispatched to this process and not yet finished
_pending_lock = threading.Lock()
_dispatched: set[uuid.UUID] = set()  # ids this process has handed to its pool


class TooManyJobs(Exception):
    """Admission refused; the route turns this into 429."""


def submit(
    db: Session,
    owner: User,
    service,
    pdf_files: list[tuple[str, bytes]],
    existing_data: tuple[str, bytes] | None,
    include_raw_texts: bool,
    study_id: uuid.UUID | None,
) -> ReportJob:
    active = db.query(ReportJob).filter(ReportJob.owner_id == owner.id, ReportJob.status.in_(JOB_ACTIVE_STATUSES)).count()
    if active >= settings.max_active_jobs_per_user:
        raise TooManyJobs("You already have uploads being processed. Wait for one to finish, then try again.")
    if not _reserve(enforce_limit=True):
        raise TooManyJobs("The server is busy reading other reports. Please try again in a few minutes.")

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
    try:
        db.add(job)
        db.flush()
        db.add_all(_file_rows(job.id, pdf_files, existing_data))
        db.commit()  # the job and its files land together
        db.refresh(job)
        _dispatch(job.id, service, pdf_files, existing_data, include_raw_texts)
    except BaseException:
        _release(None)
        raise
    return job


def _file_rows(job_id, pdf_files, existing_data) -> list[ReportJobFile]:
    rows = [ReportJobFile(job_id=job_id, position=i, kind="pdf", filename=n, content=c) for i, (n, c) in enumerate(pdf_files)]
    if existing_data:
        rows.append(ReportJobFile(job_id=job_id, position=len(rows), kind="existing", filename=existing_data[0], content=existing_data[1]))
    return rows


def _reserve(enforce_limit: bool) -> bool:
    global _pending
    with _pending_lock:
        if enforce_limit and _pending >= settings.max_pending_jobs:
            return False
        _pending += 1
        return True


def _release(job_id) -> None:
    global _pending
    with _pending_lock:
        _pending = max(0, _pending - 1)
        _dispatched.discard(job_id)


def _dispatch(job_id, service, pdf_files, existing_data, include_raw_texts=False) -> None:
    with _pending_lock:
        _dispatched.add(job_id)
    _pool.submit(_run, job_id, service, pdf_files, existing_data, include_raw_texts)


def _claim(job_id) -> RequestUser | None:
    """queued -> running, atomically. None when another worker already has the job."""
    now = datetime.utcnow()
    with SessionLocal() as db:
        claimed = db.execute(
            update(ReportJob)
            .where(ReportJob.id == job_id, ReportJob.status == JOB_QUEUED)
            .values(status=JOB_RUNNING, started_at=now, heartbeat_at=now, attempts=ReportJob.attempts + 1)
        ).rowcount
        if not claimed:
            db.rollback()
            return None
        owner = db.query(User).join(ReportJob, ReportJob.owner_id == User.id).filter(ReportJob.id == job_id).one()
        db.commit()
        return RequestUser(user_id=owner.firebase_uid, email=owner.email)


def _heartbeat(job_id, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            with SessionLocal() as db:
                db.execute(
                    update(ReportJob)
                    .where(ReportJob.id == job_id, ReportJob.status == JOB_RUNNING)
                    .values(heartbeat_at=datetime.utcnow())
                )
                db.commit()
        except Exception:
            logger.warning("Heartbeat for report job %s failed", job_id, exc_info=True)


def _run(job_id, service, pdf_files, existing_data, include_raw_texts) -> None:
    """Never leaves a claimed job running: every failure, including a failed status write, ends in failed."""
    stop = threading.Event()
    try:
        requester = _claim(job_id)
        if requester is None:
            return  # another process is running it
        threading.Thread(target=_heartbeat, args=(job_id, stop), daemon=True, name=f"heartbeat-{job_id}").start()
        result = service.analyze_reports(
            pdf_files=pdf_files,
            existing_data_file=existing_data,
            include_raw_texts=include_raw_texts,
            user=requester,
            progress_callback=lambda event: _record(job_id, event),
        )
        failed = result.pop("failed_files", None) or {}
        notes = result.pop("notes", None) or []
        result = normalize_analysis_payload(result, service)
        _set(job_id, progress_stage="saving")
        with SessionLocal() as db:
            job = db.get(ReportJob, job_id)
            if job.status != JOB_RUNNING:
                logger.warning("Report job %s is %s, not running; its result is discarded", job_id, job.status)
                return
            names = list(job.source_filenames) or ["uploaded-report.pdf"]
            if job.study_id:
                warnings = save_to_study(db, get_study_by_id(db, job.study_id), result, names, failed)
            else:
                analysis, warnings = save_to_history(db, requester.user_id, result, names, failed)
                job.analysis_id = analysis.id
            job.status = JOB_DONE
            job.finished_at = datetime.utcnow()
            job.progress = {**job.progress, "stage": "done", "warnings": warnings, "notes": notes}
            _delete_files(db, job_id)
            db.commit()  # the reports, the "done" status and the files' removal land together
    except Exception as exc:
        logger.exception("Report job %s failed", job_id)
        try:
            _finish_failed(job_id, failure_message(exc))
        except Exception:
            # The database is unreachable too; the job's heartbeat stops, and a later poll resumes or fails it.
            logger.exception("Could not record the failure of report job %s", job_id)
    finally:
        stop.set()
        _release(job_id)


def failure_message(exc: Exception) -> str:
    if isinstance(exc, RuntimeError) and str(exc).startswith(UPSTREAM_RATE_LIMIT_PREFIX):
        return "The report reader is rate limited right now. Please retry shortly."
    if isinstance(exc, ValueError):
        return str(exc) or "Could not read the uploaded reports."
    return internal_error(exc, "Analysis").detail


def _delete_files(db: Session, job_id) -> None:
    db.query(ReportJobFile).filter(ReportJobFile.job_id == job_id).delete(synchronize_session=False)


def _finish_failed(job_id, message: str) -> None:
    with SessionLocal() as db:
        db.execute(
            update(ReportJob)
            .where(ReportJob.id == job_id, ReportJob.status.in_(JOB_ACTIVE_STATUSES))
            .values(status=JOB_FAILED, error=message, finished_at=datetime.utcnow())
        )
        _delete_files(db, job_id)
        db.commit()


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
        job.heartbeat_at = datetime.utcnow()
        db.commit()


# ── Recovering jobs whose worker is gone ──────────────────────────────────────

def _is_orphaned(job: ReportJob, now: datetime) -> bool:
    if job.status == JOB_RUNNING:
        return (job.heartbeat_at or job.started_at or job.created_at) < now - HEARTBEAT_STALE_AFTER
    # Queued and not waiting in this process's pool: whoever queued it is gone.
    return job.status == JOB_QUEUED and job.id not in _dispatched and job.created_at < now - HEARTBEAT_STALE_AFTER


def recover(db: Session, job: ReportJob, service) -> ReportJob:
    """Resume an orphaned job in this process, or fail it after MAX_ATTEMPTS. Returns the job as it now stands."""
    now = datetime.utcnow()
    if job.status not in JOB_ACTIVE_STATUSES or not _is_orphaned(job, now):
        return job
    if job.attempts >= MAX_ATTEMPTS:
        _finish_failed(job.id, TOO_MANY_ATTEMPTS_MESSAGE)
    else:
        # Re-checked in the same statement: if the worker beat in the meantime, nothing changes.
        last_sign_of_life = func.coalesce(ReportJob.heartbeat_at, ReportJob.started_at, ReportJob.created_at)
        requeued = db.execute(
            update(ReportJob)
            .where(ReportJob.id == job.id, ReportJob.status == job.status, last_sign_of_life < now - HEARTBEAT_STALE_AFTER)
            .values(status=JOB_QUEUED)
        ).rowcount
        db.commit()
        if requeued:
            files = db.query(ReportJobFile).filter(ReportJobFile.job_id == job.id).order_by(ReportJobFile.position).all()
            pdfs = [(f.filename, f.content) for f in files if f.kind == "pdf"]
            existing = next(((f.filename, f.content) for f in files if f.kind == "existing"), None)
            if not pdfs and not existing:
                _finish_failed(job.id, STALE_MESSAGE)
            else:
                logger.warning("Resuming report job %s (attempt %d)", job.id, job.attempts + 1)
                _reserve(enforce_limit=False)
                _dispatch(job.id, service, pdfs, existing)
    db.expire(job)
    db.refresh(job)
    return job


def resume_orphaned_jobs(service) -> None:
    """At startup: every job whose worker is gone is resumed here (or failed after too many tries)."""
    with SessionLocal() as db:
        for job in db.query(ReportJob).filter(ReportJob.status.in_(JOB_ACTIVE_STATUSES)).all():
            recover(db, job, service)
