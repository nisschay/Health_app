"""PostgreSQL models and repository functions. Schema changes live in backend_api/migrations."""
from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    case,
    create_engine,
    func,
    text,
    update,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Session, load_only, sessionmaker

from .normalization import CONCERNING_STATUS_VALUES

LAST_LOGIN_TOUCH_INTERVAL = timedelta(minutes=10)

# Keep DB URL loading consistent with app config so persistence target is stable.
_APP_DIR = Path(__file__).resolve().parent
_ROOT_ENV_PATH = _APP_DIR.parent.parent / ".env"
_LOCAL_ENV_PATH = _APP_DIR.parent / ".env"
load_dotenv(dotenv_path=_ROOT_ENV_PATH)
load_dotenv(dotenv_path=_LOCAL_ENV_PATH)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Point it at your PostgreSQL database.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"
JOB_INTERRUPTED = "interrupted"
JOB_ACTIVE_STATUSES = (JOB_QUEUED, JOB_RUNNING)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String(128), unique=True, index=True, nullable=False)
    email = Column(String(256), nullable=True)
    display_name = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReportAnalysis(Base):
    __tablename__ = "report_analyses"

    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String(128), index=True, nullable=False)
    patient_name = Column(String(256), nullable=True)
    patient_age = Column(String(64), nullable=True)
    patient_gender = Column(String(64), nullable=True)
    patient_id = Column(String(128), nullable=True)
    lab_name = Column(String(512), nullable=True)
    report_date = Column(String(64), nullable=True)
    total_records = Column(Integer, default=0)
    analysis_json = Column(Text, nullable=False)
    source_filenames = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Profile(Base):
    __tablename__ = "profiles"
    __table_args__ = (
        # One self profile per account; enforced by the database, not by a read-then-write.
        Index(
            "uq_profiles_owner_self",
            "account_owner_id",
            unique=True,
            postgresql_where=text("lower(relationship) = 'self'"),
            sqlite_where=text("lower(relationship) = 'self'"),
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_owner_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    full_name = Column(String(256), nullable=False)
    relationship = Column(String(64), nullable=False)
    date_of_birth = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Study(Base):
    __tablename__ = "studies"
    __table_args__ = (UniqueConstraint("profile_id", "name", name="uq_studies_profile_name"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id"), index=True, nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(UUID(as_uuid=True), ForeignKey("studies.id"), index=True, nullable=False)
    file_name = Column(String(512), nullable=False)
    file_url = Column(Text, nullable=False)
    report_date = Column(Date, nullable=False)
    lab_name = Column(String(512), nullable=True)
    analysis_data = Column(JSONB, nullable=False)
    normalized_records = Column(JSONB, nullable=True)
    is_normalized = Column(Boolean, nullable=False, default=False)
    normalization_version = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ReportFinding(Base):
    """One measured value. The row every trend, dashboard and combined view is built from."""

    __tablename__ = "report_findings"
    __table_args__ = (
        Index("ix_report_findings_profile_test_date", "profile_id", "canonical_test", "test_date"),
    )

    id = Column(Integer, primary_key=True)
    report_id = Column(UUID(as_uuid=True), ForeignKey("reports.id", ondelete="CASCADE"), index=True, nullable=False)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False)
    canonical_test = Column(String(256), nullable=False)
    original_test_name = Column(String(256), nullable=True)
    category = Column(String(128), nullable=True)
    test_date = Column(Date, nullable=True)
    result_text = Column(String(256), nullable=True)
    value_numeric = Column(Float, nullable=True)
    comparator = Column(String(4), nullable=True)
    unit = Column(String(64), nullable=True)
    reference_range = Column(String(256), nullable=True)
    ref_low = Column(Float, nullable=True)
    ref_high = Column(Float, nullable=True)
    status = Column(String(32), nullable=False)
    source_filename = Column(String(512), nullable=True)
    aliases = Column(JSONB, nullable=True)


class ReportJob(Base):
    """An upload being analysed. The row outlives the browser tab that started it."""

    __tablename__ = "report_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    study_id = Column(UUID(as_uuid=True), ForeignKey("studies.id", ondelete="SET NULL"), nullable=True)
    analysis_id = Column(Integer, ForeignKey("report_analyses.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(16), nullable=False, default=JOB_QUEUED)
    source_filenames = Column(JSONB, nullable=False)
    progress = Column(JSONB, nullable=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)


class ExtractionCache(Base):
    """Model output keyed by report-text hash; survives Space restarts, unlike a dict."""

    __tablename__ = "extraction_cache"

    text_hash = Column(String(64), primary_key=True)
    payload = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def get_cached_extraction(text_hash: str) -> dict[str, Any] | None:
    with SessionLocal() as db:
        row = db.get(ExtractionCache, text_hash)
        return dict(row.payload) if row else None


def store_cached_extraction(text_hash: str, payload: dict[str, Any]) -> None:
    with SessionLocal() as db:
        db.merge(ExtractionCache(text_hash=text_hash, payload=payload))
        db.commit()


def ping_database() -> None:
    """Cheap connectivity probe; raises when the database is unreachable."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _insert(db: Session, table):
    """The dialect's INSERT ... ON CONFLICT construct; sqlite is the test database."""
    return (sqlite if db.get_bind().dialect.name == "sqlite" else postgresql).insert(table)


# ── Users and profiles ────────────────────────────────────────────────────────

def _default_self_profile_name(display_name: str | None, email: str | None) -> str:
    candidate = (display_name or "").strip()
    if candidate:
        return candidate
    if email and "@" in email:
        return email.split("@", 1)[0].replace(".", " ").strip().title() or "My Profile"
    return "My Profile"


def upsert_user(db: Session, firebase_uid: str, email: str | None, display_name: str | None) -> User:
    """One statement for the user, one for the self profile; both idempotent under concurrency."""
    now = datetime.utcnow()
    touch_before = now - LAST_LOGIN_TOUCH_INTERVAL
    stmt = _insert(db, User).values(
        firebase_uid=firebase_uid, email=email, display_name=display_name, created_at=now, last_login=now
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[User.firebase_uid],
        set_={
            "email": func.coalesce(stmt.excluded.email, User.email),
            "display_name": func.coalesce(stmt.excluded.display_name, User.display_name),
            "last_login": case((User.last_login < touch_before, now), else_=User.last_login),
        },
    ).returning(User)
    user = db.execute(stmt).scalar_one()

    profile = _insert(db, Profile).values(
        id=uuid.uuid4(),
        account_owner_id=user.id,
        full_name=_default_self_profile_name(user.display_name, user.email),
        relationship="self",
        created_at=now,
        updated_at=now,
    )
    db.execute(profile.on_conflict_do_nothing(index_elements=[Profile.account_owner_id], index_where=text("lower(relationship) = 'self'")))
    db.commit()
    return user


def get_user_by_firebase_uid(db: Session, firebase_uid: str) -> User | None:
    return db.query(User).filter(User.firebase_uid == firebase_uid).first()


def create_profile(
    db: Session,
    account_owner_id: int,
    full_name: str,
    relationship: str,
    date_of_birth: date | None = None,
) -> Profile:
    row = Profile(
        account_owner_id=account_owner_id,
        full_name=full_name,
        relationship=relationship,
        date_of_birth=date_of_birth,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_profile_by_id(db: Session, profile_id: uuid.UUID) -> Profile | None:
    return db.query(Profile).filter(Profile.id == profile_id).first()


def list_profiles_for_owner(db: Session, account_owner_id: int) -> list[Profile]:
    return (
        db.query(Profile)
        .filter(Profile.account_owner_id == account_owner_id)
        .order_by(Profile.created_at.asc())
        .all()
    )


# ── History ───────────────────────────────────────────────────────────────────

def save_analysis(
    db: Session,
    firebase_uid: str,
    patient_info: dict[str, Any],
    analysis_data: dict[str, Any],
    source_filenames: list[str],
) -> ReportAnalysis:
    record = ReportAnalysis(
        firebase_uid=firebase_uid,
        patient_name=patient_info.get("name"),
        patient_age=patient_info.get("age"),
        patient_gender=patient_info.get("gender"),
        patient_id=patient_info.get("patient_id"),
        lab_name=patient_info.get("lab_name"),
        report_date=patient_info.get("date"),
        total_records=analysis_data.get("total_records", 0),
        analysis_json=json.dumps(analysis_data),
        source_filenames=",".join(source_filenames),
    )
    db.add(record)
    db.flush()
    return record


def get_user_analyses(db: Session, firebase_uid: str, limit: int, offset: int) -> list[ReportAnalysis]:
    """A page of history rows without the full analysis text, which the list never shows."""
    return (
        db.query(ReportAnalysis)
        .options(
            load_only(
                ReportAnalysis.id,
                ReportAnalysis.patient_name,
                ReportAnalysis.patient_age,
                ReportAnalysis.patient_gender,
                ReportAnalysis.lab_name,
                ReportAnalysis.report_date,
                ReportAnalysis.total_records,
                ReportAnalysis.source_filenames,
                ReportAnalysis.created_at,
            )
        )
        .filter(ReportAnalysis.firebase_uid == firebase_uid)
        .order_by(ReportAnalysis.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_analysis_by_id(db: Session, analysis_id: int, firebase_uid: str) -> ReportAnalysis | None:
    return (
        db.query(ReportAnalysis)
        .filter(ReportAnalysis.id == analysis_id, ReportAnalysis.firebase_uid == firebase_uid)
        .first()
    )


# ── Studies and reports ───────────────────────────────────────────────────────

def create_study(db: Session, profile_id: uuid.UUID, name: str, description: str | None = None) -> Study:
    row = Study(profile_id=profile_id, name=name, description=description)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_studies_for_profile(db: Session, profile_id: uuid.UUID) -> list[Study]:
    return (
        db.query(Study)
        .filter(Study.profile_id == profile_id)
        .order_by(Study.updated_at.desc(), Study.created_at.desc())
        .all()
    )


def get_study_by_id(db: Session, study_id: uuid.UUID) -> Study | None:
    return db.query(Study).filter(Study.id == study_id).first()


def count_reports_for_study(db: Session, study_id: uuid.UUID) -> int:
    return db.query(Report).filter(Report.study_id == study_id).count()


def study_report_stats(db: Session, study_ids: list[uuid.UUID]) -> dict[uuid.UUID, tuple[int, date | None, date | None]]:
    """(count, first date, last date) per study in one grouped query."""
    if not study_ids:
        return {}
    rows = (
        db.query(Report.study_id, func.count(Report.id), func.min(Report.report_date), func.max(Report.report_date))
        .filter(Report.study_id.in_(study_ids))
        .group_by(Report.study_id)
        .all()
    )
    return {study_id: (count, first, last) for study_id, count, first, last in rows}


def list_studies_for_owner(db: Session, account_owner_id: int) -> list[Study]:
    return (
        db.query(Study)
        .join(Profile, Profile.id == Study.profile_id)
        .filter(Profile.account_owner_id == account_owner_id)
        .order_by(Study.updated_at.desc(), Study.created_at.desc())
        .all()
    )


def list_dashboard_report_rows(db: Session, account_owner_id: int):
    """Every report under the owner, without its JSON payload; the dashboard aggregates in memory."""
    return (
        db.query(Report.id, Report.study_id, Report.report_date, Report.lab_name)
        .join(Study, Study.id == Report.study_id)
        .join(Profile, Profile.id == Study.profile_id)
        .filter(Profile.account_owner_id == account_owner_id)
        .order_by(Report.report_date.asc(), Report.uploaded_at.asc())
        .all()
    )


def dashboard_alert_counts(db: Session, account_owner_id: int) -> dict[uuid.UUID, int]:
    """Concerning findings per study in one grouped query."""
    rows = (
        db.query(Report.study_id, func.count(ReportFinding.id))
        .join(Report, Report.id == ReportFinding.report_id)
        .join(Study, Study.id == Report.study_id)
        .join(Profile, Profile.id == Study.profile_id)
        .filter(Profile.account_owner_id == account_owner_id, ReportFinding.status.in_(CONCERNING_STATUS_VALUES))
        .group_by(Report.study_id)
        .all()
    )
    return dict(rows)


def create_report(
    db: Session,
    study_id: uuid.UUID,
    file_name: str,
    file_url: str,
    report_date: date,
    analysis_data: dict[str, Any],
    normalized_records: list[dict[str, Any]],
    normalization_version: int,
    findings: list[ReportFinding],
    lab_name: str | None = None,
) -> Report:
    """Adds the report and its findings to the session; the caller owns the transaction."""
    row = Report(
        study_id=study_id,
        file_name=file_name,
        file_url=file_url,
        report_date=report_date,
        lab_name=lab_name,
        analysis_data=analysis_data,
        normalized_records=normalized_records,
        is_normalized=True,
        normalization_version=normalization_version,
    )
    db.add(row)
    db.flush()
    for finding in findings:
        finding.report_id = row.id
    db.add_all(findings)
    db.execute(update(Study).where(Study.id == study_id).values(updated_at=datetime.utcnow()))
    return row


def list_reports_for_study(db: Session, study_id: uuid.UUID) -> list[Report]:
    return (
        db.query(Report)
        .filter(Report.study_id == study_id)
        .order_by(Report.report_date.asc(), Report.uploaded_at.asc())
        .all()
    )


def list_findings_for_reports(db: Session, report_ids: list[uuid.UUID]) -> list[ReportFinding]:
    if not report_ids:
        return []
    return (
        db.query(ReportFinding)
        .filter(ReportFinding.report_id.in_(report_ids))
        .order_by(ReportFinding.test_date.asc(), ReportFinding.category.asc(), ReportFinding.canonical_test.asc())
        .all()
    )


def list_trend_points(db: Session, profile_id: uuid.UUID, canonical_test: str) -> list[ReportFinding]:
    """One indexed SELECT: every reading of one test for one person, oldest first."""
    return (
        db.query(ReportFinding)
        .filter(ReportFinding.profile_id == profile_id, ReportFinding.canonical_test == canonical_test)
        .order_by(ReportFinding.test_date.asc())
        .all()
    )


def replace_findings(db: Session, report_id: uuid.UUID, findings: list[ReportFinding]) -> None:
    db.query(ReportFinding).filter(ReportFinding.report_id == report_id).delete(synchronize_session=False)
    for finding in findings:
        finding.report_id = report_id
    db.add_all(findings)


# ── Jobs ──────────────────────────────────────────────────────────────────────

def get_job(db: Session, job_id: uuid.UUID, owner_id: int) -> ReportJob | None:
    return db.query(ReportJob).filter(ReportJob.id == job_id, ReportJob.owner_id == owner_id).first()


def mark_active_jobs_interrupted(db: Session, message: str) -> int:
    """Jobs that were mid-flight when the process died can never finish; say so."""
    result = db.execute(
        update(ReportJob)
        .where(ReportJob.status.in_(JOB_ACTIVE_STATUSES))
        .values(status=JOB_INTERRUPTED, error=message, finished_at=datetime.utcnow())
    )
    db.commit()
    return result.rowcount
