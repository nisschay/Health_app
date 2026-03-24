"""PostgreSQL database connection and schema setup via SQLAlchemy."""
from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://medical_user:medical_pass@localhost:5432/medical_project",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
    # Full analysis JSON stored as text (patient_info, records, health_summary, etc.)
    analysis_json = Column(Text, nullable=False)
    source_filenames = Column(Text, nullable=True)  # comma-separated PDF names
    created_at = Column(DateTime, default=datetime.utcnow)


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_owner_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    full_name = Column(String(256), nullable=False)
    relationship = Column(String(64), nullable=False)
    date_of_birth = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Study(Base):
    __tablename__ = "studies"
    __table_args__ = (
        UniqueConstraint("profile_id", "name", name="uq_studies_profile_name"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id"), index=True, nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(UUID(as_uuid=True), ForeignKey("studies.id"), index=True, nullable=False)
    file_name = Column(String(512), nullable=False)
    file_url = Column(Text, nullable=False)
    report_date = Column(Date, nullable=False)
    lab_name = Column(String(512), nullable=True)
    analysis_data = Column(JSONB, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Repository helpers ─────────────────────────────────────────────────────────

def _default_self_profile_name(display_name: str | None, email: str | None) -> str:
    candidate = (display_name or "").strip()
    if candidate:
        return candidate
    if email and "@" in email:
        return email.split("@", 1)[0].replace(".", " ").strip().title() or "My Profile"
    return "My Profile"


def _ensure_self_profile(db: Session, user: User) -> None:
    existing = (
        db.query(Profile)
        .filter(
            Profile.account_owner_id == user.id,
            func.lower(Profile.relationship) == "self",
        )
        .first()
    )
    if existing:
        return

    db.add(
        Profile(
            account_owner_id=user.id,
            full_name=_default_self_profile_name(user.display_name, user.email),
            relationship="self",
        )
    )

def upsert_user(db: Session, firebase_uid: str, email: str | None, display_name: str | None) -> User:
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    if user:
        user.email = email
        user.display_name = display_name
        user.last_login = datetime.utcnow()
    else:
        user = User(firebase_uid=firebase_uid, email=email, display_name=display_name)
        db.add(user)

    db.flush()
    _ensure_self_profile(db, user)
    db.commit()
    db.refresh(user)
    return user


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
    db.commit()
    db.refresh(record)
    return record


def get_user_analyses(db: Session, firebase_uid: str) -> list[ReportAnalysis]:
    return (
        db.query(ReportAnalysis)
        .filter(ReportAnalysis.firebase_uid == firebase_uid)
        .order_by(ReportAnalysis.created_at.desc())
        .all()
    )


def get_analysis_by_id(db: Session, analysis_id: int, firebase_uid: str) -> ReportAnalysis | None:
    return (
        db.query(ReportAnalysis)
        .filter(
            ReportAnalysis.id == analysis_id,
            ReportAnalysis.firebase_uid == firebase_uid,
        )
        .first()
    )


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


def list_profiles_for_owner(db: Session, account_owner_id: int) -> list[Profile]:
    return (
        db.query(Profile)
        .filter(Profile.account_owner_id == account_owner_id)
        .order_by(Profile.created_at.asc())
        .all()
    )


def create_study(
    db: Session,
    profile_id: uuid.UUID,
    name: str,
    description: str | None = None,
) -> Study:
    row = Study(
        profile_id=profile_id,
        name=name,
        description=description,
    )
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


def create_report(
    db: Session,
    study_id: uuid.UUID,
    file_name: str,
    file_url: str,
    report_date: date,
    analysis_data: dict[str, Any],
    lab_name: str | None = None,
) -> Report:
    row = Report(
        study_id=study_id,
        file_name=file_name,
        file_url=file_url,
        report_date=report_date,
        lab_name=lab_name,
        analysis_data=analysis_data,
    )
    db.add(row)

    # Keep study freshness in sync when new reports are added.
    study = db.query(Study).filter(Study.id == study_id).first()
    if study:
        study.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(row)
    return row


def list_reports_for_study(db: Session, study_id: uuid.UUID) -> list[Report]:
    return (
        db.query(Report)
        .filter(Report.study_id == study_id)
        .order_by(Report.report_date.asc(), Report.uploaded_at.asc())
        .all()
    )
