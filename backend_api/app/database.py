"""PostgreSQL database connection and schema setup via SQLAlchemy."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
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

def upsert_user(db: Session, firebase_uid: str, email: str | None, display_name: str | None) -> User:
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    if user:
        user.email = email
        user.display_name = display_name
        user.last_login = datetime.utcnow()
    else:
        user = User(firebase_uid=firebase_uid, email=email, display_name=display_name)
        db.add(user)
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
