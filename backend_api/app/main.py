import json
import logging
from datetime import date
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from . import jobs
from .auth import RequestUser, get_request_user
from .config import settings
from .database import (
    ReportFinding,
    create_profile,
    create_study,
    dashboard_alert_counts,
    get_analysis_by_id,
    get_db,
    get_job,
    get_profile_by_id,
    get_study_by_id,
    get_user_analyses,
    list_dashboard_report_rows,
    list_findings_for_reports,
    list_profiles_for_owner,
    list_reports_for_study,
    list_studies_for_owner,
    list_studies_for_profile,
    list_trend_points,
    ping_database,
    study_report_stats,
    upsert_user,
)
from .findings import record_from_finding
from .migrations import run_migrations
from .saving import EMPTY_INSIGHTS, findings_for_report, normalize_analysis_payload
from .schemas import (
    AnalysisHistoryItem,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    CreateProfileRequest,
    CreateStudyRequest,
    DashboardProfileGroup,
    DashboardStudyItem,
    DashboardSummaryResponse,
    ExportPdfRequest,
    JobResponse,
    PatientInfo,
    ProfileResponse,
    RequestUserModel,
    StudySummaryResponse,
    TrendPoint,
    UserProfile,
)
from .security import (
    BodySizeLimitMiddleware,
    internal_error,
    rate_limiter,
    read_existing_data_upload,
    read_pdf_uploads,
)
from .services import MedicalAnalysisService

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="API layer for the Medical Project – auth, analysis, history.",
)

service = MedicalAnalysisService()


@app.on_event("startup")
def startup_event() -> None:
    run_migrations()
    jobs.mark_interrupted()


app.add_middleware(BodySizeLimitMiddleware)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Report database reachability too: the wake-up overlay polls this."""
    try:
        ping_database()
        database_status = "ok"
    except Exception:
        logger.exception("Health check could not reach the database")
        database_status = "unreachable"
    return {"status": "ok", "database": database_status}


# ── Auth ───────────────────────────────────────────────────────────────────────

@app.get(f"{settings.api_prefix}/auth/me", response_model=RequestUserModel)
def get_current_user(user: RequestUser = Depends(get_request_user)) -> RequestUserModel:
    return RequestUserModel(**user.__dict__)


@app.post(f"{settings.api_prefix}/auth/sync", response_model=UserProfile)
def sync_user(
    user: RequestUser = Depends(get_request_user),
    display_name: str | None = None,
    db: Session = Depends(get_db),
) -> UserProfile:
    """Called after Firebase sign-in to upsert the user in PostgreSQL."""
    row = upsert_user(db, user.user_id, user.email, display_name)
    return UserProfile(firebase_uid=row.firebase_uid, email=row.email, display_name=row.display_name)


def _current_account_owner(user: RequestUser, db: Session):
    return upsert_user(db, user.user_id, user.email, None)


def _owned_profile(db: Session, owner, profile_id: UUID):
    profile = get_profile_by_id(db, profile_id)
    if not profile or profile.account_owner_id != owner.id:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return profile


def _owned_study(db: Session, owner, study_id: UUID):
    study = get_study_by_id(db, study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found.")
    profile = get_profile_by_id(db, study.profile_id)
    if not profile or profile.account_owner_id != owner.id:
        raise HTTPException(status_code=403, detail="You do not have access to this study.")
    return study, profile


def _date_to_iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _study_summary(study, stats: tuple[int, date | None, date | None]) -> StudySummaryResponse:
    report_count, range_start, range_end = stats
    return StudySummaryResponse(
        id=study.id,
        profile_id=study.profile_id,
        name=study.name,
        description=study.description,
        report_count=report_count,
        range_start=_date_to_iso(range_start),
        range_end=_date_to_iso(range_end),
        last_updated=study.updated_at.isoformat(),
        created_at=study.created_at.isoformat(),
    )


def _parse_iso_date(value: str | None, field_name: str) -> date | None:
    if value is None or value.strip() == "":
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}. Use YYYY-MM-DD.") from exc


def _dedupe_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        signature = json.dumps(row, sort_keys=True, default=str)
        if signature not in seen:
            seen.add(signature)
            deduped.append(row)
    return deduped


def _profile_response(row) -> ProfileResponse:
    return ProfileResponse(
        id=row.id,
        account_owner_id=row.account_owner_id,
        full_name=row.full_name,
        relationship=row.relationship,
        date_of_birth=_date_to_iso(row.date_of_birth),
        created_at=row.created_at.isoformat(),
    )


# ── Profiles and studies ───────────────────────────────────────────────────────

@app.get(f"{settings.api_prefix}/studies/profiles", response_model=list[ProfileResponse])
def list_profiles(
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> list[ProfileResponse]:
    owner = _current_account_owner(user, db)
    return [_profile_response(row) for row in list_profiles_for_owner(db, owner.id)]


@app.post(f"{settings.api_prefix}/studies/profiles", response_model=ProfileResponse)
def create_profile_endpoint(
    payload: CreateProfileRequest,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> ProfileResponse:
    owner = _current_account_owner(user, db)
    row = create_profile(
        db=db,
        account_owner_id=owner.id,
        full_name=payload.full_name.strip(),
        relationship=payload.relationship.strip(),
        date_of_birth=_parse_iso_date(payload.date_of_birth, "date_of_birth"),
    )
    return _profile_response(row)


@app.get(
    f"{settings.api_prefix}/studies/profiles/{{profile_id}}/studies",
    response_model=list[StudySummaryResponse],
)
def list_profile_studies(
    profile_id: UUID,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> list[StudySummaryResponse]:
    owner = _current_account_owner(user, db)
    _owned_profile(db, owner, profile_id)
    rows = list_studies_for_profile(db, profile_id)
    stats = study_report_stats(db, [row.id for row in rows])
    return [_study_summary(row, stats.get(row.id, (0, None, None))) for row in rows]


@app.get(f"{settings.api_prefix}/studies/profiles/{{profile_id}}/trends", response_model=list[TrendPoint])
def profile_trend(
    profile_id: UUID,
    test: str = Query(min_length=1),
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> list[TrendPoint]:
    """Every reading of one canonical test for one person: a single indexed query."""
    owner = _current_account_owner(user, db)
    _owned_profile(db, owner, profile_id)
    return [
        TrendPoint(
            report_id=f.report_id,
            test_date=_date_to_iso(f.test_date),
            result_text=f.result_text,
            value_numeric=f.value_numeric,
            comparator=f.comparator,
            unit=f.unit,
            reference_range=f.reference_range,
            ref_low=f.ref_low,
            ref_high=f.ref_high,
            status=f.status,
            source_filename=f.source_filename,
        )
        for f in list_trend_points(db, profile_id, test)
    ]


@app.post(f"{settings.api_prefix}/studies", response_model=StudySummaryResponse)
def create_study_endpoint(
    payload: CreateStudyRequest,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> StudySummaryResponse:
    owner = _current_account_owner(user, db)
    _owned_profile(db, owner, payload.profile_id)
    try:
        row = create_study(
            db=db,
            profile_id=payload.profile_id,
            name=payload.name.strip(),
            description=(payload.description or "").strip() or None,
        )
    except Exception as exc:
        db.rollback()
        if "uq_studies_profile_name" in str(exc):
            raise HTTPException(status_code=409, detail="A study with this name already exists for this profile.") from exc
        raise
    return _study_summary(row, study_report_stats(db, [row.id]).get(row.id, (0, None, None)))


@app.get(f"{settings.api_prefix}/studies/dashboard", response_model=DashboardSummaryResponse)
def studies_dashboard_summary(
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> DashboardSummaryResponse:
    """Four queries regardless of how many studies or reports the owner has."""
    owner = _current_account_owner(user, db)
    profile_rows = list_profiles_for_owner(db, owner.id)
    studies_by_profile: dict[UUID, list] = {}
    for study in list_studies_for_owner(db, owner.id):
        studies_by_profile.setdefault(study.profile_id, []).append(study)

    reports_by_study: dict[UUID, list] = {}
    for row in list_dashboard_report_rows(db, owner.id):
        reports_by_study.setdefault(row.study_id, []).append(row)
    alerts_by_study = dashboard_alert_counts(db, owner.id)

    total_reports = 0
    total_alerts = 0
    groups: list[DashboardProfileGroup] = []
    for profile in profile_rows:
        study_items: list[DashboardStudyItem] = []
        for study in studies_by_profile.get(profile.id, []):
            reports = reports_by_study.get(study.id, [])
            total_reports += len(reports)
            alerts_count = alerts_by_study.get(study.id, 0)
            total_alerts += alerts_count
            lab_values = {(r.lab_name or "").strip() for r in reports if (r.lab_name or "").strip()}
            study_items.append(
                DashboardStudyItem(
                    id=study.id,
                    name=study.name,
                    description=study.description,
                    report_count=len(reports),
                    range_start=_date_to_iso(reports[0].report_date) if reports else None,
                    range_end=_date_to_iso(reports[-1].report_date) if reports else None,
                    consistent_lab_name=next(iter(lab_values)) if len(lab_values) == 1 else None,
                    has_alerts=alerts_count > 0,
                    alerts_count=alerts_count,
                    last_updated=study.updated_at.isoformat(),
                )
            )
        groups.append(
            DashboardProfileGroup(
                profile_id=profile.id,
                full_name=profile.full_name,
                relationship=profile.relationship,
                studies=study_items,
            )
        )

    return DashboardSummaryResponse(
        total_reports=total_reports,
        total_alerts=total_alerts,
        profiles_tracked=len(profile_rows),
        profiles=groups,
    )


@app.get(f"{settings.api_prefix}/studies/{{study_id}}/combined-report", response_model=AnalysisResponse)
def get_combined_study_report(
    study_id: UUID,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    owner = _current_account_owner(user, db)
    study, profile = _owned_study(db, owner, study_id)
    reports = list_reports_for_study(db, study_id)
    if not reports:
        raise HTTPException(status_code=404, detail="No reports found for this study.")

    findings: list[ReportFinding] = list_findings_for_reports(db, [report.id for report in reports])
    covered = {finding.report_id for finding in findings}
    for report in reports:
        if report.id not in covered:
            findings.extend(findings_for_report(db, report, profile.id))
    if len(covered) < len(reports):
        db.commit()

    records = _dedupe_records([record_from_finding(finding) for finding in findings])
    insights = service.get_health_insights(records) if records else EMPTY_INSIGHTS
    latest = next((r.analysis_data for r in reversed(reports) if isinstance(r.analysis_data, dict) and r.analysis_data), {})
    info = latest.get("patient_info") if isinstance(latest.get("patient_info"), dict) else {}

    return AnalysisResponse(
        user=RequestUserModel(user_id=user.user_id, email=user.email),
        patient_info=PatientInfo(
            name=str(info.get("name") or profile.full_name),
            age=str(info.get("age") or "N/A"),
            gender=str(info.get("gender") or "N/A"),
            patient_id=str(info.get("patient_id") or user.user_id),
            date=str(info.get("date") or reports[-1].report_date.isoformat()),
            lab_name=str(info.get("lab_name") or reports[0].lab_name or "N/A"),
        ),
        total_records=len(records),
        records=records,
        health_summary=insights["health_summary"],
        body_systems=insights["body_systems"],
        raw_texts=[],
        combined_report_file_names=[report.file_name for report in reports],
        reports_with_data=len({finding.report_id for finding in findings}),
    )


# ── Jobs ───────────────────────────────────────────────────────────────────────

def _job_response(job) -> JobResponse:
    return JobResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        study_id=job.study_id,
        analysis_id=job.analysis_id,
        source_filenames=list(job.source_filenames or []),
        created_at=job.created_at.isoformat(),
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
    )


@app.post(f"{settings.api_prefix}/reports/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    pdf_files: list[UploadFile] | None = File(default=None),
    existing_data: UploadFile | None = File(default=None),
    include_raw_texts: bool = Form(default=False),
    study_id: UUID | None = Form(default=None),
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> JobResponse:
    """Accept the upload and return at once; the browser polls the job until it is done."""
    rate_limiter.check(user.user_id)
    pdf_payloads = await read_pdf_uploads(pdf_files)
    existing_payload = await read_existing_data_upload(existing_data)
    if not pdf_payloads and not existing_payload:
        raise HTTPException(status_code=400, detail="At least one PDF or an existing data file is required.")
    owner = _current_account_owner(user, db)
    if study_id is not None:
        _owned_study(db, owner, study_id)
    logger.info("Report job accepted: files=%d study=%s", len(pdf_payloads), study_id)
    return _job_response(jobs.submit(db, owner, service, pdf_payloads, existing_payload, include_raw_texts, study_id))


@app.get(f"{settings.api_prefix}/reports/jobs/{{job_id}}", response_model=JobResponse)
def read_job(
    job_id: UUID,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> JobResponse:
    owner = _current_account_owner(user, db)
    job = get_job(db, job_id, owner.id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(job)


# ── History ────────────────────────────────────────────────────────────────────

@app.get(f"{settings.api_prefix}/reports/history", response_model=list[AnalysisHistoryItem])
def list_report_history(
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[AnalysisHistoryItem]:
    """Return a page of past analyses for the authenticated user, newest first."""
    owner = _current_account_owner(user, db)
    return [
        AnalysisHistoryItem(
            id=row.id,
            patient_name=row.patient_name,
            patient_age=row.patient_age,
            patient_gender=row.patient_gender,
            lab_name=row.lab_name,
            report_date=row.report_date,
            total_records=row.total_records,
            source_filenames=row.source_filenames.split(",") if row.source_filenames else [],
            created_at=row.created_at.isoformat(),
        )
        for row in get_user_analyses(db, owner.firebase_uid, limit=limit, offset=offset)
    ]


@app.get(f"{settings.api_prefix}/reports/history/{{analysis_id}}", response_model=AnalysisResponse)
def get_report_by_id(
    analysis_id: int,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Return the full AnalysisResponse for a previously saved report."""
    owner = _current_account_owner(user, db)
    row = get_analysis_by_id(db, analysis_id, owner.firebase_uid)
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    data = json.loads(row.analysis_json)
    if "health_summary" not in data:
        data = normalize_analysis_payload(data, service)  # rows saved before insights were stored
    return AnalysisResponse(**data)


# ── Chat / Export ──────────────────────────────────────────────────────────────

@app.post(f"{settings.api_prefix}/reports/chat", response_model=ChatResponse)
def chat_about_report(
    payload: ChatRequest,
    user: RequestUser = Depends(get_request_user),
) -> ChatResponse:
    rate_limiter.check(user.user_id)
    try:
        answer = service.get_chat_response(
            records=[record.model_dump() for record in payload.records],
            question=payload.question,
            history=[item.model_dump() for item in payload.history],
            analysis_id=payload.analysis_id,
            session_id=payload.session_id,
            guidelines=payload.guidelines,
            report_context=payload.report_context,
        )
        return ChatResponse(answer=answer)
    except Exception as exc:
        raise internal_error(exc, "Chat") from exc


@app.post(f"{settings.api_prefix}/reports/export/pdf")
def export_pdf(
    payload: ExportPdfRequest,
    user: RequestUser = Depends(get_request_user),
) -> StreamingResponse:
    pdf_bytes = service.export_pdf_report(
        records=[record.model_dump() for record in payload.records],
        patient_info=payload.patient_info.model_dump(),
    )
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="medical-health-report.pdf"'},
    )


@app.post(f"{settings.api_prefix}/reports/export/excel")
def export_excel(
    payload: ExportPdfRequest,
    user: RequestUser = Depends(get_request_user),
) -> StreamingResponse:
    excel_bytes = service.export_excel_report(
        records=[record.model_dump() for record in payload.records],
        patient_info=payload.patient_info.model_dump(),
    )
    return StreamingResponse(
        iter([excel_bytes]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="medical-health-report.xlsx"'},
    )
