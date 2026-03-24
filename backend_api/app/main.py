import json
import traceback
from datetime import date
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .auth import RequestUser, get_request_user
from .config import settings
from .database import (
    count_reports_for_study,
    create_profile,
    create_report,
    create_study,
    get_analysis_by_id,
    get_profile_by_id,
    get_study_by_id,
    get_study_report_date_range,
    get_db,
    get_user_by_firebase_uid,
    get_user_analyses,
    list_profiles_for_owner,
    list_reports_for_study,
    list_studies_for_profile,
    init_db,
    save_analysis,
    upsert_user,
)
from .schemas import (
    AnalysisHistoryItem,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    CreateProfileRequest,
    CreateStudyRequest,
    ExportPdfRequest,
    InsightsRequest,
    InsightsResponse,
    ProfileResponse,
    RequestUserModel,
    SaveAnalysisRequest,
    SaveStudyAnalysisRequest,
    SaveStudyAnalysisResponse,
    StudySummaryResponse,
    UserProfile,
)
from .services import MedicalAnalysisService


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="API layer for the Medical Project – auth, analysis, history.",
)

service = MedicalAnalysisService()

# Initialise DB tables on startup
@app.on_event("startup")
def startup_event() -> None:
    try:
        init_db()
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] DB init failed: {exc}")


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
    return {"status": "ok"}


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
    if not user.authenticated:
        if settings.require_auth:
            raise HTTPException(status_code=401, detail="Authentication required.")
        return UserProfile(
            firebase_uid="anonymous",
            email=None,
            display_name=display_name,
        )
    row = upsert_user(db, user.user_id, user.email, display_name)
    return UserProfile(
        firebase_uid=row.firebase_uid,
        email=row.email,
        display_name=row.display_name,
    )


def _require_authenticated_user(user: RequestUser, db: Session):
    if not user.authenticated:
        if settings.require_auth:
            raise HTTPException(status_code=401, detail="Authentication required.")
        raise HTTPException(status_code=401, detail="Sign in required.")

    owner = get_user_by_firebase_uid(db, user.user_id)
    if not owner:
        # Auto-heal if sync wasn't called for some reason.
        owner = upsert_user(db, user.user_id, user.email, None)
    return owner


def _date_to_iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _study_summary(db: Session, study) -> StudySummaryResponse:
    report_count = count_reports_for_study(db, study.id)
    range_start, range_end = get_study_report_date_range(db, study.id)
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


def _parse_report_date_flexible(value: str | None) -> date | None:
    if value is None or value.strip() == "":
        return None
    raw = value.strip()
    try:
        return date.fromisoformat(raw)
    except ValueError:
        pass

    for sep in ("-", "/"):
        parts = raw.split(sep)
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            day, month, year = parts
            if len(year) == 2:
                year = f"20{year}"
            try:
                return date(int(year), int(month), int(day))
            except ValueError:
                return None
    return None


# ── Analysis ───────────────────────────────────────────────────────────────────

@app.get(f"{settings.api_prefix}/studies/profiles", response_model=list[ProfileResponse])
def list_profiles(
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> list[ProfileResponse]:
    owner = _require_authenticated_user(user, db)
    rows = list_profiles_for_owner(db, owner.id)
    return [
        ProfileResponse(
            id=row.id,
            account_owner_id=row.account_owner_id,
            full_name=row.full_name,
            relationship=row.relationship,
            date_of_birth=_date_to_iso(row.date_of_birth),
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]


@app.post(f"{settings.api_prefix}/studies/profiles", response_model=ProfileResponse)
def create_profile_endpoint(
    payload: CreateProfileRequest,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> ProfileResponse:
    owner = _require_authenticated_user(user, db)
    row = create_profile(
        db=db,
        account_owner_id=owner.id,
        full_name=payload.full_name.strip(),
        relationship=payload.relationship.strip(),
        date_of_birth=_parse_iso_date(payload.date_of_birth, "date_of_birth"),
    )
    return ProfileResponse(
        id=row.id,
        account_owner_id=row.account_owner_id,
        full_name=row.full_name,
        relationship=row.relationship,
        date_of_birth=_date_to_iso(row.date_of_birth),
        created_at=row.created_at.isoformat(),
    )


@app.get(
    f"{settings.api_prefix}/studies/profiles/{{profile_id}}/studies",
    response_model=list[StudySummaryResponse],
)
def list_profile_studies(
    profile_id: UUID,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> list[StudySummaryResponse]:
    owner = _require_authenticated_user(user, db)
    profile = get_profile_by_id(db, profile_id)
    if not profile or profile.account_owner_id != owner.id:
        raise HTTPException(status_code=404, detail="Profile not found.")

    rows = list_studies_for_profile(db, profile_id)
    return [_study_summary(db, row) for row in rows]


@app.post(f"{settings.api_prefix}/studies", response_model=StudySummaryResponse)
def create_study_endpoint(
    payload: CreateStudyRequest,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> StudySummaryResponse:
    owner = _require_authenticated_user(user, db)
    profile = get_profile_by_id(db, payload.profile_id)
    if not profile or profile.account_owner_id != owner.id:
        raise HTTPException(status_code=404, detail="Profile not found.")

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

    return _study_summary(db, row)


@app.post(
    f"{settings.api_prefix}/studies/{{study_id}}/reports/save-analysis",
    response_model=SaveStudyAnalysisResponse,
)
def save_analysis_to_study(
    study_id: UUID,
    payload: SaveStudyAnalysisRequest,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> SaveStudyAnalysisResponse:
    owner = _require_authenticated_user(user, db)
    study = get_study_by_id(db, study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found.")

    profile = get_profile_by_id(db, study.profile_id)
    if not profile or profile.account_owner_id != owner.id:
        raise HTTPException(status_code=403, detail="You do not have access to this study.")

    analysis_dict = payload.analysis.model_dump()
    patient_info = analysis_dict.get("patient_info", {})
    report_date = _parse_report_date_flexible(patient_info.get("date"))
    if report_date is None:
        # Fallback to today when report date cannot be parsed.
        report_date = date.today()

    source_filenames = payload.source_filenames or ["uploaded-report.pdf"]
    urls = payload.source_file_urls or []
    added = 0
    for idx, name in enumerate(source_filenames):
        file_url = urls[idx] if idx < len(urls) and urls[idx] else f"uploaded://{name}"
        create_report(
            db=db,
            study_id=study_id,
            file_name=name,
            file_url=file_url,
            report_date=report_date,
            lab_name=patient_info.get("lab_name"),
            analysis_data=analysis_dict,
        )
        added += 1

    total_reports = len(list_reports_for_study(db, study_id))
    refreshed = get_study_by_id(db, study_id)
    return SaveStudyAnalysisResponse(
        study_id=study_id,
        added_reports=added,
        total_reports=total_reports,
        study_name=refreshed.name if refreshed else study.name,
    )

@app.post(f"{settings.api_prefix}/reports/analyze", response_model=AnalysisResponse)
async def analyze_reports(
    pdf_files: list[UploadFile] | None = File(default=None),
    existing_data: UploadFile | None = File(default=None),
    include_raw_texts: bool = Form(default=False),
    user: RequestUser = Depends(get_request_user),
) -> AnalysisResponse:
    print(
        f"[ANALYZE] Request received: pdf_count={len(pdf_files or [])}, "
        f"has_existing_data={existing_data is not None}, user={user.user_id}"
    )

    pdf_payloads: list[tuple[str, bytes]] = []
    for upload in pdf_files or []:
        pdf_payloads.append((upload.filename or "uploaded.pdf", await upload.read()))

    existing_payload: tuple[str, bytes] | None = None
    if existing_data is not None:
        existing_payload = (
            existing_data.filename or "medical-data.xlsx",
            await existing_data.read(),
        )

    try:
        result = service.analyze_reports(
            pdf_files=pdf_payloads,
            existing_data_file=existing_payload,
            include_raw_texts=include_raw_texts,
            user=user,
        )
    except RuntimeError as exc:
        detail = str(exc)
        if detail.startswith("RATE_LIMIT_EXCEEDED:"):
            raise HTTPException(status_code=429, detail=detail) from exc
        raise HTTPException(status_code=500, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[ANALYZE] Failed with unexpected error: {exc}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(exc)}",
        ) from exc

    print(f"[ANALYZE] Completed successfully: total_records={result.get('total_records', 0)}")

    return AnalysisResponse(**result)


@app.post(f"{settings.api_prefix}/reports/save", response_model=AnalysisHistoryItem)
def save_report_analysis(
    payload: SaveAnalysisRequest,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> AnalysisHistoryItem:
    """Save an analysis result to PostgreSQL for the authenticated user."""
    if not user.authenticated:
        if settings.require_auth:
            raise HTTPException(status_code=401, detail="Authentication required to save reports.")
        user = RequestUser(user_id="anonymous", email=None, authenticated=False)

    analysis_dict = payload.analysis.model_dump()
    row = save_analysis(
        db=db,
        firebase_uid=user.user_id,
        patient_info=analysis_dict.get("patient_info", {}),
        analysis_data=analysis_dict,
        source_filenames=payload.source_filenames,
    )
    return AnalysisHistoryItem(
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


# ── History ────────────────────────────────────────────────────────────────────

@app.get(f"{settings.api_prefix}/reports/history", response_model=list[AnalysisHistoryItem])
def list_report_history(
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> list[AnalysisHistoryItem]:
    """Return all past analyses for the authenticated user."""
    if not user.authenticated:
        if settings.require_auth:
            return []
        user = RequestUser(user_id="anonymous", email=None, authenticated=False)

    rows = get_user_analyses(db, user.user_id)
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
        for row in rows
    ]


@app.get(
    f"{settings.api_prefix}/reports/history/{{analysis_id}}",
    response_model=AnalysisResponse,
)
def get_report_by_id(
    analysis_id: int,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Return the full AnalysisResponse for a previously saved report."""
    if not user.authenticated:
        if settings.require_auth:
            raise HTTPException(status_code=401, detail="Authentication required.")
        user = RequestUser(user_id="anonymous", email=None, authenticated=False)

    row = get_analysis_by_id(db, analysis_id, user.user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    data = json.loads(row.analysis_json)
    return AnalysisResponse(**data)


# ── Chat / Insights / Export ───────────────────────────────────────────────────

@app.post(f"{settings.api_prefix}/reports/chat", response_model=ChatResponse)
def chat_about_report(
    payload: ChatRequest,
    _user: RequestUser = Depends(get_request_user),
) -> ChatResponse:
    answer = service.get_chat_response(
        records=[record.model_dump() for record in payload.records],
        question=payload.question,
        history=[item.model_dump() for item in payload.history],
    )
    return ChatResponse(answer=answer)


@app.post(f"{settings.api_prefix}/reports/insights", response_model=InsightsResponse)
def get_report_insights(
    payload: InsightsRequest,
    _user: RequestUser = Depends(get_request_user),
) -> InsightsResponse:
    result = service.get_health_insights(
        records=[record.model_dump() for record in payload.records]
    )
    return InsightsResponse(**result)


@app.post(f"{settings.api_prefix}/reports/export/pdf")
def export_pdf(
    payload: ExportPdfRequest,
    _user: RequestUser = Depends(get_request_user),
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
async def export_excel(
    payload: ExportPdfRequest,
    _user: RequestUser = Depends(get_request_user),
) -> StreamingResponse:
    from .services import dataframe_from_records
    excel_bytes = service.export_excel_report(
        records=[record.model_dump() for record in payload.records],
        patient_info=payload.patient_info.model_dump(),
    )
    return StreamingResponse(
        iter([excel_bytes]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="medical-health-report.xlsx"'},
    )

