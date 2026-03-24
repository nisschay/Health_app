import json
import traceback

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .auth import RequestUser, get_request_user
from .config import settings
from .database import (
    get_analysis_by_id,
    get_db,
    get_user_analyses,
    init_db,
    save_analysis,
    upsert_user,
)
from .schemas import (
    AnalysisHistoryItem,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    ExportPdfRequest,
    InsightsRequest,
    InsightsResponse,
    RequestUserModel,
    SaveAnalysisRequest,
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


# ── Analysis ───────────────────────────────────────────────────────────────────

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

