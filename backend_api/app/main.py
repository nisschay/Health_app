from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .auth import RequestUser, get_request_user
from .config import settings
from .schemas import (
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    ExportPdfRequest,
    InsightsRequest,
    InsightsResponse,
    RequestUserModel,
)
from .services import MedicalAnalysisService


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="API layer for the Medical Project migration from Streamlit to a Vercel-friendly frontend.",
)

service = MedicalAnalysisService()

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get(f"{settings.api_prefix}/auth/me", response_model=RequestUserModel)
def get_current_user(user: RequestUser = Depends(get_request_user)) -> RequestUserModel:
    return RequestUserModel(**user.__dict__)


@app.post(f"{settings.api_prefix}/reports/analyze", response_model=AnalysisResponse)
async def analyze_reports(
    pdf_files: list[UploadFile] | None = File(default=None),
    existing_data: UploadFile | None = File(default=None),
    include_raw_texts: bool = Form(default=False),
    user: RequestUser = Depends(get_request_user),
) -> AnalysisResponse:
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalysisResponse(**result)


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
