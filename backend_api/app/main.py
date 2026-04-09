import json
import asyncio
import threading
import traceback
from datetime import date
from pathlib import Path
from typing import Any
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
    get_user_analyses,
    list_profiles_for_owner,
    list_reports_for_study,
    list_studies_for_profile,
    init_db,
    save_analysis,
    upsert_user,
)
from .normalization import normalize_records
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
    InsightsRequest,
    InsightsResponse,
    PatientInfo,
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
CURRENT_NORMALIZATION_VERSION = 1

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
        raise HTTPException(status_code=401, detail="Authentication required.")

    # Keep user row fresh and ensure default self profile exists.
    return upsert_user(db, user.user_id, user.email, None)


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


def _normalize_filename(value: str | None) -> str:
    if value is None:
        return ""
    return Path(value).name.strip().lower()


def _extract_source_filename(record: dict[str, Any]) -> str | None:
    for key in ("Source_Filename", "source_filename", "sourceFileName", "Source Filename"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _slice_records_for_report(
    rows: list[Any],
    report_file_name: str | None,
    *,
    fallback_to_all: bool,
) -> list[dict[str, Any]]:
    valid_rows = [row for row in rows if isinstance(row, dict)]
    if not valid_rows:
        return []

    target_name = _normalize_filename(report_file_name)
    if not target_name:
        return valid_rows

    matched = [
        row
        for row in valid_rows
        if _normalize_filename(_extract_source_filename(row)) == target_name
    ]
    if matched:
        return matched

    return valid_rows if fallback_to_all else []


def _slice_records_for_report_deterministic(
    rows: list[Any],
    report_file_name: str | None,
) -> list[dict[str, Any]]:
    valid_rows = [row for row in rows if isinstance(row, dict)]
    if not valid_rows:
        return []

    target_name = _normalize_filename(report_file_name)
    if not target_name:
        return valid_rows

    exact_matches = [
        row
        for row in valid_rows
        if _normalize_filename(_extract_source_filename(row)) == target_name
    ]
    if exact_matches:
        return exact_matches

    target_stem = Path(target_name).stem
    if target_stem:
        stem_matches = []
        for row in valid_rows:
            source_name = _normalize_filename(_extract_source_filename(row))
            source_stem = Path(source_name).stem if source_name else ""
            if source_stem and (source_stem == target_stem or source_stem in target_stem or target_stem in source_stem):
                stem_matches.append(row)
        if stem_matches:
            return stem_matches

    # Legacy fallback: never drop report rows on filename mismatch.
    return valid_rows


def _with_source_filename(rows: list[dict[str, Any]], source_file_name: str) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        next_row = dict(row)
        if not _extract_source_filename(next_row):
            next_row["Source_Filename"] = source_file_name
        enriched.append(next_row)
    return enriched


def _infer_report_date_from_records(rows: list[dict[str, Any]]) -> date | None:
    for row in rows:
        raw_date = row.get("Test_Date") or row.get("date")
        if raw_date is None:
            continue
        parsed = _parse_report_date_flexible(str(raw_date))
        if parsed is not None:
            return parsed
    return None


def _dedupe_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        signature = json.dumps(row, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
    return deduped


def _extract_alerts_count(report_row) -> int:
    try:
        payload = report_row.analysis_data or {}
        concerns = payload.get("health_summary", {}).get("concerns", [])
        if isinstance(concerns, list):
            return len(concerns)
    except Exception:
        return 0
    return 0


def _empty_insights() -> dict[str, Any]:
    return {
        "health_summary": {"overall_score": 0, "category_scores": {}, "concerns": []},
        "body_systems": [],
    }


def _normalize_analysis_payload(analysis_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(analysis_payload)
    raw_records = normalized.get("records", [])
    rows = raw_records if isinstance(raw_records, list) else []
    normalized_records = normalize_records([row for row in rows if isinstance(row, dict)])

    normalized["records"] = normalized_records
    normalized["total_records"] = len(normalized_records)

    insights = service.get_health_insights(normalized_records) if normalized_records else _empty_insights()
    normalized["health_summary"] = insights.get("health_summary", _empty_insights()["health_summary"])
    normalized["body_systems"] = insights.get("body_systems", [])

    return normalized


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

    raw_analysis_dict = payload.analysis.model_dump()
    analysis_dict = _normalize_analysis_payload(raw_analysis_dict)
    patient_info = analysis_dict.get("patient_info", {})
    report_date = _parse_report_date_flexible(patient_info.get("date"))
    if report_date is None:
        # Fallback to today when report date cannot be parsed.
        report_date = date.today()

    source_filenames = payload.source_filenames or ["uploaded-report.pdf"]
    urls = payload.source_file_urls or []
    raw_analysis_records = raw_analysis_dict.get("records", []) if isinstance(raw_analysis_dict, dict) else []
    normalized_analysis_records = analysis_dict.get("records", []) if isinstance(analysis_dict, dict) else []
    if isinstance(raw_analysis_records, list):
        analysis_records = raw_analysis_records
    elif isinstance(normalized_analysis_records, list):
        analysis_records = normalized_analysis_records
    else:
        analysis_records = []

    has_record_sources = isinstance(analysis_records, list) and any(
        isinstance(row, dict) and _extract_source_filename(row)
        for row in analysis_records
    )

    added = 0
    for idx, name in enumerate(source_filenames):
        file_url = urls[idx] if idx < len(urls) and urls[idx] else f"uploaded://{name}"

        if isinstance(analysis_records, list) and analysis_records:
            if has_record_sources:
                scoped_records = _slice_records_for_report(
                    analysis_records,
                    name,
                    fallback_to_all=len(source_filenames) == 1,
                )
            else:
                scoped_records = [row for row in analysis_records if isinstance(row, dict)]
        else:
            scoped_records = []

        scoped_records = _with_source_filename([row for row in scoped_records if isinstance(row, dict)], name)
        scoped_normalized_records = normalize_records(scoped_records)

        scoped_analysis = dict(raw_analysis_dict)
        if isinstance(analysis_records, list):
            # Always scope records per report when list payload exists, including empty slices.
            scoped_analysis["records"] = scoped_records
            scoped_analysis["total_records"] = len(scoped_records)

        scoped_report_date = _infer_report_date_from_records(scoped_records) or report_date

        create_report(
            db=db,
            study_id=study_id,
            file_name=name,
            file_url=file_url,
            report_date=scoped_report_date,
            lab_name=patient_info.get("lab_name"),
            analysis_data=scoped_analysis,
            normalized_records=scoped_normalized_records,
            is_normalized=True,
            normalization_version=CURRENT_NORMALIZATION_VERSION,
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


@app.get(
    f"{settings.api_prefix}/studies/dashboard",
    response_model=DashboardSummaryResponse,
)
def studies_dashboard_summary(
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> DashboardSummaryResponse:
    owner = _require_authenticated_user(user, db)
    profile_rows = list_profiles_for_owner(db, owner.id)

    total_reports = 0
    total_alerts = 0
    groups: list[DashboardProfileGroup] = []

    for profile in profile_rows:
        studies = list_studies_for_profile(db, profile.id)
        study_items: list[DashboardStudyItem] = []
        for study in studies:
            reports = list_reports_for_study(db, study.id)
            report_count = len(reports)
            total_reports += report_count

            if reports:
                range_start = reports[0].report_date.isoformat() if reports[0].report_date else None
                range_end = reports[-1].report_date.isoformat() if reports[-1].report_date else None
            else:
                range_start = None
                range_end = None

            lab_values = sorted({(r.lab_name or "").strip() for r in reports if (r.lab_name or "").strip()})
            consistent_lab_name = lab_values[0] if len(lab_values) == 1 else None

            alerts_count = sum(_extract_alerts_count(report) for report in reports)
            total_alerts += alerts_count

            study_items.append(
                DashboardStudyItem(
                    id=study.id,
                    name=study.name,
                    description=study.description,
                    report_count=report_count,
                    range_start=range_start,
                    range_end=range_end,
                    consistent_lab_name=consistent_lab_name,
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


@app.get(
    f"{settings.api_prefix}/studies/{{study_id}}/combined-report",
    response_model=AnalysisResponse,
)
def get_combined_study_report(
    study_id: UUID,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    owner = _require_authenticated_user(user, db)
    study = get_study_by_id(db, study_id)
    if not study:
        raise HTTPException(status_code=404, detail="Study not found.")

    profile = get_profile_by_id(db, study.profile_id)
    if not profile or profile.account_owner_id != owner.id:
        raise HTTPException(status_code=403, detail="You do not have access to this study.")

    reports = list_reports_for_study(db, study_id)
    if not reports:
        raise HTTPException(status_code=404, detail="No reports found for this study.")

    report_file_names = [report.file_name for report in reports]
    combined_records: list[dict[str, Any]] = []
    reports_with_data = 0
    requires_read_path_normalization = False
    latest_payload: dict[str, Any] = {}

    for report in reports:
        payload = report.analysis_data or {}
        if isinstance(payload, dict) and payload:
            latest_payload = payload

        report_rows: list[dict[str, Any]] = []
        if (
            getattr(report, "is_normalized", False)
            and getattr(report, "normalization_version", None) == CURRENT_NORMALIZATION_VERSION
            and isinstance(getattr(report, "normalized_records", None), list)
        ):
            report_rows = [row for row in report.normalized_records if isinstance(row, dict)]
        else:
            raw_rows = payload.get("records", []) if isinstance(payload, dict) else []
            if isinstance(raw_rows, list):
                report_rows = _slice_records_for_report_deterministic(raw_rows, report.file_name)
            if report_rows:
                report_rows = normalize_records(report_rows)
                requires_read_path_normalization = True

        report_rows = _with_source_filename(report_rows, report.file_name)
        if report_rows:
            reports_with_data += 1
        combined_records.extend(report_rows)

    combined_records = _dedupe_records(combined_records)
    if requires_read_path_normalization:
        combined_records = normalize_records(combined_records)

    insights = service.get_health_insights(records=combined_records) if combined_records else _empty_insights()

    payload_patient_info = latest_payload.get("patient_info", {}) if isinstance(latest_payload, dict) else {}
    if not isinstance(payload_patient_info, dict):
        payload_patient_info = {}

    first_report = reports[0] if reports else None
    resolved_patient_info = PatientInfo(
        name=str(payload_patient_info.get("name") or profile.full_name or "N/A"),
        age=str(payload_patient_info.get("age") or "N/A"),
        gender=str(payload_patient_info.get("gender") or "N/A"),
        patient_id=str(payload_patient_info.get("patient_id") or user.user_id or "N/A"),
        date=str(payload_patient_info.get("date") or (reports[-1].report_date.isoformat() if reports and reports[-1].report_date else "N/A")),
        lab_name=str(payload_patient_info.get("lab_name") or (first_report.lab_name if first_report else None) or "N/A"),
    )

    return AnalysisResponse(
        user=RequestUserModel(
            user_id=user.user_id,
            email=user.email,
            authenticated=user.authenticated,
        ),
        patient_info=resolved_patient_info,
        total_records=len(combined_records),
        records=combined_records,
        health_summary=insights.get("health_summary", {"overall_score": 0, "category_scores": {}, "concerns": []}),
        body_systems=insights.get("body_systems", []),
        raw_texts=[],
        combined_report_file_names=report_file_names,
        reports_with_data=reports_with_data,
    )

@app.post(f"{settings.api_prefix}/reports/analyze", response_model=AnalysisResponse)
async def analyze_reports(
    pdf_files: list[UploadFile] | None = File(default=None),
    existing_data: UploadFile | None = File(default=None),
    include_raw_texts: bool = Form(default=False),
    user: RequestUser = Depends(get_request_user),
) -> AnalysisResponse:
    if not user.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required.")

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
        result = _normalize_analysis_payload(result)
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


@app.post(f"{settings.api_prefix}/reports/analyze/stream")
async def analyze_reports_stream(
    pdf_files: list[UploadFile] | None = File(default=None),
    existing_data: UploadFile | None = File(default=None),
    include_raw_texts: bool = Form(default=False),
    user: RequestUser = Depends(get_request_user),
) -> StreamingResponse:
    if not user.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required.")

    pdf_payloads: list[tuple[str, bytes]] = []
    for upload in pdf_files or []:
        pdf_payloads.append((upload.filename or "uploaded.pdf", await upload.read()))

    existing_payload: tuple[str, bytes] | None = None
    if existing_data is not None:
        existing_payload = (
            existing_data.filename or "medical-data.xlsx",
            await existing_data.read(),
        )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def emit(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def worker() -> None:
        try:
            emit({"type": "stage", "step": "validating", "status": "active"})
            if not pdf_payloads and not existing_payload:
                raise ValueError("At least one PDF or an existing data file is required.")
            emit({"type": "stage", "step": "validating", "status": "complete"})

            emit({"type": "stage", "step": "uploading", "status": "active"})
            emit({"type": "stage", "step": "uploading", "status": "complete"})

            emit({"type": "stage", "step": "processing", "status": "active"})
            result = service.analyze_reports(
                pdf_files=pdf_payloads,
                existing_data_file=existing_payload,
                include_raw_texts=include_raw_texts,
                user=user,
                progress_callback=emit,
            )
            result = _normalize_analysis_payload(result)
            emit({"type": "stage", "step": "processing", "status": "complete"})

            emit({"type": "stage", "step": "saving", "status": "active"})
            emit({"type": "stage", "step": "saving", "status": "complete"})
            emit({"type": "done", "result": result})
        except RuntimeError as exc:
            detail = str(exc)
            if detail.startswith("RATE_LIMIT_EXCEEDED:"):
                emit({"type": "error", "status": 429, "message": detail})
                return
            emit({"type": "error", "status": 500, "message": detail})
        except ValueError as exc:
            emit({"type": "error", "status": 400, "message": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            emit({"type": "error", "status": 500, "message": f"Analysis failed: {str(exc)}"})

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0)
            if event.get("type") in {"done", "error"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(f"{settings.api_prefix}/reports/save", response_model=AnalysisHistoryItem)
def save_report_analysis(
    payload: SaveAnalysisRequest,
    user: RequestUser = Depends(get_request_user),
    db: Session = Depends(get_db),
) -> AnalysisHistoryItem:
    """Save an analysis result to PostgreSQL for the authenticated user."""
    owner = _require_authenticated_user(user, db)

    analysis_dict = _normalize_analysis_payload(payload.analysis.model_dump())
    row = save_analysis(
        db=db,
        firebase_uid=owner.firebase_uid,
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
    owner = _require_authenticated_user(user, db)

    rows = get_user_analyses(db, owner.firebase_uid)
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
    owner = _require_authenticated_user(user, db)

    row = get_analysis_by_id(db, analysis_id, owner.firebase_uid)
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    data = _normalize_analysis_payload(json.loads(row.analysis_json))
    return AnalysisResponse(**data)


# ── Chat / Insights / Export ───────────────────────────────────────────────────

@app.post(f"{settings.api_prefix}/reports/chat", response_model=ChatResponse)
def chat_about_report(
    payload: ChatRequest,
    user: RequestUser = Depends(get_request_user),
) -> ChatResponse:
    if not user.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        answer = service.get_chat_response(
            records=[record.model_dump() for record in payload.records],
            question=payload.question,
            history=[item.model_dump() for item in payload.history],
            analysis_id=payload.analysis_id,
            session_id=payload.session_id,
            system_prompt=payload.system_prompt,
            report_context=payload.report_context,
        )
        return ChatResponse(answer=answer)
    except Exception as exc:
        print(f"[CHAT] Failed: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(exc)}") from exc


@app.post(f"{settings.api_prefix}/reports/insights", response_model=InsightsResponse)
def get_report_insights(
    payload: InsightsRequest,
    user: RequestUser = Depends(get_request_user),
) -> InsightsResponse:
    if not user.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required.")
    result = service.get_health_insights(
        records=[record.model_dump() for record in payload.records]
    )
    return InsightsResponse(**result)


@app.post(f"{settings.api_prefix}/reports/export/pdf")
def export_pdf(
    payload: ExportPdfRequest,
    user: RequestUser = Depends(get_request_user),
) -> StreamingResponse:
    if not user.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required.")
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
    user: RequestUser = Depends(get_request_user),
) -> StreamingResponse:
    if not user.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required.")
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

