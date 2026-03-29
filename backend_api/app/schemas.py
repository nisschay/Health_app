from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    name: str = "N/A"
    age: str = "N/A"
    gender: str = "N/A"
    patient_id: str = "N/A"
    date: str = "N/A"
    lab_name: str = "N/A"


class MedicalRecord(BaseModel):
    Source_Filename: str | None = None
    Patient_ID: str | None = None
    Patient_Name: str | None = None
    Age: str | None = None
    Gender: str | None = None
    Test_Date: str | None = None
    Lab_Name: str | None = None
    Test_Category: str | None = None
    Original_Test_Name: str | None = None
    Test_Name: str | None = None
    Result: Any = None
    Unit: str | None = None
    Reference_Range: str | None = None
    Status: str | None = None
    Processed_Date: str | None = None
    Result_Numeric: float | None = None
    Test_Date_dt: str | None = None


class RawTextPreview(BaseModel):
    name: str
    text: str


class RequestUserModel(BaseModel):
    user_id: str
    email: str | None = None
    authenticated: bool


class AnalysisResponse(BaseModel):
    user: RequestUserModel
    patient_info: PatientInfo
    total_records: int
    records: list[MedicalRecord]
    health_summary: dict[str, Any]
    body_systems: list[dict[str, Any]]
    raw_texts: list[RawTextPreview] = Field(default_factory=list)
    combined_report_file_names: list[str] = Field(default_factory=list)
    reports_with_data: int | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    sessionId: str | None = None
    reportContext: list[MedicalRecord] | dict[str, Any] | list[Any] | None = None
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str


class InsightsRequest(BaseModel):
    records: list[MedicalRecord]


class InsightsResponse(BaseModel):
    health_summary: dict[str, Any]
    body_systems: list[dict[str, Any]]


class ExportPdfRequest(BaseModel):
    patient_info: PatientInfo
    records: list[MedicalRecord]


# ── History / persistence schemas ─────────────────────────────────────────────

class AnalysisHistoryItem(BaseModel):
    id: int
    patient_name: str | None
    patient_age: str | None
    patient_gender: str | None
    lab_name: str | None
    report_date: str | None
    total_records: int
    source_filenames: list[str]
    created_at: str  # ISO datetime


class UserProfile(BaseModel):
    firebase_uid: str
    email: str | None
    display_name: str | None


class SaveAnalysisRequest(BaseModel):
    analysis: AnalysisResponse
    source_filenames: list[str] = Field(default_factory=list)


class MergeAnalysisRequest(BaseModel):
    """Merge new PDFs into an existing saved analysis and save the result."""
    existing_analysis_id: int
    new_analysis: AnalysisResponse


# ── Study management schemas ──────────────────────────────────────────────────

class ProfileResponse(BaseModel):
    id: UUID
    account_owner_id: int
    full_name: str
    relationship: str
    date_of_birth: str | None
    created_at: str


class CreateProfileRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=256)
    relationship: str = Field(min_length=1, max_length=64)
    date_of_birth: str | None = None


class StudySummaryResponse(BaseModel):
    id: UUID
    profile_id: UUID
    name: str
    description: str | None
    report_count: int
    range_start: str | None
    range_end: str | None
    last_updated: str
    created_at: str


class CreateStudyRequest(BaseModel):
    profile_id: UUID
    name: str = Field(min_length=1, max_length=60)
    description: str | None = Field(default=None, max_length=200)


class SaveStudyAnalysisRequest(BaseModel):
    analysis: AnalysisResponse
    source_filenames: list[str] = Field(default_factory=list)
    source_file_urls: list[str] = Field(default_factory=list)


class SaveStudyAnalysisResponse(BaseModel):
    study_id: UUID
    added_reports: int
    total_reports: int
    study_name: str


class DashboardStudyItem(BaseModel):
    id: UUID
    name: str
    description: str | None
    report_count: int
    range_start: str | None
    range_end: str | None
    consistent_lab_name: str | None
    has_alerts: bool
    alerts_count: int
    last_updated: str


class DashboardProfileGroup(BaseModel):
    profile_id: UUID
    full_name: str
    relationship: str
    studies: list[DashboardStudyItem]


class DashboardSummaryResponse(BaseModel):
    total_reports: int
    total_alerts: int
    profiles_tracked: int
    profiles: list[DashboardProfileGroup]
