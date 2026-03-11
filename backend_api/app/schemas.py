from typing import Any

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


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    records: list[MedicalRecord]
    question: str
    history: list[ChatTurn] = Field(default_factory=list)


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
