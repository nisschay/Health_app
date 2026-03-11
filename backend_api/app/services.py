from __future__ import annotations

import io
from dataclasses import asdict
from datetime import datetime
from typing import Any

import pandas as pd

from Helper_Functions import (
    analyze_medical_report_with_gemini,
    calculate_health_score,
    combine_duplicate_tests,
    consolidate_patient_info,
    create_consolidated_info_with_smart_selection,
    create_structured_dataframe,
    extract_text_from_pdf,
    generate_pdf_health_report,
    get_body_system_analysis,
    get_chatbot_response,
    parse_date_dd_mm_yyyy,
    process_existing_excel_csv,
    smart_consolidate_patient_info,
)
from unify_test_names import unify_test_names

from .auth import RequestUser
from .config import settings


class NamedBytesIO(io.BytesIO):
    def __init__(self, payload: bytes, name: str):
        super().__init__(payload)
        self.name = name


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _serialize_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        cleaned = {key: _clean_value(value) for key, value in row.items()}
        records.append(cleaned)
    return records


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    if "Result" in df.columns:
        df["Result_Numeric"] = pd.to_numeric(df["Result"], errors="coerce")
    if "Test_Date" in df.columns:
        df["Test_Date_dt"] = df["Test_Date"].apply(parse_date_dd_mm_yyyy)
    return df


class MedicalAnalysisService:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.gemini_api_key

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured for the backend API.")
        return self.api_key

    def analyze_reports(
        self,
        pdf_files: list[tuple[str, bytes]],
        existing_data_file: tuple[str, bytes] | None = None,
        include_raw_texts: bool = False,
        user: RequestUser | None = None,
    ) -> dict[str, Any]:
        api_key = self._require_api_key()

        if not pdf_files and not existing_data_file:
            raise ValueError("At least one PDF or an existing data file is required.")

        all_dfs: list[pd.DataFrame] = []
        all_patient_infos_from_pdfs: list[dict[str, Any]] = []
        raw_texts: list[dict[str, str]] = []

        for filename, payload in pdf_files:
            report_text = extract_text_from_pdf(payload)
            if not report_text:
                continue

            if include_raw_texts:
                raw_texts.append(
                    {
                        "name": filename,
                        "text": report_text[:2000],
                    }
                )

            gemini_analysis_json = analyze_medical_report_with_gemini(report_text, api_key)
            if not gemini_analysis_json:
                continue

            df_single, patient_info_single = create_structured_dataframe(
                gemini_analysis_json,
                filename,
            )
            if not df_single.empty:
                all_dfs.append(df_single)
            if patient_info_single:
                all_patient_infos_from_pdfs.append(patient_info_single)

        existing_df = pd.DataFrame()
        existing_patient_info: dict[str, Any] = {}
        if existing_data_file:
            existing_name, existing_payload = existing_data_file
            existing_file_like = NamedBytesIO(existing_payload, existing_name)
            existing_df, existing_patient_info = process_existing_excel_csv(
                existing_file_like,
                all_patient_infos_from_pdfs,
            )

        if not existing_df.empty and all_dfs:
            new_data_df = pd.concat(all_dfs, ignore_index=True)
            combined_raw_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        elif not existing_df.empty:
            combined_raw_df = existing_df
        elif all_dfs:
            combined_raw_df = pd.concat(all_dfs, ignore_index=True)
        else:
            raise ValueError("No medical data could be extracted from the provided files.")

        if existing_patient_info and all_patient_infos_from_pdfs:
            consolidated_info = smart_consolidate_patient_info(
                existing_patient_info,
                all_patient_infos_from_pdfs,
            )
        elif existing_patient_info:
            consolidated_info = existing_patient_info
        elif all_patient_infos_from_pdfs:
            consolidated_info = consolidate_patient_info(all_patient_infos_from_pdfs)
            if consolidated_info.get("error") == "name_mismatch":
                consolidated_info = create_consolidated_info_with_smart_selection(
                    all_patient_infos_from_pdfs
                )
        else:
            consolidated_info = {}

        combined_raw_df = combined_raw_df.dropna(subset=["Test_Name", "Result"], how="all")
        combined_raw_df["Result_Numeric"] = pd.to_numeric(
            combined_raw_df["Result"],
            errors="coerce",
        )
        combined_raw_df["Test_Date_dt"] = combined_raw_df["Test_Date"].apply(
            parse_date_dd_mm_yyyy
        )
        combined_raw_df = combined_raw_df.sort_values(
            by=["Test_Date_dt", "Test_Category", "Test_Name"],
            na_position="last",
        ).reset_index(drop=True)

        combined_raw_df = unify_test_names(combined_raw_df)
        combined_raw_df = combine_duplicate_tests(combined_raw_df)

        health_summary = calculate_health_score(combined_raw_df)
        body_systems = get_body_system_analysis(combined_raw_df)

        return {
            "user": asdict(user) if user else None,
            "patient_info": consolidated_info,
            "total_records": len(combined_raw_df),
            "records": _serialize_dataframe(combined_raw_df),
            "health_summary": health_summary,
            "body_systems": body_systems,
            "raw_texts": raw_texts,
        }

    def get_chat_response(
        self,
        records: list[dict[str, Any]],
        question: str,
        history: list[dict[str, str]],
    ) -> str:
        api_key = self._require_api_key()
        report_df = dataframe_from_records(records)
        return get_chatbot_response(report_df, question, history, api_key)

    def get_health_insights(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        report_df = dataframe_from_records(records)
        return {
            "health_summary": calculate_health_score(report_df),
            "body_systems": get_body_system_analysis(report_df),
        }

    def export_pdf_report(
        self,
        records: list[dict[str, Any]],
        patient_info: dict[str, Any],
    ) -> bytes:
        api_key = self._require_api_key()
        report_df = dataframe_from_records(records)
        return generate_pdf_health_report(report_df, patient_info, api_key)
