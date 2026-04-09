from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from Helper_Functions import (
    analyze_medical_report_with_gemini,
    calculate_health_score,
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
    get_last_extraction_error,
)

from .auth import RequestUser
from .config import settings
from .normalization import normalize_dataframe, normalize_records


def _is_rate_limit_reason(reason: str) -> bool:
    text = (reason or "").lower()
    return "rate limit" in text or "quota" in text or " 429" in text or ":429" in text


class NamedBytesIO(io.BytesIO):
    def __init__(self, payload: bytes, name: str):
        super().__init__(payload)
        self.name = name


def _clean_value(value: Any) -> Any:
    if value is None:
        return None

    # pd.isna can return an array for array-like inputs; avoid ambiguous truth checks.
    try:
        na_result = pd.isna(value)
    except Exception:
        na_result = False

    if isinstance(na_result, bool):
        if na_result:
            return None
    elif hasattr(na_result, "all"):
        try:
            if bool(na_result.all()):
                return None
        except Exception:
            pass

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

    df = normalize_dataframe(df)

    if "Result" in df.columns:
        df["Result_Numeric"] = pd.to_numeric(df["Result"], errors="coerce")
    if "Test_Date" in df.columns:
        df["Test_Date_dt"] = df["Test_Date"].apply(parse_date_dd_mm_yyyy)
    return df


def _process_single_pdf(
    filename: str,
    payload: bytes,
    api_key: str,
    include_raw_texts: bool,
) -> dict[str, Any]:
    try:
        report_text = extract_text_from_pdf(payload)
    except Exception:
        report_text = None

    if not report_text:
        return {
            "file": filename,
            "ok": False,
            "reason": "No extractable text found.",
        }

    gemini_analysis_json = analyze_medical_report_with_gemini(report_text, api_key)
    if not gemini_analysis_json:
        reason = get_last_extraction_error() or "Gemini extraction failed"
        return {
            "file": filename,
            "ok": False,
            "reason": reason,
        }

    df_single, patient_info_single = create_structured_dataframe(
        gemini_analysis_json,
        filename,
        api_key_for_gemini=api_key,
    )

    return {
        "file": filename,
        "ok": True,
        "df": df_single,
        "patient_info": patient_info_single,
        "raw_preview": (
            {
                "name": filename,
                "text": report_text[:2000],
            }
            if include_raw_texts
            else None
        ),
    }


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
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        api_key = self._require_api_key()

        def emit(event: dict[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback(event)

        total_files = len(pdf_files or [])
        processed_files = 0
        started_at = datetime.utcnow()

        if not pdf_files and not existing_data_file:
            raise ValueError("At least one PDF or an existing data file is required.")

        for idx, (filename, _) in enumerate(pdf_files or []):
            emit(
                {
                    "type": "file",
                    "file": filename,
                    "step": "queued",
                    "percent": 0,
                    "processed": processed_files,
                    "total": total_files,
                    "index": idx,
                }
            )

        all_dfs: list[pd.DataFrame] = []
        all_patient_infos_from_pdfs: list[dict[str, Any]] = []
        raw_texts: list[dict[str, str]] = []
        failed_files: list[str] = []

        queue_mode_active = (
            settings.enable_batch_ingestion_queue
            and total_files >= settings.batch_queue_min_files
        )
        worker_count = (
            min(settings.batch_ingestion_workers, max(1, total_files))
            if queue_mode_active
            else 1
        )

        if queue_mode_active:
            print(
                "[ANALYZE] Phase C batch worker mode enabled: "
                f"files={total_files}, workers={worker_count}"
            )

        for filename, _ in pdf_files:
            emit(
                {
                    "type": "file",
                    "file": filename,
                    "step": "extracting",
                    "percent": 20,
                    "processed": processed_files,
                    "total": total_files,
                }
            )

        if worker_count <= 1:
            results = [
                _process_single_pdf(filename, payload, api_key, include_raw_texts)
                for filename, payload in pdf_files
            ]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _process_single_pdf,
                        filename,
                        payload,
                        api_key,
                        include_raw_texts,
                    )
                    for filename, payload in pdf_files
                ]
                results = [future.result() for future in as_completed(futures)]

        for result in results:
            filename = str(result.get("file") or "uploaded.pdf")

            if not result.get("ok"):
                reason = str(result.get("reason") or "Processing failed")
                failed_files.append(f"{filename}: {reason}")
                processed_files += 1
                emit(
                    {
                        "type": "file",
                        "file": filename,
                        "step": "failed",
                        "percent": 100,
                        "processed": processed_files,
                        "total": total_files,
                        "error": reason,
                    }
                )
                continue

            emit(
                {
                    "type": "file",
                    "file": filename,
                    "step": "parsing",
                    "percent": 75,
                    "processed": processed_files,
                    "total": total_files,
                }
            )

            df_single = result.get("df")
            patient_info_single = result.get("patient_info")
            raw_preview = result.get("raw_preview")

            if isinstance(df_single, pd.DataFrame) and not df_single.empty:
                all_dfs.append(df_single)
            if isinstance(patient_info_single, dict) and patient_info_single:
                all_patient_infos_from_pdfs.append(patient_info_single)
            if isinstance(raw_preview, dict):
                raw_texts.append(raw_preview)

            processed_files += 1
            elapsed = (datetime.utcnow() - started_at).total_seconds()
            avg_per_file = elapsed / processed_files if processed_files > 0 else 0
            remaining = max(total_files - processed_files, 0)
            eta_seconds = int(avg_per_file * remaining)
            emit(
                {
                    "type": "file",
                    "file": filename,
                    "step": "done",
                    "percent": 100,
                    "processed": processed_files,
                    "total": total_files,
                    "eta_seconds": eta_seconds,
                }
            )

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
            details = "; ".join(failed_files[:5]) if failed_files else "Unknown extraction failure."
            if failed_files and all(_is_rate_limit_reason(item) for item in failed_files):
                raise RuntimeError(
                    "RATE_LIMIT_EXCEEDED: Gemini free-tier request quota reached for all uploaded files. "
                    f"Details: {details}"
                )
            raise ValueError(
                "No medical data could be extracted from the provided files. "
                f"Details: {details}"
            )

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

        combined_raw_df = normalize_dataframe(combined_raw_df)

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
        analysis_id: str | None = None,
        session_id: str | None = None,
        system_prompt: str | None = None,
        report_context: dict[str, Any] | None = None,
    ) -> str:
        api_key = self._require_api_key()
        report_df = dataframe_from_records(records)
        return get_chatbot_response(
            report_df,
            question,
            history,
            api_key,
            analysis_id=analysis_id,
            session_id=session_id,
            system_prompt=system_prompt,
            report_context=report_context or {},
        )

    def get_health_insights(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        report_df = dataframe_from_records(normalize_records(records))
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

    def export_excel_report(
        self,
        records: list[dict[str, Any]],
        patient_info: dict[str, Any],
    ) -> bytes:
        """Generate an Excel file with organized test data."""
        from Helper_Functions import create_enhanced_excel_with_trends, parse_date_dd_mm_yyyy

        report_df = dataframe_from_records(records)
        if report_df.empty:
            raise ValueError("No records to export.")

        report_df["Lab_Name_Clean"] = report_df.get("Lab_Name", pd.Series()).fillna("Unknown Lab")
        report_df["Test_Date"] = report_df.get("Test_Date", pd.Series()).fillna("Unknown Date")
        report_df["Date_Lab"] = (
            report_df["Test_Date"].astype(str) + "_" + report_df["Lab_Name_Clean"].astype(str)
        )

        valid = report_df[
            report_df["Test_Name"].notna()
            & (report_df["Test_Name"] != "N/A")
            & report_df["Result"].notna()
        ]

        if valid.empty:
            raise ValueError("No valid test data to export.")

        organized_df = valid.pivot_table(
            index=["Test_Category", "Test_Name"],
            columns="Date_Lab",
            values="Result",
            aggfunc="first",
        ).reset_index()

        ref_range_df = valid.pivot_table(
            index=["Test_Category", "Test_Name"],
            columns="Date_Lab",
            values="Reference_Range",
            aggfunc="first",
        ).reset_index()

        date_lab_cols = [c for c in organized_df.columns if c not in ["Test_Category", "Test_Name"]]

        def _sort_key(col: str) -> Any:
            try:
                return parse_date_dd_mm_yyyy(col.split("_")[0]) or pd.Timestamp.min
            except Exception:
                return pd.Timestamp.min

        date_lab_cols_sorted = sorted(date_lab_cols, key=_sort_key)
        required_cols = ["Test_Category", "Test_Name"] + date_lab_cols_sorted
        organized_df = organized_df.reindex(columns=required_cols, fill_value="").sort_values(
            ["Test_Category", "Test_Name"]
        ).reset_index(drop=True)
        ref_range_df = ref_range_df.reindex(columns=required_cols, fill_value="").sort_values(
            ["Test_Category", "Test_Name"]
        ).reset_index(drop=True)

        return create_enhanced_excel_with_trends(
            organized_df, ref_range_df, date_lab_cols_sorted, patient_info
        )
