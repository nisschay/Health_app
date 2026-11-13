"""Turn a PDF into structured test results: text and OCR, the model prompt, and validation."""
from __future__ import annotations

import io
import json
import logging
import os
import re
from datetime import datetime

import pandas as pd
from pypdf import PdfReader

try:
    import pytesseract
except Exception:  # optional: OCR for scanned PDFs
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:  # optional: OCR for scanned PDFs
    convert_from_bytes = None

from .llm import EXTRACTION_MODEL_CANDIDATES, EXTRACTION_MODEL_TIMEOUT_SECONDS, ModelUnavailable, _env_bool, generate
from .normalization import CANONICAL_STATUS_VALUES, canonicalize_category, normalize_test_name
from .values import (
    _canonical_status,
    _clean_text_value,
    derive_status,
    format_date_dd_mm_yyyy,
    parse_date_dd_mm_yyyy,
    result_series_to_numeric,
)

logger = logging.getLogger(__name__)


# Longer reports are read in parts of at most this size, split at page boundaries.
EXTRACTION_CHUNK_CHARS = 60000
# Part of the extraction cache key: bump it when the prompt, schema or parsing changes,
# so reports already in the cache are read again with the fix.
EXTRACTION_VERSION = 2


PDF_OCR_FALLBACK_ENABLED = _env_bool("PDF_OCR_FALLBACK_ENABLED", default=True)
# Pages read by OCR per PDF; each takes a few seconds on the free server.
PDF_OCR_MAX_PAGES = max(1, int(os.getenv("PDF_OCR_MAX_PAGES", "15")))


CANONICAL_CATEGORY_VALUES = [
    "Haematology",
    "Lipid Profile",
    "Liver Function",
    "Kidney Function",
    "Diabetes & Glucose",
    "Thyroid Function",
    "Vitamins & Minerals",
    "Hormones",
    "Cardiac Markers",
    "Immunology",
    "Urinalysis",
    "Inflammation",
    "Proteins",
    "Other",
]


# Bounds on one PDF, so a crafted or huge file cannot hold a worker for long.
PDF_MAX_PAGES = 50
OCR_PAGE_TIMEOUT_SECONDS = 60
# A page with less text than this is treated as a scanned image and read by OCR.
PAGE_MIN_TEXT_CHARS = 20


def _ocr_page(file_content: bytes, page_number: int) -> str:
    if convert_from_bytes is None or pytesseract is None:
        return ""
    try:
        images = convert_from_bytes(
            file_content, first_page=page_number, last_page=page_number, timeout=OCR_PAGE_TIMEOUT_SECONDS
        )
        return "".join(pytesseract.image_to_string(image, timeout=OCR_PAGE_TIMEOUT_SECONDS) for image in images)
    except Exception as exc:
        logger.warning("OCR failed on page %d: %s", page_number, exc)
        return ""


def extract_text_from_pdf(file_content: bytes) -> str | None:
    """Text of each page in order; pages without a text layer go through OCR.

    The old reader ran OCR only when the whole file had almost no text, and then only on the
    first 5 pages, so a digital first page followed by scanned pages lost the scanned ones.
    """
    try:
        reader = PdfReader(io.BytesIO(file_content))
        pages = list(reader.pages)
    except Exception as exc:
        logger.error("Could not open PDF: %s", exc)
        return None
    if len(pages) > PDF_MAX_PAGES:
        logger.warning("PDF has %d pages; reading the first %d", len(pages), PDF_MAX_PAGES)

    ocr_budget = PDF_OCR_MAX_PAGES if PDF_OCR_FALLBACK_ENABLED else 0
    parts = []
    for number, page in enumerate(pages[:PDF_MAX_PAGES], start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Could not read text on page %d: %s", number, exc)
            text = ""
        if len(text.strip()) < PAGE_MIN_TEXT_CHARS and ocr_budget > 0:
            ocr_budget -= 1
            text = _ocr_page(file_content, number) or text
        if text.strip():
            parts.append(f"\n--- Page {number} ---\n{text}")
    return "".join(parts).strip() or None


def _extract_first_json_object(response_text: str) -> str | None:
    """Extract the first balanced JSON object from model text output."""
    start_index = response_text.find('{')
    if start_index == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start_index, len(response_text)):
        ch = response_text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return response_text[start_index:i + 1]
    return None


def _normalize_patient_info_payload(patient_info) -> dict:
    if not isinstance(patient_info, dict):
        patient_info = {}

    normalized = {
        "name": _clean_text_value(patient_info.get("name"), default="N/A"),
        "age": _clean_text_value(patient_info.get("age"), default="N/A"),
        "gender": _clean_text_value(patient_info.get("gender"), default="N/A"),
        "patient_id": _clean_text_value(patient_info.get("patient_id"), default="N/A"),
        "date": "N/A",
        "lab_name": _clean_text_value(patient_info.get("lab_name"), default="N/A"),
    }

    report_date_text = _clean_text_value(patient_info.get("date"), default="N/A")
    parsed_report_date = parse_date_dd_mm_yyyy(report_date_text)
    if parsed_report_date is not None:
        normalized["date"] = format_date_dd_mm_yyyy(parsed_report_date)

    return normalized


def _normalize_single_test_result(raw_result) -> dict | None:
    if not isinstance(raw_result, dict):
        return None

    test_name = _clean_text_value(raw_result.get("test_name"), default="")
    result = _clean_text_value(raw_result.get("result"), default="")
    if not test_name or not result:
        return None

    raw_category = _clean_text_value(raw_result.get("category"), default="Other")
    normalized_category = canonicalize_category(raw_category.title())

    return {
        "test_name": test_name,
        "result": result,
        "unit": _clean_text_value(raw_result.get("unit"), default=""),
        "reference_range": _clean_text_value(raw_result.get("reference_range"), default="N/A"),
        "status": _canonical_status(raw_result.get("status")),
        "category": normalized_category if normalized_category else "Other",
    }


NO_RESULTS_ISSUE = "No valid entries were present in 'test_results'."


def pdf_page_count(file_content: bytes) -> int:
    try:
        return len(PdfReader(io.BytesIO(file_content)).pages)
    except Exception:
        return 0


def _validate_extraction_payload(parsed_payload) -> tuple[bool, dict, list[str]]:
    if not isinstance(parsed_payload, dict):
        return False, {}, ["Top-level payload must be a JSON object."]

    issues = []

    raw_patient_info = parsed_payload.get("patient_info")
    if not isinstance(raw_patient_info, dict):
        issues.append("Field 'patient_info' must be an object.")
        raw_patient_info = {}

    raw_test_results = parsed_payload.get("test_results")
    if not isinstance(raw_test_results, list):
        issues.append("Field 'test_results' must be an array.")
        raw_test_results = []

    normalized_test_results = []
    malformed_entries = 0
    for item in raw_test_results:
        normalized = _normalize_single_test_result(item)
        if normalized is None:
            malformed_entries += 1
            continue
        normalized_test_results.append(normalized)

    if malformed_entries:
        issues.append(f"Dropped {malformed_entries} malformed test entries.")
    if not normalized_test_results:
        issues.append(NO_RESULTS_ISSUE)

    raw_summary = parsed_payload.get("abnormal_findings_summary_from_report", [])
    if isinstance(raw_summary, str):
        raw_summary = [raw_summary]
    if not isinstance(raw_summary, list):
        raw_summary = []

    summary_items = []
    for item in raw_summary:
        cleaned = _clean_text_value(item, default="").strip()
        if cleaned:
            summary_items.append(cleaned)

    normalized_payload = {
        "patient_info": _normalize_patient_info_payload(raw_patient_info),
        "test_results": normalized_test_results,
        "abnormal_findings_summary_from_report": summary_items,
    }

    is_valid = (
        isinstance(parsed_payload.get("patient_info"), dict)
        and isinstance(parsed_payload.get("test_results"), list)
        and len(normalized_test_results) > 0
    )
    return is_valid, normalized_payload, issues


def _build_structured_extraction_prompt(report_text: str, validation_feedback: str = "") -> str:
    categories_text = ", ".join(CANONICAL_CATEGORY_VALUES)
    feedback_block = ""
    if validation_feedback:
        feedback_block = (
            "\nIMPORTANT CORRECTION:\n"
            f"Your previous response failed validation for: {validation_feedback}\n"
            "Return corrected JSON only and satisfy all schema constraints.\n"
        )

    return f"""
Analyze this medical laboratory report and return ONLY a JSON object with the schema below.{feedback_block}

Hard requirements:
1. Return valid JSON only. Do not use markdown fences.
2. Include all required top-level fields: patient_info, test_results, abnormal_findings_summary_from_report.
3. Keep dates in DD-MM-YYYY format; if unknown use \"N/A\".
4. test_results must be an array of objects and each object must include:
   test_name, result, unit, reference_range, status, category
5. Use category values from: {categories_text}
6. status must be one of: Low, Normal, High, Critical, Positive, Negative, N/A
7. If report does not explicitly provide an interpretation flag, set status to \"N/A\".
8. Preserve exact textual result for non-numeric outcomes (e.g., Detected, Non Reactive).

Schema:
{{
  "patient_info": {{
    "name": "Full patient name or N/A",
    "age": "Age or N/A",
    "gender": "Gender or N/A",
    "patient_id": "Patient ID or registration number or N/A",
    "date": "DD-MM-YYYY or N/A",
    "lab_name": "Primary laboratory or hospital name or N/A"
  }},
  "test_results": [
    {{
      "test_name": "Name of test",
      "result": "Observed value or finding",
      "unit": "Unit or empty string",
      "reference_range": "Reference range or N/A",
      "status": "Low/Normal/High/Critical/Positive/Negative/N/A",
      "category": "Canonical category"
    }}
  ],
  "abnormal_findings_summary_from_report": [
    "Only explicit abnormalities or summary notes stated in the report"
  ]
}}

Medical Report Text:
---
{report_text}
---
"""


_STATUS_ENUM = sorted(CANONICAL_STATUS_VALUES)


EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["patient_info", "test_results", "abnormal_findings_summary_from_report"],
    "properties": {
        "patient_info": {
            "type": "object",
            "required": ["name", "age", "gender", "patient_id", "date", "lab_name"],
            "properties": {key: {"type": "string"} for key in ("name", "age", "gender", "patient_id", "date", "lab_name")},
        },
        "test_results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["test_name", "result", "unit", "reference_range", "status", "category"],
                "properties": {
                    "test_name": {"type": "string"},
                    "result": {"type": "string"},
                    "unit": {"type": "string"},
                    "reference_range": {"type": "string"},
                    "status": {"type": "string", "enum": _STATUS_ENUM},
                    "category": {"type": "string", "enum": CANONICAL_CATEGORY_VALUES},
                },
            },
        },
        "abnormal_findings_summary_from_report": {"type": "array", "items": {"type": "string"}},
    },
}


_EXTRACTION_GENERATION_CONFIG = {
    "response_mime_type": "application/json",
    "response_json_schema": EXTRACTION_RESPONSE_SCHEMA,
    "temperature": 0.0,
    # Copying a lab table needs little reasoning; the default thinking made each report ~5x slower.
    "thinking_config": {"thinking_level": "low"},
}


def _parse_extraction_response(response_text: str) -> tuple[dict | None, str]:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    json_str = fenced.group(1) if fenced else _extract_first_json_object(response_text)
    if not json_str:
        return None, "Response did not contain a JSON object."
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"
    is_valid, payload, issues = _validate_extraction_payload(parsed)
    if is_valid:
        return payload, ""
    return None, "; ".join(issues[:6]) or "Schema validation failed."


def split_into_chunks(text: str, max_chars: int | None = None) -> list[str]:
    """Whole pages per part, so a table is never cut mid-row; a single oversized page is cut by length."""
    max_chars = max_chars or EXTRACTION_CHUNK_CHARS
    pages = re.split(r"(?=\n--- Page \d+ ---\n)", text)
    chunks: list[str] = []
    current = ""
    for page in pages:
        while len(page) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(page[:max_chars])
            page = page[max_chars:]
        if len(current) + len(page) > max_chars and current:
            chunks.append(current)
            current = ""
        current += page
    if current.strip():
        chunks.append(current)
    return chunks


def _merge_parts(parts: list[dict]) -> dict:
    """One payload from the parts of a long report: every result once, patient details from the first part that has them."""
    patient_info = dict(parts[0]["patient_info"])
    for part in parts[1:]:
        for key, value in part["patient_info"].items():
            if patient_info.get(key) in (None, "", "N/A") and value not in (None, "", "N/A"):
                patient_info[key] = value
    results, seen = [], set()
    for part in parts:
        for row in part["test_results"]:
            key = (row.get("test_name"), row.get("result"), row.get("unit"))
            if key not in seen:
                seen.add(key)
                results.append(row)
    summary = list(dict.fromkeys(item for part in parts for item in part.get("abnormal_findings_summary_from_report", [])))
    return {"patient_info": patient_info, "test_results": results, "abnormal_findings_summary_from_report": summary}


def _extract_part(text: str, api_key: str) -> tuple[dict | None, str]:
    feedback = ""
    for _attempt in range(2):
        prompt = _build_structured_extraction_prompt(text, validation_feedback=feedback)
        try:
            response_text = generate(
                prompt,
                api_key=api_key,
                models=EXTRACTION_MODEL_CANDIDATES,
                config=_EXTRACTION_GENERATION_CONFIG,
                timeout_seconds=EXTRACTION_MODEL_TIMEOUT_SECONDS,
            )
        except ModelUnavailable as exc:
            if exc.rate_limited:
                return None, "Every free model has used up its limit for now." + (f" Try again in {exc.retry_after}." if exc.retry_after else "")
            return None, "The report reader is unavailable right now. Please try again shortly."
        payload, feedback = _parse_extraction_response(response_text)
        if payload is not None:
            return payload, ""
    return None, feedback or "Extraction failed after retry."


def analyze_medical_report_with_gemini(text_content, api_key_for_gemini) -> tuple[dict | None, str]:
    """Extract structured findings from report text. Returns (payload, "") or (None, reason).

    A long report is read in parts; a part with no results (a cover page, notes) is fine
    as long as some part has them. Thread-safe: holds no module state.
    """
    if not api_key_for_gemini:
        return None, "Gemini API key is missing or invalid."
    if not text_content or not text_content.strip():
        return None, "No extractable text was provided."

    chunks = split_into_chunks(text_content)
    if len(chunks) == 1:
        return _extract_part(chunks[0], api_key_for_gemini)
    parts = []
    for number, chunk in enumerate(chunks, start=1):
        payload, error = _extract_part(chunk, api_key_for_gemini)
        if payload is not None:
            parts.append(payload)
        elif NO_RESULTS_ISSUE not in error:
            return None, f"Part {number} of {len(chunks)} could not be read: {error}"
    if not parts:
        return None, NO_RESULTS_ISSUE
    return _merge_parts(parts), ""


def create_structured_dataframe(ai_results_json, source_filename="Uploaded PDF"):
    if not ai_results_json or not isinstance(ai_results_json, dict):
        return pd.DataFrame(), {}

    _, normalized_payload, _ = _validate_extraction_payload(ai_results_json)
    if not normalized_payload:
        return pd.DataFrame(), {}

    patient_info_dict = normalized_payload.get('patient_info', {})
    report_date_str = patient_info_dict.get('date', 'N/A')
    parsed_date = 'N/A'
    if report_date_str and report_date_str != 'N/A':
        dt = parse_date_dd_mm_yyyy(report_date_str)
        if dt is not None:
            parsed_date = format_date_dd_mm_yyyy(dt)
    patient_info_dict['date'] = parsed_date

    test_results = normalized_payload.get('test_results', [])
    if not test_results:
        return pd.DataFrame(), patient_info_dict

    all_rows = []
    for test_result in test_results:
        raw_category = _clean_text_value(test_result.get('category'), default='N/A').title()
        raw_test_name = _clean_text_value(test_result.get('test_name'), default='UnknownTest')
        final_status = derive_status(
            test_result.get('result'), test_result.get('reference_range'), test_result.get('status')
        )
        row = {
            'Source_Filename': source_filename,
            'Patient_ID': patient_info_dict.get('patient_id', 'N/A'),
            'Patient_Name': patient_info_dict.get('name', 'N/A'),
            'Age': patient_info_dict.get('age', 'N/A'),
            'Gender': patient_info_dict.get('gender', 'N/A'),
            'Test_Date': parsed_date,
            'Lab_Name': patient_info_dict.get('lab_name', 'N/A'),
            'Test_Category': canonicalize_category(raw_category),
            'Original_Test_Name': test_result.get('test_name', 'UnknownTest'),
            'Test_Name': normalize_test_name(raw_test_name),
            'Result': test_result.get('result', ''),
            'Unit': _clean_text_value(test_result.get('unit'), default=''),
            'Reference_Range': test_result.get('reference_range', ''),
            'Status': final_status,
            'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(), patient_info_dict

    df = pd.DataFrame(all_rows)
    df['Result_Numeric'] = result_series_to_numeric(df['Result'])
    # Use the new date parsing function
    df['Test_Date_dt'] = df['Test_Date'].apply(parse_date_dd_mm_yyyy)
    df = df.sort_values(by=['Test_Date_dt', 'Test_Category', 'Test_Name']).reset_index(drop=True)

    return df, patient_info_dict
