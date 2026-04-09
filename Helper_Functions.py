import streamlit as st
import pandas as pd
from datetime import datetime
import re
import PyPDF2
import io
from collections import Counter
import json
import plotly.graph_objs as go
import google.generativeai as genai
from test_category_mapping import TEST_CATEGORY_TO_BODY_PARTS, BODY_PARTS_TO_EMOJI, TEST_NAME_MAPPING, UNIT_MAPPING, STATUS_MAPPING
try:
    from backend_api.app.normalization import canonicalize_category, normalize_test_name
except Exception:
    # Streamlit-only fallback when backend package imports are unavailable.
    def canonicalize_category(raw):
        text = str(raw).strip() if raw is not None else ""
        return text.title() if text else "Other"

    def normalize_test_name(raw):
        text = str(raw).strip() if raw is not None else ""
        return text.title() if text else "Unknown Test"
import sys
import os
from collections import Counter
import hashlib
import copy
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover - streamlit internals can vary
    get_script_run_ctx = None

logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)


gemini_model_extraction = None
gemini_model_chat = None
_last_extraction_error = ""
_analysis_cache: dict[str, dict] = {}
_analysis_cache_max_items = 128
_active_extraction_model_name = ""
_active_chat_model_name = ""


def _model_candidates_from_env(var_name: str, default: str) -> list[str]:
    raw = os.getenv(var_name, default)
    return [m.strip() for m in raw.split(",") if m.strip()]


EXTRACTION_MODEL_CANDIDATES = _model_candidates_from_env(
    "GEMINI_EXTRACTION_MODELS",
    "gemini-2.5-flash,gemini-1.5-flash",
)
CHAT_MODEL_CANDIDATES = _model_candidates_from_env(
    "GEMINI_CHAT_MODELS",
    "gemini-2.5-flash,gemini-1.5-flash",
)

CHAT_HISTORY_LIMIT = max(1, int(os.getenv("CHAT_HISTORY_LIMIT", "8")))
CHAT_PROMPT_MAX_ROWS = max(20, int(os.getenv("CHAT_PROMPT_MAX_ROWS", "80")))
CHAT_TIMELINE_LIMIT = max(1, int(os.getenv("CHAT_TIMELINE_LIMIT", "12")))
CHAT_SNAPSHOT_LIMIT = max(5, int(os.getenv("CHAT_SNAPSHOT_LIMIT", "40")))
CHAT_MODEL_TIMEOUT_SECONDS = max(5, int(os.getenv("CHAT_MODEL_TIMEOUT_SECONDS", "50")))

CANONICAL_STATUS_VALUES = {
    "Low",
    "Normal",
    "High",
    "Critical",
    "Positive",
    "Negative",
    "N/A",
}

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


def _has_streamlit_context() -> bool:
    if get_script_run_ctx is None:
        return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _ui_warn(message: str) -> None:
    """Show warning in Streamlit UI when available, otherwise log to stdout."""
    if _has_streamlit_context():
        st.warning(message)
    else:
        print(f"[WARN] {message}")


def _ui_error(message: str) -> None:
    """Show error in Streamlit UI when available, otherwise log to stdout."""
    if _has_streamlit_context():
        st.error(message)
    else:
        print(f"[ERROR] {message}")


def _ui_debug_text(label: str, value: str, height: int = 150) -> None:
    """Optional debug output that is safe outside Streamlit execution context."""
    if _has_streamlit_context():
        st.text_area(label, value, height=height)
    else:
        print(f"[DEBUG] {label}: {value[:500]}")


def get_last_extraction_error() -> str:
    return _last_extraction_error


def _is_rate_limit_error(error_text: str) -> bool:
    lowered = (error_text or "").lower()
    return "429" in lowered or "quota exceeded" in lowered or "rate limit" in lowered


def _extract_retry_delay(error_text: str) -> str | None:
    if not error_text:
        return None
    match_sec = re.search(r"retry in\s+([\d\.]+)s", error_text, flags=re.IGNORECASE)
    if match_sec:
        return f"{match_sec.group(1)}s"
    match_ms = re.search(r"retry in\s+([\d\.]+)ms", error_text, flags=re.IGNORECASE)
    if match_ms:
        return f"{match_ms.group(1)}ms"
    return None


def _cache_get(cache_key: str):
    cached = _analysis_cache.get(cache_key)
    return copy.deepcopy(cached) if cached is not None else None


def _cache_set(cache_key: str, payload: dict) -> None:
    if len(_analysis_cache) >= _analysis_cache_max_items:
        oldest_key = next(iter(_analysis_cache))
        _analysis_cache.pop(oldest_key, None)
    _analysis_cache[cache_key] = copy.deepcopy(payload)



def format_date_dd_mm_yyyy(date_obj):
    """Format datetime object to DD-MM-YYYY string"""
    if pd.isna(date_obj) or date_obj is None:
        return 'N/A'
    return date_obj.strftime('%d-%m-%Y')

def extract_patient_info_from_excel_filename(filename):
    """Try to extract patient info from Excel filename if possible"""
    # This is a fallback - try to extract patient name from filename
    name_match = re.search(r'medical_reports?_.*?_([^_]+(?:_[^_]+)*)', filename.lower())
    if name_match:
        # Replace underscores with spaces and title case
        extracted_name = name_match.group(1).replace('_', ' ').title()
        return extracted_name
    return "N/A"



def standardize_value(value, mapping_dict, default_case='title'):
    if not isinstance(value, str):
        return value
    original_value = value.strip()
    for pattern, standard_form in mapping_dict.items():
        if pattern.search(original_value):
            if default_case == 'title':
                return standard_form.title()
            elif default_case == 'lower':
                return standard_form.lower()
            return standard_form
    if default_case == 'title':
        return original_value.title()
    elif default_case == 'lower':
        return original_value.lower()
    return original_value


def create_consolidated_info_with_smart_selection(patient_info_list):
    """Create consolidated patient info with smart selection for name and age."""
    if not patient_info_list:
        return {}

    # Get all valid data
    names = [pi.get('name') for pi in patient_info_list if pi.get('name') and pi.get('name') not in ['N/A', '']]
    ages = [pi.get('age') for pi in patient_info_list if pi.get('age') and pi.get('age') not in ['N/A', '']]
    genders = [pi.get('gender') for pi in patient_info_list if pi.get('gender') and pi.get('gender') not in ['N/A', '']]
    patient_ids = [pi.get('patient_id') for pi in patient_info_list if pi.get('patient_id') and pi.get('patient_id') not in ['N/A', '']]
    dates = [pi.get('date') for pi in patient_info_list if pi.get('date') and pi.get('date') not in ['N/A', '']]
    lab_names = [pi.get('lab_name') for pi in patient_info_list if pi.get('lab_name') and pi.get('lab_name') not in ['N/A', '']]

    # 1. Smart name selection - prefer the longest, most complete name
    final_name = "N/A"
    if names:
        # Normalize names first
        normalized_names = [normalize_name(name) for name in names]
        normalized_names = [name for name in normalized_names if name]  # Remove empty strings
        
        if normalized_names:
            # Count occurrences of normalized names
            name_counts = Counter(normalized_names)
            
            # Get the most frequent names
            max_count = max(name_counts.values())
            most_frequent_names = [name for name, count in name_counts.items() if count == max_count]
            
            # Among the most frequent names, choose the longest one (most complete)
            final_name = max(most_frequent_names, key=len) if most_frequent_names else normalized_names[0]

    # 2. Age from the PDF with the most recent date
    final_age = "N/A"
    if dates and ages:
        # Create list of (date, patient_info) pairs
        date_info_pairs = []
        for pi in patient_info_list:
            pi_date = pi.get('date')
            pi_age = pi.get('age')
            if pi_date and pi_date not in ['N/A', ''] and pi_age and pi_age not in ['N/A', '']:
                parsed_date = parse_date_dd_mm_yyyy(pi_date)
                if parsed_date is not None:
                    date_info_pairs.append((parsed_date, pi_age))
        
        if date_info_pairs:
            # Sort by date and get the age from the most recent date
            date_info_pairs.sort(key=lambda x: x[0], reverse=True)
            final_age = date_info_pairs[0][1]  # Age from most recent date
        elif ages:
            # If no valid dates, just use the most common age
            final_age = Counter(ages).most_common(1)[0][0]

    # 3. Other fields: most frequent valid value
    final_gender = Counter(genders).most_common(1)[0][0] if genders else "N/A"
    final_patient_id = Counter(patient_ids).most_common(1)[0][0] if patient_ids else "N/A"
    final_lab_name = Counter(lab_names).most_common(1)[0][0] if lab_names else "N/A"
    
    # For date, use the most recent one
    parsed_dates = [parse_date_dd_mm_yyyy(d) for d in dates]
    valid_parsed_dates = [d for d in parsed_dates if d is not None]
    final_date = format_date_dd_mm_yyyy(max(valid_parsed_dates)) if valid_parsed_dates else "N/A"
    
    return {
        'name': final_name,
        'age': final_age,
        'gender': final_gender,
        'patient_id': final_patient_id,
        'date': final_date,
        'lab_name': final_lab_name
    }

def extract_text_from_pdf(file_content):
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"\n--- Page {page_num+1} ---\n{page_text}")
        text = "".join(text_parts)
        if not text.strip():
            _ui_warn("No text extracted from PDF. The PDF may be image-based or empty.")
        return text
    except Exception as e:
        _ui_error(f"Error reading PDF: {str(e)}")
        return None

def init_gemini_models(api_key_for_gemini):
    global gemini_model_extraction, gemini_model_chat, _active_extraction_model_name, _active_chat_model_name
    try:
        if not api_key_for_gemini:
            _ui_error("Gemini API key is missing. Cannot initialize models.")
            return False
        
        genai.configure(api_key=api_key_for_gemini)

        last_error = None
        for model_name in EXTRACTION_MODEL_CANDIDATES:
            try:
                gemini_model_extraction = genai.GenerativeModel(model_name)
                _active_extraction_model_name = model_name
                break
            except Exception as model_exc:
                last_error = model_exc
                continue

        if gemini_model_extraction is None:
            _ui_error(f"Could not initialize Gemini extraction model: {last_error}")
            gemini_model_chat = None
            return False

        chat_last_error = None
        for model_name in CHAT_MODEL_CANDIDATES:
            try:
                gemini_model_chat = genai.GenerativeModel(model_name)
                _active_chat_model_name = model_name
                return True
            except Exception as model_exc:
                chat_last_error = model_exc
                continue

        _ui_error(f"Could not initialize Gemini chat model: {chat_last_error}")
        gemini_model_extraction = None
        gemini_model_chat = None
        _active_extraction_model_name = ""
        _active_chat_model_name = ""
        return False
        
    except Exception as e:
        _ui_error(f"Error configuring Gemini: {e}. Please ensure your API key is correct and valid.")
        gemini_model_extraction = None
        gemini_model_chat = None
        _active_extraction_model_name = ""
        _active_chat_model_name = ""
        return False


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


def _extract_text_from_response(response) -> str:
    response_text = getattr(response, "text", "") or ""
    if response_text:
        return response_text

    candidates = getattr(response, "candidates", None) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts).strip()


def _generate_with_extraction_models(prompt: str, generation_config: dict | None = None) -> tuple[str, Exception | None]:
    global gemini_model_extraction
    global _active_extraction_model_name

    response_text = ""
    last_exception = None
    models_to_try = []

    if _active_extraction_model_name:
        models_to_try.append(_active_extraction_model_name)
    for model_name in EXTRACTION_MODEL_CANDIDATES:
        if model_name not in models_to_try:
            models_to_try.append(model_name)

    for model_name in models_to_try:
        if model_name != _active_extraction_model_name:
            try:
                gemini_model_extraction = genai.GenerativeModel(model_name)
                _active_extraction_model_name = model_name
            except Exception as model_exc:
                last_exception = model_exc
                continue

        try:
            call_kwargs = {}
            if generation_config:
                call_kwargs["generation_config"] = generation_config
            response = gemini_model_extraction.generate_content(prompt, **call_kwargs)
            response_text = _extract_text_from_response(response)
            if response_text:
                return response_text, None
        except Exception as call_exc:
            last_exception = call_exc
            if _is_rate_limit_error(str(call_exc)):
                continue
            if generation_config:
                try:
                    # Retry once without structured output hints for model compatibility.
                    response = gemini_model_extraction.generate_content(prompt)
                    response_text = _extract_text_from_response(response)
                    if response_text:
                        return response_text, None
                except Exception as fallback_exc:
                    last_exception = fallback_exc
            continue

    return "", last_exception


def _clean_text_value(value, default: str = "N/A") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _canonical_status(raw_status) -> str:
    status_text = _clean_text_value(raw_status, default="N/A")
    lowered = status_text.lower()

    if lowered in {"positive", "detected", "reactive", "present", "pos"}:
        return "Positive"
    if lowered in {"negative", "not detected", "non reactive", "absent", "neg"}:
        return "Negative"
    if "critical" in lowered or "panic" in lowered:
        return "Critical"
    if "high" in lowered or lowered in {"h", "elevated", "above range", "borderline high"}:
        return "High"
    if "low" in lowered or lowered in {"l", "decreased", "below range"}:
        return "Low"
    if "normal" in lowered or "within range" in lowered or "within normal" in lowered:
        return "Normal"
    if lowered in {"na", "n/a", "not applicable", "none", "unknown"}:
        return "N/A"

    standardized = str(standardize_value(status_text, STATUS_MAPPING, default_case='title')).strip()
    if standardized in CANONICAL_STATUS_VALUES:
        return standardized
    return "N/A"


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
    normalized_category = canonicalize_category(standardize_value(raw_category, {}, default_case='title'))

    return {
        "test_name": test_name,
        "result": result,
        "unit": _clean_text_value(raw_result.get("unit"), default=""),
        "reference_range": _clean_text_value(raw_result.get("reference_range"), default="N/A"),
        "status": _canonical_status(raw_result.get("status")),
        "category": normalized_category if normalized_category else "Other",
    }


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
        issues.append("No valid entries were present in 'test_results'.")

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


def extract_patient_info_from_normalized_data(df):
    """Extract patient info from normalized dataframe"""
    if df.empty:
        return {}
    
    # Get the most common/recent patient info
    patient_info = {}
    
    # Get most common values
    if 'Patient_Name' in df.columns:
        names = df['Patient_Name'].dropna()
        names = names[names != 'N/A']
        patient_info['name'] = names.mode().iloc[0] if not names.empty else 'N/A'
    
    if 'Age' in df.columns:
        ages = df['Age'].dropna()
        ages = ages[ages != 'N/A']
        patient_info['age'] = ages.mode().iloc[0] if not ages.empty else 'N/A'
    
    if 'Gender' in df.columns:
        genders = df['Gender'].dropna()
        genders = genders[genders != 'N/A']
        patient_info['gender'] = genders.mode().iloc[0] if not genders.empty else 'N/A'
    
    if 'Patient_ID' in df.columns:
        ids = df['Patient_ID'].dropna()
        ids = ids[ids != 'N/A']
        patient_info['patient_id'] = ids.mode().iloc[0] if not ids.empty else 'N/A'
    
    if 'Lab_Name' in df.columns:
        labs = df['Lab_Name'].dropna()
        labs = labs[labs != 'N/A']
        patient_info['lab_name'] = labs.mode().iloc[0] if not labs.empty else 'N/A'
    
    # Get most recent date
    if 'Test_Date' in df.columns:
        dates = df['Test_Date'].dropna()
        dates = dates[dates != 'N/A']
        if not dates.empty:
            # Parse dates and get the most recent one
            parsed_dates = [parse_date_dd_mm_yyyy(d) for d in dates]
            valid_dates = [d for d in parsed_dates if d is not None]
            if valid_dates:
                most_recent = max(valid_dates)
                patient_info['date'] = format_date_dd_mm_yyyy(most_recent)
            else:
                patient_info['date'] = 'N/A'
        else:
            patient_info['date'] = 'N/A'
    
    return patient_info
def auto_detect_and_process(df, filename, new_patient_info_list):
    """Auto-detect the format and process accordingly"""
    
    # Check if it looks like a simple test results table
    if len(df.columns) >= 2:
        # Assume first column is test name, others are dates/results
        test_col = df.columns[0]
        result_cols = df.columns[1:]
        
        # Try to detect if columns are dates
        date_like_cols = []
        for col in result_cols:
            if parse_date_dd_mm_yyyy(str(col)) is not None:
                date_like_cols.append(col)
        
        if date_like_cols:
            # Treat as a simple pivoted format
            print(f"Detected simple pivoted format with {len(date_like_cols)} date columns")
            
            # Convert to our standard pivoted format
            simple_df = df.copy()
            simple_df['Test_Category'] = 'General'  # Default category
            simple_df = simple_df.rename(columns={test_col: 'Test_Name'})
            
            # Reorder columns
            cols = ['Test_Category', 'Test_Name'] + [col for col in simple_df.columns if col not in ['Test_Category', 'Test_Name']]
            simple_df = simple_df[cols]
            
            return process_pivoted_excel_data(simple_df, filename, new_patient_info_list)
    
    print("Could not detect format, returning empty DataFrame")
    return pd.DataFrame(), {}



def analyze_medical_report_with_gemini(text_content, api_key_for_gemini):
    global gemini_model_extraction
    global _last_extraction_error
    _last_extraction_error = ""
    if not gemini_model_extraction and not init_gemini_models(api_key_for_gemini):
        _last_extraction_error = "Gemini extraction model not initialized. API key may be invalid or missing."
        _ui_error(_last_extraction_error)
        return None

    if not text_content or not text_content.strip():
        _last_extraction_error = "No extractable text was provided to Gemini."
        _ui_warn(_last_extraction_error)
        return None

    # Keep prompt size bounded to reduce timeout/limit failures on very large PDFs.
    max_chars = 120000
    bounded_text = text_content[:max_chars]
    cache_key = hashlib.md5(bounded_text.encode("utf-8", errors="ignore")).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    validation_feedback = ""
    last_response_text = ""
    last_exception = None
    try:
        for attempt in range(2):
            prompt = _build_structured_extraction_prompt(
                bounded_text,
                validation_feedback=validation_feedback,
            )
            response_text, last_exception = _generate_with_extraction_models(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                },
            )
            last_response_text = response_text

            if not response_text:
                if last_exception and _is_rate_limit_error(str(last_exception)):
                    retry_delay = _extract_retry_delay(str(last_exception))
                    _last_extraction_error = (
                        f"Rate limit reached for Gemini extraction ({_active_extraction_model_name or 'configured models'})."
                        + (f" Retry after {retry_delay}." if retry_delay else "")
                    )
                    _ui_error(_last_extraction_error)
                    return None
                if attempt == 0:
                    validation_feedback = "Empty response from model."
                    continue
                break

            match_json_block = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if match_json_block:
                json_str = match_json_block.group(1)
            else:
                json_str = _extract_first_json_object(response_text)
                if not json_str:
                    validation_feedback = "Response did not contain a valid JSON object."
                    if attempt == 0:
                        _ui_warn("Gemini extraction response was not valid JSON; retrying with corrective instructions.")
                        continue
                    _last_extraction_error = validation_feedback
                    _ui_error(_last_extraction_error)
                    _ui_debug_text("Gemini API Response (text)", response_text, height=150)
                    return None

            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                validation_feedback = f"JSON parse error: {json_e}"
                if attempt == 0:
                    _ui_warn("Gemini extraction JSON parse failed; retrying once with corrective instructions.")
                    continue
                _last_extraction_error = f"JSON parse error from Gemini response: {json_e}"
                _ui_error(_last_extraction_error)
                _ui_debug_text("Problematic JSON string", json_str, height=150)
                _ui_debug_text("Full Gemini Response (text)", response_text, height=150)
                return None

            is_valid, normalized_payload, validation_issues = _validate_extraction_payload(parsed)
            if is_valid:
                _cache_set(cache_key, normalized_payload)
                return normalized_payload

            validation_feedback = "; ".join(validation_issues[:6]) or "Schema validation failed."
            if attempt == 0:
                _ui_warn("Gemini extraction response failed schema validation; retrying once with corrective instructions.")
                continue

            _last_extraction_error = f"Gemini JSON failed validation after retry: {validation_feedback}"
            _ui_error(_last_extraction_error)
            _ui_debug_text("Gemini JSON payload", json.dumps(parsed)[:2500], height=180)
            return None

        if last_exception and _is_rate_limit_error(str(last_exception)):
                retry_delay = _extract_retry_delay(str(last_exception))
                _last_extraction_error = (
                    f"Rate limit reached for Gemini extraction ({_active_extraction_model_name or 'configured models'})."
                    + (f" Retry after {retry_delay}." if retry_delay else "")
                )
                _ui_error(_last_extraction_error)
                return None

        _last_extraction_error = "Gemini extraction failed after retry attempts."
        if last_exception:
            _last_extraction_error = f"Gemini extraction exception: {str(last_exception)}"
        _ui_error(_last_extraction_error)
        if last_response_text:
            _ui_debug_text("Last Gemini Response (text)", last_response_text, height=150)
        return None
    except Exception as e:
        _last_extraction_error = f"Gemini extraction exception: {str(e)}"
        _ui_error(_last_extraction_error)
        if "API key not valid" in str(e) or "PERMISSION_DENIED" in str(e):
            _ui_error("Please ensure your Gemini API key is correct and has the necessary permissions.")
        return None

def _classify_test_statuses_with_gemini(test_results: list[dict], api_key_for_gemini: str | None) -> list[str]:
    fallback_statuses = [_canonical_status(item.get("status")) for item in test_results]
    if not test_results:
        return fallback_statuses

    if not api_key_for_gemini:
        return fallback_statuses

    if not gemini_model_extraction and not init_gemini_models(api_key_for_gemini):
        return fallback_statuses

    classifier_input = []
    for idx, test_result in enumerate(test_results):
        classifier_input.append(
            {
                "index": idx,
                "test_name": _clean_text_value(test_result.get("test_name"), default="N/A"),
                "result": _clean_text_value(test_result.get("result"), default="N/A"),
                "unit": _clean_text_value(test_result.get("unit"), default=""),
                "reference_range": _clean_text_value(test_result.get("reference_range"), default="N/A"),
                "reported_status": _canonical_status(test_result.get("status")),
            }
        )

    prompt = f"""
Classify the clinical status for each laboratory finding.
Return ONLY JSON in this shape:
{{
  "classifications": [
    {{"index": 0, "status": "Low|Normal|High|Critical|Positive|Negative|N/A", "reason": "short rationale"}}
  ]
}}

Rules:
- Use each finding's result and reference_range.
- Respect textual outcomes like Positive/Negative/Detected/Not Detected.
- If data is insufficient, use N/A.
- Do not skip indexes.

Findings:
{json.dumps(classifier_input)}
"""

    response_text, last_exception = _generate_with_extraction_models(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    )
    if not response_text:
        if last_exception:
            _ui_warn(f"Severity classification fallback used due to model error: {last_exception}")
        return fallback_statuses

    match_json_block = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    json_str = match_json_block.group(1) if match_json_block else _extract_first_json_object(response_text)
    if not json_str:
        return fallback_statuses

    try:
        parsed = json.loads(json_str)
    except Exception:
        return fallback_statuses

    raw_classifications = parsed.get("classifications") if isinstance(parsed, dict) else None
    if not isinstance(raw_classifications, list):
        return fallback_statuses

    final_statuses = list(fallback_statuses)
    for item in raw_classifications:
        if not isinstance(item, dict):
            continue
        index_val = item.get("index")
        if isinstance(index_val, bool):
            continue
        try:
            idx = int(index_val)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(final_statuses):
            continue

        candidate_status = _canonical_status(item.get("status"))
        if candidate_status == "N/A" and fallback_statuses[idx] != "N/A":
            candidate_status = fallback_statuses[idx]
        final_statuses[idx] = candidate_status

    return final_statuses


def create_structured_dataframe(
    ai_results_json,
    source_filename="Uploaded PDF",
    api_key_for_gemini=None,
):
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

    classified_statuses = _classify_test_statuses_with_gemini(test_results, api_key_for_gemini)
    if len(classified_statuses) != len(test_results):
        classified_statuses = [_canonical_status(item.get('status')) for item in test_results]

    all_rows = []
    for idx, test_result in enumerate(test_results):
        raw_category = standardize_value(test_result.get('category', 'N/A'), {}, default_case='title')
        raw_test_name = standardize_value(test_result.get('test_name', 'UnknownTest'), TEST_NAME_MAPPING, default_case='title')
        final_status = classified_statuses[idx] if idx < len(classified_statuses) else _canonical_status(test_result.get('status'))
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
            'Unit': standardize_value(test_result.get('unit', ''), UNIT_MAPPING, default_case='original'),
            'Reference_Range': test_result.get('reference_range', ''),
            'Status': final_status,
            'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(), patient_info_dict

    df = pd.DataFrame(all_rows)
    df['Result_Numeric'] = pd.to_numeric(df['Result'], errors='coerce')
    # Use the new date parsing function
    df['Test_Date_dt'] = df['Test_Date'].apply(parse_date_dd_mm_yyyy)
    df = df.sort_values(by=['Test_Date_dt', 'Test_Category', 'Test_Name']).reset_index(drop=True)
    
    return df, patient_info_dict



def normalize_name(name):
    """Normalize name by removing titles and standardizing format"""
    if not name or name == 'N/A':
        return ''
    name = name.strip()
    titles = ['mr', 'mrs', 'ms', 'dr', 'prof', 'self', 'mr.', 'mrs.', 'ms.', 'dr.', 'prof.']
    name_lower = name.lower()
    for title in titles:
        name_lower = re.sub(rf'\b{title}\b\.?\s*', '', name_lower)
    words = [word for word in name_lower.split() if word]
    return ' '.join(word.title() for word in words)

def are_names_matching(name1, name2):
    name1 = normalize_name(name1)
    name2 = normalize_name(name2)
    if not name1 or not name2:
        return True
    name1_parts = set(name1.split())
    name2_parts = set(name2.split())
    if name1_parts.issubset(name2_parts) or name2_parts.issubset(name1_parts):
        return True
    matching_parts = name1_parts.intersection(name2_parts)
    total_unique_parts = name1_parts.union(name2_parts)
    if len(matching_parts) >= 2 and len(matching_parts) / len(total_unique_parts) >= 0.6:
        return True
    return False

def get_most_common_or_latest(items, is_date=False):
    if not items:
        return "N/A"
    if is_date:
        parsed_dates = [parse_date_dd_mm_yyyy(d) for d in items]
        valid_parsed_dates = [d for d in parsed_dates if d is not None]
        return format_date_dd_mm_yyyy(max(valid_parsed_dates)) if valid_parsed_dates else "N/A"
    return Counter(items).most_common(1)[0][0]

def consolidate_patient_info(patient_info_list):
    if not patient_info_list:
        return {}

    names = [pi.get('name') for pi in patient_info_list if pi.get('name') and pi.get('name') not in ['N/A', '']]
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if not are_names_matching(names[i], names[j]):
                    return {
                        'error': 'name_mismatch',
                        'names': names,
                        'conflicting_names': [names[i], names[j]]
                    }

    final_name = "N/A"
    if names:
        normalized_names = [normalize_name(name) for name in names if name]
        if normalized_names:
            name_counts = Counter(normalized_names)
            max_count = max(name_counts.values())
            most_frequent_names = [name for name, count in name_counts.items() if count == max_count]
            final_name = max(most_frequent_names, key=len) if most_frequent_names else normalized_names[0]

    final_age = "N/A"
    ages = [pi.get('age') for pi in patient_info_list if pi.get('age') and pi.get('age') not in ['N/A', '']]
    if ages:
        date_age_pairs = []
        for pi in patient_info_list:
            pi_date_str = pi.get('date')
            pi_age = pi.get('age')
            if pi_date_str and pi_date_str not in ['N/A', ''] and pi_age and pi_age not in ['N/A', '']:
                parsed_date = parse_date_dd_mm_yyyy(pi_date_str)
                if parsed_date is not None:
                    date_age_pairs.append((parsed_date, pi_age))
        if date_age_pairs:
            date_age_pairs.sort(key=lambda x: x[0], reverse=True)
            final_age = date_age_pairs[0][1]
        else:
            final_age = Counter(ages).most_common(1)[0][0]

    return {
        'name': final_name,
        'age': final_age,
        'gender': get_most_common_or_latest([pi.get('gender') for pi in patient_info_list if pi.get('gender') and pi.get('gender') not in ['N/A', '']]),
        'patient_id': get_most_common_or_latest([pi.get('patient_id') for pi in patient_info_list if pi.get('patient_id') and pi.get('patient_id') not in ['N/A', '']]),
        'date': get_most_common_or_latest([pi.get('date') for pi in patient_info_list if pi.get('date') and pi.get('date') not in ['N/A', '']], is_date=True),
        'lab_name': get_most_common_or_latest([pi.get('lab_name') for pi in patient_info_list if pi.get('lab_name') and pi.get('lab_name') not in ['N/A', '']])
    }

def get_chatbot_response(
    report_df_for_prompt,
    user_question,
    chat_history_for_prompt,
    api_key_for_gemini,
    analysis_id=None,
    session_id=None,
    system_prompt=None,
    report_context=None,
):
    global gemini_model_chat, _active_chat_model_name
    if not gemini_model_chat and not init_gemini_models(api_key_for_gemini):
        return "Chatbot model not initialized. API key might be missing or invalid."
    if report_df_for_prompt.empty:
        return "No report data available to answer questions. Please analyze a report first."

    expected_cols = [
        'Test_Date',
        'Test_Category',
        'Test_Name',
        'Result',
        'Unit',
        'Reference_Range',
        'Status',
    ]
    safe_df = report_df_for_prompt.copy()
    for col in expected_cols:
        if col not in safe_df.columns:
            safe_df[col] = "N/A"

    if "Lab_Name" not in safe_df.columns:
        safe_df["Lab_Name"] = "Unknown Lab"

    safe_df["Test_Date"] = safe_df["Test_Date"].fillna("Unknown date").astype(str)
    safe_df["Lab_Name"] = safe_df["Lab_Name"].fillna("Unknown Lab").astype(str)
    safe_df["_date_sort"] = safe_df["Test_Date"].apply(
        lambda value: parse_date_dd_mm_yyyy(value) or datetime.max
    )
    safe_df = safe_df.sort_values(by=["_date_sort", "Lab_Name", "Test_Name"], na_position="last").reset_index(drop=True)

    df_string = safe_df[expected_cols].to_string(index=False, max_rows=CHAT_PROMPT_MAX_ROWS)

    timeline_lines = []
    grouped = safe_df.groupby(["Test_Date", "Lab_Name"], dropna=False, sort=False)
    for (date_label, lab_label), group in grouped:
        status_series = group["Status"].fillna("").astype(str).str.lower()
        abnormal_rows = group[
            (~status_series.isin(["normal", "negative", "n/a", "na", ""]))
        ]
        abnormal_preview = ", ".join(
            abnormal_rows.apply(
                lambda row: (
                    f"{row.get('Test_Name', 'Unknown')}: {row.get('Result', 'N/A')} "
                    f"{row.get('Unit', '')} ({row.get('Status', 'ABNORMAL')})"
                ).strip(),
                axis=1,
            ).tolist()[:10]
        )
        timeline_lines.append(
            f"Date: {date_label} | Lab: {lab_label}\n"
            f"Tests performed: {len(group)}\n"
            f"Abnormal findings: {abnormal_preview or 'None'}"
        )

    timeline_summary = "\n---\n".join(timeline_lines[:CHAT_TIMELINE_LIMIT])

    latest_snapshot_rows = safe_df.drop_duplicates(subset=["Test_Name"], keep="last")
    latest_snapshot = "\n".join(
        latest_snapshot_rows.apply(
            lambda row: (
                f"{row.get('Test_Name', 'Unknown')}: {row.get('Result', 'N/A')} {row.get('Unit', '')} "
                f"[{row.get('Status', 'UNKNOWN')}] as of {row.get('Test_Date', 'Unknown date')}"
            ).strip(),
            axis=1,
        ).tolist()[:CHAT_SNAPSHOT_LIMIT]
    )

    report_context = report_context or {}
    source_names = report_context.get("sourceFileNames") or []
    if isinstance(source_names, list):
        source_names = [str(item) for item in source_names if item]
    else:
        source_names = []

    history_context = ""
    for entry in chat_history_for_prompt[-CHAT_HISTORY_LIMIT:]:
        role = str(entry.get("role", "user")).capitalize()
        content = str(entry.get("content", "")).strip()
        if content:
            history_context += f"{role}: {content}\n"

    if system_prompt:
        prompt = f"""{system_prompt}

THREAD CONTEXT:
{history_context or 'No previous conversation in this session.'}

ADDITIONAL TABULAR DATA:
{df_string}

REQUEST METADATA:
- Analysis ID: {analysis_id or 'unknown-analysis'}
- Session ID: {session_id or 'unknown-session'}
- Source files: {', '.join(source_names) if source_names else 'Unknown'}

User Question: {user_question}

Assistant Response (markdown):
"""
    else:
        prompt = f"""You are a Clinical Assistant helping patients understand longitudinal lab trends.
You must use ALL available reports, not only the most recent data.

INSTRUCTIONS:
- Answer in plain language
- Include value, unit, and date when citing labs
- Distinguish historical abnormalities from current status
- If a test has not been repeated recently, call out that current status is unknown
- Never diagnose; recommend clinician follow-up for treatment decisions
- Use markdown formatting with short sections and bullet points

REPORT TIMELINE:
{timeline_summary}

LATEST VALUES SNAPSHOT:
{latest_snapshot}

THREAD CONTEXT:
{history_context or 'No previous conversation in this session.'}

AVAILABLE MEDICAL REPORT DATA (tabular snapshot):
{df_string}

REQUEST METADATA:
- Analysis ID: {analysis_id or 'unknown-analysis'}
- Session ID: {session_id or 'unknown-session'}
- Source files: {', '.join(source_names) if source_names else 'Unknown'}

User Question: {user_question}

Assistant Response (markdown):
"""

    def _response_text_from_model_response(model_response):
        text = (getattr(model_response, "text", "") or "").strip()
        if text:
            return text

        parts = []
        for candidate in getattr(model_response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)

        return "\n".join(parts).strip()

    model_names_to_try = []
    if _active_chat_model_name:
        model_names_to_try.append(_active_chat_model_name)
    for model_name in CHAT_MODEL_CANDIDATES:
        if model_name not in model_names_to_try:
            model_names_to_try.append(model_name)

    last_error_text = ""
    for model_name in model_names_to_try:
        try:
            if model_name != _active_chat_model_name:
                gemini_model_chat = genai.GenerativeModel(model_name)
                _active_chat_model_name = model_name

            with ThreadPoolExecutor(max_workers=1) as executor:
                response_future = executor.submit(
                    gemini_model_chat.generate_content,
                    prompt,
                    generation_config={"temperature": 0.3},
                )
                response = response_future.result(timeout=CHAT_MODEL_TIMEOUT_SECONDS)
            response_text = _response_text_from_model_response(response)
            if response_text:
                return response_text
            last_error_text = "Gemini chat returned an empty response."
            continue
        except FutureTimeoutError:
            last_error_text = (
                f"Gemini chat timed out after {CHAT_MODEL_TIMEOUT_SECONDS}s "
                f"for model {model_name}."
            )
            continue
        except Exception as e:
            last_error_text = str(e)
            if _is_rate_limit_error(last_error_text):
                retry_delay = _extract_retry_delay(last_error_text)
                return (
                    "Gemini chat rate limit reached. "
                    + (f"Please retry after {retry_delay}." if retry_delay else "Please try again shortly.")
                )
            continue

    if last_error_text:
        _ui_error(f"Error getting chatbot response from Gemini: {last_error_text}")
    return "Sorry, I encountered an error trying to respond."

def parse_reference_range(ref_range_str):
    if not isinstance(ref_range_str, str) or ref_range_str.lower() == 'n/a' or not ref_range_str.strip():
        return None, None, None
    
    ref_range_str = ref_range_str.strip()
    
    match_range = re.search(r'([\d.]+)\s*-\s*([\d.]+)', ref_range_str)
    if match_range:
        try: return float(match_range.group(1)), float(match_range.group(2)), "range"
        except ValueError: pass

    match_less_than = re.search(r'(?:<|Less than|upto)\s*([\d.]+)', ref_range_str, re.IGNORECASE)
    if match_less_than:
        try: return None, float(match_less_than.group(1)), "less_than"
        except ValueError: pass

    match_greater_than = re.search(r'(?:>|Greater than|above)\s*([\d.]+)', ref_range_str, re.IGNORECASE)
    if match_greater_than:
        try: return float(match_greater_than.group(1)), None, "greater_than"
        except ValueError: pass
    
    if ref_range_str.lower() in ["negative", "non reactive", "not detected"]:
        return None, None, "qualitative_normal"
    if ref_range_str.lower() in ["positive", "reactive", "detected"]:
        return None, None, "qualitative_abnormal"
        
    return None, None, None

def generate_test_plot(df_report, selected_test_name, selected_date=None):
    test_data_for_plot = df_report[df_report['Test_Name'] == selected_test_name]
    
    if selected_date and selected_date != "All Dates":
        test_data_for_plot = test_data_for_plot[test_data_for_plot['Test_Date'] == selected_date]
    
    if test_data_for_plot.empty:
        st.warning(f"No data found for test: {selected_test_name}" + (f" on {selected_date}" if selected_date and selected_date != "All Dates" else ""))
        return None

    if "All Dates" == selected_date and len(test_data_for_plot['Test_Date_dt'].unique()) > 1:
        fig = go.Figure()
        test_data_for_plot = test_data_for_plot.sort_values('Test_Date_dt')
        
        if pd.to_numeric(test_data_for_plot['Result'], errors='coerce').notna().all():
            # Format dates for display using DD-MM-YYYY
            test_data_for_plot['Date_Display'] = test_data_for_plot['Test_Date_dt'].apply(format_date_dd_mm_yyyy)
            
            fig.add_trace(go.Scatter(
                x=test_data_for_plot['Date_Display'],
                y=test_data_for_plot['Result_Numeric'],
                mode='lines+markers',
                name='Result Trend',
                line=dict(width=3),
                marker=dict(size=8)
            ))
            
            latest_entry = test_data_for_plot.iloc[-1]
            low_ref, high_ref, ref_type = parse_reference_range(latest_entry['Reference_Range'])
            
            data_min = test_data_for_plot['Result_Numeric'].min()
            data_max = test_data_for_plot['Result_Numeric'].max()
            data_range = data_max - data_min
            
            if ref_type == "range" and low_ref is not None and high_ref is not None:
                fig.add_hline(y=high_ref, line_dash="dash", line_color="red", annotation_text="Upper Reference", annotation_position="bottom right")
                fig.add_hline(y=low_ref, line_dash="dash", line_color="green", annotation_text="Lower Reference", annotation_position="top right")
                fig.add_hrect(y0=low_ref, y1=high_ref, fillcolor="green", opacity=0.1, annotation_text="Normal Range", annotation_position="top left")
                y_min = min(data_min, low_ref) - abs(data_range * 0.1 if data_range > 0 else low_ref * 0.1)
                y_max = max(data_max, high_ref) + abs(data_range * 0.1 if data_range > 0 else high_ref * 0.1)
            elif ref_type == "less_than" and high_ref is not None:
                fig.add_hline(y=high_ref, line_dash="dash", line_color="red", annotation_text=f"< {high_ref}", annotation_position="bottom right")
                y_min = min(data_min, 0) - abs(data_range * 0.1 if data_range > 0 else data_min * 0.1)
                y_max = max(data_max, high_ref) + abs(data_range * 0.1 if data_range > 0 else high_ref * 0.1)
            elif ref_type == "greater_than" and low_ref is not None:
                fig.add_hline(y=low_ref, line_dash="dash", line_color="green", annotation_text=f"> {low_ref}", annotation_position="top right")
                y_min = min(data_min, low_ref) - abs(data_range * 0.1 if data_range > 0 else low_ref * 0.1)
                y_max = max(data_max, data_max * 1.2) + abs(data_range * 0.1 if data_range > 0 else data_max * 0.1)
            else:
                padding = abs(data_range * 0.15 if data_range > 0 else data_max * 0.15)
                y_min = data_min - padding
                y_max = data_max + padding
            
            unit = latest_entry['Unit']
            fig.update_layout(
                title_text=f"{selected_test_name} Trend ({unit})",
                xaxis_title="Date",
                yaxis_title=f"Result ({unit})",
                yaxis=dict(range=[y_min, y_max]),
                height=500,
                margin=dict(l=20, r=20, t=60, b=80),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                autosize=True,
                xaxis=dict(tickangle=45, tickfont=dict(size=10))
            )
            return fig
        else:
            st.info(f"Cannot plot trend for '{selected_test_name}' as some results are non-numeric.")
            return None

    test_entry = test_data_for_plot.sort_values('Test_Date_dt', ascending=False).iloc[0]
    result_val_numeric = test_entry['Result_Numeric']
    result_val_str = test_entry['Result']
    ref_range_str = test_entry['Reference_Range']
    unit = test_entry['Unit']
    status = test_entry['Status']

    if pd.isna(result_val_numeric):
        st.info(f"Result for '{selected_test_name}' is '{result_val_str}' (non-numeric). Status: {status}")
        return None

    fig = go.Figure()
    
    low_ref, high_ref, ref_type = parse_reference_range(ref_range_str)
    
    if ref_type == "range" and low_ref is not None and high_ref is not None:
        ref_range_span = high_ref - low_ref
        axis_min = min(low_ref - ref_range_span * 0.2, result_val_numeric - ref_range_span * 0.2)
        axis_max = max(high_ref + ref_range_span * 0.2, result_val_numeric + ref_range_span * 0.2)
        steps = [
            {'range': [axis_min, low_ref], 'color': 'lightcoral'},
            {'range': [low_ref, high_ref], 'color': 'lightgreen'},
            {'range': [high_ref, axis_max], 'color': 'lightcoral'}
        ]
    elif ref_type == "less_than" and high_ref is not None:
        axis_min = 0
        axis_max = max(high_ref * 1.5, result_val_numeric * 1.2)
        steps = [
            {'range': [axis_min, high_ref], 'color': 'lightgreen'},
            {'range': [high_ref, axis_max], 'color': 'lightcoral'}
        ]
    elif ref_type == "greater_than" and low_ref is not None:
        axis_min = min(low_ref * 0.5, result_val_numeric * 0.8)
        axis_max = max(low_ref * 1.5, result_val_numeric * 1.2)
        steps = [
            {'range': [axis_min, low_ref], 'color': 'lightcoral'},
            {'range': [low_ref, axis_max], 'color': 'lightgreen'}
        ]
    else:
        value_range = abs(result_val_numeric * 0.5) if result_val_numeric != 0 else 10
        axis_min = result_val_numeric - value_range
        axis_max = result_val_numeric + value_range
        steps = [
            {'range': [axis_min, axis_max], 'color': 'lightblue'}
        ]

    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=result_val_numeric,
        domain={'x': [0.1, 1], 'y': [0.2, 0.9]},
        number={'suffix': f" {unit}", 'font': {'size': 20}, 'valueformat': '.2f'},
        title={
            'text': f"<br>{selected_test_name}<br><span style='font-size:0.8em;color:gray'>Reference: {ref_range_str}<br>Status: {status}</span>",
            'font': {"size": 12},
            'align': "center"
        },
        gauge={
            'shape': "bullet",
            'axis': {'range': [axis_min, axis_max]},
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.75,
                'value': result_val_numeric
            },
            'steps': steps,
            'bar': {'color': "rgba(0,0,0,0)"}
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        autosize=True
    )
    return fig

# Replace the existing function in your Medical_Project.py file with this enhanced version
def create_enhanced_excel_with_trends(organized_df, ref_range_df, date_lab_cols_sorted, patient_info):
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        # Write the data without headers first, starting from row 4 (index 3)
        organized_df.to_excel(writer, index=False, sheet_name='Medical Data with Trends', startrow=3, header=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Medical Data with Trends']
        
        # Title format
        title_format = workbook.add_format({'bold': True, 'font_size': 16, 'align': 'center', 'bg_color': '#4472C4', 'font_color': 'white'})
        worksheet.merge_range('A1:' + chr(65 + len(organized_df.columns) - 1) + '1', 'Medical Test Results - Organized by Date', title_format)
        
        # Header formats
        date_format = workbook.add_format({'bold': True, 'bg_color': '#E7E6E6', 'border': 1, 'align': 'center', 'font_color': '#2E5C8F'})
        lab_format = workbook.add_format({'bold': True, 'bg_color': '#F0F8FF', 'border': 1, 'align': 'center', 'font_color': '#1E7B3E', 'italic': True})
        empty_format = workbook.add_format({'border': 1, 'bg_color': '#FFFFFF'})
        
        # Write date row (row 2)
        worksheet.write(1, 0, '📅 Date', date_format)
        worksheet.write(1, 1, '', date_format)
        
        # Write lab row (row 3)
        worksheet.write(2, 0, '🏥 Lab', lab_format)
        worksheet.write(2, 1, '', lab_format)
        
        # Fill in date and lab information for data columns
        for i, date_lab_col in enumerate(date_lab_cols_sorted):
            col_idx = i + 2
            parts = date_lab_col.split('_', 1)
            date_part = parts[0] if len(parts) > 0 else 'N/A'
            lab_part = parts[1] if len(parts) > 1 else 'N/A'
            worksheet.write(1, col_idx, date_part, date_format)
            worksheet.write(2, col_idx, lab_part, lab_format)
        
        # Add reference ranges column
        ref_col_idx = len(organized_df.columns)
        worksheet.write(1, ref_col_idx, '', empty_format)  # Empty instead of date
        worksheet.write(2, ref_col_idx, '', empty_format)  # Empty instead of lab
        
        # Write custom column headers (row 4) - clean headers without the date_lab combinations
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D9E2F3', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        
        # Write the first two column headers manually
        worksheet.write(3, 0, 'Test_Category', header_format)
        worksheet.write(3, 1, 'Test_Name', header_format)
        
        # Write empty headers for the data columns (instead of the date_lab combinations)
        for i, date_lab_col in enumerate(date_lab_cols_sorted):
            col_idx = i + 2
            worksheet.write(3, col_idx, '', header_format)  # Empty header instead of date_lab combination
        
        # Add "Reference Ranges" header
        worksheet.write(3, ref_col_idx, "Reference Ranges", header_format)
        
        # Data formatting
        data_format = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
        numeric_format = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter', 'num_format': '0.00'})
        ref_format = workbook.add_format({'border': 1, 'align': 'left', 'valign': 'vcenter', 'text_wrap': True})
        
        # Write data manually with proper formatting
        for row_num in range(len(organized_df)):
            for col_num in range(len(organized_df.columns)):
                value = organized_df.iloc[row_num, col_num]
                if col_num < 2:  # Category and Test Name columns
                    worksheet.write(row_num + 4, col_num, value, data_format)
                else:  # Data columns
                    try:
                        if pd.notna(value) and str(value).strip() != '':
                            # Try to convert to float for proper numeric storage
                            float_val = float(value)
                            worksheet.write_number(row_num + 4, col_num, float_val, numeric_format)
                        else:
                            worksheet.write(row_num + 4, col_num, value if pd.notna(value) else '', data_format)
                    except (ValueError, TypeError):
                        # If conversion fails, write as text
                        worksheet.write(row_num + 4, col_num, str(value) if pd.notna(value) else '', data_format)
        
        # Write reference ranges for each row
        for row_num in range(len(organized_df)):
            # Add reference ranges from ref_range_df
            try:
                # Get all non-null reference ranges for this test across all dates
                ref_values = []
                for col_idx, col_name in enumerate(date_lab_cols_sorted):
                    if col_idx + 2 < len(ref_range_df.columns):
                        ref_value = ref_range_df.iloc[row_num, col_idx + 2]
                        if pd.notna(ref_value) and str(ref_value).strip() != '' and str(ref_value) != 'N/A':
                            ref_values.append(str(ref_value))
                
                # Use the most common reference range or the first valid one
                if ref_values:
                    from collections import Counter
                    most_common_ref = Counter(ref_values).most_common(1)[0][0]
                    worksheet.write(row_num + 4, ref_col_idx, most_common_ref, ref_format)
                else:
                    worksheet.write(row_num + 4, ref_col_idx, '', ref_format)
                    
            except Exception as e:
                worksheet.write(row_num + 4, ref_col_idx, '', ref_format)
        
        # Set row heights
        worksheet.set_row(1, 25)
        worksheet.set_row(2, 25)
        worksheet.set_row(3, 35)
        for row_num in range(len(organized_df)):
            worksheet.set_row(row_num + 4, 25)
        
        # Set column widths
        worksheet.set_column('A:A', 20)  # Test Category
        worksheet.set_column('B:B', 25)  # Test Name
        for i, col in enumerate(date_lab_cols_sorted):
            worksheet.set_column(i + 2, i + 2, 12)  # Data columns
        worksheet.set_column(ref_col_idx, ref_col_idx, 20)  # Reference ranges column
        
        # Add autofilter and freeze panes
        worksheet.autofilter(3, 0, len(organized_df) + 3, len(organized_df.columns))
        worksheet.freeze_panes(4, 2)
        
        # =================== NEW CHARTS TAB ===================
        # Create a new worksheet for individual test charts
        chart_worksheet = workbook.add_worksheet('Test Trend Charts')
        
        # Add title to charts sheet
        chart_worksheet.merge_range('A1:H1', 'Individual Test Trend Charts', title_format)
        chart_worksheet.write('A2', f'Patient: {patient_info.get("name", "N/A")} | Age: {patient_info.get("age", "N/A")} | Gender: {patient_info.get("gender", "N/A")}', 
                             workbook.add_format({'font_size': 12, 'italic': True}))
        
        # Prepare data for charts
        chart_row = 4  # Starting row for charts
        charts_per_row = 2  # Number of charts per row
        chart_width = 480  # Width of each chart in pixels
        chart_height = 300  # Height of each chart in pixels
        chart_spacing_x = 8  # Column spacing between charts (in Excel columns)
        chart_spacing_y = 20  # Row spacing between charts (in Excel rows)
        
        chart_count = 0
        
        # Extract dates for x-axis (parse them properly for sorting)
        dates_for_charts = []
        for date_lab_col in date_lab_cols_sorted:
            date_part = date_lab_col.split('_')[0]
            dates_for_charts.append(date_part)
        
        # Group tests by category for better organization
        tests_by_category = {}
        for idx, row in organized_df.iterrows():
            category = row['Test_Category']
            test_name = row['Test_Name']
            if category not in tests_by_category:
                tests_by_category[category] = []
            tests_by_category[category].append((idx, test_name))
        
        # Create charts for each test that has numeric data
        for category, tests in tests_by_category.items():
            for test_idx, test_name in tests:
                # Get the data for this test
                test_data = []
                has_numeric_data = False
                
                for col_idx, date_lab_col in enumerate(date_lab_cols_sorted):
                    value = organized_df.iloc[test_idx, col_idx + 2]  # +2 to skip category and test name columns
                    try:
                        if pd.notna(value) and str(value).strip() != '':
                            numeric_value = float(value)
                            test_data.append(numeric_value)
                            has_numeric_data = True
                        else:
                            test_data.append(None)
                    except (ValueError, TypeError):
                        test_data.append(None)
                
                # Only create chart if there's numeric data and more than one data point
                valid_data_points = [x for x in test_data if x is not None]
                if has_numeric_data and len(valid_data_points) > 1:
                    # Calculate chart position
                    row_position = chart_count // charts_per_row
                    col_position = chart_count % charts_per_row
                    
                    chart_start_row = chart_row + (row_position * chart_spacing_y)
                    chart_start_col = col_position * chart_spacing_x
                    
                    # Create a line chart
                    chart = workbook.add_chart({'type': 'line'})
                    
                    # Create a temporary data table on the chart sheet for this specific test
                    data_start_row = chart_start_row + 16  # Place data below the chart
                    
                    # Write dates header
                    chart_worksheet.write(data_start_row, chart_start_col, 'Date', 
                                        workbook.add_format({'bold': True, 'bg_color': '#D9E2F3'}))
                    chart_worksheet.write(data_start_row + 1, chart_start_col, test_name, 
                                        workbook.add_format({'bold': True, 'bg_color': '#D9E2F3'}))
                    
                    # Write dates and values - only write actual data points (no empty cells)
                    valid_dates = []
                    valid_values = []
                    date_col_positions = []
                    
                    for i, (date, value) in enumerate(zip(dates_for_charts, test_data)):
                        if value is not None:  # Only include dates with actual values
                            valid_dates.append(date)
                            valid_values.append(value)
                            date_col_positions.append(chart_start_col + 1 + len(valid_dates) - 1)
                    
                    # Write only the valid dates and values (no gaps)
                    for i, (date, value) in enumerate(zip(valid_dates, valid_values)):
                        chart_worksheet.write(data_start_row, chart_start_col + 1 + i, date)
                        chart_worksheet.write_number(data_start_row + 1, chart_start_col + 1 + i, value)
                    
                    # Add data series to chart - only for valid data points
                    chart.add_series({
                        'name': test_name,
                        'categories': ['Test Trend Charts', data_start_row, chart_start_col + 1, 
                                     data_start_row, chart_start_col + len(valid_dates)],
                        'values': ['Test Trend Charts', data_start_row + 1, chart_start_col + 1, 
                                 data_start_row + 1, chart_start_col + len(valid_dates)],
                        'line': {'color': '#4472C4', 'width': 2},
                        'marker': {'type': 'circle', 'size': 6, 'border': {'color': '#4472C4'}, 'fill': {'color': '#4472C4'}},
                    })
                    
                    # Get reference range for this test and parse it
                    ref_range = ''
                    low_ref = None
                    high_ref = None
                    ref_type = None
                    
                    try:
                        for col_idx, col_name in enumerate(date_lab_cols_sorted):
                            if col_idx + 2 < len(ref_range_df.columns):
                                ref_value = ref_range_df.iloc[test_idx, col_idx + 2]
                                if pd.notna(ref_value) and str(ref_value).strip() != '' and str(ref_value) != 'N/A':
                                    ref_range = str(ref_value)
                                    break
                    except:
                        pass
                    
                    # Parse reference range to extract numeric values
                    if ref_range:
                        import re
                        # Try to parse range like "13.0 - 17.0" or "13.0-17.0"
                        match_range = re.search(r'([\d.]+)\s*-\s*([\d.]+)', ref_range)
                        if match_range:
                            try:
                                low_ref = float(match_range.group(1))
                                high_ref = float(match_range.group(2))
                                ref_type = "range"
                            except ValueError:
                                pass
                        else:
                            # Try to parse "< 5.0" or "Less than 5.0"
                            match_less_than = re.search(r'(?:<|Less than|upto)\s*([\d.]+)', ref_range, re.IGNORECASE)
                            if match_less_than:
                                try:
                                    high_ref = float(match_less_than.group(1))
                                    ref_type = "less_than"
                                except ValueError:
                                    pass
                            else:
                                # Try to parse "> 5.0" or "Greater than 5.0"
                                match_greater_than = re.search(r'(?:>|Greater than|above)\s*([\d.]+)', ref_range, re.IGNORECASE)
                                if match_greater_than:
                                    try:
                                        low_ref = float(match_greater_than.group(1))
                                        ref_type = "greater_than"
                                    except ValueError:
                                        pass
                    
                    # Add reference range lines if we have valid numeric ranges
                    if ref_type == "range" and low_ref is not None and high_ref is not None:
                        # Create data for reference lines (same valid dates, constant values)
                        ref_low_row = data_start_row + 2
                        ref_high_row = data_start_row + 3
                        
                        # Write reference line labels
                        chart_worksheet.write(ref_low_row, chart_start_col, 'Lower Limit', 
                                            workbook.add_format({'bold': True, 'font_color': '#228B22'}))
                        chart_worksheet.write(ref_high_row, chart_start_col, 'Upper Limit', 
                                            workbook.add_format({'bold': True, 'font_color': '#DC143C'}))
                        
                        # Write reference values ONLY for valid dates (same pattern as main data)
                        for i in range(len(valid_dates)):
                            chart_worksheet.write_number(ref_low_row, chart_start_col + 1 + i, low_ref)
                            chart_worksheet.write_number(ref_high_row, chart_start_col + 1 + i, high_ref)
                        
                        # Add lower reference line series
                        chart.add_series({
                            'name': f'Lower Limit ({low_ref})',
                            'categories': ['Test Trend Charts', data_start_row, chart_start_col + 1, 
                                         data_start_row, chart_start_col + len(valid_dates)],
                            'values': ['Test Trend Charts', ref_low_row, chart_start_col + 1, 
                                     ref_low_row, chart_start_col + len(valid_dates)],
                            'line': {'color': '#228B22', 'width': 2, 'dash_type': 'dash'},
                            'marker': {'type': 'none'},
                        })
                        
                        # Add upper reference line series
                        chart.add_series({
                            'name': f'Upper Limit ({high_ref})',
                            'categories': ['Test Trend Charts', data_start_row, chart_start_col + 1, 
                                         data_start_row, chart_start_col + len(valid_dates)],
                            'values': ['Test Trend Charts', ref_high_row, chart_start_col + 1, 
                                     ref_high_row, chart_start_col + len(valid_dates)],
                            'line': {'color': '#DC143C', 'width': 2, 'dash_type': 'dash'},
                            'marker': {'type': 'none'},
                        })
                        
                    elif ref_type == "less_than" and high_ref is not None:
                        # Add only upper limit line
                        ref_high_row = data_start_row + 2
                        
                        chart_worksheet.write(ref_high_row, chart_start_col, f'Upper Limit (<{high_ref})', 
                                            workbook.add_format({'bold': True, 'font_color': '#DC143C'}))
                        
                        # Write reference values ONLY for valid dates
                        for i in range(len(valid_dates)):
                            chart_worksheet.write_number(ref_high_row, chart_start_col + 1 + i, high_ref)
                        
                        chart.add_series({
                            'name': f'Upper Limit (<{high_ref})',
                            'categories': ['Test Trend Charts', data_start_row, chart_start_col + 1, 
                                         data_start_row, chart_start_col + len(valid_dates)],
                            'values': ['Test Trend Charts', ref_high_row, chart_start_col + 1, 
                                     ref_high_row, chart_start_col + len(valid_dates)],
                            'line': {'color': '#DC143C', 'width': 2, 'dash_type': 'dash'},
                            'marker': {'type': 'none'},
                        })
                        
                    elif ref_type == "greater_than" and low_ref is not None:
                        # Add only lower limit line
                        ref_low_row = data_start_row + 2
                        
                        chart_worksheet.write(ref_low_row, chart_start_col, f'Lower Limit (>{low_ref})', 
                                            workbook.add_format({'bold': True, 'font_color': '#228B22'}))
                        
                        # Write reference values ONLY for valid dates
                        for i in range(len(valid_dates)):
                            chart_worksheet.write_number(ref_low_row, chart_start_col + 1 + i, low_ref)
                        
                        chart.add_series({
                            'name': f'Lower Limit (>{low_ref})',
                            'categories': ['Test Trend Charts', data_start_row, chart_start_col + 1, 
                                         data_start_row, chart_start_col + len(valid_dates)],
                            'values': ['Test Trend Charts', ref_low_row, chart_start_col + 1, 
                                     ref_low_row, chart_start_col + len(valid_dates)],
                            'line': {'color': '#228B22', 'width': 2, 'dash_type': 'dash'},
                            'marker': {'type': 'none'},
                        })
                    
                    # Adjust Y-axis range to include reference ranges
                    min_val = min(valid_data_points)
                    max_val = max(valid_data_points)
                    
                    # Include reference range values in axis calculation
                    all_values = valid_data_points[:]
                    if low_ref is not None:
                        all_values.append(low_ref)
                    if high_ref is not None:
                        all_values.append(high_ref)
                    
                    final_min = min(all_values)
                    final_max = max(all_values)
                    value_range = final_max - final_min if final_max != final_min else final_max * 0.1
                    y_min = final_min - (value_range * 0.15) if value_range > 0 else final_min * 0.85
                    y_max = final_max + (value_range * 0.15) if value_range > 0 else final_max * 1.15
                    
                    # Configure chart
                    chart.set_title({
                        'name': f'{test_name}\n({category})',
                        'name_font': {'size': 11, 'bold': True}
                    })
                    chart.set_x_axis({
                        'name': 'Test Date',
                        'name_font': {'size': 10},
                        'num_font': {'size': 9, 'rotation': 45}
                    })
                    chart.set_y_axis({
                        'name': f'Value {ref_range}' if ref_range else 'Value',
                        'name_font': {'size': 10},
                        'min': y_min,
                        'max': y_max
                    })
                    
                    # Show legend if we have reference ranges
                    if ref_type in ["range", "less_than", "greater_than"]:
                        chart.set_legend({
                            'position': 'bottom',
                            'font': {'size': 8}
                        })
                    else:
                        chart.set_legend({'none': True})
                        
                    chart.set_size({'width': chart_width, 'height': chart_height})
                    chart.set_style(10)  # Use a clean style
                    
                    # Insert chart
                    chart_worksheet.insert_chart(chart_start_row, chart_start_col, chart)
                    
                    chart_count += 1
        
        # Add summary information to charts sheet
        if chart_count > 0:
            summary_row = 2 + ((chart_count // charts_per_row + 1) * chart_spacing_y)
            chart_worksheet.write(summary_row, 0, f'Total Charts Generated: {chart_count}', 
                                workbook.add_format({'bold': True, 'font_size': 12}))
            chart_worksheet.write(summary_row + 1, 0, 'Note: Only tests with multiple numeric values are charted', 
                                workbook.add_format({'italic': True, 'font_color': '#666666'}))
        else:
            chart_worksheet.write(4, 0, 'No charts could be generated - insufficient numeric data points', 
                                workbook.add_format({'font_color': 'red', 'bold': True}))
        
        # =================== END CHARTS TAB ===================
        
        # Create summary sheet
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.merge_range('A1:D1', 'Medical Report Summary', title_format)
        
        info_format = workbook.add_format({'bold': True, 'bg_color': '#E7E6E6'})
        
        # Patient information
        summary_sheet.write('A3', 'Patient Information:', info_format)
        summary_sheet.write('A4', f"Name: {patient_info.get('name', 'N/A')}")
        summary_sheet.write('A5', f"Age: {patient_info.get('age', 'N/A')}")
        summary_sheet.write('A6', f"Gender: {patient_info.get('gender', 'N/A')}")
        summary_sheet.write('A7', f"Patient ID: {patient_info.get('patient_id', 'N/A')}")
        summary_sheet.write('A8', f"Primary Lab: {patient_info.get('lab_name', 'N/A')}")
        
        # Report statistics
        summary_sheet.write('A10', 'Report Statistics:', info_format)
        summary_sheet.write('A11', f"Total Test Categories: {organized_df['Test_Category'].nunique()}")
        summary_sheet.write('A12', f"Total Tests: {len(organized_df)}")
        summary_sheet.write('A13', f"Total Date-Lab Combinations: {len(date_lab_cols_sorted)}")
        summary_sheet.write('A14', f"Charts Generated: {chart_count}")
        
        if date_lab_cols_sorted:
            first_date = date_lab_cols_sorted[0].split('_')[0]
            last_date = date_lab_cols_sorted[-1].split('_')[0]
            summary_sheet.write('A15', f"Date Range: {first_date} to {last_date}")
        
        summary_sheet.write('A16', f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
        
        # Test categories
        summary_sheet.write('A18', 'Test Categories:', info_format)
        categories = organized_df['Test_Category'].value_counts()
        for i, (category, count) in enumerate(categories.items()):
            summary_sheet.write(f'A{19+i}', f"• {category}: {count} tests")
        
        # Labs used
        summary_sheet.write('A' + str(19 + len(categories) + 2), 'Labs Used:', info_format)
        unique_labs = set()
        for col in date_lab_cols_sorted:
            if '_' in col:
                lab_part = col.split('_', 1)[1]
                unique_labs.add(lab_part)
        
        for i, lab in enumerate(sorted(unique_labs)):
            summary_sheet.write(f'A{19 + len(categories) + 3 + i}', f"• {lab}")

    return output_excel.getvalue()

    
def smart_consolidate_patient_info(existing_patient_info, new_patient_info_list):
    """
    Intelligently consolidate patient info from existing Excel and new PDFs
    Priority: New PDF data > Existing Excel data (as PDFs are more detailed and recent)
    """
    
    if not new_patient_info_list and not existing_patient_info:
        return {}
    
    if not new_patient_info_list:
        return existing_patient_info
    
    if not existing_patient_info:
        # Use your existing consolidate_patient_info function that's already in your file
        return consolidate_patient_info(new_patient_info_list)
    
    # Both exist - merge intelligently
    consolidated = {}
    
    # Name: prefer from new PDFs if available, otherwise use existing
    new_names = [pi.get('name', '') for pi in new_patient_info_list if pi.get('name') and pi.get('name') != 'N/A']
    if new_names:
        # Use your existing smart name selection logic
        normalized_names = [normalize_name(name) for name in new_names if name]
        if normalized_names:
            name_counts = Counter(normalized_names)
            max_count = max(name_counts.values())
            most_frequent_names = [name for name, count in name_counts.items() if count == max_count]
            consolidated['name'] = max(most_frequent_names, key=len) if most_frequent_names else normalized_names[0]
        else:
            consolidated['name'] = existing_patient_info.get('name', 'N/A')
    else:
        consolidated['name'] = existing_patient_info.get('name', 'N/A')
    
    # Age: prefer from most recent PDF, otherwise use existing
    new_ages = [(pi.get('date', ''), pi.get('age', '')) for pi in new_patient_info_list if pi.get('age') and pi.get('age') != 'N/A']
    if new_ages:
        # Sort by date and get most recent age
        date_age_pairs = []
        for date_str, age in new_ages:
            if date_str:
                parsed_date = parse_date_dd_mm_yyyy(date_str)
                if parsed_date:
                    date_age_pairs.append((parsed_date, age))
        
        if date_age_pairs:
            date_age_pairs.sort(key=lambda x: x[0], reverse=True)
            consolidated['age'] = date_age_pairs[0][1]
        else:
            consolidated['age'] = new_ages[0][1]  # Use first available
    else:
        consolidated['age'] = existing_patient_info.get('age', 'N/A')
    
    # Gender: prefer from new PDFs
    new_genders = [pi.get('gender', '') for pi in new_patient_info_list if pi.get('gender') and pi.get('gender') != 'N/A']
    consolidated['gender'] = Counter(new_genders).most_common(1)[0][0] if new_genders else existing_patient_info.get('gender', 'N/A')
    
    # Patient ID: prefer from new PDFs
    new_ids = [pi.get('patient_id', '') for pi in new_patient_info_list if pi.get('patient_id') and pi.get('patient_id') != 'N/A']
    consolidated['patient_id'] = Counter(new_ids).most_common(1)[0][0] if new_ids else existing_patient_info.get('patient_id', 'N/A')
    
    # Lab name: combine both (could be different labs over time)
    new_labs = [pi.get('lab_name', '') for pi in new_patient_info_list if pi.get('lab_name') and pi.get('lab_name') != 'N/A']
    all_labs = new_labs + ([existing_patient_info.get('lab_name', '')] if existing_patient_info.get('lab_name') and existing_patient_info.get('lab_name') != 'N/A' else [])
    consolidated['lab_name'] = Counter(all_labs).most_common(1)[0][0] if all_labs else 'N/A'
    
    # Date: use most recent from all sources
    all_dates = []
    for pi in new_patient_info_list:
        if pi.get('date') and pi.get('date') != 'N/A':
            all_dates.append(pi.get('date'))
    if existing_patient_info.get('date') and existing_patient_info.get('date') != 'N/A':
        all_dates.append(existing_patient_info.get('date'))
    
    if all_dates:
        parsed_dates = [parse_date_dd_mm_yyyy(d) for d in all_dates]
        valid_dates = [d for d in parsed_dates if d is not None]
        consolidated['date'] = format_date_dd_mm_yyyy(max(valid_dates)) if valid_dates else 'N/A'
    else:
        consolidated['date'] = 'N/A'
    
    return consolidated

def combine_duplicate_tests(df):
    df = df.copy()
    test_cat_counts = df.groupby(['Test_Name', 'Test_Category'])['Test_Date'].nunique().reset_index().rename(columns={'Test_Date': 'date_count'})
    test_cat_total = df.groupby(['Test_Name', 'Test_Category']).size().reset_index(name='row_count')
    merged = pd.merge(test_cat_counts, test_cat_total, on=['Test_Name', 'Test_Category'])

    # Pick best category per test by max distinct dates, then max rows, then stable name order.
    ranked = merged.sort_values(
        by=['Test_Name', 'date_count', 'row_count', 'Test_Category'],
        ascending=[True, False, False, True],
    )
    best_cats = ranked.drop_duplicates(subset=['Test_Name'])[['Test_Name', 'Test_Category']]
    best_cats = best_cats.rename(columns={'Test_Category': 'Best_Test_Category'})
    df = pd.merge(df, best_cats, on='Test_Name', how='left')
    df['Test_Category'] = df['Best_Test_Category']
    df = df.drop(columns=['Best_Test_Category'])
    df = df.sort_values(['Test_Name', 'Test_Date', 'Test_Category']).drop_duplicates(['Test_Name', 'Test_Date'])
    df = df.reset_index(drop=True)
    return df

def validate_data_consistency(df):
    """Validate the consistency of the combined dataset"""
    issues = []
    
    if df.empty:
        return ["No data available"]
    
    # Check for missing critical fields
    critical_fields = ['Test_Name', 'Result', 'Test_Date']
    for field in critical_fields:
        missing_count = df[field].isna().sum() + (df[field] == 'N/A').sum()
        if missing_count > 0:
            issues.append(f"Missing {field} in {missing_count} records")
    
    # Check date consistency
    date_issues = df[df['Test_Date_dt'].isna() & (df['Test_Date'] != 'N/A')]
    if not date_issues.empty:
        issues.append(f"Invalid date formats in {len(date_issues)} records")
    
    # Check for duplicate test entries on same date
    duplicates = df.groupby(['Test_Name', 'Test_Date']).size()
    duplicate_count = (duplicates > 1).sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate test entries for same dates")
    
    return issues

def display_combination_summary(existing_count, new_count, total_count, patient_info):
    """Display a nice summary of the data combination process"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📋 Existing Records", 
            value=existing_count,
            help="Records from uploaded Excel/CSV file"
        )
    
    with col2:
        st.metric(
            label="📄 New PDF Records", 
            value=new_count,
            help="Records extracted from new PDF reports"
        )
    
    with col3:
        st.metric(
            label="📊 Total Records", 
            value=total_count,
            delta=f"+{new_count}" if new_count > 0 else None,
            help="Combined total records available"
        )
    
    # Show patient info summary
    if patient_info:
        st.info(f"👤 **Patient**: {patient_info.get('name', 'N/A')} | "
                f"📅 **Latest Date**: {patient_info.get('date', 'N/A')} | "
                f"🏥 **Primary Lab**: {patient_info.get('lab_name', 'N/A')}")
def safe_file_processing(file_processor_func, file_obj, *args, **kwargs):
    """Safely process files with comprehensive error handling"""
    try:
        return file_processor_func(file_obj, *args, **kwargs)
    except pd.errors.EmptyDataError:
        st.error(f"❌ File {file_obj.name} appears to be empty or corrupted")
        return pd.DataFrame(), {}
    except pd.errors.ParserError as e:
        st.error(f"❌ Could not parse file {file_obj.name}: {str(e)}")
        return pd.DataFrame(), {}
    except Exception as e:
        st.error(f"❌ Unexpected error processing {file_obj.name}: {str(e)}")
        return pd.DataFrame(), {}


def handle_patient_info_conflicts(existing_info, new_info_list):
    """Handle conflicts in patient information with user-friendly messages"""
    
    if not existing_info and not new_info_list:
        return {}
    
    # Collect all names for comparison
    all_names = []
    if existing_info.get('name') and existing_info.get('name') != 'N/A':
        all_names.append(existing_info.get('name'))
    
    for info in new_info_list:
        if info.get('name') and info.get('name') != 'N/A':
            all_names.append(info.get('name'))
    
    # Check for name conflicts
    normalized_names = [normalize_name(name) for name in all_names]
    unique_names = set(normalized_names)
    
    if len(unique_names) > 1:
        st.warning("⚠️ **Patient Name Conflict Detected**")
        
        with st.expander("🔍 View Name Conflict Details", expanded=False):
            st.write("**Names found across all sources:**")
            for i, name in enumerate(set(all_names), 1):
                st.write(f"{i}. {name}")
            
            st.write("**Normalized names:**")
            for i, name in enumerate(unique_names, 1):
                st.write(f"{i}. {name}")
        
        # Use smart consolidation
        consolidated = smart_consolidate_patient_info(existing_info, new_info_list)
        
        st.success(f"✅ **Auto-resolved to**: {consolidated.get('name', 'N/A')}")
        st.info("💡 The system selected the most complete and frequently occurring name")
        
        return consolidated
    
    else:
        # No conflicts, proceed with normal consolidation
        return smart_consolidate_patient_info(existing_info, new_info_list)

def enhanced_data_combination_workflow(uploaded_excel_file, new_pdf_data_list, new_patient_info_list):
    """
    FIXED: Ensure both Excel and PDF data are properly combined
    """
    
    # Process existing Excel/CSV file
    existing_df, existing_patient_info = process_existing_excel_csv(uploaded_excel_file, new_patient_info_list)
    
    print(f"Existing data: {len(existing_df)} rows")
    print(f"New PDF data: {len(new_pdf_data_list)} DataFrames")
    
    # Start with existing data
    if not existing_df.empty:
        combined_df = existing_df.copy()
    else:
        combined_df = pd.DataFrame()
    
    # Add new PDF data
    if new_pdf_data_list:
        new_data_df = pd.concat(new_pdf_data_list, ignore_index=True)
        
        if not combined_df.empty:
            # Make sure columns match before combining
            all_cols = list(set(combined_df.columns.tolist() + new_data_df.columns.tolist()))
            
            # Add missing columns
            for col in all_cols:
                if col not in combined_df.columns:
                    combined_df[col] = 'N/A'
                if col not in new_data_df.columns:
                    new_data_df[col] = 'N/A'
            
            # Reorder columns and combine
            combined_df = combined_df[all_cols]
            new_data_df = new_data_df[all_cols]
            combined_df = pd.concat([combined_df, new_data_df], ignore_index=True)
        else:
            combined_df = new_data_df
    
    # Consolidate patient info
    final_patient_info = smart_consolidate_patient_info(existing_patient_info, new_patient_info_list)
    
    print(f"Final combined data: {len(combined_df)} rows")
    
    return combined_df, final_patient_info

def process_normalized_excel_data(df, filename, new_patient_info_list):
    """
    FIXED: Process Excel data that's already in normalized format
    """
    print("Processing normalized Excel data...")
    
    # Ensure all required columns exist
    expected_columns = [
        'Source_Filename', 'Patient_ID', 'Patient_Name', 'Age', 'Gender', 
        'Test_Date', 'Lab_Name', 'Test_Category', 'Original_Test_Name', 
        'Test_Name', 'Result', 'Unit', 'Reference_Range', 'Status', 
        'Processed_Date', 'Result_Numeric', 'Test_Date_dt'
    ]
    
    # Add missing columns with default values
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 'N/A'
            print(f"Added missing column: {col}")
    
    # Update computed columns
    df['Source_Filename'] = filename
    df['Processed_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Fix numeric conversion for Result column
    df['Result_Numeric'] = pd.to_numeric(df['Result'], errors='coerce')
    
    # Fix date parsing
    df['Test_Date_dt'] = df['Test_Date'].apply(parse_date_dd_mm_yyyy)
    
    # Clean up the dataframe - remove completely empty rows
    df = df.dropna(subset=['Test_Name'], how='all')
    df = df[df['Test_Name'] != 'N/A']
    df = df[df['Test_Name'].notna()]
    
    print(f"Processed normalized data shape: {df.shape}")
    
    # Extract patient info from the existing data
    existing_patient_info = extract_patient_info_from_normalized_data(df)
    
    # Reorder columns to match expected format
    df = df[expected_columns]
    
    return df, existing_patient_info

def process_pivoted_excel_data(df, filename, new_patient_info_list):
    """
    FIXED: Process Excel data that's in pivoted format (organized by date)
    """
    print("Processing pivoted Excel data...")
    
    # Remove header rows if they exist (Date and Lab rows)
    original_shape = df.shape
    
    # Check for and remove header rows that contain dates or lab info
    rows_to_skip = 0
    for idx in range(min(3, len(df))):  # Check first 3 rows
        row_content = str(df.iloc[idx, 0]).lower() if not pd.isna(df.iloc[idx, 0]) else ""
        if any(indicator in row_content for indicator in ['📅', 'date', '🏥', 'lab']):
            rows_to_skip = idx + 1
    
    if rows_to_skip > 0:
        df = df.iloc[rows_to_skip:].reset_index(drop=True)
        print(f"Skipped {rows_to_skip} header rows. New shape: {df.shape}")
    
    # Identify date-lab columns (everything except Test_Category and Test_Name)
    non_date_cols = ['Test_Category', 'Test_Name']
    date_lab_cols = [col for col in df.columns if col not in non_date_cols and not col.startswith('Unnamed')]
    
    print(f"Found {len(date_lab_cols)} date-lab columns: {date_lab_cols[:5]}...")  # Show first 5
    
    if not date_lab_cols:
        print("No date columns found in pivoted data")
        return pd.DataFrame(), {}
    
    # Convert from pivoted to normalized format
    normalized_rows = []
    
    print(f"Processing {len(df)} test rows...")
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:  # Progress indicator
            print(f"Processing row {idx}/{len(df)}")
            
        test_category = row.get('Test_Category', 'N/A')
        test_name = row.get('Test_Name', 'N/A')
        
        # Skip if test name is not valid
        if pd.isna(test_name) or test_name in ['N/A', '', 'Test_Name'] or str(test_name).strip() == '':
            continue
        
        for date_lab_col in date_lab_cols:
            result_value = row.get(date_lab_col, '')
            
            # Skip empty results
            if pd.isna(result_value) or result_value in ['', 'N/A'] or str(result_value).strip() == '':
                continue
            
            # Parse date and lab from column name
            if '_' in str(date_lab_col):
                parts = str(date_lab_col).split('_', 1)
                test_date = parts[0] if len(parts) > 0 else 'N/A'
                lab_name = parts[1] if len(parts) > 1 else 'N/A'
            else:
                # If no underscore, assume it's just a date
                test_date = str(date_lab_col)
                lab_name = 'N/A'
            
            # Normalize the date format
            parsed_date = parse_date_dd_mm_yyyy(test_date)
            formatted_date = format_date_dd_mm_yyyy(parsed_date) if parsed_date else test_date
            
            # Determine patient info - use from new PDFs if available, otherwise extract from filename
            patient_name = 'N/A'
            patient_age = 'N/A'
            patient_gender = 'N/A'
            patient_id = 'N/A'
            
            if new_patient_info_list:
                # Use the most recent patient info from new PDFs
                latest_info = new_patient_info_list[-1]  # Assuming the last one is most recent
                patient_name = latest_info.get('name', extract_patient_info_from_excel_filename(filename))
                patient_age = latest_info.get('age', 'N/A')
                patient_gender = latest_info.get('gender', 'N/A')
                patient_id = latest_info.get('patient_id', 'N/A')
            else:
                patient_name = extract_patient_info_from_excel_filename(filename)
            
            normalized_row = {
                'Source_Filename': filename,
                'Patient_ID': patient_id,
                'Patient_Name': patient_name,
                'Age': patient_age,
                'Gender': patient_gender,
                'Test_Date': formatted_date,
                'Lab_Name': lab_name,
                'Test_Category': test_category,
                'Original_Test_Name': test_name,
                'Test_Name': test_name,  # Will be standardized later if needed
                'Result': str(result_value),
                'Unit': 'N/A',  # Not available in pivoted format
                'Reference_Range': 'N/A',  # Not available in pivoted format
                'Status': 'N/A',  # Not available in pivoted format
                'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Result_Numeric': pd.to_numeric(result_value, errors='coerce'),
                'Test_Date_dt': parsed_date
            }
            
            normalized_rows.append(normalized_row)
    
    print(f"Created {len(normalized_rows)} normalized rows")
    
    if not normalized_rows:
        print("No valid data found in pivoted format")
        return pd.DataFrame(), {}
    
    normalized_df = pd.DataFrame(normalized_rows)
    
    # Extract patient info (use the most recent date)
    if normalized_df.empty:
        existing_patient_info = {}
    else:
        # Get the most recent date entry for patient info
        most_recent_idx = normalized_df['Test_Date_dt'].idxmax() if normalized_df['Test_Date_dt'].notna().any() else 0
        recent_row = normalized_df.iloc[most_recent_idx] if most_recent_idx is not None else normalized_df.iloc[0]
        
        existing_patient_info = {
            'name': recent_row['Patient_Name'],
            'age': recent_row['Age'],
            'gender': recent_row['Gender'],
            'patient_id': recent_row['Patient_ID'],
            'date': recent_row['Test_Date'],
            'lab_name': recent_row['Lab_Name']
        }
    
    print(f"Final normalized data shape: {normalized_df.shape}")
    return normalized_df, existing_patient_info

def debug_data_combination(existing_df, new_pdf_data_list, step_name=""):
    """
    Helper function to debug data combination issues
    """
    print(f"\n=== DEBUG: {step_name} ===")
    
    if not existing_df.empty:
        print(f"Existing data: {existing_df.shape} rows")
        print(f"Existing columns: {list(existing_df.columns)}")
        print(f"Sample existing data:")
        print(existing_df[['Test_Name', 'Result', 'Test_Date']].head(3).to_string())
    else:
        print("Existing data: EMPTY")
    
    if new_pdf_data_list:
        total_new_rows = sum(len(df) for df in new_pdf_data_list)
        print(f"New PDF data: {len(new_pdf_data_list)} dataframes, {total_new_rows} total rows")
        if new_pdf_data_list:
            sample_df = new_pdf_data_list[0]
            print(f"Sample new data:")
            print(sample_df[['Test_Name', 'Result', 'Test_Date']].head(3).to_string())
    else:
        print("New PDF data: NONE")
    
    print("=" * 50)

# Add this debugging version of the main workflow
def enhanced_data_combination_workflow_debug(uploaded_excel_file, new_pdf_data_list, new_patient_info_list):
    """
    DEBUG VERSION: Main workflow with extensive logging
    """
    print("=== STARTING DEBUG DATA COMBINATION WORKFLOW ===")
    
    # Step 1: Process existing Excel/CSV file
    print(f"\nSTEP 1: Processing existing file: {uploaded_excel_file.name}")
    existing_df, existing_patient_info = process_existing_excel_csv(uploaded_excel_file, new_patient_info_list)
    
    debug_data_combination(existing_df, new_pdf_data_list, "After processing existing file")
    
    # Step 2: Initialize combined dataframe
    combined_df = pd.DataFrame()
    
    # Step 3: Add existing data
    if not existing_df.empty:
        print(f"\nSTEP 3: Adding existing data to combined dataset...")
        combined_df = existing_df.copy()
        print(f"Combined data after adding existing: {combined_df.shape}")
        print(f"Sample combined data:")
        if not combined_df.empty:
            print(combined_df[['Test_Name', 'Result', 'Test_Date']].head(3).to_string())
    else:
        print(f"\nSTEP 3: No existing data to add (existing_df is empty)")
    
    # Step 4: Add new PDF data if available
    if new_pdf_data_list:
        print(f"\nSTEP 4: Processing {len(new_pdf_data_list)} new PDF datasets...")
        new_data_df = pd.concat(new_pdf_data_list, ignore_index=True)
        print(f"New PDF data shape: {new_data_df.shape}")
        
        if not combined_df.empty:
            print("STEP 4a: Combining existing + new data...")
            
            # Show column comparison
            existing_cols = set(combined_df.columns)
            new_cols = set(new_data_df.columns)
            print(f"Existing columns: {len(existing_cols)}")
            print(f"New columns: {len(new_cols)}")
            print(f"Common columns: {len(existing_cols & new_cols)}")
            print(f"Missing from existing: {new_cols - existing_cols}")
            print(f"Missing from new: {existing_cols - new_cols}")
            
            # Ensure both dataframes have the same columns
            all_columns = list(existing_cols | new_cols)
            
            # Add missing columns to both dataframes
            for col in all_columns:
                if col not in combined_df.columns:
                    combined_df[col] = 'N/A'
                    print(f"Added column '{col}' to existing data")
                if col not in new_data_df.columns:
                    new_data_df[col] = 'N/A'
                    print(f"Added column '{col}' to new data")
            
            # Reorder columns to match
            combined_df = combined_df[all_columns]
            new_data_df = new_data_df[all_columns]
            
            print(f"Before concat - Combined: {combined_df.shape}, New: {new_data_df.shape}")
            
            # Combine the data
            combined_df = pd.concat([combined_df, new_data_df], ignore_index=True)
            print(f"After concat - Final combined data shape: {combined_df.shape}")
            
        else:
            print("STEP 4b: No existing data, using only new PDF data...")
            combined_df = new_data_df
    else:
        print(f"\nSTEP 4: No new PDF data to add...")
    
    # Step 5: Final validation
    print(f"\nSTEP 5: Final validation")
    print(f"Final combined data shape: {combined_df.shape}")
    
    if not combined_df.empty:
        print("Sample final data:")
        print(combined_df[['Source_Filename', 'Test_Name', 'Result', 'Test_Date']].head(5).to_string())
        
        # Check data sources
        source_counts = combined_df['Source_Filename'].value_counts()
        print(f"\nData sources breakdown:")
        for source, count in source_counts.items():
            print(f"  {source}: {count} records")
    else:
        print("WARNING: Final combined data is EMPTY!")
    
    # Smart consolidation of patient info
    final_patient_info = smart_consolidate_patient_info(existing_patient_info, new_patient_info_list)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Combined data shape: {combined_df.shape}")
    print(f"Patient info: {final_patient_info}")
    print("=== END DEBUG WORKFLOW ===\n")
    
    return combined_df, final_patient_info

# Replace the existing parse_date_dd_mm_yyyy function if it has issues
def parse_date_dd_mm_yyyy(date_str):
    """FIXED: Parse date string ensuring DD/MM/YYYY format"""
    if not date_str or pd.isna(date_str) or date_str in ['N/A', '']:
        return None
    
    # Convert to string in case it's not already
    date_str = str(date_str).strip()
    
    try:
        # First try to parse as DD/MM/YYYY
        return pd.to_datetime(date_str, format='%d/%m/%Y', errors='raise')
    except:
        try:
            # Try DD-MM-YYYY
            return pd.to_datetime(date_str, format='%d-%m-%Y', errors='raise')
        except:
            try:
                # Try DD.MM.YYYY
                return pd.to_datetime(date_str, format='%d.%m.%Y', errors='raise')
            except:
                try:
                    # If all else fails, try to parse and assume it's in DD/MM/YYYY format
                    # Force dayfirst=True to ensure DD/MM/YYYY interpretation
                    return pd.to_datetime(date_str, dayfirst=True, errors='raise')
                except:
                    print(f"Could not parse date: {date_str}")
                    return None

def process_pivoted_excel_data(df, filename, new_patient_info_list):
    """
    FIXED: Better handling of pivoted Excel data
    """
    
    # Remove any header rows with emojis or labels
    cleaned_df = df.copy()
    
    # Remove rows that look like headers
    rows_to_keep = []
    for idx, row in df.iterrows():
        first_val = str(row.iloc[0]).strip().lower()
        
        # Skip header-like rows
        if any(x in first_val for x in ['📅', '🏥', 'date', 'lab', 'test_category', 'unnamed']):
            continue
        
        # Skip completely empty rows
        if pd.isna(row.iloc[0]) and pd.isna(row.iloc[1]):
            continue
            
        rows_to_keep.append(idx)
    
    if rows_to_keep:
        cleaned_df = df.loc[rows_to_keep].reset_index(drop=True)
    else:
        return pd.DataFrame(), {}
    
    # Find data columns (exclude Test_Category and Test_Name)
    data_columns = []
    for col in cleaned_df.columns:
        if col not in ['Test_Category', 'Test_Name'] and not str(col).startswith('Unnamed'):
            data_columns.append(col)
    
    # Convert to normalized format
    normalized_rows = []
    
    for idx, row in cleaned_df.iterrows():
        test_category = row.get('Test_Category', 'General')
        test_name = row.get('Test_Name', '')
        
        if not test_name or pd.isna(test_name):
            continue
        
        for data_col in data_columns:
            result = row.get(data_col, '')
            
            if pd.isna(result) or str(result).strip() == '':
                continue
            
            # Parse date and lab from column name
            if '_' in str(data_col):
                date_part, lab_part = str(data_col).split('_', 1)
            else:
                date_part = str(data_col)
                lab_part = 'General Lab'
            
            # Get patient info
            patient_name = new_patient_info_list[-1].get('name', 'N/A') if new_patient_info_list else 'N/A'
            patient_age = new_patient_info_list[-1].get('age', 'N/A') if new_patient_info_list else 'N/A'
            patient_gender = new_patient_info_list[-1].get('gender', 'N/A') if new_patient_info_list else 'N/A'
            patient_id = new_patient_info_list[-1].get('patient_id', 'N/A') if new_patient_info_list else 'N/A'
            
            normalized_row = {
                'Source_Filename': filename,
                'Patient_ID': patient_id,
                'Patient_Name': patient_name,
                'Age': patient_age,
                'Gender': patient_gender,
                'Test_Date': date_part,
                'Lab_Name': lab_part,
                'Test_Category': test_category,
                'Original_Test_Name': test_name,
                'Test_Name': test_name,
                'Result': str(result),
                'Unit': 'N/A',
                'Reference_Range': 'N/A',
                'Status': 'N/A',
                'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Result_Numeric': pd.to_numeric(result, errors='coerce'),
                'Test_Date_dt': parse_date_dd_mm_yyyy(date_part)
            }
            
            normalized_rows.append(normalized_row)
    
    if normalized_rows:
        return pd.DataFrame(normalized_rows), {'name': patient_name, 'age': patient_age, 'gender': patient_gender, 'patient_id': patient_id, 'date': date_part, 'lab_name': lab_part}
    else:
        return pd.DataFrame(), {}



def process_pivoted_excel_data_simple(df, filename, new_patient_info_list):
    """
    SIMPLE: Process pivoted Excel data from tab 1
    """
    print(f"Processing pivoted data: {df.shape}")
    
    # Get data columns (everything except Test_Category and Test_Name)
    data_cols = [col for col in df.columns if col not in ['Test_Category', 'Test_Name']]
    print(f"Found {len(data_cols)} data columns: {data_cols[:3]}...")
    
    normalized_rows = []
    
    for idx, row in df.iterrows():
        test_category = str(row.get('Test_Category', 'General')).strip()
        test_name = str(row.get('Test_Name', '')).strip()
        
        # Skip if no valid test name
        if not test_name or test_name.lower() in ['nan', 'test_name', '']:
            continue
        
        # Process each data column
        for col in data_cols:
            value = row[col]
            
            # Skip empty values
            if pd.isna(value) or str(value).strip() == '':
                continue
            
            # Extract date and lab from column name
            col_str = str(col)
            if '_' in col_str:
                parts = col_str.split('_', 1)
                date_part = parts[0]
                lab_part = parts[1] if len(parts) > 1 else 'Lab'
            else:
                date_part = col_str
                lab_part = 'Lab'
            
            # Use patient info from PDFs if available
            if new_patient_info_list:
                latest_info = new_patient_info_list[-1]
                patient_name = latest_info.get('name', 'Patient')
                patient_age = latest_info.get('age', 'N/A')
                patient_gender = latest_info.get('gender', 'N/A')
                patient_id = latest_info.get('patient_id', 'N/A')
            else:
                patient_name = 'Patient'
                patient_age = 'N/A'
                patient_gender = 'N/A'
                patient_id = 'N/A'
            
            # Create normalized row
            normalized_row = {
                'Source_Filename': filename,
                'Patient_ID': patient_id,
                'Patient_Name': patient_name,
                'Age': patient_age,
                'Gender': patient_gender,
                'Test_Date': date_part,
                'Lab_Name': lab_part,
                'Test_Category': test_category,
                'Original_Test_Name': test_name,
                'Test_Name': test_name,
                'Result': str(value),
                'Unit': 'N/A',
                'Reference_Range': 'N/A',
                'Status': 'N/A',
                'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Result_Numeric': pd.to_numeric(value, errors='coerce'),
                'Test_Date_dt': parse_date_dd_mm_yyyy(date_part)
            }
            
            normalized_rows.append(normalized_row)
    
    print(f"Created {len(normalized_rows)} data rows from Excel")
    
    if normalized_rows:
        result_df = pd.DataFrame(normalized_rows)
        
        # Create patient info
        patient_info = {
            'name': patient_name,
            'age': patient_age,
            'gender': patient_gender,
            'patient_id': patient_id,
            'date': date_part,
            'lab_name': lab_part
        }
        
        return result_df, patient_info
    else:
        return pd.DataFrame(), {}

def process_excel_pivoted_format(df, filename, new_patient_info_list):
    """
    Process the pivoted Excel format from your screenshot
    """
    print(f"Processing Excel pivoted format: {df.shape}")
    
    # Get data columns (everything except Test_Category, Test_Name, and Reference Ranges)
    data_cols = []
    for col in df.columns:
        col_str = str(col).lower()
        if col not in ['Test_Category', 'Test_Name'] and 'reference' not in col_str and 'range' not in col_str:
            data_cols.append(col)
    
    print(f"Found {len(data_cols)} data columns: {data_cols[:5]}...")
    
    normalized_rows = []
    
    for idx, row in df.iterrows():
        test_category = str(row.get('Test_Category', 'General')).strip()
        test_name = str(row.get('Test_Name', '')).strip()
        
        # Skip if no valid test name
        if not test_name or test_name.lower() in ['nan', 'test_name', ''] or pd.isna(test_name):
            print(f"Skipping row {idx}: invalid test_name '{test_name}'")
            continue
        
        print(f"Processing test: {test_category} - {test_name}")
        
        # Process each data column
        for col in data_cols:
            value = row[col]
            
            # Skip empty values
            if pd.isna(value) or str(value).strip() in ['', 'None', 'nan']:
                continue
            
            print(f"  Found value in {col}: {value}")
            
            # Parse date and lab from column name
            col_str = str(col)
            if '_' in col_str:
                parts = col_str.split('_', 1)
                date_part = parts[0]
                lab_part = parts[1] if len(parts) > 1 else 'Lab'
            else:
                date_part = col_str
                lab_part = 'Lab'
            
            # Use patient info from PDFs if available
            if new_patient_info_list:
                latest_info = new_patient_info_list[-1]
                patient_name = latest_info.get('name', 'Patient')
                patient_age = latest_info.get('age', 'N/A')
                patient_gender = latest_info.get('gender', 'N/A')
                patient_id = latest_info.get('patient_id', 'N/A')
            else:
                patient_name = extract_patient_info_from_excel_filename(filename)
                patient_age = 'N/A'
                patient_gender = 'N/A'
                patient_id = 'N/A'
            
            # Create normalized row
            normalized_row = {
                'Source_Filename': filename,
                'Patient_ID': patient_id,
                'Patient_Name': patient_name,
                'Age': patient_age,
                'Gender': patient_gender,
                'Test_Date': date_part,
                'Lab_Name': lab_part,
                'Test_Category': test_category,
                'Original_Test_Name': test_name,
                'Test_Name': test_name,
                'Result': str(value),
                'Unit': 'N/A',
                'Reference_Range': 'N/A',
                'Status': 'N/A',
                'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Result_Numeric': pd.to_numeric(value, errors='coerce'),
                'Test_Date_dt': parse_date_dd_mm_yyyy(date_part)
            }
            
            normalized_rows.append(normalized_row)
    
    print(f"✅ Created {len(normalized_rows)} normalized rows from Excel")
    
    if normalized_rows:
        result_df = pd.DataFrame(normalized_rows)
        
        # Create patient info
        if new_patient_info_list:
            patient_info = new_patient_info_list[-1].copy()
        else:
            patient_info = {
                'name': patient_name,
                'age': patient_age,
                'gender': patient_gender,
                'patient_id': patient_id,
                'date': date_part,
                'lab_name': lab_part
            }
        
        print(f"✅ Successfully processed Excel data:")
        print(f"   - {len(result_df)} total records")
        print(f"   - Patient: {patient_info.get('name', 'N/A')}")
        print(f"   - Date range: {result_df['Test_Date'].min()} to {result_df['Test_Date'].max()}")
        
        return result_df, patient_info
    else:
        print("❌ No data rows created from Excel")
        return pd.DataFrame(), {}

def process_existing_excel_csv(uploaded_excel_file, new_patient_info_list):
    """
    FIXED: Handle the exact Excel structure from your screenshot without creating unwanted columns
    """
    try:
        filename = uploaded_excel_file.name
        print(f"Processing file: {filename}")
        
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_excel_file)
            print(f"✓ Read CSV: {df.shape}")
        else:
            # For Excel - read the exact structure shown in your image
            print("Reading Excel with your specific structure...")
            
            # Read the entire first sheet without skipping rows initially
            full_df = pd.read_excel(uploaded_excel_file, sheet_name=0, header=None)
            print(f"Full Excel shape: {full_df.shape}")
            
            # Extract the date row (row 2, index 1)
            date_row = full_df.iloc[1, :].values
            print(f"Date row: {date_row[:10]}...")  # Show first 10 values
            
            # Extract the lab row (row 3, index 2) 
            lab_row = full_df.iloc[2, :].values
            print(f"Lab row: {lab_row[:10]}...")  # Show first 10 values
            
            # Create column headers by combining date + lab
            new_columns = []
            for i, (date_val, lab_val) in enumerate(zip(date_row, lab_row)):
                if i == 0:  # First column is Test_Category
                    new_columns.append('Test_Category')
                elif i == 1:  # Second column is Test_Name
                    new_columns.append('Test_Name')
                else:  # Data columns - combine date_lab
                    date_str = str(date_val).strip() if not pd.isna(date_val) else ""
                    lab_str = str(lab_val).strip() if not pd.isna(lab_val) else ""
                    
                    # FIXED: Skip columns that are not actual data columns
                    # Skip if both date and lab are empty or contain unwanted values
                    if (not date_str or date_str.lower() in ['nan', 'col', '', 'reference', 'ranges']) and \
                       (not lab_str or lab_str.lower() in ['nan', '7', '', 'reference', 'ranges']):
                        continue
                    
                    # Skip if this looks like a reference range column header
                    if 'reference' in date_str.lower() or 'reference' in lab_str.lower():
                        new_columns.append('Reference_Ranges')  # Keep as reference column, don't treat as data
                        continue
                    
                    # Create column name as date_lab for actual data columns
                    if date_str and date_str != 'nan' and lab_str and lab_str != 'nan':
                        col_name = f"{date_str}_{lab_str}"
                    elif date_str and date_str != 'nan':
                        col_name = date_str
                    else:
                        continue  # Skip this column entirely
                    
                    new_columns.append(col_name)
            
            print(f"Created column names: {new_columns}")
            
            # Extract the actual data (starting from row 5, index 4)
            data_df = full_df.iloc[4:].copy()
            
            # FIXED: Only keep the columns we actually want
            data_df = data_df.iloc[:, :len(new_columns)]  # Trim to match our column count
            
            # Set the new column names
            data_df.columns = new_columns
            
            # Remove completely empty rows
            data_df = data_df.dropna(how='all').reset_index(drop=True)
            
            df = data_df
            print(f"✓ Processed Excel structure: {df.shape}")
            print(f"Final columns: {list(df.columns)}")
        
        if df is None or df.empty:
            print("❌ Could not extract any data")
            return pd.DataFrame(), {}
        
        # Show what we extracted
        print(f"Final dataframe shape: {df.shape}")
        print(f"Sample data:")
        if not df.empty and len(df.columns) >= 2:
            print(df[['Test_Category', 'Test_Name']].head(3).to_string())
        
        # Process as pivoted format (since that's what your Excel structure is)
        return process_excel_pivoted_format_fixed(df, filename, new_patient_info_list)
        
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame(), {}


def process_excel_pivoted_format_fixed(df, filename, new_patient_info_list):
    """
    FIXED: Process the pivoted Excel format without including unwanted columns
    """
    print(f"Processing Excel pivoted format: {df.shape}")
    print(f"All columns in dataframe: {list(df.columns)}")
    
    # Get data columns - exclude Test_Category, Test_Name, and Reference_Ranges
    data_cols = []
    excluded_cols = []
    
    for col in df.columns:
        col_str = str(col).lower()
        
        # Skip the main structural columns
        if col in ['Test_Category', 'Test_Name']:
            continue
            
        # Skip reference range columns
        if ('reference' in col_str and 'range' in col_str) or col_str in ['reference_ranges', 'ref_range']:
            excluded_cols.append(col)
            continue
            
        # Skip columns that look like unwanted metadata
        if col_str in ['col', 'column', 'unnamed'] or col_str.startswith('unnamed'):
            excluded_cols.append(col)
            continue
            
        # This should be a data column
        data_cols.append(col)
    
    print(f"Data columns to process: {data_cols}")
    print(f"Excluded columns: {excluded_cols}")
    
    normalized_rows = []
    
    for idx, row in df.iterrows():
        test_category = str(row.get('Test_Category', 'General')).strip()
        test_name = str(row.get('Test_Name', '')).strip()
        
        # Skip if no valid test name
        if not test_name or test_name.lower() in ['nan', 'test_name', ''] or pd.isna(test_name):
            continue
        
        # Get reference range if available
        ref_range = 'N/A'
        if 'Reference_Ranges' in df.columns:
            ref_range = str(row.get('Reference_Ranges', 'N/A')).strip()
            if ref_range.lower() in ['nan', ''] or pd.isna(ref_range):
                ref_range = 'N/A'
        
        # Process each data column
        for col in data_cols:
            value = row[col]
            
            # Skip empty values
            if pd.isna(value) or str(value).strip() in ['', 'None', 'nan']:
                continue
            
            # Parse date and lab from column name
            col_str = str(col)
            if '_' in col_str:
                parts = col_str.split('_', 1)
                date_part = parts[0]
                lab_part = parts[1] if len(parts) > 1 else 'Lab'
            else:
                date_part = col_str
                lab_part = 'Lab'
            
            # Use patient info from PDFs if available
            if new_patient_info_list:
                latest_info = new_patient_info_list[-1]
                patient_name = latest_info.get('name', 'Patient')
                patient_age = latest_info.get('age', 'N/A')
                patient_gender = latest_info.get('gender', 'N/A')
                patient_id = latest_info.get('patient_id', 'N/A')
            else:
                patient_name = extract_patient_info_from_excel_filename(filename)
                patient_age = 'N/A'
                patient_gender = 'N/A'
                patient_id = 'N/A'
            
            # Create normalized row
            normalized_row = {
                'Source_Filename': filename,
                'Patient_ID': patient_id,
                'Patient_Name': patient_name,
                'Age': patient_age,
                'Gender': patient_gender,
                'Test_Date': date_part,
                'Lab_Name': lab_part,
                'Test_Category': test_category,
                'Original_Test_Name': test_name,
                'Test_Name': test_name,
                'Result': str(value),
                'Unit': 'N/A',
                'Reference_Range': ref_range,  # Use the reference range from the Excel file
                'Status': 'N/A',
                'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Result_Numeric': pd.to_numeric(value, errors='coerce'),
                'Test_Date_dt': parse_date_dd_mm_yyyy(date_part)
            }
            
            normalized_rows.append(normalized_row)
    
    print(f"✅ Created {len(normalized_rows)} normalized rows from Excel")
    
    if normalized_rows:
        result_df = pd.DataFrame(normalized_rows)
        
        # Create patient info
        if new_patient_info_list:
            patient_info = new_patient_info_list[-1].copy()
        else:
            patient_info = {
                'name': patient_name if 'patient_name' in locals() else 'Patient',
                'age': patient_age if 'patient_age' in locals() else 'N/A',
                'gender': patient_gender if 'patient_gender' in locals() else 'N/A',
                'patient_id': patient_id if 'patient_id' in locals() else 'N/A',
                'date': date_part if 'date_part' in locals() else 'N/A',
                'lab_name': lab_part if 'lab_part' in locals() else 'N/A'
            }
        
        print(f"✅ Successfully processed Excel data:")
        print(f"   - {len(result_df)} total records")
        print(f"   - Patient: {patient_info.get('name', 'N/A')}")
        if not result_df.empty:
            print(f"   - Date range: {result_df['Test_Date'].min()} to {result_df['Test_Date'].max()}")
        
        return result_df, patient_info
    else:
        print("❌ No data rows created from Excel")
        return pd.DataFrame(), {}


# =========================================
# HEALTH INSIGHTS DASHBOARD FUNCTIONS
# =========================================

def calculate_health_score(df):
    """
    Calculate an overall health score based on test results.
    Returns a score from 0-100 and breakdown by category.
    """
    if df.empty:
        return {'overall_score': 0, 'category_scores': {}, 'concerns': []}
    
    # Count status distribution
    status_counts = df['Status'].value_counts().to_dict()
    total_tests = len(df)
    
    # Weights for different statuses
    status_weights = {
        'Normal': 100,
        'N/A': 80,  # Unknown is neutral
        'Low': 60,
        'High': 40,
        'Critical': 10,
        'Positive': 30,  # For disease markers
        'Negative': 100  # For disease markers - good
    }
    
    # Calculate weighted score
    weighted_sum = 0
    for status, count in status_counts.items():
        status_clean = str(status).strip().title()
        weight = status_weights.get(status_clean, 70)
        weighted_sum += weight * count
    
    overall_score = int(weighted_sum / total_tests) if total_tests > 0 else 0
    
    # Calculate scores by category
    category_scores = {}
    for category in df['Test_Category'].unique():
        if pd.isna(category) or category == 'N/A':
            continue
        cat_df = df[df['Test_Category'] == category]
        cat_total = len(cat_df)
        if cat_total == 0:
            continue
        
        cat_weighted_sum = 0
        for _, row in cat_df.iterrows():
            status = str(row.get('Status', 'N/A')).strip().title()
            weight = status_weights.get(status, 70)
            cat_weighted_sum += weight
        
        category_scores[category] = {
            'score': int(cat_weighted_sum / cat_total),
            'total_tests': cat_total,
            'abnormal_count': len(cat_df[cat_df['Status'].isin(['High', 'Low', 'Critical', 'Positive'])])
        }
    
    # Identify concerns (tests with abnormal values)
    concerns = []
    abnormal_df = df[df['Status'].isin(['High', 'Low', 'Critical', 'Positive'])]
    for _, row in abnormal_df.iterrows():
        concerns.append({
            'test_name': row.get('Test_Name', 'Unknown'),
            'result': row.get('Result', 'N/A'),
            'status': row.get('Status', 'N/A'),
            'reference': row.get('Reference_Range', 'N/A'),
            'category': row.get('Test_Category', 'N/A'),
            'date': row.get('Test_Date', 'N/A')
        })
    
    return {
        'overall_score': overall_score,
        'category_scores': category_scores,
        'concerns': concerns
    }


def get_body_system_analysis(df):
    """
    Analyze health by body system with concern levels.
    Returns sorted list from most concerning to least.
    """
    if df.empty:
        return []
    
    body_systems = {}
    
    for category in df['Test_Category'].unique():
        if pd.isna(category) or category == 'N/A':
            continue
            
        cat_df = df[df['Test_Category'] == category]
        
        # Get body parts for this category
        body_parts = TEST_CATEGORY_TO_BODY_PARTS.get(category, ['General'])
        
        for body_part in body_parts:
            if body_part not in body_systems:
                body_systems[body_part] = {
                    'tests': [],
                    'abnormal_count': 0,
                    'total_count': 0,
                    'categories': set()
                }
            
            body_systems[body_part]['categories'].add(category)
            body_systems[body_part]['total_count'] += len(cat_df)
            
            abnormal = cat_df[cat_df['Status'].isin(['High', 'Low', 'Critical', 'Positive'])]
            body_systems[body_part]['abnormal_count'] += len(abnormal)
            
            for _, row in cat_df.iterrows():
                body_systems[body_part]['tests'].append({
                    'name': row.get('Test_Name', 'Unknown'),
                    'result': row.get('Result', 'N/A'),
                    'status': row.get('Status', 'N/A'),
                    'category': category
                })
    
    # Calculate concern level for each body system
    # CRITICAL: Only flag systems where >= 50% of tests are abnormal
    # This prevents over-alarming patients with misleading visual indicators
    result = []
    for system, data in body_systems.items():
        if data['total_count'] == 0:
            continue
            
        abnormal_ratio = data['abnormal_count'] / data['total_count']
        
        # MEDICAL LOGIC: Only flag if >= 50% abnormal (strict threshold)
        if abnormal_ratio >= 0.50:
            concern_level = 'flagged'
            concern_score = 2
        else:
            concern_level = 'normal'
            concern_score = 1
        
        result.append({
            'system': system,
            'emoji': BODY_PARTS_TO_EMOJI.get(system, '📊'),
            'concern_level': concern_level,
            'concern_score': concern_score,
            'abnormal_count': data['abnormal_count'],
            'total_count': data['total_count'],
            'abnormal_ratio': abnormal_ratio,
            'categories': list(data['categories']),
            'tests': data['tests']
        })
    
    # Sort by concern score (highest first), then by abnormal count
    result.sort(key=lambda x: (-x['concern_score'], -x['abnormal_count']))
    
    return result


def create_normalized_chart(df, test_name, height=300):
    """
    Create a normalized chart that handles different value ranges gracefully.
    Uses percentage of reference range for normalization.
    """
    test_data = df[df['Test_Name'] == test_name].copy()
    
    if test_data.empty:
        return None
    
    test_data = test_data.sort_values('Test_Date_dt')
    test_data = test_data[test_data['Result_Numeric'].notna()]
    
    if len(test_data) == 0:
        return None
    
    fig = go.Figure()
    
    # Get reference range
    ref_range_str = test_data.iloc[-1].get('Reference_Range', '')
    low_ref, high_ref, ref_type = parse_reference_range(ref_range_str)
    
    # Format dates for display
    test_data['Date_Display'] = test_data['Test_Date_dt'].apply(
        lambda x: format_date_dd_mm_yyyy(x) if pd.notna(x) else 'N/A'
    )
    
    # Create normalized values (percentage of reference range midpoint)
    if ref_type == 'range' and low_ref is not None and high_ref is not None:
        ref_mid = (low_ref + high_ref) / 2
        test_data['Normalized'] = (test_data['Result_Numeric'] / ref_mid) * 100
        y_label = "% of Normal"
        
        # Reference lines at normalized positions
        low_norm = (low_ref / ref_mid) * 100
        high_norm = (high_ref / ref_mid) * 100
        
        fig.add_hline(y=high_norm, line_dash="dash", line_color="#ef4444", 
                      annotation_text="Upper Limit", annotation_position="bottom right")
        fig.add_hline(y=low_norm, line_dash="dash", line_color="#22c55e",
                      annotation_text="Lower Limit", annotation_position="top right")
        fig.add_hrect(y0=low_norm, y1=high_norm, fillcolor="#22c55e", opacity=0.1)
        
        y_values = test_data['Normalized']
    else:
        # Just use the actual values
        y_values = test_data['Result_Numeric']
        y_label = "Value"
    
    # Determine color based on latest status
    latest_status = test_data.iloc[-1].get('Status', 'N/A')
    if latest_status in ['High', 'Low', 'Critical']:
        line_color = '#ef4444'
    elif latest_status == 'Normal':
        line_color = '#22c55e'
    else:
        line_color = '#a855f7'
    
    # Add the line trace
    fig.add_trace(go.Scatter(
        x=test_data['Date_Display'],
        y=y_values,
        mode='lines+markers',
        name=test_name,
        line=dict(color=line_color, width=3),
        marker=dict(size=10, color=line_color),
        hovertemplate='<b>%{x}</b><br>Value: %{customdata:.2f}<extra></extra>',
        customdata=test_data['Result_Numeric']
    ))
    
    # Style the chart - using correct Plotly syntax (no deprecated titlefont)
    fig.update_layout(
        title=dict(text=test_name, font=dict(size=14, color='#f5f5f5')),
        xaxis=dict(
            title=dict(text='Date', font=dict(color='#a3a3a3')),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            tickfont=dict(color='#a3a3a3')
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color='#a3a3a3')),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            tickfont=dict(color='#a3a3a3')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        font=dict(color='#f5f5f5')
    )
    
    return fig


def generate_health_summary_with_ai(df, patient_info, api_key):
    """
    Generate an AI-powered health summary for the PDF report.
    """
    global gemini_model_chat
    if not gemini_model_chat:
        init_gemini_models(api_key)
    
    if df.empty:
        return "No data available for analysis."
    
    # Prepare summary data
    health_data = calculate_health_score(df)
    body_systems = get_body_system_analysis(df)
    
    # Create a concise data summary for the AI
    data_summary = f"""
Patient: {patient_info.get('name', 'N/A')}, Age: {patient_info.get('age', 'N/A')}, Gender: {patient_info.get('gender', 'N/A')}
Overall Health Score: {health_data['overall_score']}/100

Test Results Summary:
"""
    
    for category, scores in health_data['category_scores'].items():
        data_summary += f"- {category}: Score {scores['score']}/100, {scores['abnormal_count']} abnormal out of {scores['total_tests']} tests\n"
    
    if health_data['concerns']:
        data_summary += "\nAbnormal Results:\n"
        for concern in health_data['concerns'][:10]:  # Limit to top 10
            data_summary += f"- {concern['test_name']}: {concern['result']} ({concern['status']}) - Ref: {concern['reference']}\n"
    
    prompt = f"""Based on this medical report data, provide a professional health summary suitable for a PDF report.
Include:
1. A brief overall health assessment (2-3 sentences)
2. Key findings and areas of concern (if any)
3. General recommendations (no specific medical advice)

Keep the response concise, professional, and suitable for a patient to share with their doctor.
Do not provide specific medical diagnoses or treatment recommendations.

{data_summary}
"""
    
    try:
        response = gemini_model_chat.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate AI summary: {str(e)}"


def create_pdf_report(df, patient_info, api_key):
    """
    Generate a comprehensive PDF health report with charts and insights.
    Returns PDF bytes.
    """
    from io import BytesIO
    import base64
    
    # Get analysis data
    health_data = calculate_health_score(df)
    body_systems = get_body_system_analysis(df)
    ai_summary = generate_health_summary_with_ai(df, patient_info, api_key)
    
    # Generate HTML for PDF
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Health Report - {patient_info.get('name', 'Patient')}</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', -apple-system, sans-serif;
                color: #1a1a1a;
                line-height: 1.6;
                padding: 40px;
                max-width: 800px;
                margin: 0 auto;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 3px solid #a855f7;
            }}
            
            .header h1 {{
                font-size: 28px;
                color: #a855f7;
                margin-bottom: 10px;
            }}
            
            .header p {{
                color: #666;
                font-size: 14px;
            }}
            
            .patient-info {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-bottom: 30px;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 12px;
            }}
            
            .patient-info-item {{
                text-align: center;
            }}
            
            .patient-info-item label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
                display: block;
                margin-bottom: 5px;
            }}
            
            .patient-info-item value {{
                font-size: 16px;
                font-weight: 600;
                color: #1a1a1a;
            }}
            
            .score-section {{
                text-align: center;
                margin: 30px 0;
                padding: 30px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 16px;
            }}
            
            .health-score {{
                font-size: 72px;
                font-weight: 700;
                background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            
            .score-label {{
                font-size: 14px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 2px;
            }}
            
            .section {{
                margin: 30px 0;
            }}
            
            .section h2 {{
                font-size: 20px;
                color: #1a1a1a;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #e9ecef;
            }}
            
            .ai-summary {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 12px;
                border-left: 4px solid #a855f7;
            }}
            
            .body-system {{
                display: flex;
                align-items: center;
                padding: 15px;
                margin: 10px 0;
                border-radius: 10px;
                background: #fff;
                border: 1px solid #e9ecef;
            }}
            
            .body-system.high {{
                border-left: 4px solid #ef4444;
                background: #fef2f2;
            }}
            
            .body-system.medium {{
                border-left: 4px solid #f97316;
                background: #fff7ed;
            }}
            
            .body-system.low {{
                border-left: 4px solid #22c55e;
                background: #f0fdf4;
            }}
            
            .body-system-emoji {{
                font-size: 24px;
                margin-right: 15px;
            }}
            
            .body-system-name {{
                flex: 1;
                font-weight: 600;
            }}
            
            .body-system-stats {{
                text-align: right;
                font-size: 14px;
                color: #666;
            }}
            
            .concerns-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            
            .concerns-table th {{
                background: #f8f9fa;
                padding: 12px;
                text-align: left;
                font-size: 12px;
                text-transform: uppercase;
                color: #666;
                border-bottom: 2px solid #e9ecef;
            }}
            
            .concerns-table td {{
                padding: 12px;
                border-bottom: 1px solid #e9ecef;
            }}
            
            .status-badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
            }}
            
            .status-high {{
                background: #fee2e2;
                color: #dc2626;
            }}
            
            .status-low {{
                background: #fed7aa;
                color: #ea580c;
            }}
            
            .status-normal {{
                background: #dcfce7;
                color: #16a34a;
            }}
            
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #e9ecef;
                text-align: center;
                color: #999;
                font-size: 12px;
            }}
            
            @media print {{
                body {{
                    padding: 20px;
                }}
                .section {{
                    page-break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏥 Health Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="patient-info">
            <div class="patient-info-item">
                <label>Patient Name</label>
                <value>{patient_info.get('name', 'N/A')}</value>
            </div>
            <div class="patient-info-item">
                <label>Age</label>
                <value>{patient_info.get('age', 'N/A')}</value>
            </div>
            <div class="patient-info-item">
                <label>Gender</label>
                <value>{patient_info.get('gender', 'N/A')}</value>
            </div>
            <div class="patient-info-item">
                <label>Report Date</label>
                <value>{patient_info.get('date', 'N/A')}</value>
            </div>
        </div>
        
        <div class="score-section">
            <div class="score-label">Overall Health Score</div>
            <div class="health-score">{health_data['overall_score']}</div>
            <div class="score-label">out of 100</div>
        </div>
        
        <div class="section">
            <h2>📋 AI Health Summary</h2>
            <div class="ai-summary">
                {ai_summary.replace(chr(10), '<br>')}
            </div>
        </div>
        
        <div class="section">
            <h2>🫀 Body Systems Analysis</h2>
            <p style="color: #666; margin-bottom: 15px;">Ordered from most concerning to least concerning:</p>
    """
    
    for system in body_systems:
        concern_class = system['concern_level']
        html_content += f"""
            <div class="body-system {concern_class}">
                <span class="body-system-emoji">{system['emoji']}</span>
                <span class="body-system-name">{system['system']}</span>
                <span class="body-system-stats">
                    {system['abnormal_count']} / {system['total_count']} tests flagged
                </span>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>⚠️ Items Requiring Attention</h2>
    """
    
    if health_data['concerns']:
        html_content += """
            <table class="concerns-table">
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Result</th>
                        <th>Reference</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for concern in health_data['concerns']:
            status_class = 'status-high' if concern['status'] in ['High', 'Critical'] else 'status-low'
            html_content += f"""
                <tr>
                    <td><strong>{concern['test_name']}</strong></td>
                    <td>{concern['result']}</td>
                    <td>{concern['reference']}</td>
                    <td><span class="status-badge {status_class}">{concern['status']}</span></td>
                </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        """
    else:
        html_content += """
            <p style="color: #22c55e; padding: 20px; background: #f0fdf4; border-radius: 10px;">
                ✅ All test results are within normal ranges. Keep up the good work!
            </p>
        """
    
    html_content += f"""
        </div>
        
        <div class="footer">
            <p><strong>Disclaimer:</strong> This report is for informational purposes only and should not be considered medical advice.</p>
            <p>Always consult with a qualified healthcare provider for medical decisions.</p>
            <p style="margin-top: 10px;">Generated by Medical Report Analyzer • {datetime.now().strftime('%Y')}</p>
        </div>
    </body>
    </html>
    """
    
    return html_content


def generate_health_insights_dashboard(df):
    """
    Generate a comprehensive health insights dashboard in Streamlit.
    Shows body systems analysis with clinical, restrained design.
    Uses 50% threshold for flagging - only truly concerning systems are highlighted.
    """
    import streamlit as st
    
    if df.empty:
        st.warning("No data available for health insights.")
        return
    
    # Calculate health score and body systems analysis
    health_data = calculate_health_score(df)
    body_systems = get_body_system_analysis(df)
    
    # Overall Health Score Card - CLEAN, NO GRADIENTS
    score = health_data['overall_score']
    if score >= 80:
        score_color = "#10b981"
        score_text = "Good"
    elif score >= 60:
        score_color = "#f59e0b"
        score_text = "Fair"
    else:
        score_color = "#dc2626"
        score_text = "Needs Review"
    
    # Clean, clinical score display - solid background, no text-shadow/glow
    st.markdown(f"""
    <div style='text-align: center; padding: 1.5rem; background: #1c1c1c; 
                border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #2a2a2a;'>
        <p style='font-size: 0.75rem; color: #737373; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;'>
            Overall Health Score
        </p>
        <div style='font-size: 3rem; font-weight: 700; color: {score_color};'>
            {score}
        </div>
        <p style='font-size: 1rem; color: {score_color}; font-weight: 500;'>{score_text}</p>
        <p style='font-size: 0.75rem; color: #737373; margin-top: 0.5rem;'>
            Based on {len(df)} test results across {len(health_data['category_scores'])} categories
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Body Systems Overview - CLINICAL DESIGN
    st.markdown("### Body Systems Overview")
    
    # Count flagged vs normal systems for summary
    flagged_count = sum(1 for s in body_systems if s['concern_level'] == 'flagged')
    normal_count = len(body_systems) - flagged_count
    
    if flagged_count > 0:
        st.markdown(f"<p style='color: #a3a3a3; margin-bottom: 1rem;'>{flagged_count} system(s) flagged for review (≥50% abnormal tests). {normal_count} system(s) within normal limits.</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color: #10b981; margin-bottom: 1rem;'>✓ All {len(body_systems)} body systems are within normal limits.</p>", unsafe_allow_html=True)
    
    if not body_systems:
        st.info("No body system data available.")
    else:
        # Show flagged systems first (if any), then normal systems
        flagged_systems = [s for s in body_systems if s['concern_level'] == 'flagged']
        normal_systems = [s for s in body_systems if s['concern_level'] == 'normal']
        
        # Flagged systems - only these get visual emphasis
        if flagged_systems:
            st.markdown("#### ⚠️ Systems Requiring Attention")
            for system in flagged_systems:
                # Solid background with subtle left border - NO gradient
                st.markdown(f"""
                <div style='background: #1c1c1c; border-radius: 8px; padding: 1rem; 
                            margin-bottom: 0.5rem; border: 1px solid #2a2a2a;
                            border-left: 3px solid #dc2626;'>
                    <div style='display: flex; align-items: center; gap: 0.75rem;'>
                        <span style='font-size: 1.25rem;'>{system['emoji']}</span>
                        <div style='flex: 1;'>
                            <div style='font-weight: 600; color: #f5f5f5; font-size: 0.9rem;'>{system['system']}</div>
                            <div style='font-size: 0.75rem; color: #dc2626;'>
                                {system['abnormal_count']} of {system['total_count']} tests abnormal ({int(system['abnormal_ratio']*100)}%)
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Normal systems - muted, understated
        if normal_systems:
            with st.expander(f"✓ Normal Systems ({len(normal_systems)})", expanded=False):
                cols = st.columns(min(len(normal_systems), 4))
                for i, system in enumerate(normal_systems):
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        # Very muted, calm display - solid colors only
                        st.markdown(f"""
                        <div style='background: #1c1c1c; border-radius: 8px; padding: 0.75rem; 
                                    margin-bottom: 0.5rem; border: 1px solid #2a2a2a;
                                    border-left: 3px solid #10b981;'>
                            <div style='font-size: 1rem; margin-bottom: 0.25rem;'>{system['emoji']}</div>
                            <div style='font-weight: 500; color: #a3a3a3; font-size: 0.8rem;'>{system['system']}</div>
                            <div style='font-size: 0.7rem; color: #737373;'>
                                {system['abnormal_count']}/{system['total_count']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Expandable details for flagged systems only (to reduce visual clutter)
        if flagged_systems:
            st.markdown("#### Detailed Test Results")
            for system in flagged_systems:
                with st.expander(f"{system['emoji']} {system['system']} - {system['abnormal_count']}/{system['total_count']} abnormal"):
                    tests_data = []
                    for test in system['tests']:
                        status = test.get('status', 'N/A')
                        if status in ['High', 'Low', 'Critical']:
                            status_badge = f"⚠️ {status}"
                        elif status == 'Normal':
                            status_badge = f"✓ Normal"
                        else:
                            status_badge = f"— {status}"
                        
                        tests_data.append({
                            'Test': test['name'],
                            'Result': test['result'],
                            'Status': status_badge,
                            'Category': test.get('category', 'N/A')
                        })
                    
                    if tests_data:
                        tests_df = pd.DataFrame(tests_data)
                        st.dataframe(tests_df, use_container_width=True, hide_index=True)


def generate_pdf_health_report(df, patient_info, api_key):
    """
    Generate a PDF health report and return bytes.
    Uses ReportLab for reliable PDF generation without system dependencies.
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from datetime import datetime
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=24, spaceAfter=30, textColor=HexColor('#1e293b'),
        alignment=1  # Center
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'],
        fontSize=14, spaceAfter=12, textColor=HexColor('#6366f1'),
        spaceBefore=20
    )
    normal_style = ParagraphStyle(
        'CustomNormal', parent=styles['Normal'],
        fontSize=10, spaceAfter=6, textColor=HexColor('#374151')
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Health Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Patient Info
    story.append(Paragraph("Patient Information", heading_style))
    patient_name = patient_info.get('name', 'N/A') if patient_info else 'N/A'
    patient_age = patient_info.get('age', 'N/A') if patient_info else 'N/A'
    patient_gender = patient_info.get('gender', 'N/A') if patient_info else 'N/A'
    story.append(Paragraph(f"<b>Name:</b> {patient_name}", normal_style))
    story.append(Paragraph(f"<b>Age:</b> {patient_age}", normal_style))
    story.append(Paragraph(f"<b>Gender:</b> {patient_gender}", normal_style))
    story.append(Spacer(1, 15))
    
    # Test Results Table
    story.append(Paragraph("Test Results Summary", heading_style))
    
    if df is not None and not df.empty:
        # Build table data
        table_data = [['Test Name', 'Value', 'Unit', 'Reference Range', 'Status']]
        
        for _, row in df.iterrows():
            test_name = str(row.get('Test_Name', row.get('test_name', 'N/A')))[:30]
            value = str(row.get('Value', row.get('value', 'N/A')))
            unit = str(row.get('Unit', row.get('unit', '')))
            ref_range = str(row.get('Reference_Range', row.get('reference_range', 'N/A')))
            status = str(row.get('Status', row.get('status', 'Normal')))
            table_data.append([test_name, value, unit, ref_range, status])
        
        # Create table
        table = Table(table_data, colWidths=[2*inch, 0.8*inch, 0.6*inch, 1.5*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e5e7eb')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f9fafb')]),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No test results available.", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Paragraph("Disclaimer", heading_style))
    disclaimer_text = """This report is generated by an AI-powered analysis tool and is intended 
    for informational purposes only. It does not constitute medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare professionals for medical decisions."""
    story.append(Paragraph(disclaimer_text, normal_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()