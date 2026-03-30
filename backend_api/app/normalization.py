from __future__ import annotations

import re
from typing import Any

CANONICAL_CATEGORIES = [
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

CATEGORY_MAP: dict[str, str] = {
    # Vitamins
    "vitamin b-12": "Vitamins & Minerals",
    "vitamin b12": "Vitamins & Minerals",
    "vitamin d": "Vitamins & Minerals",
    "vitamin levels": "Vitamins & Minerals",
    "vitamin profile": "Vitamins & Minerals",
    "vitamins": "Vitamins & Minerals",
    "minerals": "Vitamins & Minerals",
    "iron profile": "Vitamins & Minerals",
    "iron studies": "Vitamins & Minerals",
    # Diabetes
    "diabetes": "Diabetes & Glucose",
    "diabetes panel": "Diabetes & Glucose",
    "glucose metabolism": "Diabetes & Glucose",
    "blood sugar": "Diabetes & Glucose",
    "diabetes / glucose metabolism": "Diabetes & Glucose",
    "diabetes markers": "Diabetes & Glucose",
    "hba1c": "Diabetes & Glucose",
    # Liver
    "liver function test": "Liver Function",
    "liver function": "Liver Function",
    "lft": "Liver Function",
    "hepatic": "Liver Function",
    # Kidney
    "kidney function test": "Kidney Function",
    "kidney function": "Kidney Function",
    "renal function": "Kidney Function",
    "kft": "Kidney Function",
    "renal": "Kidney Function",
    # Thyroid
    "thyroid function test": "Thyroid Function",
    "thyroid profile": "Thyroid Function",
    "thyroid": "Thyroid Function",
    "tft": "Thyroid Function",
    # Lipids
    "lipid profile": "Lipid Profile",
    "lipids": "Lipid Profile",
    "cholesterol": "Lipid Profile",
    # Haematology
    "haematology": "Haematology",
    "hematology": "Haematology",
    "cbc": "Haematology",
    "complete blood count": "Haematology",
    "blood count": "Haematology",
    # Hormones
    "hormone": "Hormones",
    "hormones": "Hormones",
    "hormone profile": "Hormones",
    "sex hormones": "Hormones",
    "reproductive hormones": "Hormones",
    # Cardiac
    "cardiac risk markers": "Cardiac Markers",
    "cardiac markers": "Cardiac Markers",
    "cardiac": "Cardiac Markers",
    "heart": "Cardiac Markers",
    # Inflammation
    "inflammation marker": "Inflammation",
    "inflammation markers": "Inflammation",
    "inflammatory": "Inflammation",
    "crp": "Inflammation",
    # Proteins
    "proteins": "Proteins",
    "protein profile": "Proteins",
    # Immunology
    "immunology": "Immunology",
    "immune": "Immunology",
    "serology": "Immunology",
    # Urine
    "urinalysis": "Urinalysis",
    "urine": "Urinalysis",
    "urine routine": "Urinalysis",
    "urine analysis": "Urinalysis",
    # Additional observed categories
    "electrolytes": "Kidney Function",
    "blood glucose": "Diabetes & Glucose",
    "biochemistry": "Other",
    "bone health": "Vitamins & Minerals",
    "other": "Other",
}

TEST_NAME_MAP: dict[str, str] = {
    "sgot": "Aspartate Aminotransferase (SGOT/AST)",
    "ast": "Aspartate Aminotransferase (SGOT/AST)",
    "aspartate aminotransferase": "Aspartate Aminotransferase (SGOT/AST)",
    "sgot/ast": "Aspartate Aminotransferase (SGOT/AST)",
    "sgpt": "Alanine Aminotransferase (SGPT/ALT)",
    "alt": "Alanine Aminotransferase (SGPT/ALT)",
    "alanine aminotransferase": "Alanine Aminotransferase (SGPT/ALT)",
    "hba1c": "Glycated Haemoglobin (HbA1C)",
    "glycated haemoglobin": "Glycated Haemoglobin (HbA1C)",
    "haemoglobin a1c": "Glycated Haemoglobin (HbA1C)",
    "25-oh vitamin d": "25-OH Vitamin D (Total)",
    "25 oh vitamin d": "25-OH Vitamin D (Total)",
    "vitamin d total": "25-OH Vitamin D (Total)",
    "vitamin d3": "25-OH Vitamin D (Total)",
    "cholecalciferol": "25-OH Vitamin D (Total)",
    "25 oh cholecalciferol": "25-OH Vitamin D (Total)",
    "ldl": "LDL Cholesterol",
    "ldl cholesterol": "LDL Cholesterol",
    "low density lipoprotein": "LDL Cholesterol",
    "hdl": "HDL Cholesterol",
    "hdl cholesterol": "HDL Cholesterol",
    "high density lipoprotein": "HDL Cholesterol",
    "tsh": "Thyroid Stimulating Hormone (TSH)",
    "thyroid stimulating hormone": "Thyroid Stimulating Hormone (TSH)",
    "t3": "Triiodothyronine (T3)",
    "t4": "Thyroxine (T4)",
    "free t3": "Free Triiodothyronine (fT3)",
    "free t4": "Free Thyroxine (fT4)",
    "creatinine": "Serum Creatinine",
    "serum creatinine": "Serum Creatinine",
    "urea": "Blood Urea Nitrogen (BUN)",
    "bun": "Blood Urea Nitrogen (BUN)",
    "blood urea nitrogen": "Blood Urea Nitrogen (BUN)",
    "vitamin b12": "Vitamin B12 (Cobalamin)",
    "vitamin b-12": "Vitamin B12 (Cobalamin)",
    "cobalamin": "Vitamin B12 (Cobalamin)",
    "cyanocobalamin": "Vitamin B12 (Cobalamin)",
    "ferritin": "Serum Ferritin",
    "serum ferritin": "Serum Ferritin",
    "haemoglobin": "Haemoglobin (Hb)",
    "hemoglobin": "Haemoglobin (Hb)",
    "hb": "Haemoglobin (Hb)",
    "triglycerides": "Triglycerides",
    "trigs": "Triglycerides",
    "tg": "Triglycerides",
    "glucose": "Fasting Blood Glucose",
    "fasting glucose": "Fasting Blood Glucose",
    "fasting blood sugar": "Fasting Blood Glucose",
    "fbs": "Fasting Blood Glucose",
}

_warned_categories: set[str] = set()
_warned_tests: set[str] = set()


def _normalize_lookup_key(raw: str) -> str:
    key = raw.lower().strip()
    key = re.sub(r"[()]", "", key)
    key = re.sub(r"\\+", " ", key)
    key = re.sub(r"[_]+", " ", key)
    key = re.sub(r"\s*/\s*", " / ", key)
    key = re.sub(r"[^a-z0-9\-&/+\s]", " ", key)
    key = re.sub(r"\s+", " ", key)
    return key.strip(" -")


def _titlecase_fallback(raw: str) -> str:
    return " ".join(part.capitalize() for part in raw.strip().split())


def canonicalize_category(raw: str | None) -> str:
    if raw is None:
        return "Other"

    raw_text = str(raw).strip()
    if not raw_text or raw_text.lower() in {"n/a", "na", "none"}:
        return "Other"

    key = _normalize_lookup_key(raw_text)
    if key in CATEGORY_MAP:
        return CATEGORY_MAP[key]

    for map_key, canonical in CATEGORY_MAP.items():
        if key in map_key or map_key in key:
            return canonical

    if raw_text not in _warned_categories:
        _warned_categories.add(raw_text)
        print(f"[Category] Unmapped category: \"{raw_text}\" - add to CATEGORY_MAP")

    return "Other"


def normalize_test_name(raw: str | None) -> str:
    if raw is None:
        return "Unknown Test"

    raw_text = str(raw).strip()
    if not raw_text or raw_text.lower() in {"n/a", "na", "none"}:
        return "Unknown Test"

    key = _normalize_lookup_key(raw_text)
    if key in TEST_NAME_MAP:
        return TEST_NAME_MAP[key]

    compact_key = key.replace(" ", "")
    for map_key, canonical in TEST_NAME_MAP.items():
        map_compact_key = map_key.replace(" ", "")
        if compact_key == map_compact_key:
            return canonical

    for map_key, canonical in TEST_NAME_MAP.items():
        map_compact_key = map_key.replace(" ", "")
        if key in map_key or map_key in key or compact_key in map_compact_key or map_compact_key in compact_key:
            return canonical

    if raw_text not in _warned_tests:
        _warned_tests.add(raw_text)
        print(f"[TestName] Unmapped test: \"{raw_text}\" - consider adding to TEST_NAME_MAP")

    return _titlecase_fallback(raw_text)


def _normalize_status(raw: str | None) -> str:
    if raw is None:
        return "N/A"

    status = str(raw).strip()
    if not status:
        return "N/A"

    key = status.lower()
    if key in {"n/a", "na", "not applicable"}:
        return "N/A"
    if "critical" in key:
        return "Critical"
    if "insufficient" in key:
        return "Insufficient"
    if "borderline" in key:
        return "Borderline"
    if "positive" in key:
        return "Positive"
    if "negative" in key:
        return "Negative"
    if "flag" in key:
        return "Flagged"
    if "high" in key:
        return "High"
    if "low" in key:
        return "Low"
    if "normal" in key or "within" in key:
        return "Normal"

    return _titlecase_fallback(status)


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)

    raw_name = (
        normalized.get("Test_Name")
        or normalized.get("Original_Test_Name")
        or ""
    )

    normalized["Original_Test_Name"] = normalized.get("Original_Test_Name") or (str(raw_name).strip() or None)
    normalized["Test_Name"] = normalize_test_name(str(raw_name) if raw_name is not None else None)
    normalized["Test_Category"] = canonicalize_category(normalized.get("Test_Category"))
    normalized["Status"] = _normalize_status(normalized.get("Status"))

    return normalized


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        normalized_rows.append(normalize_record(row))
    return normalized_rows


def normalize_dataframe(df):
    if df is None or df.empty:
        return df

    normalized = df.copy()

    if "Test_Name" in normalized.columns:
        if "Original_Test_Name" not in normalized.columns:
            normalized["Original_Test_Name"] = normalized["Test_Name"]

        normalized["Original_Test_Name"] = normalized["Original_Test_Name"].fillna(normalized["Test_Name"])
        normalized["Test_Name"] = normalized["Test_Name"].fillna("").apply(lambda value: normalize_test_name(str(value)))

    if "Test_Category" in normalized.columns:
        normalized["Test_Category"] = normalized["Test_Category"].fillna("Other").apply(lambda value: canonicalize_category(str(value)))

    if "Status" in normalized.columns:
        normalized["Status"] = normalized["Status"].fillna("N/A").apply(lambda value: _normalize_status(str(value)))

    return normalized
