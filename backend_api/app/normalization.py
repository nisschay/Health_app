from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import pandas as pd

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

CATEGORY_PRIORITY: dict[str, int] = {
    "Haematology": 10,
    "Lipid Profile": 10,
    "Liver Function": 10,
    "Kidney Function": 10,
    "Diabetes & Glucose": 10,
    "Thyroid Function": 10,
    "Vitamins & Minerals": 10,
    "Hormones": 10,
    "Cardiac Markers": 10,
    "Immunology": 10,
    "Urinalysis": 10,
    "Inflammation": 10,
    "Proteins": 5,
    "Other": 1,
}

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
    "calcium": "Vitamins & Minerals",
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
    "mean platelet volume": "Mean Platelet Volume (MPV)",
    "mpv": "Mean Platelet Volume (MPV)",
    "platelet distribution width": "Platelet Distribution Width (PDW)",
    "pdw": "Platelet Distribution Width (PDW)",
    "rdw": "RDW (RBC Histogram)",
    "rdw rbc histogram": "RDW (RBC Histogram)",
    "rbc": "Red Blood Cell Count (RBC)",
    "red blood cell count": "Red Blood Cell Count (RBC)",
    "total rbc": "Red Blood Cell Count (RBC)",
    "rbc electrical impedance": "Red Blood Cell Count (RBC)",
    "total leucocyte count": "Total WBC Count",
    "total leucocytes count": "Total WBC Count",
    "total wbc count": "Total WBC Count",
    "wbc": "Total WBC Count",
    "tlc": "Total WBC Count",
    "cholesterol total": "Total Cholesterol",
    "cholesterol-total": "Total Cholesterol",
    "protein total": "Total Protein",
    "proteins total": "Total Protein",
    "bilirubin direct": "Bilirubin Direct (Conjugated)",
    "bilirubin conjugated": "Bilirubin Direct (Conjugated)",
    "malaria parasite": "Malarial Parasite Screen",
    "malarial parasite": "Malarial Parasite Screen",
    "r a test": "Rheumatoid Factor (RA)",
    "rheumatoid factor": "Rheumatoid Factor (RA)",
    "rheumatoid factor ra test": "Rheumatoid Factor (RA)",

    # Haematology - cell counts and differentials
    "hematocrit": "Haematocrit (HCT/PCV)",
    "hematocrit (pcv)": "Haematocrit (HCT/PCV)",
    "pcv(calc)": "Haematocrit (HCT/PCV)",
    "lymphocytes": "Lymphocytes %",
    "lymphocyte": "Lymphocytes %",
    "lymphocytes percentage": "Lymphocytes %",
    "lymphocytes - absolute count": "Lymphocytes (Absolute)",
    "lymphocytes-absolute count": "Lymphocytes (Absolute)",
    "lymphocytes absolute count": "Lymphocytes (Absolute)",
    "neutrophils": "Neutrophils %",
    "neutrophil": "Neutrophils %",
    "neutrophils percentage": "Neutrophils %",
    "neutrophils - absolute count": "Neutrophils (Absolute)",
    "neutrophils-absolute count": "Neutrophils (Absolute)",
    "neutrophil absolute": "Neutrophils (Absolute)",
    "neutrophil (absolute)": "Neutrophils (Absolute)",
    "eosinophils": "Eosinophils %",
    "eosinophil": "Eosinophils %",
    "eosinophils percentage": "Eosinophils %",
    "eosinophils - absolute count": "Eosinophils (Absolute)",
    "eosinophil (absolute)": "Eosinophils (Absolute)",
    "eosinophils-absolute count": "Eosinophils (Absolute)",
    "basophils": "Basophils %",
    "basophil": "Basophils %",
    "basophils percentage": "Basophils %",
    "basophils - absolute count": "Basophils (Absolute)",
    "basophil (absolute)": "Basophils (Absolute)",
    "basophils-absolute count": "Basophils (Absolute)",
    "monocytes": "Monocytes %",
    "monocytes percentage": "Monocytes %",
    "monocytes - absolute count": "Monocytes (Absolute)",
    "monocytes absolute": "Monocytes (Absolute)",
    "monocytes (absolute)": "Monocytes (Absolute)",
    "monocytes-absolute count": "Monocytes (Absolute)",
    "band cell": "Band Cells %",
    "platelet count": "Platelet Count",
    "platelets": "Platelet Count",
    "platelet (smear study)": "Platelet Count (Smear)",
    "neut/lympho ratio (nlr)": "Neutrophil/Lymphocyte Ratio (NLR)",
    "neut/lympho ratio nlr": "Neutrophil/Lymphocyte Ratio (NLR)",
    "immature granulocyte percentage": "Immature Granulocytes %",
    "immature granulocyte percentage(ig%)": "Immature Granulocytes %",
    "immature granulocyte percentage (ig%)": "Immature Granulocytes %",
    "immature granulocytes (ig)": "Immature Granulocytes (Absolute)",
    "nucleated red blood cells": "Nucleated RBC",
    "nucleated red blood cells %": "Nucleated RBC %",
    "platelet to large cell ratio (plcr)": "Platelet Large Cell Ratio (PLCR)",
    "platelet to large cell ratio plcr": "Platelet Large Cell Ratio (PLCR)",
    "plateletcrit (pct)": "Plateletcrit (PCT)",
    "plateletcrit pct": "Plateletcrit (PCT)",
    "red cell distribution width index (rdwi)": "RDW Index (RDWI)",
    "mentzer index": "Mentzer Index",
    "mch (calc)": "MCH",
    "mean corpuscular haemoglobin": "MCH",
    "mchc (calc)": "MCHC",
    "mean corp.hemo.conc(mchc)": "MCHC",
    "mcv (rbc histogram)": "Mean Corpuscular Volume (MCV)",
    "mean corpuscular volume (mcv)": "Mean Corpuscular Volume (MCV)",
    "mean corpuscular volume (mcv) ": "Mean Corpuscular Volume (MCV)",
    "mean corpuscular volume": "Mean Corpuscular Volume (MCV)",

    # Liver Function
    "albumin": "Albumin",
    "albumin - serum": "Albumin",
    "albumin serum": "Albumin",
    "globulin": "Globulin",
    "serum globulin": "Globulin",
    "albumin/globulin ratio (a/g ratio)": "Albumin/Globulin Ratio (A/G)",
    "albumin/globulin ratio": "Albumin/Globulin Ratio (A/G)",
    "a/g ratio": "Albumin/Globulin Ratio (A/G)",
    "serum alb/globulin ratio": "Albumin/Globulin Ratio (A/G)",
    "alb creat ratio": "Albumin/Creatinine Ratio",
    "alkaline phosphatase (alp)": "Alkaline Phosphatase (ALP)",
    "alkaline phosphatase": "Alkaline Phosphatase (ALP)",
    "gamma glutamyl transferase": "Gamma-Glutamyl Transferase (GGT)",
    "gamma glutamyl transferase (ggt)": "Gamma-Glutamyl Transferase (GGT)",
    "bilirubin total": "Bilirubin Total",
    "bilirubin - total": "Bilirubin Total",
    "bilirubin (indirect)": "Bilirubin Indirect",
    "bilirubin indirect": "Bilirubin Indirect",
    "bilirubin unconjugated": "Bilirubin Indirect",

    # Vitamins & Minerals
    "calcium": "Calcium",
    "iron": "Serum Iron",
    "total iron binding capacity (tibc)": "Total Iron Binding Capacity (TIBC)",
    "total iron binding capacity": "Total Iron Binding Capacity (TIBC)",
    "unsat.iron-binding capacity(uibc)": "Unsaturated Iron-Binding Capacity (UIBC)",
    "% transferrin saturation": "Transferrin Saturation %",
    "transferrin saturation": "Transferrin Saturation %",
    "folate": "Folate (Folic Acid)",
    "magnesium": "Magnesium",
    "phosphorous": "Phosphorus",

    # Electrolytes
    "sodium": "Sodium (Na)",
    "potassium": "Potassium (K)",
    "chloride": "Chloride (Cl)",

    # Kidney
    "uric acid": "Uric Acid",
    "est. glomerular filtration rate (egfr)": "eGFR (Estimated)",
    "microalbuminuria": "Microalbuminuria",

    # Lipids
    "vldl": "VLDL Cholesterol",
    "vldl cholesterol": "VLDL Cholesterol",
    "apolipoprotein - a1 (apo-a1)": "Apolipoprotein A1 (Apo-A1)",
    "apolipoprotein-a1 apo-a1": "Apolipoprotein A1 (Apo-A1)",
    "apolipoprotein - b (apo-b)": "Apolipoprotein B (Apo-B)",
    "apolipoprotein-b apo-b": "Apolipoprotein B (Apo-B)",
    "apo b / apo a1 ratio (apo b/a1)": "Apo B/A1 Ratio",
    "apo b/apo a1 ratio": "Apo B/A1 Ratio",
    "apo b/apo a1 ratio apo b/a1": "Apo B/A1 Ratio",
    "lipoprotein (a) [lp(a)]": "Lipoprotein (a) [Lp(a)]",

    # Hormones / Markers
    "testosterone": "Testosterone",
    "insulin fasting": "Insulin (Fasting)",
    "prostate specific antigen (psa)": "PSA (Total)",
    "carcino embryonic antigen (cea)": "CEA (Carcinoembryonic Antigen)",
    "anti ccp (accp)": "Anti-CCP Antibodies",
    "anti nuclear antibodies (ana)": "ANA (Anti-Nuclear Antibodies)",
    "high sensitivity c-reactive protein (hs-crp)": "hs-CRP",

    # Urinalysis
    "colour": "Urine Colour",
    "transparency": "Urine Transparency",
    "sp.gravity": "Urine Specific Gravity",
    "ph": "Urine pH",
    "leucocytes (esterase)": "Leucocyte Esterase (Urine)",
    "ketone bodies urine": "Urine Ketones",
    "urobilinogen": "Urobilinogen (Urine)",
    "nitrite": "Urine Nitrite",
    "epithelial cell (urine)": "Epithelial Cells (Urine)",
    "crystals (urine)": "Crystals (Urine)",
    "cast (urine)": "Casts (Urine)",
}

TEST_CATEGORY_HINTS: dict[str, str] = {
    "aspartate aminotransferase": "Liver Function",
    "sgot/ast": "Liver Function",
    "ldl cholesterol": "Lipid Profile",
    "total leucocyte count": "Haematology",
    "albumin": "Liver Function",
    "albumin/globulin ratio": "Liver Function",
    "globulin": "Liver Function",
    "triglycerides": "Lipid Profile",
    "calcium": "Vitamins & Minerals",
    "malarial parasite": "Haematology",
    "rheumatoid factor": "Immunology",
}

MUST_NOT_MERGE_RAW = [
    ("globulin", "albumin/globulin ratio"),
    ("albumin", "albumin/globulin ratio"),
    ("albumin", "alb creat ratio"),
    ("creatinine", "alb creat ratio"),
    ("iron", "total iron binding capacity"),
    ("iron", "transferrin saturation"),
    ("iron", "unsat iron-binding capacity"),
    ("apolipoprotein-a1", "apolipoprotein-b"),
    ("apo-a1", "apo-b"),
    ("apolipoprotein a", "apolipoprotein b"),
    ("bilirubin total", "bilirubin direct"),
    ("bilirubin total", "bilirubin indirect"),
    ("bilirubin total", "bilirubin conjugated"),
    ("bilirubin direct", "bilirubin indirect"),
    ("bilirubin conjugated", "bilirubin unconjugated"),
    ("neutrophils", "neut/lympho ratio"),
    ("neutrophil", "neut/lympho ratio"),
    ("prostate specific antigen", "free psa"),
    ("t3", "t4"),
    ("free t3", "free t4"),
    ("t3", "tsh"),
    ("t4", "tsh"),
]

_warned_categories: set[str] = set()
_warned_tests: set[str] = set()
_warned_other_category_tests: set[str] = set()
_normalized_test_name_map_cache: dict[str, str] | None = None


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


NORMALIZATION_ALIAS_WHITELIST_ONLY = _as_bool(
    os.getenv("NORMALIZATION_ALIAS_WHITELIST_ONLY"),
    default=True,
)


def _normalize_lookup_key(raw: str) -> str:
    key = raw.lower().strip()
    key = re.sub(r"[()\[\]]", " ", key)
    key = re.sub(r"\\+", " ", key)
    key = re.sub(r"[_]+", " ", key)
    key = re.sub(r"\s*/\s*", " / ", key)
    key = re.sub(r"[^a-z0-9\-&/+,.\s]", " ", key)
    key = re.sub(r"\s+", " ", key)
    return key.strip(" -")


def _normalized_test_name_map() -> dict[str, str]:
    global _normalized_test_name_map_cache
    if _normalized_test_name_map_cache is not None:
        return _normalized_test_name_map_cache

    mapped: dict[str, str] = {}
    for map_key, canonical in TEST_NAME_MAP.items():
        normalized_key = _normalize_lookup_key(map_key)
        if normalized_key:
            mapped[normalized_key] = canonical

    _normalized_test_name_map_cache = mapped
    return _normalized_test_name_map_cache


def _canonical_from_lookup(value: str) -> str | None:
    normalized_map = _normalized_test_name_map()
    lookup_key = _normalize_lookup_key(value)
    if lookup_key in normalized_map:
        return normalized_map[lookup_key]

    compact_lookup = _compact(lookup_key)
    for map_key, canonical in normalized_map.items():
        if _compact(map_key) == compact_lookup:
            return canonical
    return None


def _titlecase_fallback(raw: str) -> str:
    words = [part for part in re.split(r"\s+", raw.strip()) if part]
    return " ".join(word.capitalize() if len(word) > 4 else word.upper() for word in words)


def _unique_strings(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _coerce_aliases(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique_strings([str(item) for item in value])
    if isinstance(value, str):
        if not value.strip():
            return []
        return [value.strip()]
    return []


def _tokenize(value: str) -> list[str]:
    tokens = [token for token in re.split(r"[\s\-/,\.]+", value.lower()) if token]
    return tokens


def _initials(value: str) -> str:
    tokens = _tokenize(value)
    return "".join(token[0] for token in tokens if token)


def _compact(value: str) -> str:
    return re.sub(r"[\s\-/,\.]+", "", value.lower())


def structural_clean(raw: str | None) -> str:
    if raw is None:
        return "unknown test"

    cleaned = str(raw).lower().strip()
    if not cleaned:
        return "unknown test"

    qualifier_pattern = re.compile(
        r"\s*\(\s*(urine|serum|plasma|blood|routine|total|electrical\s+impedance|rbc\s+histogram|appearance)\s*\)",
        flags=re.IGNORECASE,
    )
    cleaned = qualifier_pattern.sub("", cleaned)

    abbr_pattern = re.compile(r"\(\s*([a-z]{2,6})\s*\)", flags=re.IGNORECASE)

    source = cleaned

    def _replace_abbr(match: re.Match[str]) -> str:
        abbr = match.group(1).lower().strip()
        base_segment = source[: match.start()].strip()
        base_tokens = [token for token in re.split(r"\s+", base_segment) if token]
        initials = "".join(token[0] for token in base_tokens if token)
        compact_base = "".join(base_tokens)
        if (initials and abbr in initials) or (compact_base and abbr in compact_base):
            return ""
        return match.group(0)

    cleaned = abbr_pattern.sub(_replace_abbr, source)
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)
    cleaned = re.sub(r"\s*-\s*", "-", cleaned)
    cleaned = re.sub(r"[^a-z0-9%\-/,\.\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned or "unknown test"


def _normalized_blocklist_key(value: str) -> str:
    cleaned = structural_clean(value)
    cleaned = re.sub(r"[\-/,\.]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_blocklisted(a: str, b: str) -> bool:
    a_norm = _normalized_blocklist_key(a)
    b_norm = _normalized_blocklist_key(b)
    shorter, longer = (a_norm, b_norm) if len(a_norm) <= len(b_norm) else (b_norm, a_norm)

    for p1, p2 in MUST_NOT_MERGE_RAW:
        p1_norm = _normalized_blocklist_key(p1)
        p2_norm = _normalized_blocklist_key(p2)
        if (p1_norm in shorter and p2_norm in longer) or (p2_norm in shorter and p1_norm in longer):
            return True
    return False


def are_same_test(a: str, b: str) -> bool:
    a_key = structural_clean(a)
    b_key = structural_clean(b)

    if a_key == b_key:
        return True

    if NORMALIZATION_ALIAS_WHITELIST_ONLY:
        a_canonical = _canonical_from_lookup(a_key)
        b_canonical = _canonical_from_lookup(b_key)
        return bool(a_canonical and b_canonical and a_canonical == b_canonical)

    # Hard blocklist first: these pairs must never merge regardless of similarity.
    if _is_blocklisted(a_key, b_key):
        return False

    # Structural rule 1: ratio tests should not merge with non-ratio tests.
    a_is_ratio = "ratio" in a_key or "/" in a_key
    b_is_ratio = "ratio" in b_key or "/" in b_key
    if a_is_ratio != b_is_ratio:
        return False

    # Structural rule 2: percentage and absolute count represent different measurements.
    a_is_percent = "percent" in a_key or a_key.endswith("%") or "(%)" in a_key
    b_is_percent = "percent" in b_key or b_key.endswith("%") or "(%)" in b_key
    a_is_absolute = bool(re.search(r"(?:absolute|count|(?:\\u00d7|x)10|10\^)", a_key))
    b_is_absolute = bool(re.search(r"(?:absolute|count|(?:\\u00d7|x)10|10\^)", b_key))
    if (a_is_percent and b_is_absolute) or (b_is_percent and a_is_absolute):
        return False

    # Structural rule 3: bilirubin qualifiers are distinct analytes.
    bilirubin_qualifiers = ["direct", "indirect", "total", "conjugated", "unconjugated"]
    if "bilirubin" in a_key and "bilirubin" in b_key:
        a_qualifier = next((q for q in bilirubin_qualifiers if q in a_key), None)
        b_qualifier = next((q for q in bilirubin_qualifiers if q in b_key), None)
        if a_qualifier and b_qualifier and a_qualifier != b_qualifier:
            return False
        if (a_qualifier and not b_qualifier) or (b_qualifier and not a_qualifier):
            return False

    initials_a = _initials(a_key)
    initials_b = _initials(b_key)
    compact_a = _compact(a_key)
    compact_b = _compact(b_key)
    if compact_a == initials_b or compact_b == initials_a:
        return True

    # Structural rule 4: token overlap applies only when both names have at least 2 meaningful tokens.
    tokens_a = [token for token in _tokenize(a_key) if len(token) > 2]
    tokens_b = [token for token in _tokenize(b_key) if len(token) > 2]
    if len(tokens_a) >= 2 and len(tokens_b) >= 2:
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        shorter = set_a if len(set_a) <= len(set_b) else set_b
        longer = set_b if len(set_a) <= len(set_b) else set_a
        if shorter:
            overlap = len([token for token in shorter if token in longer])
            if overlap / len(shorter) >= 0.8:
                return True

    strip_punct_a = _compact(a_key)
    strip_punct_b = _compact(b_key)
    if strip_punct_a == strip_punct_b:
        return True

    shorter_text, longer_text = (a_key, b_key) if len(a_key) <= len(b_key) else (b_key, a_key)
    if len(shorter_text) >= 8 and longer_text.startswith(shorter_text):
        return True

    return False


def _display_name_from_key(value: str) -> str:
    special_tokens = {
        "ph": "pH",
        "sgot": "SGOT",
        "sgpt": "SGPT",
        "ast": "AST",
        "alt": "ALT",
        "tlc": "TLC",
        "rbc": "RBC",
        "rdw": "RDW",
        "mpv": "MPV",
        "pdw": "PDW",
        "ldl": "LDL",
        "hdl": "HDL",
        "vldl": "VLDL",
        "ra": "RA",
        "hba1c": "HbA1C",
    }

    words = [token for token in re.split(r"\s+", value.strip()) if token]
    rendered: list[str] = []
    for word in words:
        key = word.lower()
        if key in special_tokens:
            rendered.append(special_tokens[key])
        elif len(key) <= 3 and key.isalpha():
            rendered.append(key.upper())
        else:
            rendered.append(key.capitalize())
    return " ".join(rendered)


def pick_canonical_name(names: list[str]) -> str:
    unique_names = _unique_strings(names)
    if not unique_names:
        return "Unknown Test"

    scored: list[tuple[int, str]] = []
    for name in unique_names:
        words = [w for w in re.split(r"\s+", name.strip()) if w]
        has_brackets = bool(re.search(r"\(.*\)", name))
        has_full_words = any(len(word) > 4 for word in words)
        letters = re.sub(r"[^A-Za-z]", "", name)
        is_all_caps = bool(letters) and letters == letters.upper()
        is_pure_abbr = bool(words) and all(len(re.sub(r"[^A-Za-z]", "", word)) <= 4 for word in words)

        score = len(name)
        if has_brackets and has_full_words:
            score += 20
        if is_all_caps:
            score -= 15
        if is_pure_abbr:
            score -= 20
        scored.append((score, name.strip()))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_name = scored[0][1]
    return best_name


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


def _normalize_test_map_key(value: str) -> str:
    key = structural_clean(value)
    key = key.replace("(", " ").replace(")", " ")
    key = re.sub(r"\s+", " ", key).strip()
    return key


def normalize_test_name(raw: str | None) -> str:
    if raw is None:
        return "Unknown Test"

    raw_text = str(raw).strip()
    if not raw_text or raw_text.lower() in {"n/a", "na", "none"}:
        return "Unknown Test"

    key = _normalize_test_map_key(raw_text)
    if key in TEST_NAME_MAP:
        return TEST_NAME_MAP[key]

    normalized_map = _normalized_test_name_map()
    normalized_key = _normalize_lookup_key(key)
    if normalized_key in normalized_map:
        return normalized_map[normalized_key]

    compact_key = _compact(key)
    for map_key, canonical in TEST_NAME_MAP.items():
        map_compact = _compact(map_key)
        if compact_key == map_compact:
            return canonical

    for map_key, canonical in normalized_map.items():
        map_compact = _compact(map_key)
        if compact_key == map_compact:
            return canonical

    if not NORMALIZATION_ALIAS_WHITELIST_ONLY:
        for map_key, canonical in TEST_NAME_MAP.items():
            if are_same_test(key, map_key):
                return canonical

    if raw_text not in _warned_tests:
        _warned_tests.add(raw_text)
        print(f"[TestName] Unmapped test: \"{raw_text}\" - consider adding to TEST_NAME_MAP")

    return _display_name_from_key(key)


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


def _parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value).replace(",", "").strip())
        if parsed != parsed:
            return None
        return parsed
    except Exception:
        return None


def _parse_date_sort_value(value: str | None) -> float:
    if value is None:
        return float("inf")

    raw = str(value).strip()
    if not raw or raw.lower() == "n/a":
        return float("inf")

    candidates = (
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%y",
        "%d/%m/%y",
    )
    for fmt in candidates:
        try:
            return datetime.strptime(raw, fmt).timestamp()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(raw).timestamp()
    except Exception:
        return float("inf")


def _row_completeness_score(entry: dict[str, Any]) -> int:
    score = 0
    if _parse_numeric(entry.get("Result")) is not None:
        score += 2
    if str(entry.get("Reference_Range") or "").strip():
        score += 2
    if str(entry.get("Unit") or "").strip():
        score += 1
    status = str(entry.get("Status") or "").strip().lower()
    if status and status != "n/a":
        score += 1
    return score


def merge_readings_by_date(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for finding in findings:
        date_key = str(finding.get("Test_Date") or "N/A").strip() or "N/A"
        by_date[date_key].append(finding)

    merged_rows: list[dict[str, Any]] = []

    for date_key, entries in by_date.items():
        if len(entries) == 1:
            merged_rows.append(dict(entries[0]))
            continue

        with_numeric = [entry for entry in entries if _parse_numeric(entry.get("Result")) is not None]
        candidate_pool = with_numeric if with_numeric else entries

        best = sorted(
            candidate_pool,
            key=lambda item: (
                _row_completeness_score(item),
                len(str(item.get("Reference_Range") or "")),
                len(str(item.get("Unit") or "")),
            ),
            reverse=True,
        )[0]

        merged_rows.append(dict(best))

    merged_rows.sort(key=lambda item: _parse_date_sort_value(item.get("Test_Date")))
    return merged_rows


def _category_from_test_hint(test_name: str | None) -> str | None:
    if not test_name:
        return None

    cleaned = structural_clean(test_name)
    for hint_key, category in TEST_CATEGORY_HINTS.items():
        if are_same_test(cleaned, hint_key) or hint_key in cleaned or cleaned in hint_key:
            return category
    return None


def resolve_category(categories: list[str], canonical_name: str | None = None) -> str:
    hinted = _category_from_test_hint(canonical_name)
    if hinted:
        return hinted

    canonical_categories = [canonicalize_category(category) for category in categories]
    non_other = [category for category in canonical_categories if category != "Other"]
    if not non_other:
        return "Other"

    freq = Counter(non_other)
    ranked = sorted(
        freq.items(),
        key=lambda item: (
            CATEGORY_PRIORITY.get(item[0], 0),
            item[1],
        ),
        reverse=True,
    )
    return ranked[0][0]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)

    raw_name = normalized.get("Original_Test_Name") or normalized.get("Test_Name") or ""
    raw_name_text = str(raw_name).strip()

    normalized["Original_Test_Name"] = raw_name_text or None
    normalized["Test_Name"] = normalize_test_name(raw_name_text)
    normalized["Test_Category"] = canonicalize_category(normalized.get("Test_Category"))
    normalized["Status"] = _normalize_status(normalized.get("Status"))

    existing_aliases = _coerce_aliases(normalized.get("Aliases"))
    normalized["Aliases"] = _unique_strings(existing_aliases + [raw_name_text, normalized.get("Test_Name")])

    return normalized


def deduplicate_findings(
    findings: list[dict[str, Any]],
    *,
    assume_normalized: bool = False,
) -> list[dict[str, Any]]:
    if not findings:
        return []

    key_map: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for finding in findings:
        if not isinstance(finding, dict):
            continue

        normalized_row = dict(finding) if assume_normalized else normalize_record(finding)
        row_name = (
            normalized_row.get("Test_Name")
            or normalized_row.get("Original_Test_Name")
            or "Unknown Test"
        )
        key = structural_clean(str(row_name))
        key_map[key].append(normalized_row)

    keys = list(key_map.keys())
    adjacency: dict[str, set[str]] = {key: set() for key in keys}

    for i, key in enumerate(keys):
        for j in range(i + 1, len(keys)):
            candidate = keys[j]
            if are_same_test(key, candidate):
                adjacency[key].add(candidate)
                adjacency[candidate].add(key)

    clusters: dict[str, list[str]] = {}
    visited: set[str] = set()

    for key in keys:
        if key in visited:
            continue

        stack = [key]
        cluster: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

        canonical_key = sorted(cluster, key=len, reverse=True)[0]
        clusters[canonical_key] = cluster

    deduped_rows: list[dict[str, Any]] = []

    for canonical_key, alias_keys in clusters.items():
        all_rows = [row for key in alias_keys for row in key_map.get(key, [])]
        raw_names = _unique_strings(
            [
                str(row.get("Original_Test_Name") or row.get("Test_Name") or "").strip()
                for row in all_rows
            ]
        )
        aliases = _unique_strings(
            [
                alias
                for row in all_rows
                for alias in _coerce_aliases(row.get("Aliases"))
            ]
            + raw_names
        )

        canonical_name = normalize_test_name(pick_canonical_name(raw_names))
        canonical_category = resolve_category(
            [str(row.get("Test_Category") or "Other") for row in all_rows],
            canonical_name=canonical_name,
        )

        if len(alias_keys) > 1:
            print(
                f"[Dedup] Merged {len(alias_keys)} variants -> \"{canonical_name}\": "
                + ", ".join(f'\"{key}\"' for key in alias_keys)
            )

        merged_rows = merge_readings_by_date(all_rows)
        for row in merged_rows:
            merged = dict(row)
            previous_name = str(merged.get("Test_Name") or merged.get("Original_Test_Name") or canonical_name).strip()
            merged["Test_Name"] = canonical_name
            merged["Test_Category"] = canonical_category
            merged["Aliases"] = _unique_strings(aliases + [canonical_name])
            if not merged.get("Original_Test_Name"):
                merged["Original_Test_Name"] = previous_name or canonical_name

            if previous_name and previous_name != canonical_name:
                date_text = str(merged.get("Test_Date") or "N/A")
                print(f"[MERGE] \"{previous_name}\" -> \"{canonical_name}\" ({date_text})")

            deduped_rows.append(merged)

        if canonical_category == "Other":
            category_warning_key = f"{canonical_name}|{canonical_key}"
            if category_warning_key not in _warned_other_category_tests:
                _warned_other_category_tests.add(category_warning_key)
                raw_category = str(all_rows[0].get("Test_Category") or "N/A") if all_rows else "N/A"
                print(
                    "[Normalization] Unknown category for: "
                    f"\"{canonical_name}\" - raw category was: \"{raw_category}\". "
                    "Add to CATEGORY_MAP."
                )

    deduped_rows.sort(
        key=lambda row: (
            _parse_date_sort_value(str(row.get("Test_Date") if row.get("Test_Date") is not None else "N/A")),
            str(row.get("Test_Name") or ""),
        )
    )
    return deduped_rows


def normalize_records(records: list[dict[str, Any]], *, deduplicate: bool = True) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        normalized_rows.append(normalize_record(row))

    if not deduplicate:
        return normalized_rows

    return deduplicate_findings(normalized_rows, assume_normalized=True)


def normalize_dataframe(df: pd.DataFrame, *, deduplicate: bool = True) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    original_columns = list(df.columns)
    records = df.to_dict(orient="records")
    normalized_records = normalize_records(records, deduplicate=deduplicate)

    normalized_df = pd.DataFrame(normalized_records)

    required_columns = list(original_columns)
    for column in ("Original_Test_Name", "Aliases", "Test_Name", "Test_Category", "Status"):
        if column not in required_columns:
            required_columns.append(column)

    for column in required_columns:
        if column not in normalized_df.columns:
            normalized_df[column] = None

    return normalized_df.reindex(columns=required_columns)
