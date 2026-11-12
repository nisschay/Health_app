"""Read lab values: numbers, comparators, reference ranges, statuses and report dates."""
from __future__ import annotations

import logging
import math
import re

import pandas as pd

from .normalization import (
    CANONICAL_STATUS_VALUES,
    STATUS_HIGH,
    STATUS_LOW,
    STATUS_NEGATIVE,
    STATUS_NORMAL,
    STATUS_NOT_APPLICABLE,
    STATUS_POSITIVE,
    normalize_status,
)

logger = logging.getLogger(__name__)


def format_date_dd_mm_yyyy(date_obj):
    """Format datetime object to DD-MM-YYYY string"""
    if pd.isna(date_obj) or date_obj is None:
        return 'N/A'
    return date_obj.strftime('%d-%m-%Y')


def _clean_text_value(value, default: str = "N/A") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


_RESULT_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


_COMPARATOR_CANONICAL = {
    "<": "<",
    "≤": "<=",
    "<=": "<=",
    "=<": "<=",
    ">": ">",
    "≥": ">=",
    ">=": ">=",
    "=>": ">=",
}


# Longest first so "<=" is not shadowed by "<".
_RESULT_COMPARATOR_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(op) for op in sorted(_COMPARATOR_CANONICAL, key=len, reverse=True)) + r")\s*"
)


# Thousands grouping in both styles: Indian "2,50,000" and Western "250,000".
_DIGIT_GROUPING_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{2})*,\d{3}\b|\b\d{1,3}(?:,\d{3})+\b")


def strip_digit_grouping(text: str) -> str:
    """'2,50,000' -> '250000'. Reading only the last group turned a platelet count of 2.5 lakh into 2."""
    return _DIGIT_GROUPING_PATTERN.sub(lambda match: match.group(0).replace(",", ""), text)


def parse_result_numeric(value) -> tuple[float | None, str | None]:
    """Parse a lab result into (numeric_value, comparator); "< 0.5" -> (0.5, "<")."""
    if value is None:
        return None, None

    if isinstance(value, bool):
        return None, None

    if isinstance(value, (int, float)):
        numeric = float(value)
        return (None, None) if math.isnan(numeric) else (numeric, None)

    text = str(value).strip()
    if not text:
        return None, None

    comparator = None
    comparator_match = _RESULT_COMPARATOR_PATTERN.match(text)
    if comparator_match:
        comparator = _COMPARATOR_CANONICAL.get(comparator_match.group(1))
        text = text[comparator_match.end():]

    text = strip_digit_grouping(text)

    number_match = _RESULT_NUMBER_PATTERN.search(text)
    if not number_match:
        return None, comparator

    try:
        numeric = float(number_match.group(0))
    except (TypeError, ValueError):
        return None, comparator

    if math.isnan(numeric) or math.isinf(numeric):
        return None, comparator

    return numeric, comparator


def result_series_to_numeric(series) -> pd.Series:
    """Vectorised parse_result_numeric for a dataframe column, values only."""
    return series.map(lambda item: parse_result_numeric(item)[0])


def _canonical_status(raw_status) -> str:
    """Map lab-supplied status text onto one canonical value, or N/A."""
    status = normalize_status(_clean_text_value(raw_status, default="N/A"))
    return status if status in CANONICAL_STATUS_VALUES else "N/A"


def _status_from_numbers(result, reference_range) -> str | None:
    value, _ = parse_result_numeric(result)
    low, high, kind = parse_reference_range(reference_range)
    if value is None or kind not in {"range", "less_than", "greater_than"}:
        return None
    if low is not None and value < low:
        return STATUS_LOW
    if high is not None and value > high:
        return STATUS_HIGH
    return STATUS_NORMAL


def derive_status(result, reference_range, reported_status) -> str:
    """The report's own flag, unless it says Normal for a value outside the report's own range.

    The model copies the lab's flag and sometimes misses one. A High or Low flag is kept even
    when the numbers look normal: labs flag on criteria a single range cannot express.
    """
    reported = _canonical_status(reported_status)
    computed = _status_from_numbers(result, reference_range)
    if reported == STATUS_NORMAL and computed in (STATUS_HIGH, STATUS_LOW):
        return computed
    if reported != STATUS_NOT_APPLICABLE:
        return reported
    if computed:
        return computed

    textual = _canonical_status(result)
    if textual in {STATUS_POSITIVE, STATUS_NEGATIVE}:
        return textual
    return reported


_RANGE_NUMBER = r"-?\d+(?:\.\d+)?"


_RANGE_PAIR_PATTERN = re.compile(rf"(?<![\d.])({_RANGE_NUMBER})\s*(?:-|–|to)\s*({_RANGE_NUMBER})", re.IGNORECASE)


_RANGE_UPPER_PATTERN = re.compile(rf"(?:<=?|≤|less\s+than|upto|up\s+to|below)\s*({_RANGE_NUMBER})", re.IGNORECASE)


_RANGE_LOWER_PATTERN = re.compile(rf"(?:>=?|≥|greater\s+than|more\s+than|above)\s*({_RANGE_NUMBER})", re.IGNORECASE)


_RANGE_SEGMENT_SPLIT = re.compile(r"[;,\n|]")


_RANGE_EXPRESSION = (
    rf"(?:(?:<=?|>=?|≤|≥|below|less\s+than|above|greater\s+than|upto|up\s+to)\s*{_RANGE_NUMBER}"
    rf"|{_RANGE_NUMBER}\s*(?:-|–|to)\s*{_RANGE_NUMBER})"
)


# Labels a lab uses for its healthy tier. "Near optimal" and "above optimal" are not healthy,
# and risk labels are left out on purpose: "low risk" for HDL is not the normal band.
_NORMAL_TIER_LABEL = r"(?<!near\s)(?<!above\s)\b(?:normal|desirable|optimal|sufficien(?:t|cy))\b"


_TIER_NUMBER_THEN_LABEL = re.compile(rf"({_RANGE_EXPRESSION})\s*[:\-]?\s*{_NORMAL_TIER_LABEL}", re.IGNORECASE)


_TIER_LABEL_THEN_NUMBER = re.compile(rf"{_NORMAL_TIER_LABEL}(?:\s+level)?\s*[:\-]?\s*({_RANGE_EXPRESSION})", re.IGNORECASE)


def _is_ambiguous_range(text: str) -> bool:
    """More than one band or threshold: which one is "normal" cannot be read off the numbers."""
    numeric_segments = [part for part in _RANGE_SEGMENT_SPLIT.split(text) if re.search(r"\d", part)]
    pairs = _RANGE_PAIR_PATTERN.findall(text)
    without_pairs = _RANGE_PAIR_PATTERN.sub(" ", text)
    thresholds = _RANGE_UPPER_PATTERN.findall(without_pairs) + _RANGE_LOWER_PATTERN.findall(without_pairs)
    return len(numeric_segments) > 1 or len(pairs) + len(thresholds) > 1


def _labelled_normal_tier(text: str) -> str | None:
    """The one band the lab itself labels healthy ("< 200 Desirable", "Normal : Below 5.7"), else None.

    Two healthy bands (adult and paediatric vitamin D) are as ambiguous as none.
    """
    for pattern in (_TIER_NUMBER_THEN_LABEL, _TIER_LABEL_THEN_NUMBER):
        found = {re.sub(r"\s+", " ", match.strip().lower()) for match in pattern.findall(text)}
        if found:
            return found.pop() if len(found) == 1 else None
    return None


def parse_reference_range(ref_range_str):
    """(low, high, kind) for a single range or threshold; (None, None, None) when ambiguous.

    Tiered ranges ("Desirable <200, Borderline 200-239") use the one tier the lab labels
    healthy. Taking the first number pair instead turned a desirable cholesterol into "Low".
    Ranges per sex or age ("Male 13-17 Female 12-15") stay unparsed, and the status falls
    back to the lab's own flag: unknown is better than wrong.
    """
    if not isinstance(ref_range_str, str) or ref_range_str.lower() == 'n/a' or not ref_range_str.strip():
        return None, None, None

    text = strip_digit_grouping(ref_range_str.strip())
    lowered = text.lower()
    if lowered in ["negative", "non reactive", "not detected", "nil", "absent", "not present"]:
        return None, None, "qualitative_normal"
    if lowered in ["positive", "reactive", "detected"]:
        return None, None, "qualitative_abnormal"

    if _is_ambiguous_range(text):
        normal_tier = _labelled_normal_tier(text)
        if normal_tier is None or _is_ambiguous_range(normal_tier):
            return None, None, None
        text = normal_tier

    pairs = _RANGE_PAIR_PATTERN.findall(text)
    uppers = _RANGE_UPPER_PATTERN.findall(_RANGE_PAIR_PATTERN.sub(" ", text))
    lowers = _RANGE_LOWER_PATTERN.findall(_RANGE_PAIR_PATTERN.sub(" ", text))

    try:
        if pairs:
            low, high = float(pairs[0][0]), float(pairs[0][1])
            return (low, high, "range") if low <= high else (None, None, None)
        if uppers:
            return None, float(uppers[0]), "less_than"
        if lowers:
            return float(lowers[0]), None, "greater_than"
    except ValueError:
        pass
    return None, None, None


# Add this debugging version of the main workflow
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
    except (ValueError, TypeError, OverflowError):
        try:
            # Try DD-MM-YYYY
            return pd.to_datetime(date_str, format='%d-%m-%Y', errors='raise')
        except (ValueError, TypeError, OverflowError):
            try:
                # Try DD.MM.YYYY
                return pd.to_datetime(date_str, format='%d.%m.%Y', errors='raise')
            except (ValueError, TypeError, OverflowError):
                try:
                    # If all else fails, try to parse and assume it's in DD/MM/YYYY format
                    # Force dayfirst=True to ensure DD/MM/YYYY interpretation
                    return pd.to_datetime(date_str, dayfirst=True, errors='raise')
                except (ValueError, TypeError, OverflowError):
                    logger.debug("Could not parse date: %s", date_str)
                    return None
