"""Convert between the record dicts the API speaks and the report_findings rows the database keeps."""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from Helper_Functions import parse_date_dd_mm_yyyy, parse_reference_range, parse_result_numeric

from .database import ReportFinding
from .normalization import STATUS_NOT_APPLICABLE


def _text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text and text.upper() != "N/A" else None


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = parse_date_dd_mm_yyyy(str(value)) if value else None
    return parsed.date() if isinstance(parsed, datetime) else parsed


def findings_from_records(profile_id: uuid.UUID, rows: list[dict[str, Any]], source_filename: str) -> list[ReportFinding]:
    findings = []
    for row in rows:
        name = _text(row.get("Test_Name"), 256)
        if not name:
            continue
        value, comparator = parse_result_numeric(row.get("Result"))
        ref_low, ref_high, _ = parse_reference_range(row.get("Reference_Range"))
        findings.append(
            ReportFinding(
                profile_id=profile_id,
                canonical_test=name,
                original_test_name=_text(row.get("Original_Test_Name"), 256),
                category=_text(row.get("Test_Category"), 128),
                test_date=_as_date(row.get("Test_Date")),
                result_text=_text(row.get("Result"), 256),
                value_numeric=value,
                comparator=comparator,
                unit=_text(row.get("Unit"), 64),
                reference_range=_text(row.get("Reference_Range"), 256),
                ref_low=ref_low,
                ref_high=ref_high,
                status=_text(row.get("Status"), 32) or STATUS_NOT_APPLICABLE,
                source_filename=_text(row.get("Source_Filename"), 512) or source_filename,
                aliases=[str(a) for a in row.get("Aliases") or []] or None,
            )
        )
    return findings


def record_from_finding(finding: ReportFinding) -> dict[str, Any]:
    """The MedicalRecord shape the frontend renders; the JSON payload is no longer read for this."""
    test_date = finding.test_date.strftime("%d-%m-%Y") if finding.test_date else "N/A"
    return {
        "Source_Filename": finding.source_filename,
        "Test_Date": test_date,
        "Test_Category": finding.category or "Other",
        "Original_Test_Name": finding.original_test_name or finding.canonical_test,
        "Test_Name": finding.canonical_test,
        "Aliases": finding.aliases or [finding.canonical_test],
        "Result": finding.result_text,
        "Unit": finding.unit or "",
        "Reference_Range": finding.reference_range or "",
        "Status": finding.status,
        "Result_Numeric": finding.value_numeric,
        "Test_Date_dt": finding.test_date.isoformat() if finding.test_date else None,
    }
