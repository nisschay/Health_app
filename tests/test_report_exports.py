import io

import pandas as pd
import pytest
from pypdf import PdfReader

from Helper_Functions import (
    calculate_health_score,
    create_enhanced_excel_with_trends,
    generate_pdf_health_report,
)

PATIENT = {"name": "Test Patient", "age": "40", "gender": "F", "patient_id": "P1"}


def _records_df():
    return pd.DataFrame(
        [
            {
                "Test_Name": "Haemoglobin (Hb)",
                "Result": "12.4",
                "Unit": "g/dL",
                "Reference_Range": "12 - 15",
                "Status": "Normal",
                "Test_Category": "Haematology",
                "Test_Date": "05-03-2024",
            }
        ]
    )


def test_pdf_export_renders_the_result_value():
    """The table read a 'Value' column that never existed, so every row was N/A."""
    pdf_bytes = generate_pdf_health_report(_records_df(), PATIENT, api_key=None)
    text = "".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_bytes)).pages)
    assert "12.4" in text
    assert "N/A" not in text.split("Test Results Summary")[-1]


def test_excel_export_handles_more_than_26_columns():
    """chr(65 + n) produced '[' past column Z and xlsxwriter rejected the range."""
    columns = {"Test_Name": ["Haemoglobin (Hb)"], "Test_Category": ["Haematology"]}
    date_lab_cols = [f"2024-01-{day:02d}|Lab" for day in range(1, 28)]
    for column in date_lab_cols:
        columns[column] = ["12.4"]
    organized_df = pd.DataFrame(columns)

    payload = create_enhanced_excel_with_trends(
        organized_df,
        pd.DataFrame([{"Test_Name": "Haemoglobin (Hb)", "Reference_Range": "12 - 15"}]),
        date_lab_cols,
        PATIENT,
    )
    assert len(payload) > 0


@pytest.mark.parametrize("status", ["Abnormal", "Flagged", "Borderline", "Insufficient"])
def test_non_directional_statuses_reach_the_concerns_list(status):
    df = _records_df()
    df.loc[0, "Status"] = status
    summary = calculate_health_score(df)
    assert len(summary["concerns"]) == 1
    assert summary["overall_score"] < 100
