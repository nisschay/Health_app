import pytest

from Helper_Functions import create_structured_dataframe


def _payload(test_name, unit="mg/L", result="5.0"):
    return {
        "patient_info": {
            "name": "Test Patient",
            "age": "40",
            "gender": "F",
            "patient_id": "P1",
            "date": "05-03-2024",
            "lab_name": "Example Lab",
        },
        "test_results": [
            {
                "test_name": test_name,
                "result": result,
                "unit": unit,
                "reference_range": "1 - 10",
                "status": "Normal",
                "category": "Biochemistry",
            }
        ],
    }


def _single_row(monkeypatch, test_name, unit="mg/L"):
    # The classifier is a second Gemini call; bypass it so this stays a unit test.
    monkeypatch.setattr(
        "Helper_Functions._classify_test_statuses_with_gemini",
        lambda results, api_key: [],
    )
    df, _ = create_structured_dataframe(_payload(test_name, unit), "report.pdf")
    assert len(df) == 1
    return df.iloc[0]


@pytest.mark.parametrize(
    ("raw_name", "must_not_contain"),
    [
        ("Fasting Blood Sugar", "Aminotransferase"),
        ("Glucose Fasting", "Aminotransferase"),
        ("Mean Corpuscular Haemoglobin", "Haemoglobin (Hb)"),
        ("Alpha Fetoprotein", "Alkaline Phosphatase"),
    ],
)
def test_unanchored_regex_no_longer_renames_tests(monkeypatch, raw_name, must_not_contain):
    row = _single_row(monkeypatch, raw_name)
    assert must_not_contain not in row["Test_Name"]
    assert row["Original_Test_Name"] == raw_name


def test_fasting_glucose_stays_glucose(monkeypatch):
    row = _single_row(monkeypatch, "Fasting Blood Sugar")
    assert "Glucose" in row["Test_Name"]


@pytest.mark.parametrize("unit", ["mg/L", "mIU/L", "µg/L", "ng/mL"])
def test_units_are_preserved_verbatim(monkeypatch, unit):
    row = _single_row(monkeypatch, "C Reactive Protein", unit=unit)
    assert row["Unit"] == unit
