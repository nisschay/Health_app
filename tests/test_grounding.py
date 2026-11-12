"""Extracted values must come from the report text."""
from backend_api.app import services
from backend_api.app.grounding import ground_results

TEXT = """--- Page 1 ---
Haemoglobin 12.40 g/dL 13.0 - 17.0
Platelet Count 2,50,000 /cumm 1,50,000 - 4,10,000
WBC 7.8 10^3/uL
TSH < 0.01 uIU/mL
Urine Glucose Present (++)
Ketones Nil
"""


def _payload(*rows):
    return {"patient_info": {}, "test_results": [{"test_name": n, "result": r} for n, r in rows]}


def test_values_the_report_contains_are_kept_however_they_are_written():
    payload = _payload(
        ("Haemoglobin", "12.4"),        # 12.40 in the text
        ("Platelets", "250000"),        # 2,50,000
        ("WBC", "7800"),                # 7.8 x10^3
        ("TSH", "<0.01"),
        ("Urine Glucose", "Present"),
        ("Ketones", "Nil"),
    )
    grounded, dropped = ground_results(payload, TEXT)
    assert dropped == []
    assert len(grounded["test_results"]) == 6


def test_an_invented_value_is_left_out_and_named():
    grounded, dropped = ground_results(_payload(("Haemoglobin", "12.4"), ("LDL", "131"), ("HbA1c", "6.2")), TEXT)
    assert [r["test_name"] for r in grounded["test_results"]] == ["Haemoglobin"]
    assert dropped == ["LDL", "HbA1c"]


def test_an_altered_value_is_left_out():
    """A model that rounds 12.40 to 12 changes the result."""
    _, dropped = ground_results(_payload(("Haemoglobin", "12")), TEXT)
    assert dropped == ["Haemoglobin"]


def test_a_word_result_must_appear_as_a_word():
    _, dropped = ground_results(_payload(("Ketones", "Positive")), TEXT)
    assert dropped == ["Ketones"]


def test_the_upload_reports_what_was_left_out(monkeypatch):
    monkeypatch.setattr(services, "extract_text_from_pdf", lambda payload: TEXT)
    monkeypatch.setattr(services, "get_cached_extraction", lambda h: None)
    monkeypatch.setattr(services, "store_cached_extraction", lambda h, p, owner: None)
    payload = {
        "patient_info": {"name": "A", "date": "05-03-2024"},
        "test_results": [
            {"test_name": "Haemoglobin", "result": "12.4", "unit": "g/dL", "reference_range": "13.0 - 17.0", "status": "Low", "category": "Haematology"},
            {"test_name": "LDL", "result": "131", "unit": "mg/dL", "reference_range": "<100", "status": "High", "category": "Lipid Profile"},
        ],
    }
    monkeypatch.setattr(services, "analyze_medical_report_with_gemini", lambda text, key: (payload, ""))
    result = services.MedicalAnalysisService(api_key="k").analyze_reports(pdf_files=[("r.pdf", b"%PDF")])
    assert [r["Test_Name"] for r in result["records"]] == ["Haemoglobin (Hb)"]
    assert result["notes"] == ["r.pdf: left out 1 value(s) the report does not contain (LDL)"]
