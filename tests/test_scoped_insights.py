from backend_api.app.services import MedicalAnalysisService

service = MedicalAnalysisService()


def _record(test_name, status, source):
    return {
        "Test_Name": test_name,
        "Test_Category": "Haematology",
        "Test_Date": "05-03-2024",
        "Result": "12.4",
        "Unit": "g/dL",
        "Reference_Range": "12 - 15",
        "Status": status,
        "Source_Filename": source,
    }


BATCH = [
    _record("Haemoglobin (Hb)", "High", "report_a.pdf"),
    _record("Serum Creatinine", "Low", "report_b.pdf"),
    _record("Serum Ferritin", "Critical", "report_c.pdf"),
]


def test_insights_describe_only_the_records_given():
    """Each saved report used to carry the whole batch's concerns."""
    whole_batch = service.get_health_insights(BATCH)
    assert len(whole_batch["health_summary"]["concerns"]) == 3

    for record in BATCH:
        scoped = service.get_health_insights([record])
        assert len(scoped["health_summary"]["concerns"]) == 1


def test_a_clean_report_has_no_concerns():
    scoped = service.get_health_insights([_record("Haemoglobin (Hb)", "Normal", "report_a.pdf")])
    assert scoped["health_summary"]["concerns"] == []


def test_empty_slice_does_not_raise():
    scoped = service.get_health_insights([])
    assert scoped["health_summary"]["concerns"] == []
