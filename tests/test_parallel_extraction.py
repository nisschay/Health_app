import threading
import time

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend_api.app import database, services
from backend_api.app.services import MedicalAnalysisService

PAYLOAD = {
    "patient_info": {"name": "T", "age": "40", "gender": "F", "patient_id": "P1", "date": "05-03-2024", "lab_name": "L"},
    "test_results": [
        {"test_name": "HbA1c", "result": "7.1", "unit": "%", "reference_range": "4 - 5.6", "status": "High", "category": "Diabetes & Glucose"}
    ],
    "abnormal_findings_summary_from_report": [],
}


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(services, "extract_text_from_pdf", lambda payload: payload.decode())
    monkeypatch.setattr(services, "get_cached_extraction", lambda h: None)
    monkeypatch.setattr(services, "store_cached_extraction", lambda h, p: None)


def test_small_batches_run_in_parallel(offline, monkeypatch):
    """Under ten files everything used to run one at a time."""
    threads: set[str] = set()

    def slow_extract(text, key):
        threads.add(threading.current_thread().name)
        time.sleep(0.05)
        return PAYLOAD, ""

    monkeypatch.setattr(services, "analyze_medical_report_with_gemini", slow_extract)
    events = []
    result = MedicalAnalysisService(api_key="k").analyze_reports(
        pdf_files=[(f"r{i}.pdf", f"report {i}".encode()) for i in range(3)],
        progress_callback=events.append,
    )
    assert len(threads) >= 2, "three files should not share one worker thread"
    assert result["total_records"] == 3
    assert [e["file"] for e in events if e.get("step") == "done"].__len__() == 3


def test_a_failing_file_reports_its_own_reason(offline, monkeypatch):
    def extract(text, key):
        return (None, f"cannot read {text}") if "bad" in text else (PAYLOAD, "")

    monkeypatch.setattr(services, "analyze_medical_report_with_gemini", extract)
    events = []
    MedicalAnalysisService(api_key="k").analyze_reports(
        pdf_files=[("good.pdf", b"good report"), ("bad.pdf", b"bad report")],
        progress_callback=events.append,
    )
    failed = [e for e in events if e.get("step") == "failed"]
    assert len(failed) == 1
    assert failed[0]["file"] == "bad.pdf"
    assert failed[0]["error"] == "cannot read bad report"


def test_cache_hit_skips_the_model(monkeypatch):
    monkeypatch.setattr(services, "extract_text_from_pdf", lambda payload: "text")
    monkeypatch.setattr(services, "get_cached_extraction", lambda h: dict(PAYLOAD))

    def must_not_run(text, key):
        raise AssertionError("model called despite a cache hit")

    monkeypatch.setattr(services, "analyze_medical_report_with_gemini", must_not_run)
    outcome = services._process_single_pdf("r.pdf", b"x", "k", False)
    assert outcome["ok"] and isinstance(outcome["df"], pd.DataFrame)


def test_cache_failure_never_fails_extraction(offline, monkeypatch):
    def broken(*args):
        raise RuntimeError("database asleep")

    monkeypatch.setattr(services, "get_cached_extraction", broken)
    monkeypatch.setattr(services, "store_cached_extraction", broken)
    monkeypatch.setattr(services, "analyze_medical_report_with_gemini", lambda t, k: (PAYLOAD, ""))
    assert services._process_single_pdf("r.pdf", b"x", "k", False)["ok"]


def test_extraction_cache_round_trips(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    database.Base.metadata.create_all(engine, tables=[database.ExtractionCache.__table__])
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine))

    assert database.get_cached_extraction("abc") is None
    database.store_cached_extraction("abc", PAYLOAD)
    database.store_cached_extraction("abc", PAYLOAD)  # idempotent
    assert database.get_cached_extraction("abc") == PAYLOAD
