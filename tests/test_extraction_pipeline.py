import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend_api.app import extraction, llm, values

VALID_PAYLOAD = {
    "patient_info": {"name": "T", "age": "40", "gender": "F", "patient_id": "P1", "date": "05-03-2024", "lab_name": "L"},
    "test_results": [
        {"test_name": "HbA1c", "result": "7.1", "unit": "%", "reference_range": "4 - 5.6", "status": "N/A", "category": "Diabetes & Glucose"}
    ],
    "abnormal_findings_summary_from_report": [],
}


class _FakeModel:
    """Echoes a payload, or fails when the prompt asks it to."""

    calls = 0
    lock = threading.Lock()

    @staticmethod
    def call(api_key, model, prompt, config, timeout_seconds):
        with _FakeModel.lock:
            _FakeModel.calls += 1
        if "FAIL" in prompt:
            raise RuntimeError("boom")
        return json.dumps(VALID_PAYLOAD)


@pytest.fixture
def fake_gemini(monkeypatch):
    _FakeModel.calls = 0
    monkeypatch.setattr(llm, "_call_model", _FakeModel.call)
    monkeypatch.setattr(llm._model_call_pacer, "wait", lambda: None)
    monkeypatch.setattr(llm, "_cooling_until", {})
    return _FakeModel


def test_missing_key_and_empty_text_fail_without_calling_the_model(fake_gemini):
    assert extraction.analyze_medical_report_with_gemini("some text", None) == (None, "Gemini API key is missing or invalid.")
    assert extraction.analyze_medical_report_with_gemini("   ", "key")[0] is None
    assert fake_gemini.calls == 0


def test_a_report_costs_exactly_one_model_call(fake_gemini):
    """A second classification call used to double every upload's latency and quota."""
    payload, error = extraction.analyze_medical_report_with_gemini("Haemoglobin 12.4 g/dL", "key")
    assert error == ""
    assert payload["test_results"][0]["test_name"] == "HbA1c"
    assert fake_gemini.calls == 1

    df, _ = extraction.create_structured_dataframe(payload, "r.pdf")
    assert len(df) == 1
    assert fake_gemini.calls == 1, "building the dataframe must not call the model again"


def test_concurrent_calls_keep_their_own_result(fake_gemini):
    """The old module-level error string let one thread report another file's failure."""
    inputs = [f"report {'FAIL' if i % 2 else 'ok'} {i} text" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda text: extraction.analyze_medical_report_with_gemini(text, "key"), inputs))
    for i, (payload, error) in enumerate(results):
        if i % 2:
            assert payload is None and "unavailable" in error, f"input {i} got {error!r}"
        else:
            assert payload is not None and error == "", f"input {i} got {error!r}"


def test_a_long_rate_limit_is_reported_as_such(fake_gemini, monkeypatch):
    def limited(*args):
        raise llm.errors.ClientError(429, {"error": {"code": 429, "message": "Quota exceeded. Please retry in 300s."}})

    monkeypatch.setattr(llm, "_call_model", limited)
    payload, error = extraction.analyze_medical_report_with_gemini("text", "key")
    assert payload is None
    assert "used up its limit" in error and "5m" in error


def test_pacer_blocks_once_the_window_is_full(monkeypatch):
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        pacer._calls.clear()  # simulate the window rolling over

    monkeypatch.setattr(llm.time, "sleep", fake_sleep)
    pacer = llm._CallPacer(per_minute=2)
    pacer.wait()
    pacer.wait()
    assert sleeps == []
    pacer.wait()
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 60


@pytest.mark.parametrize(
    ("result", "reference", "reported", "expected"),
    [
        ("12.4", "12 - 15", "High", "High"),  # the report's own flag wins
        ("12.4", "12 - 15", "N/A", "Normal"),
        ("11.0", "12 - 15", "N/A", "Low"),
        ("16.0", "12 - 15", "N/A", "High"),
        ("< 0.5", "< 1.0", "N/A", "Normal"),
        ("250", "< 200", "N/A", "High"),
        ("45", "> 60", "N/A", "Low"),
        ("Not Detected", "Negative", "N/A", "Negative"),
        ("Positive", "N/A", "N/A", "Positive"),
        ("abc", "N/A", "N/A", "N/A"),
    ],
)
def test_derive_status(result, reference, reported, expected):
    assert values.derive_status(result, reference, reported) == expected
