"""A report longer than one model call is read in parts, never cut short."""
import json

import pytest

from backend_api.app import extraction, llm


def _row(name, result):
    return {"test_name": name, "result": result, "unit": "mg/dL", "reference_range": "", "status": "N/A", "category": "Other"}


def _pages(n, chars):
    return "".join(f"\n--- Page {i} ---\n" + ("x" * chars) for i in range(1, n + 1))


def test_parts_are_whole_pages_and_lose_nothing():
    text = _pages(5, 400)
    chunks = extraction.split_into_chunks(text, max_chars=1000)
    assert "".join(chunks) == text
    assert all(len(c) <= 1000 for c in chunks)
    assert all(c.startswith("\n--- Page") for c in chunks), "no page is split across parts"


def test_a_single_huge_page_is_cut_by_length():
    chunks = extraction.split_into_chunks(_pages(1, 2500), max_chars=1000)
    assert len(chunks) == 3 and "".join(chunks) == _pages(1, 2500)


@pytest.fixture
def parts(monkeypatch):
    """Each call answers with the rows named in its own part of the text."""
    monkeypatch.setattr(extraction, "EXTRACTION_CHUNK_CHARS", 40)
    monkeypatch.setattr(llm._model_call_pacer, "wait", lambda: None)
    replies = {
        "PAGE-A": {"patient_info": {"name": "Asha", "date": "N/A"}, "test_results": [_row("LDL", "140")]},
        "PAGE-B": {"patient_info": {"name": "N/A", "date": "05-03-2024"}, "test_results": [_row("LDL", "140"), _row("HDL", "45")]},
        "PAGE-C": {"patient_info": {}, "test_results": []},
    }

    def call(api_key, model, prompt, config, timeout_seconds):
        for marker, payload in replies.items():
            if marker in prompt:
                return json.dumps({"abnormal_findings_summary_from_report": [], **payload})
        raise AssertionError("no marker in prompt")

    monkeypatch.setattr(llm, "_call_model", call)
    return replies


def test_a_long_report_is_read_in_parts_and_merged(parts):
    text = "\n--- Page 1 ---\nPAGE-A results" + "\n--- Page 2 ---\nPAGE-B results" + "\n--- Page 3 ---\nPAGE-C notes"
    payload, error = extraction.analyze_medical_report_with_gemini(text, "k")
    assert error == ""
    assert [(r["test_name"], r["result"]) for r in payload["test_results"]] == [("LDL", "140"), ("HDL", "45")], "LDL repeated on two pages is kept once"
    assert payload["patient_info"]["name"] == "Asha" and payload["patient_info"]["date"] == "05-03-2024"


def test_a_part_the_model_cannot_read_fails_the_file(parts, monkeypatch):
    def unavailable(*args):
        raise llm.errors.ServerError(503, {"error": {"code": 503, "message": "high demand"}})

    monkeypatch.setattr(llm, "_call_model", unavailable)
    monkeypatch.setattr(llm, "_cooling_until", {})
    payload, error = extraction.analyze_medical_report_with_gemini("\n--- Page 1 ---\nPAGE-A" + "\n--- Page 2 ---\nPAGE-B" * 3, "k")
    assert payload is None and error.startswith("Part 1 of")
