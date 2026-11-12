import pandas as pd
from google.api_core.exceptions import DeadlineExceeded

import Helper_Functions as helpers


class _StallingModel:
    """Records how it was called, then behaves like a request the SDK gave up on."""

    calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    def generate_content(self, prompt, **kwargs):
        _StallingModel.calls.append(kwargs)
        raise DeadlineExceeded("504 Deadline Exceeded")


def _stub(monkeypatch):
    _StallingModel.calls = []
    monkeypatch.setattr(helpers.genai, "GenerativeModel", _StallingModel)
    monkeypatch.setattr(helpers, "init_gemini_models", lambda *a, **k: True, raising=False)
    monkeypatch.setattr(helpers, "gemini_model_chat", None, raising=False)
    monkeypatch.setattr(helpers, "_active_chat_model_name", None, raising=False)
    monkeypatch.setattr(helpers._model_call_pacer, "wait", lambda: None)


def test_chat_call_carries_a_real_deadline_and_reports_the_timeout(monkeypatch):
    """The old code timed out a future, then blocked in the executor's exit until Gemini answered anyway."""
    _stub(monkeypatch)
    seen = {}
    monkeypatch.setattr(helpers, "_ui_error", lambda msg: seen.setdefault("msg", msg))

    answer = helpers.get_chatbot_response(pd.DataFrame([{"Test_Name": "HbA1c", "Result": "7.1"}]), "Why is my HbA1c high?", [], "test-key")

    assert _StallingModel.calls
    assert all(c["request_options"]["timeout"] == helpers.CHAT_MODEL_TIMEOUT_SECONDS for c in _StallingModel.calls)
    assert "timed out" in seen["msg"]
    assert "error" in answer.lower()


def test_extraction_call_carries_a_real_deadline(monkeypatch):
    _stub(monkeypatch)
    text, error = helpers._generate_with_extraction_models("extract this")
    assert text == "" and isinstance(error, DeadlineExceeded)
    assert _StallingModel.calls[0]["request_options"]["timeout"] == helpers.EXTRACTION_MODEL_TIMEOUT_SECONDS
