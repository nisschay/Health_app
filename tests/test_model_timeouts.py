import logging

import pandas as pd
from google.api_core.exceptions import DeadlineExceeded

from backend_api.app import chat, extraction, llm


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
    monkeypatch.setattr(llm.genai, "GenerativeModel", _StallingModel)
    monkeypatch.setattr(chat, "init_gemini_models", lambda *a, **k: True)
    monkeypatch.setattr(chat, "gemini_model_chat", None)
    monkeypatch.setattr(chat, "_active_chat_model_name", None)
    monkeypatch.setattr(llm._model_call_pacer, "wait", lambda: None)


def test_chat_call_carries_a_real_deadline_and_reports_the_timeout(monkeypatch, caplog):
    """The old code timed out a future, then blocked in the executor's exit until Gemini answered anyway."""
    _stub(monkeypatch)
    with caplog.at_level(logging.ERROR, logger=chat.__name__):
        answer = chat.get_chatbot_response(pd.DataFrame([{"Test_Name": "HbA1c", "Result": "7.1"}]), "Why is my HbA1c high?", [], "test-key")

    assert _StallingModel.calls
    assert all(c["request_options"]["timeout"] == llm.CHAT_MODEL_TIMEOUT_SECONDS for c in _StallingModel.calls)
    assert "timed out" in caplog.text
    assert "error" in answer.lower()


def test_extraction_call_carries_a_real_deadline(monkeypatch):
    _stub(monkeypatch)
    text, error = extraction._generate_with_extraction_models("extract this")
    assert text == "" and isinstance(error, DeadlineExceeded)
    assert _StallingModel.calls[0]["request_options"]["timeout"] == llm.EXTRACTION_MODEL_TIMEOUT_SECONDS
