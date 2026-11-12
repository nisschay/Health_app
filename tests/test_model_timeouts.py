import logging

import pytest
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
    monkeypatch.setattr(llm.genai, "configure", lambda **k: None)
    monkeypatch.setattr(llm, "_genai_configured_key", None)
    monkeypatch.setattr(llm._model_call_pacer, "wait", lambda: None)


def test_chat_call_carries_a_real_deadline_and_reports_the_timeout(monkeypatch, caplog):
    """The old code timed out a future, then blocked in the executor's exit until Gemini answered anyway."""
    _stub(monkeypatch)
    with caplog.at_level(logging.ERROR, logger=chat.__name__), pytest.raises(chat.ChatUnavailable) as err:
        chat.ask_model("Why is my HbA1c high?", "test-key")

    assert _StallingModel.calls
    assert all(c["request_options"]["timeout"] == llm.CHAT_MODEL_TIMEOUT_SECONDS for c in _StallingModel.calls)
    assert "timed out" in caplog.text
    assert not err.value.rate_limited


def test_extraction_call_carries_a_real_deadline(monkeypatch):
    _stub(monkeypatch)
    text, error = extraction._generate_with_extraction_models("extract this")
    assert text == "" and isinstance(error, DeadlineExceeded)
    assert _StallingModel.calls[0]["request_options"]["timeout"] == llm.EXTRACTION_MODEL_TIMEOUT_SECONDS
