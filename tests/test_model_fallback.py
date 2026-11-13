"""How a Gemini call survives an overloaded model, a rate limit or a timeout."""
import httpx
import pytest

from backend_api.app import chat, llm

MODELS = ["primary", "fallback"]


def _error(code, message):
    cls = llm.errors.ClientError if code < 500 else llm.errors.ServerError
    return cls(code, {"error": {"code": code, "message": message}})


@pytest.fixture
def model(monkeypatch):
    """Scripted replies per model: an exception to raise or a string to return, consumed in order."""
    script, calls, pauses = {}, [], []

    def call(api_key, name, prompt, config, timeout_seconds):
        calls.append((name, timeout_seconds))
        reply = script[name].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "_call_model", call)
    monkeypatch.setattr(llm, "_cooling_until", {})
    monkeypatch.setattr(llm._model_call_pacer, "wait", lambda: None)
    monkeypatch.setattr(llm._model_call_pacer, "pause", pauses.append)
    return script, calls, pauses


def _generate(models=MODELS):
    return llm.generate("p", api_key="k", models=models, config={}, timeout_seconds=7)


def test_an_overloaded_model_hands_over_to_the_next(model):
    script, calls, _ = model
    script.update(primary=[_error(503, "This model is currently experiencing high demand.")], fallback=["answer"])
    assert _generate() == "answer"
    assert calls == [("primary", 7), ("fallback", 7)]


def test_an_overloaded_model_is_tried_last_for_a_while(model):
    """Without this, every file in an upload waited on the same overloaded model first."""
    script, calls, _ = model
    script.update(primary=[_error(503, "high demand")], fallback=["one", "two"])
    assert _generate() == "one"
    assert _generate() == "two"
    assert [name for name, _ in calls] == ["primary", "fallback", "fallback"]


def test_a_timeout_or_empty_answer_hands_over_to_the_next(model):
    script, calls, _ = model
    script.update(primary=[httpx.ReadTimeout("timed out"), ""], fallback=["", ""])
    with pytest.raises(llm.ModelUnavailable) as err:
        _generate()
    assert not err.value.rate_limited
    assert [name for name, _ in calls] == ["primary", "fallback", "fallback", "primary"]


def test_when_every_model_is_overloaded_the_list_is_tried_once_more(model):
    """Both models returned 503 at the same moment during a live test; a few seconds later they answered."""
    script, calls, _ = model
    busy = _error(503, "high demand")
    script.update(primary=[busy, busy], fallback=[busy, "answer"])
    assert _generate() == "answer"
    assert [name for name, _ in calls] == ["primary", "fallback", "primary", "fallback"]


def test_a_model_that_does_not_exist_is_not_retried(model):
    script, calls, _ = model
    script.update(primary=[_error(404, "not found")], fallback=[_error(404, "not found")])
    with pytest.raises(llm.ModelUnavailable):
        _generate()
    assert len(calls) == 2


def test_a_short_rate_limit_is_waited_out_on_the_same_model(model):
    """Two of three files in one upload used to fail with 'Retry after 52s'."""
    script, calls, pauses = model
    script.update(primary=[_error(429, "Quota exceeded. Please retry in 52.3s."), "answer"])
    assert _generate() == "answer"
    assert [name for name, _ in calls] == ["primary", "primary"]
    assert pauses == [pytest.approx(52.3)]


def test_waits_are_capped_per_call(model):
    script, calls, pauses = model
    limited = _error(429, "Quota exceeded. Please retry in 5s.")
    script.update(primary=[limited, limited, limited], fallback=["answer"])
    assert _generate() == "answer"
    assert len(pauses) == llm.RATE_LIMIT_WAITS_PER_CALL
    assert [name for name, _ in calls] == ["primary"] * 3 + ["fallback"]


def test_a_long_rate_limit_is_not_waited_and_is_reported(model):
    script, _, pauses = model
    script.update(primary=[_error(429, "Quota exceeded. Please retry in 3600s.")])
    with pytest.raises(llm.ModelUnavailable) as err:
        _generate(["primary"])
    assert pauses == []
    assert err.value.rate_limited and err.value.retry_after == "60m"


def test_a_missing_key_never_calls_the_model(model):
    _, calls, _ = model
    with pytest.raises(llm.ModelUnavailable):
        llm.generate("p", api_key=None, models=MODELS, config={}, timeout_seconds=7)
    assert calls == []


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("Please retry in 52.3s.", 52.3), ("retry in 900ms", 0.9), ('"retryDelay": "20s"', 20.0), ("no hint", None)],
)
def test_retry_after_seconds(text, seconds):
    assert llm.retry_after_seconds(text) == (pytest.approx(seconds) if seconds else None)


def test_chat_uses_its_own_deadline_and_maps_rate_limits(model):
    script, calls, _ = model
    limited = _error(429, "Quota exceeded. Please retry in 3600s.")
    script.update({name: [limited] for name in llm.CHAT_MODEL_CANDIDATES})
    with pytest.raises(chat.ChatUnavailable) as err:
        chat.ask_model("Why is my HbA1c high?", "k")
    assert err.value.rate_limited
    assert all(timeout == llm.CHAT_MODEL_TIMEOUT_SECONDS for _, timeout in calls)


def test_pacer_pause_holds_every_caller(monkeypatch):
    clock = [100.0]
    sleeps = []
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock[0])

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(llm.time, "sleep", fake_sleep)
    pacer = llm._CallPacer(per_minute=10)
    pacer.pause(30)
    pacer.wait()
    assert sum(sleeps) == pytest.approx(30)
