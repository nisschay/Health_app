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
    def pause(seconds):  # stands in for the wait: the resting models are free again afterwards
        pauses.append(seconds)
        llm._resting_until.clear()

    monkeypatch.setattr(llm._model_call_pacer, "pause", pause)
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


def test_a_rate_limited_model_steps_aside_for_the_next(model):
    """Waiting a minute on one model while another is free made uploads crawl."""
    script, calls, pauses = model
    script.update(primary=[_error(429, "Quota exceeded. Please retry in 52.3s.")], fallback=["answer"])
    assert _generate() == "answer"
    assert [name for name, _ in calls] == ["primary", "fallback"]
    assert pauses == []


def test_when_every_model_is_briefly_limited_the_call_waits_for_the_first(model):
    """Two of three files in one upload used to fail with 'Retry after 52s'."""
    script, calls, pauses = model
    script.update(primary=[_error(429, "Please retry in 52.3s."), "answer"], fallback=[_error(429, "Please retry in 70s.")])
    assert _generate() == "answer"
    assert [name for name, _ in calls] == ["primary", "fallback", "primary"]
    assert pauses == [pytest.approx(52.3, abs=1)]


def test_waits_are_capped_per_call(model):
    script, calls, pauses = model
    limited = _error(429, "Quota exceeded. Please retry in 30s.")
    script.update(primary=[limited] * 5)
    with pytest.raises(llm.ModelUnavailable) as err:
        _generate(["primary"])
    assert err.value.rate_limited
    assert len(pauses) == llm.RATE_LIMIT_WAITS_PER_CALL
    assert len(calls) == llm.RATE_LIMIT_WAITS_PER_CALL + 1


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


def test_a_model_whose_daily_quota_is_spent_is_skipped_until_it_resets(model):
    """Each free model has its own daily quota; asking a spent one again wastes a call per file."""
    script, calls, _ = model
    spent = _error(429, "Quota exceeded. Please retry in 79200s.")
    script.update(primary=[spent], fallback=["one", "two"])
    assert _generate() == "one"
    assert _generate() == "two"
    assert [name for name, _ in calls] == ["primary", "fallback", "fallback"]


def test_a_provider_without_a_key_is_left_out(model, monkeypatch):
    script, calls, _ = model
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    script.update({"groq:some-model": ["never"], "fallback": ["answer"]})
    assert _generate(["groq:some-model", "fallback"]) == "answer"
    assert [name for name, _ in calls] == ["fallback"]


def test_other_hosts_get_the_same_rate_limit_handling(model, monkeypatch):
    script, calls, pauses = model
    monkeypatch.setenv("GROQ_API_KEY", "g")
    script.update({"groq:some-model": [llm.ProviderError(429, "groq 429: slow down. Please retry in 7s."), "answer"]})
    assert _generate(["groq:some-model"]) == "answer"
    assert pauses == [pytest.approx(7, abs=1)]


def test_openai_compatible_request_shape(monkeypatch):
    """Extraction asks for JSON; the schema itself is in the prompt, since not every host accepts one."""
    sent = {}

    class Reply:
        status_code = 200
        headers: dict = {}

        def json(self):
            return {"choices": [{"message": {"content": " {\"ok\": 1} "}}]}

    def post(url, json, headers, timeout):
        sent.update(url=url, body=json, auth=headers["Authorization"], timeout=timeout)
        return Reply()

    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setattr(llm.httpx, "post", post)
    out = llm._call_model(None, "groq:llama-x", "prompt", {"temperature": 0.0, "response_mime_type": "application/json", "thinking_config": {}}, 30)
    assert out == '{"ok": 1}'
    assert sent["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert sent["body"] == {"model": "llama-x", "messages": [{"role": "user", "content": "prompt"}], "temperature": 0.0, "response_format": {"type": "json_object"}}
    assert sent["auth"] == "Bearer secret" and sent["timeout"] == 30


def test_a_few_seconds_limit_is_waited_on_the_same_fast_model(model):
    """Groq asked for 1s; moving on to an overloaded Gemini model cost two minutes instead."""
    script, calls, pauses = model
    script.update(primary=[_error(429, "Please retry in 1.2s."), "answer"], fallback=["slow answer"])
    assert _generate() == "answer"
    assert [name for name, _ in calls] == ["primary", "primary"]
    assert pauses == [pytest.approx(1.2)]
