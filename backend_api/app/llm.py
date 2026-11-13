"""Model calls shared by extraction and chat: a chain of free models, pacing, and waiting out rate limits.

A model is named "gemini-3.5-flash" (Gemini, with GEMINI_API_KEY) or "<provider>:<model>" for an
OpenAI-compatible host listed in PROVIDERS, e.g. "groq:llama-3.3-70b-versatile". A provider whose key
is not set is skipped, so the chain can list more than is configured.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

logger = logging.getLogger(__name__)


def _model_candidates_from_env(var_name: str, default: str) -> list[str]:
    raw = os.getenv(var_name, default)
    return [m.strip() for m in raw.split(",") if m.strip()]


def _env_bool(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Free hosts that speak the OpenAI chat API: name -> (base URL, environment variable holding the key).
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "github": ("https://models.github.ai/inference", "GITHUB_MODELS_TOKEN"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
}

# Every free Gemini model has its own daily quota, so the chain walks through them before other hosts.
# The first model is preferred; the next ones answer when it is overloaded, missing or out of quota.
_FREE_CHAIN = ",".join([
    # Groq first: fast, read every test report correctly, and does not train on API data.
    # Its free tier allows about 8,000 tokens a minute (three reports), so busy minutes spill over to Gemini.
    "groq:openai/gpt-oss-120b",
    "gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-2.5-flash",
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "groq:qwen/qwen3.8-27b",
])
EXTRACTION_MODEL_CANDIDATES = _model_candidates_from_env("EXTRACTION_MODELS", os.getenv("GEMINI_EXTRACTION_MODELS", _FREE_CHAIN))
CHAT_MODEL_CANDIDATES = _model_candidates_from_env("CHAT_MODELS", os.getenv("GEMINI_CHAT_MODELS", _FREE_CHAIN))
GEMINI_RPM = max(1, int(os.getenv("GEMINI_RPM", "10")))
CHAT_MODEL_TIMEOUT_SECONDS = max(5, int(os.getenv("CHAT_MODEL_TIMEOUT_SECONDS", "50")))
# Gemini sets its server deadline from this, so a hung model fails at 60s instead of 120s; reads take 2-15s.
EXTRACTION_MODEL_TIMEOUT_SECONDS = max(30, int(os.getenv("EXTRACTION_MODEL_TIMEOUT_SECONDS", "60")))

# A rate limit that clears within this many seconds is waited out instead of failing the file.
RATE_LIMIT_MAX_WAIT_SECONDS = 60
RATE_LIMIT_WAITS_PER_CALL = 2
# A host asking for this little is waited for on the same model instead of moving down the chain.
SHORT_RATE_LIMIT_SECONDS = 10
_DEFAULT_RATE_LIMIT_WAIT_SECONDS = 15
# A model that was overloaded or timed out is tried last for this long, so each file does not wait on it again.
MODEL_COOLDOWN_SECONDS = 300
# A model that refused the request (retired, bad key) or spent its quota is tried last for this long.
MISSING_MODEL_COOLDOWN_SECONDS = 3600
DAILY_LIMIT_COOLDOWN_SECONDS = 24 * 3600
# When every model is overloaded at once, wait this long and go round the list one more time.
OVERLOAD_RETRY_SECONDS = 10


class ModelUnavailable(Exception):
    """No model answered. `rate_limited` says whether quota was the reason."""

    def __init__(self, message: str, *, rate_limited: bool, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.rate_limited = rate_limited
        self.retry_after = retry_after


def format_delay(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    if seconds < 120:
        return f"{int(round(seconds))}s"
    if seconds < 7200:
        return f"{int(round(seconds / 60))}m"
    return f"{int(round(seconds / 3600))}h"


_RETRY_PATTERNS = (
    (re.compile(r"retry in\s+([\d.]+)\s*ms", re.IGNORECASE), 0.001),
    (re.compile(r"retry in\s+([\d.]+)\s*s", re.IGNORECASE), 1.0),
    (re.compile(r"retryDelay['\"]?\s*:\s*['\"]([\d.]+)s", re.IGNORECASE), 1.0),
)


def retry_after_seconds(error_text: str) -> float | None:
    """The wait Gemini asks for in a rate-limit error, if it names one."""
    for pattern, scale in _RETRY_PATTERNS:
        match = pattern.search(error_text or "")
        if match:
            try:
                return float(match.group(1)) * scale
            except ValueError:
                return None
    return None


class _CallPacer:
    """Sliding-window limiter shared by all threads, paced to the provider's RPM.

    `pause` holds every caller back after a rate limit, so parallel workers stop
    hammering a quota that is already spent.
    """

    def __init__(self, per_minute: int) -> None:
        self._per_minute = per_minute
        self._calls: deque[float] = deque()
        self._resume_at = 0.0
        self._lock = threading.Lock()

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._resume_at = max(self._resume_at, time.monotonic() + seconds)

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60:
                    self._calls.popleft()
                if now >= self._resume_at and len(self._calls) < self._per_minute:
                    self._calls.append(now)
                    return
                delay = max(self._resume_at - now, 60 - (now - self._calls[0]) if len(self._calls) >= self._per_minute else 0)
            time.sleep(max(delay, 0.05))


_model_call_pacer = _CallPacer(GEMINI_RPM)

_cooling_until: dict[str, float] = {}  # overloaded or slow: tried last
_resting_until: dict[str, float] = {}  # rate limited, retired or refused: skipped until then
_cooling_lock = threading.Lock()


def _cool_down(model: str, seconds: float = MODEL_COOLDOWN_SECONDS) -> None:
    with _cooling_lock:
        _cooling_until[model] = max(_cooling_until.get(model, 0.0), time.monotonic() + seconds)


def _in_preferred_order(models: list[str]) -> list[str]:
    """Healthy models first, in their configured order; cooling ones after."""
    now = time.monotonic()
    with _cooling_lock:
        return sorted(models, key=lambda m: _cooling_until.get(m, 0.0) > now)


_clients: dict[str, genai.Client] = {}
_clients_lock = threading.Lock()


def _client(api_key: str) -> genai.Client:
    with _clients_lock:
        if api_key not in _clients:
            _clients[api_key] = genai.Client(api_key=api_key)
        return _clients[api_key]


class ProviderError(Exception):
    """An HTTP error from an OpenAI-compatible host, with the status code generate() decides on."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _provider_of(model: str) -> tuple[str, str]:
    provider, sep, name = model.partition(":")
    return (provider, name) if sep and provider in PROVIDERS else ("gemini", model)


def _key_for(model: str, gemini_key: str | None) -> str | None:
    provider, _ = _provider_of(model)
    return gemini_key if provider == "gemini" else os.getenv(PROVIDERS[provider][1])


def _call_openai_compatible(provider: str, model: str, prompt: str, config: dict[str, Any], timeout_seconds: int) -> str:
    base_url, key_env = PROVIDERS[provider]
    body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if "temperature" in config:
        body["temperature"] = config["temperature"]
    if config.get("response_mime_type") == "application/json":
        body["response_format"] = {"type": "json_object"}  # the prompt itself spells out the schema
    response = httpx.post(
        f"{base_url}/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {os.getenv(key_env)}"},
        timeout=timeout_seconds,
    )
    if response.status_code >= 400:
        retry = response.headers.get("retry-after")
        hint = f" Please retry in {retry}s." if retry and retry.replace(".", "", 1).isdigit() else ""
        raise ProviderError(response.status_code, f"{provider} {response.status_code}: {response.text[:300]}{hint}")
    choices = response.json().get("choices") or [{}]
    return ((choices[0].get("message") or {}).get("content") or "").strip()


_THINKING_BUDGETS = {"minimal": 0, "low": 1024, "medium": 4096, "high": 8192}


def _call_model(api_key: str, model: str, prompt: str, config: dict[str, Any], timeout_seconds: int) -> str:
    """One request to one model. The only function that talks to a model host."""
    provider, name = _provider_of(model)
    if provider != "gemini":
        return _call_openai_compatible(provider, name, prompt, config, timeout_seconds)
    level = (config.get("thinking_config") or {}).get("thinking_level")
    if level and name.startswith("gemini-2."):
        # Gemini 2.x takes a token budget instead of a level.
        config = {**config, "thinking_config": {"thinking_budget": _THINKING_BUDGETS.get(level, 1024)}}
    response = _client(api_key).models.generate_content(
        model=model,
        contents=prompt,
        # v1beta: the v1 API refuses JSON mode for some models (gemini-2.5-flash).
        config=types.GenerateContentConfig(**config, http_options=types.HttpOptions(timeout=timeout_seconds * 1000, api_version="v1beta")),
    )
    return (response.text or "").strip()


def _rest(model: str, seconds: float) -> None:
    with _cooling_lock:
        _resting_until[model] = max(_resting_until.get(model, 0.0), time.monotonic() + seconds)


def _rest_left(model: str) -> float:
    with _cooling_lock:
        return max(0.0, _resting_until.get(model, 0.0) - time.monotonic())


def generate(prompt: str, *, api_key: str | None, models: list[str], config: dict[str, Any], timeout_seconds: int) -> str:
    """Ask each model in turn and return the first non-empty answer.

    A rate-limited model rests for as long as its host asks and the next model answers
    instead; only when every model is resting for a short while does the call wait. An
    overloaded, missing or slow model hands over to the next one, and if every model was
    overloaded the list is tried once more after a short wait. Raises ModelUnavailable
    when none answers.
    """
    models = [m for m in models if _key_for(m, api_key)]
    if not models:
        raise ModelUnavailable("No model is configured: set GEMINI_API_KEY or a provider key.", rate_limited=False)
    waits_left = RATE_LIMIT_WAITS_PER_CALL
    short_waits_left = 3
    overload_retry_used = False
    while True:
        overloaded = False
        for model in _in_preferred_order(models):
            if _rest_left(model) > 0:
                continue
            _model_call_pacer.wait()
            try:
                text = _call_model(_key_for(model, api_key), model, prompt, config, timeout_seconds)
            except (errors.APIError, ProviderError) as exc:
                if exc.code == 429:
                    delay = retry_after_seconds(str(exc))
                    wait = _DEFAULT_RATE_LIMIT_WAIT_SECONDS if delay is None else delay
                    if wait <= SHORT_RATE_LIMIT_SECONDS and short_waits_left > 0:
                        # A few seconds on a fast model beats a slow or overloaded one further down.
                        short_waits_left -= 1
                        _model_call_pacer.pause(wait)
                        _model_call_pacer.wait()
                        try:
                            text = _call_model(_key_for(model, api_key), model, prompt, config, timeout_seconds)
                        except Exception as retry_exc:  # still limited or failed: let it rest and move on
                            logger.warning("Model %s failed after a short wait: %s", model, str(retry_exc)[:200])
                            _rest(model, max(wait, _DEFAULT_RATE_LIMIT_WAIT_SECONDS))
                            continue
                        if text:
                            return text
                        continue
                    _rest(model, min(wait, DAILY_LIMIT_COOLDOWN_SECONDS))
                    logger.warning("Model %s is rate limited for %s; trying the next model", model, format_delay(wait))
                    continue
                logger.warning("Model %s failed with %s: %s", model, exc.code, str(exc)[:200])
                if exc.code >= 500:
                    _cool_down(model)
                    overloaded = True
                else:  # retired model, bad key or a request it refuses: no point asking again soon
                    _rest(model, MISSING_MODEL_COOLDOWN_SECONDS)
                continue
            except Exception as exc:  # timeouts and network errors: the next model may still answer
                logger.warning("Model %s failed: %s: %s", model, type(exc).__name__, str(exc)[:200])
                _cool_down(model)
                overloaded = True
                continue
            if text:
                return text
            logger.warning("Model %s returned an empty answer", model)

        soonest = min(_rest_left(m) for m in models)
        if 0 < soonest <= RATE_LIMIT_MAX_WAIT_SECONDS and waits_left > 0:
            waits_left -= 1
            logger.warning("Every model is rate limited; waiting %.0fs for the first to free up", soonest)
            _model_call_pacer.pause(soonest)
            continue
        if overloaded and not overload_retry_used:
            overload_retry_used = True
            logger.warning("Every model is overloaded; trying again in %ss", OVERLOAD_RETRY_SECONDS)
            time.sleep(OVERLOAD_RETRY_SECONDS)
            continue
        if soonest > 0:
            raise ModelUnavailable("Rate limit reached.", rate_limited=True, retry_after=format_delay(soonest))
        raise ModelUnavailable("No model answered.", rate_limited=False)
