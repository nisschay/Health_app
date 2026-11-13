"""Gemini calls shared by extraction and chat: model fallback, pacing, and waiting out rate limits."""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from typing import Any

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


# The first model is preferred; the next ones answer when it is overloaded, missing or out of quota.
EXTRACTION_MODEL_CANDIDATES = _model_candidates_from_env("GEMINI_EXTRACTION_MODELS", "gemini-3.8-flash,gemini-3.5-flash")
CHAT_MODEL_CANDIDATES = _model_candidates_from_env("GEMINI_CHAT_MODELS", "gemini-3.8-flash,gemini-3.5-flash")
GEMINI_RPM = max(1, int(os.getenv("GEMINI_RPM", "10")))
CHAT_MODEL_TIMEOUT_SECONDS = max(5, int(os.getenv("CHAT_MODEL_TIMEOUT_SECONDS", "50")))
EXTRACTION_MODEL_TIMEOUT_SECONDS = max(30, int(os.getenv("EXTRACTION_MODEL_TIMEOUT_SECONDS", "120")))

# A rate limit that clears within this many seconds is waited out instead of failing the file.
RATE_LIMIT_MAX_WAIT_SECONDS = 60
RATE_LIMIT_WAITS_PER_CALL = 2
_DEFAULT_RATE_LIMIT_WAIT_SECONDS = 15
# A model that was overloaded or timed out is tried last for this long, so each file does not wait on it again.
MODEL_COOLDOWN_SECONDS = 300


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

_cooling_until: dict[str, float] = {}
_cooling_lock = threading.Lock()


def _cool_down(model: str) -> None:
    with _cooling_lock:
        _cooling_until[model] = time.monotonic() + MODEL_COOLDOWN_SECONDS


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


def _call_model(api_key: str, model: str, prompt: str, config: dict[str, Any], timeout_seconds: int) -> str:
    """One request to one model. The only function that talks to Gemini."""
    response = _client(api_key).models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config, http_options=types.HttpOptions(timeout=timeout_seconds * 1000)),
    )
    return (response.text or "").strip()


def generate(prompt: str, *, api_key: str | None, models: list[str], config: dict[str, Any], timeout_seconds: int) -> str:
    """Ask each model in turn and return the first non-empty answer.

    A short rate limit is waited out on the same model; an overloaded, missing or
    slow model hands over to the next one. Raises ModelUnavailable when none answers.
    """
    if not api_key:
        raise ModelUnavailable("GEMINI_API_KEY is not configured.", rate_limited=False)
    waits_left = RATE_LIMIT_WAITS_PER_CALL
    rate_limited, retry_after = False, None
    for model in _in_preferred_order(models):
        while True:
            _model_call_pacer.wait()
            try:
                text = _call_model(api_key, model, prompt, config, timeout_seconds)
            except errors.APIError as exc:
                if exc.code != 429:
                    logger.warning("Model %s failed with %s: %s", model, exc.code, str(exc)[:200])
                    if exc.code >= 500:
                        _cool_down(model)
                    break
                delay = retry_after_seconds(str(exc))
                wait = _DEFAULT_RATE_LIMIT_WAIT_SECONDS if delay is None else delay
                if waits_left > 0 and wait <= RATE_LIMIT_MAX_WAIT_SECONDS:
                    waits_left -= 1
                    logger.warning("Model %s is rate limited; waiting %.0fs", model, wait)
                    _model_call_pacer.pause(wait)
                    continue
                rate_limited, retry_after = True, format_delay(wait)
                logger.warning("Model %s is rate limited for %s; trying the next model", model, retry_after)
                break
            except Exception as exc:  # timeouts and network errors: the next model may still answer
                logger.warning("Model %s failed: %s: %s", model, type(exc).__name__, str(exc)[:200])
                _cool_down(model)
                break
            if text:
                return text
            logger.warning("Model %s returned an empty answer", model)
            break
    if rate_limited:
        raise ModelUnavailable("Rate limit reached.", rate_limited=True, retry_after=retry_after)
    raise ModelUnavailable("No model answered.", rate_limited=False)
