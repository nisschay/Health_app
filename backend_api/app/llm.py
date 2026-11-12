"""Gemini plumbing shared by extraction and chat: configuration, pacing, rate-limit parsing."""
from __future__ import annotations

import os
import re
import threading
import time
from collections import deque

import google.generativeai as genai

_genai_configured_key: str | None = None


_genai_configure_lock = threading.Lock()


def _model_candidates_from_env(var_name: str, default: str) -> list[str]:
    raw = os.getenv(var_name, default)
    return [m.strip() for m in raw.split(",") if m.strip()]


def _env_bool(var_name: str, default: bool = False) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


EXTRACTION_MODEL_CANDIDATES = _model_candidates_from_env(
    "GEMINI_EXTRACTION_MODELS",
    "gemini-3.8-flash",
)


CHAT_MODEL_CANDIDATES = _model_candidates_from_env(
    "GEMINI_CHAT_MODELS",
    "gemini-3.8-flash",
)


GEMINI_RPM = max(1, int(os.getenv("GEMINI_RPM", "10")))


CHAT_MODEL_TIMEOUT_SECONDS = max(5, int(os.getenv("CHAT_MODEL_TIMEOUT_SECONDS", "50")))


EXTRACTION_MODEL_TIMEOUT_SECONDS = max(30, int(os.getenv("EXTRACTION_MODEL_TIMEOUT_SECONDS", "120")))


def _is_rate_limit_error(error_text: str) -> bool:
    lowered = (error_text or "").lower()
    return "429" in lowered or "quota exceeded" in lowered or "rate limit" in lowered


def _extract_retry_delay(error_text: str) -> str | None:
    if not error_text:
        return None

    def _format_delay(seconds: float) -> str:
        if seconds < 1:
            return "<1s"
        if seconds < 120:
            return f"{int(round(seconds))}s"
        if seconds < 7200:
            return f"{int(round(seconds / 60))}m"
        if seconds < 172800:
            return f"{int(round(seconds / 3600))}h"
        return f"{int(round(seconds / 86400))}d"

    match_sec = re.search(r"retry in\s+([\d\.]+)s", error_text, flags=re.IGNORECASE)
    if match_sec:
        try:
            return _format_delay(float(match_sec.group(1)))
        except ValueError:
            return f"{match_sec.group(1)}s"

    match_ms = re.search(r"retry in\s+([\d\.]+)ms", error_text, flags=re.IGNORECASE)
    if match_ms:
        try:
            return _format_delay(float(match_ms.group(1)) / 1000.0)
        except ValueError:
            return f"{match_ms.group(1)}ms"

    return None


class _CallPacer:
    """Sliding-window limiter shared by all threads, paced to the provider's RPM."""

    def __init__(self, per_minute: int) -> None:
        self._per_minute = per_minute
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60:
                    self._calls.popleft()
                if len(self._calls) < self._per_minute:
                    self._calls.append(now)
                    return
                delay = 60 - (now - self._calls[0])
            time.sleep(max(delay, 0.05))


_model_call_pacer = _CallPacer(GEMINI_RPM)


def _ensure_genai_configured(api_key: str | None) -> bool:
    global _genai_configured_key
    if not api_key:
        return False
    with _genai_configure_lock:
        if _genai_configured_key != api_key:
            genai.configure(api_key=api_key)
            _genai_configured_key = api_key
    return True


def _extract_text_from_response(response) -> str:
    response_text = getattr(response, "text", "") or ""
    if response_text:
        return response_text

    candidates = getattr(response, "candidates", None) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts).strip()
