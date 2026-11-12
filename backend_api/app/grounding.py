"""Keep only extracted results the report's own text contains.

The model sees nothing but the report text, so a value that is not in that text was
invented or altered. Such a value is left out and reported, never shown as a result.
"""
from __future__ import annotations

import re
from typing import Any

from .values import parse_result_numeric, strip_digit_grouping

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Labs write large counts as "2.5 lakh" or "250 x10^3"; the model may write 250000.
_SCALES = (1, 1_000, 100_000)


def _numbers_in(text: str) -> set[float]:
    return {round(float(n), 6) for n in _NUMBER.findall(strip_digit_grouping(text))}


def _words_in(text: str) -> str:
    return f" {re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()} "


def _found(result: str, numbers: set[float], words: str) -> bool:
    value, _ = parse_result_numeric(result)
    if value is not None:
        return any(round(abs(value) / scale, 6) in numbers for scale in _SCALES)
    phrase = _words_in(result).strip()
    return bool(phrase) and f" {phrase} " in words


def ground_results(payload: dict[str, Any], report_text: str) -> tuple[dict[str, Any], list[str]]:
    """The payload without results absent from the text, and the names of those left out."""
    numbers, words = _numbers_in(report_text), _words_in(report_text)
    kept, dropped = [], []
    for row in payload.get("test_results") or []:
        result = str((row or {}).get("result") or "")
        (kept if _found(result, numbers, words) else dropped).append(row)
    return {**payload, "test_results": kept}, [str((row or {}).get("test_name") or "unnamed") for row in dropped]
