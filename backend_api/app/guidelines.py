"""Guideline extracts for the chat prompt, chosen by whole-word matches on the question and abnormal results."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA = Path(__file__).resolve().parent / "data" / "guidelines.json"

# Everyday words people use for a group of tests.
_CONCEPTS = {
    "sugar": ["glucose", "hba1c", "diabetes", "prediabetes"],
    "diabetes": ["glucose", "hba1c"],
    "cholesterol": ["ldl", "hdl", "triglycerides", "lipid"],
    "lipids": ["ldl", "hdl", "triglycerides"],
    "liver": ["ast", "alt", "sgot", "sgpt", "bilirubin"],
    "kidney": ["creatinine", "egfr", "urea"],
    "thyroid": ["tsh", "free t4", "ft4"],
    "anaemia": ["haemoglobin", "hemoglobin", "ferritin", "b12"],
    "anemia": ["haemoglobin", "hemoglobin", "ferritin", "b12"],
    "inflammation": ["crp", "hs-crp"],
}

MAX_GUIDELINES = 3


@lru_cache(maxsize=1)
def _entries() -> list[dict[str, Any]]:
    return json.loads(_DATA.read_text(encoding="utf-8"))


def _words(text: str) -> str:
    """Lowercase, punctuation to spaces, padded so ' alt ' only matches the whole word."""
    return f" {re.sub(r'[^a-z0-9%+]+', ' ', text.lower()).strip()} "


def _mentions(haystack: str, term: str) -> bool:
    term = _words(term).strip()
    return bool(term) and f" {term} " in haystack


def _score(entry: dict[str, Any], question: str, abnormal: list[str]) -> int:
    names = entry["testNames"] + entry["aliases"]
    score = 0
    for finding in abnormal:
        if any(_mentions(finding, name) for name in names):
            score += 30  # the patient has an abnormal result for this test
    if any(_mentions(question, name) for name in names):
        score += 25  # the question names the test
    if any(_mentions(question, word) for word in entry["keywords"]):
        score += 10
    return score


def select_guidelines(question: str, abnormal_tests: list[str]) -> list[str]:
    """Up to three extracts, formatted for the prompt. Empty when nothing clearly applies."""
    asked = _words(question)
    for word, related in _CONCEPTS.items():
        if f" {word} " in asked:
            asked += " ".join(related) + " "
    abnormal = [_words(name) for name in abnormal_tests]

    ranked = sorted(
        ((score, entry) for entry in _entries() if (score := _score(entry, asked, abnormal)) >= 25),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [_format(entry) for _, entry in ranked[:MAX_GUIDELINES]]


def _format(entry: dict[str, Any]) -> str:
    bands = "\n".join(
        f"- {band['label']} ({band['range']}): {band['interpretation']}" for band in entry["interpretationBands"]
    )
    return (
        f"{entry['title']}\n{entry['content']}\nBands:\n{bands}\n"
        f"Escalate when: {'; '.join(entry['escalationTriggers'])}\n"
        f"Cite as: [{entry['source']}]({entry['sourceUrl']})"
    )
