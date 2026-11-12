"""One unit per cell count, so a trend line never mixes 2.61 lakh/cumm with 284 x10^3/uL.

Only counts are converted: their units differ by a power of ten and nothing else, so the
conversion is exact. A result that would land outside a plausible range is left as written.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PER_UL = "/µL"
MILLIONS_PER_UL = "x10^6/µL"


@dataclass(frozen=True)
class Count:
    unit: str        # the unit every reading of this test is stored in
    divisor: float   # cells per µL divided by this gives the stored value
    low: float       # plausible bounds, in cells per µL, for a converted value
    high: float


def _count_for(test: str) -> Count | None:
    name = test.lower()
    if "platelet count" in name:
        return Count(PER_UL, 1, 5_000, 2_000_000)
    if "wbc count" in name or "leucocyte count" in name or "leukocyte count" in name:
        return Count(PER_UL, 1, 300, 200_000)
    if "(absolute)" in name:
        return Count(PER_UL, 1, 0, 100_000)
    if "red blood cell count" in name or name == "rbc count":
        return Count(MILLIONS_PER_UL, 1_000_000, 500_000, 10_000_000)
    return None


def _cells_per_ul_multiplier(unit: str) -> float | None:
    """What one reported unit is in cells per µL, or None when the unit says nothing usable."""
    u = re.sub(r"\s+", "", (unit or "").lower()).replace("µ", "u").replace("μ", "u")
    if re.search(r"lac|lakh", u):
        return 100_000
    if re.search(r"10(\^|\*\*|e)?6|10⁶|mill", u):
        return 1_000_000
    if re.search(r"10(\^|\*\*|e)?3|10³|thou|k/u", u):
        return 1_000
    if re.search(r"/(ul|cumm|c\.?mm|mm3|mm³|cmm)\.?$|^cumm$|^cells/ul$", u):
        return 1
    return None


def to_common_unit(test: str, unit: str | None, *values: float | None) -> tuple[str | None, tuple[float | None, ...]]:
    """(unit, values) in the test's common unit; unchanged when the test or unit is not a known count."""
    count = _count_for(test)
    multiplier = _cells_per_ul_multiplier(unit or "")
    if count is None or multiplier is None:
        return unit, values
    first = next((v for v in values if v is not None), None)
    if first is not None and not count.low <= first * multiplier <= count.high:
        return unit, values  # implausible after conversion: the unit was probably misread
    return count.unit, tuple(None if v is None else round(v * multiplier / count.divisor, 4) for v in values)


def format_number(value: float) -> str:
    return f"{value:,.0f}".replace(",", "") if value >= 1000 or value == int(value) else f"{value:g}"
