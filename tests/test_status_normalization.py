import pytest

from backend_api.app.normalization import (
    CONCERNING_STATUS_VALUES,
    normalize_status,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Abnormal", "Flagged"),
        ("ABNORMAL", "Flagged"),
        ("abnormal result", "Flagged"),
        ("Out of range", "Flagged"),
        ("Flagged", "Flagged"),
        ("Low Normal", "Normal"),
        ("High Normal", "Normal"),
        ("Normal", "Normal"),
        ("within normal limits", "Normal"),
        ("WNL", "Normal"),
        ("HIGH", "High"),
        ("h", "High"),
        ("elevated", "High"),
        ("Low", "Low"),
        ("decreased", "Low"),
        ("Critical high", "Critical"),
        ("panic value", "Critical"),
        ("Borderline", "Borderline"),
        ("Borderline High", "Borderline"),
        ("Insufficient", "Insufficient"),
        ("Positive", "Positive"),
        ("Not Detected", "Negative"),
        ("non-reactive", "Negative"),
        ("N/A", "N/A"),
        ("", "N/A"),
        (None, "N/A"),
    ],
)
def test_normalize_status(raw, expected):
    assert normalize_status(raw) == expected


def test_abnormal_is_never_normal():
    """The substring 'normal' inside 'abnormal' must not read as in-range."""
    assert normalize_status("Abnormal") != "Normal"
    assert normalize_status("Abnormal") in CONCERNING_STATUS_VALUES


def test_concerning_excludes_reassuring_values():
    assert "Normal" not in CONCERNING_STATUS_VALUES
    assert "Negative" not in CONCERNING_STATUS_VALUES
    assert "N/A" not in CONCERNING_STATUS_VALUES
