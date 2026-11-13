import pytest

from backend_api.app.normalization import normalize_test_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Same tests the lab spelled differently: they must share one trend line.
        ("EGFR", "eGFR (Estimated)"),
        ("Sgpt/alt", "Alanine Aminotransferase (SGPT/ALT)"),
        ("Vitamin D 25-OH", "25-OH Vitamin D (Total)"),
        ("TSH Ultrasensitive", "Thyroid Stimulating Hormone (TSH)"),
        ("RBC Count", "Red Blood Cell Count (RBC)"),
        # Names with no alias keep the lab's casing conventions.
        ("HBSAG", "HBsAg"),
        ("Dengue NS1 Antigen", "Dengue NS1 Antigen"),
        ("PUS CELLS", "Pus Cells"),
        ("RH Factor", "Rh Factor"),
        ("T3, Total", "T3, Total"),
        ("random xyz-ab test", "Random XYZ-AB Test"),
    ],
)
def test_display_names(raw: str, expected: str) -> None:
    assert normalize_test_name(raw) == expected
