"""Cell counts are stored in one unit per test, so their trend lines compare like with like."""
import uuid

import pytest

from backend_api.app.findings import findings_from_records
from backend_api.app.units import to_common_unit


def _finding(name, result, unit, reference="N/A"):
    rows = [{"Test_Name": name, "Result": result, "Unit": unit, "Reference_Range": reference, "Status": "Normal"}]
    return findings_from_records(uuid.uuid4(), rows, "r.pdf")[0]


@pytest.mark.parametrize(
    ("result", "unit", "reference", "expected"),
    [
        ("2.61", "lacs/c.mm.", "1.5 - 4.0 lacs/c.mm", (261000, 150000, 400000)),
        ("284", "X 10³ / µL", "150-400", (284000, 150000, 400000)),
        ("225000", "/µL", "150000 - 410000", (225000, 150000, 410000)),
        ("2,61,000", "/cumm", "1,50,000 - 4,10,000", (261000, 150000, 410000)),
    ],
)
def test_every_platelet_count_lands_in_cells_per_microlitre(result, unit, reference, expected):
    """Three labs wrote one count three ways; the trend line plotted 284, 2.61 and 225000 together."""
    f = _finding("Platelet Count", result, unit, reference)
    assert (f.value_numeric, f.ref_low, f.ref_high) == expected
    assert f.unit == "/µL" and f.result_text == str(expected[0])


def test_absolute_counts_and_wbc_convert_too():
    assert _finding("Neutrophils (Absolute)", "5.01", "X 10³ / µL").value_numeric == 5010
    assert _finding("Total WBC Count", "9580", "cumm").unit == "/µL"


def test_red_cells_stay_in_millions():
    f = _finding("Red Blood Cell Count (RBC)", "4.63", "x 10^6 /uL", "4.50 - 5.50")
    assert (f.value_numeric, f.unit, f.ref_low) == (4.63, "x10^6/µL", 4.5)


def test_a_conversion_that_would_be_implausible_is_not_made():
    """2.61 'thousand per µL' is 2,610 platelets: the unit was misread, so the value is left as written."""
    f = _finding("Platelet Count", "2.61", "X 10³ / µL")
    assert (f.value_numeric, f.unit) == (2.61, "X 10³ / µL")


def test_tests_that_are_not_counts_are_untouched():
    assert to_common_unit("Haemoglobin (Hb)", "g/dL", 13.2) == ("g/dL", (13.2,))
    assert to_common_unit("Platelet Count", "", 250000) == ("", (250000,))


def test_a_comparator_survives_conversion():
    assert _finding("Eosinophils (Absolute)", "<0.1", "X 10³ / µL").result_text == "<100"
