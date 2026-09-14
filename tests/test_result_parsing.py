import pandas as pd
import pytest

from Helper_Functions import parse_result_numeric, result_series_to_numeric


@pytest.mark.parametrize(
    ("raw", "expected_value", "expected_comparator"),
    [
        ("< 0.5", 0.5, "<"),
        ("<0.5", 0.5, "<"),
        ("≤ 0.5", 0.5, "<="),
        ("> 200", 200.0, ">"),
        ("≥ 200", 200.0, ">="),
        ("1,200", 1200.0, None),
        ("1,200.5", 1200.5, None),
        ("12.5 H", 12.5, None),
        ("12.5 L", 12.5, None),
        ("5.6*", 5.6, None),
        ("12.5 g/dL", 12.5, None),
        ("  7.9  ", 7.9, None),
        ("-1.5", -1.5, None),
        (13.4, 13.4, None),
        (42, 42.0, None),
        ("Negative", None, None),
        ("", None, None),
        (None, None, None),
        (float("nan"), None, None),
    ],
)
def test_parse_result_numeric(raw, expected_value, expected_comparator):
    value, comparator = parse_result_numeric(raw)
    assert value == expected_value
    assert comparator == expected_comparator


def test_qualified_values_survive_a_dataframe_column():
    """pd.to_numeric dropped all three of these; the parser keeps them."""
    series = pd.Series(["< 0.5", "1,200", "12.5 H"])
    assert list(result_series_to_numeric(series)) == [0.5, 1200.0, 12.5]


def test_non_numeric_still_becomes_nan():
    series = pd.Series(["Negative", "Not Detected"])
    assert result_series_to_numeric(series).isna().all()
