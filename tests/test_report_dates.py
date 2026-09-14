from datetime import date

import pytest

from backend_api.app.main import _parse_report_date_flexible


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-03-05", date(2024, 3, 5)),
        ("05-03-2024", date(2024, 3, 5)),
        ("05/03/2024", date(2024, 3, 5)),
        ("05.03.2024", date(2024, 3, 5)),
        ("5-3-24", date(2024, 3, 5)),
        ("2024/03/05", date(2024, 3, 5)),
        ("", None),
        (None, None),
        ("not a date", None),
        ("32-13-2024", None),
    ],
)
def test_parse_report_date_flexible(raw, expected):
    assert _parse_report_date_flexible(raw) == expected


def test_slash_year_first_is_not_read_as_a_day():
    """'2024/03/05' used to parse as day=2024 and return None."""
    assert _parse_report_date_flexible("2024/03/05") == date(2024, 3, 5)


def test_a_bad_separator_does_not_abort_the_remaining_formats():
    assert _parse_report_date_flexible("05/03/2024") is not None
