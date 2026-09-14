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

