from datetime import date

import pytest

from backend_api.app.saving import parse_report_date


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
def testparse_report_date(raw, expected):
    assert parse_report_date(raw) == expected

