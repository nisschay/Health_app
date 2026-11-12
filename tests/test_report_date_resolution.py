from datetime import date

import pytest

from backend_api.app.saving import _resolve_report_date

FALLBACK = date(2024, 1, 1)


def test_a_record_date_wins_over_the_payload_date():
    rows = [{"Test_Date": "05-03-2024"}]
    assert _resolve_report_date(rows, FALLBACK, "r.pdf") == date(2024, 3, 5)


def test_the_payload_date_is_used_when_records_have_none():
    assert _resolve_report_date([{"Test_Date": "not a date"}], FALLBACK, "r.pdf") == FALLBACK


def test_no_readable_date_anywhere_is_refused():
    """Unreadable dates used to be stored as today, which corrupted every timeline."""
    with pytest.raises(ValueError, match="report.pdf"):
        _resolve_report_date([{"Test_Date": "not a date"}], None, "report.pdf")


def test_todays_date_is_never_invented():
    with pytest.raises(ValueError):
        _resolve_report_date([], None, "r.pdf")
