from backend_api.app.normalization import merge_readings_by_date


def _reading(source, result, test_date="05-03-2024"):
    return {
        "Test_Name": "Haemoglobin (Hb)",
        "Test_Date": test_date,
        "Result": result,
        "Unit": "g/dL",
        "Reference_Range": "12 - 15",
        "Status": "Normal",
        "Source_Filename": source,
    }


def test_two_labs_on_the_same_day_both_survive():
    """Keying on date alone silently dropped one of two same-day readings."""
    merged = merge_readings_by_date([_reading("lab_a.pdf", "12.4"), _reading("lab_b.pdf", "13.1")])
    assert len(merged) == 2
    assert {row["Result"] for row in merged} == {"12.4", "13.1"}


def test_alias_duplicates_within_one_file_still_collapse():
    merged = merge_readings_by_date([_reading("lab_a.pdf", "12.4"), _reading("lab_a.pdf", "12.4")])
    assert len(merged) == 1


def test_readings_stay_sorted_by_date():
    merged = merge_readings_by_date(
        [
            _reading("lab_a.pdf", "13.1", test_date="09-03-2024"),
            _reading("lab_a.pdf", "12.4", test_date="05-03-2024"),
        ]
    )
    assert [row["Test_Date"] for row in merged] == ["05-03-2024", "09-03-2024"]
