"""Values a patient would see wrong: each case was a real wrong output before the fix."""
import pytest

from backend_api.app.normalization import normalize_records, normalize_status, normalize_test_name
from backend_api.app.values import derive_status, parse_reference_range, parse_result_numeric


def _row(name, result, status="N/A", category="Other", reference=""):
    return {"Test_Name": name, "Result": result, "Status": status, "Reference_Range": reference,
            "Test_Category": category, "Test_Date": "05-03-2024", "Source_Filename": "a.pdf"}


# ── Urine and blood tests stay apart ──────────────────────────────────────────

@pytest.mark.parametrize("name", ["Glucose (Urine)", "Urine Glucose", "Glucose, Urine", "URINE SUGAR", "Urine Glucose Sugar"])
def test_urine_glucose_is_never_named_as_blood_glucose(name):
    assert normalize_test_name(name) == "Glucose (Urine)"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Creatinine (Urine)", "Creatinine (Urine)"),
        ("Haemoglobin (Urine)", "Haemoglobin (Urine)"),
        ("Protein (Urine)", "Protein (Urine)"),
        ("Creatinine (Serum)", "Serum Creatinine"),
        ("Specific Gravity (Urine)", "Specific Gravity"),
        ("Ketone Bodies Urine", "Urine Ketones"),
    ],
)
def test_specimen_decides_the_name_only_for_tests_measured_in_both(name, expected):
    assert normalize_test_name(name) == expected


def test_a_urine_reading_no_longer_replaces_the_blood_reading_on_the_same_day():
    rows = normalize_records([
        _row("Glucose", "92", "Normal", "Diabetes", "70-100"),
        _row("Glucose (Urine)", "Present ++", category="Urinalysis"),
    ])
    assert {(r["Test_Name"], r["Result"]) for r in rows} == {
        ("Fasting Blood Glucose", "92"),
        ("Glucose (Urine)", "Present ++"),
    }


def test_two_different_values_for_one_test_in_one_file_are_both_kept():
    rows = normalize_records([_row("Haemoglobin", "12.4", "Normal"), _row("Hb", "9.1", "Low")])
    assert sorted(r["Result"] for r in rows) == ["12.4", "9.1"]


def test_true_duplicates_still_collapse_whatever_the_formatting():
    rows = normalize_records([_row("Haemoglobin", "12.4", "Normal"), _row("Hb", "12.40 g/dL", "Normal")])
    assert len(rows) == 1


# ── Negations ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["Not Present", "not present", "Nil", "Not Seen", "None seen", "Absent", "Not detected"])
def test_negated_results_are_negative(text):
    assert normalize_status(text) == "Negative"
    assert derive_status(text, "", "N/A") == "Negative"


def test_present_is_still_positive():
    assert normalize_status("Present ++") == "Positive"


# ── Reference ranges ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("reference", "value", "status"),
    [
        # The first number pair used to win: desirable cholesterol was "Low", prediabetic HbA1c "Normal".
        ("Desirable <200, Borderline 200-239", "180", "Normal"),
        ("Normal <5.7; Prediabetes 5.7-6.4", "6.0", "High"),
        ("< 100 Optimal, 100-129 Near optimal/above optimal, 130-159 Borderline High", "80", "Normal"),
        ("< 5.7 : Normal\n5.7-6.4 : Prediabetes\n>= 6.5 : Diabetes", "5.2", "Normal"),
        ("Normal : Below 5.7 % Prediabetic : 5.7 - 6.4 % Diabetic : >= 6.5 %", "6.1", "High"),
        ("DEFICIENCY : <20 ng/ml || INSUFFICIENCY : 20-<30 ng/ml || SUFFICIENCY : 30-100 ng/ml || TOXICITY : >100 ng/ml", "18", "Low"),
        (">= 90 : Normal; 60 - 89 : Mild Decrease; 45 - 59 : Mild to Moderate Decrease", "72", "Low"),
    ],
)
def test_tiered_ranges_use_the_tier_the_lab_labels_healthy(reference, value, status):
    assert derive_status(value, reference, "N/A") == status


@pytest.mark.parametrize(
    ("reference", "value"),
    [
        ("Low risk >60, High risk <40", "45"),             # was High; no tier is labelled normal
        ("Male 13-17 Female 12-15", "12.5"),               # was Low for a woman
        ("Adult: 30 - 100 Normal Level; Pediatric 20 - 100 Normal Level", "25"),  # two healthy tiers
        ("90 - 120 mg/dl : Good Control; 121 - 150 mg/dl : Fair Control", "130"),
    ],
)
def test_ranges_without_one_labelled_healthy_tier_are_not_guessed(reference, value):
    assert parse_reference_range(reference) == (None, None, None)
    assert derive_status(value, reference, "N/A") == "N/A", "unknown beats wrong"
    assert derive_status(value, reference, "High") == "High", "the lab's own flag still wins"


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("70-100", (70.0, 100.0, "range")),
        ("13.0 - 17.0 g/dL", (13.0, 17.0, "range")),
        ("4.0 - 11.0 10^3/uL", (4.0, 11.0, "range")),
        ("1,50,000 - 4,10,000", (150000.0, 410000.0, "range")),
        ("4,000-10,500/cumm", (4000.0, 10500.0, "range")),    # was 0 - 10
        ("Adult: 142.39 to 923.14", (142.39, 923.14, "range")),
        ("-2 - 2", (-2.0, 2.0, "range")),
        ("0.5 to 1.2", (0.5, 1.2, "range")),
        ("< 200", (None, 200.0, "less_than")),
        ("Less than 5", (None, 5.0, "less_than")),
        (">40", (40.0, None, "greater_than")),
        ("Negative", (None, None, "qualitative_normal")),
    ],
)
def test_single_ranges_still_parse(reference, expected):
    assert parse_reference_range(reference) == expected


# ── Numbers ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2,50,000", 250000.0),   # was 2.0
        ("1,50,000", 150000.0),
        ("250,000", 250000.0),
        ("1,234.5", 1234.5),
        ("12.4", 12.4),
    ],
)
def test_indian_and_western_digit_grouping(text, expected):
    assert parse_result_numeric(text)[0] == expected


def test_platelets_in_lakh_format_get_the_right_status():
    assert derive_status("2,50,000", "1,50,000 - 4,10,000", "N/A") == "Normal"
    assert derive_status("90,000", "1,50,000 - 4,10,000", "N/A") == "Low"


# ── The model's flag against the numbers ─────────────────────────────────────

def test_a_normal_flag_on_a_value_outside_its_range_is_corrected():
    """The model copies the lab's flag and sometimes misses one."""
    assert derive_status("212", "< 200 Desirable, 200-239 Borderline", "Normal") == "High"
    assert derive_status("3.1", "3.5 - 5.1", "Normal") == "Low"


def test_a_lab_flag_of_high_is_kept_even_when_the_numbers_look_normal():
    assert derive_status("5.0", "3.5 - 5.1", "High") == "High"


# ── Not results ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["Blood Group", "ABO Group", "Rh Factor", "Rh Type", "Blood Group & Rh"])
def test_blood_group_and_rh_are_never_flagged(name):
    """Rh "Positive" counted as an abnormal result and lowered the health score."""
    rows = normalize_records([_row(name, "Positive", "Positive")])
    assert rows[0]["Status"] == "N/A"


def test_a_positive_infection_test_is_still_flagged():
    rows = normalize_records([_row("HBsAg", "Reactive", "Positive")])
    assert rows[0]["Status"] == "Positive"
