import pytest

from backend_api.app import normalization as n
from backend_api.app.normalization import (
    TEST_NAME_MAP,
    are_same_test,
    normalize_records,
    normalize_test_name,
)

# Raw spellings seen in reports, and the canonical name each must resolve to.
GOLDEN = [
    ("HbA1C", "Glycated Haemoglobin (HbA1C)"),
    ("haemoglobin a1c", "Glycated Haemoglobin (HbA1C)"),
    ("TSH", "Thyroid Stimulating Hormone (TSH)"),
    ("thyroid stimulating hormone", "Thyroid Stimulating Hormone (TSH)"),
    ("high sensitivity c-reactive protein (hs-crp)", "hs-CRP"),
    ("LDL", "LDL Cholesterol"),
    ("albumin - serum", "Albumin"),
    ("vitamin b-12", "Vitamin B12 (Cobalamin)"),
    ("Fasting Blood Sugar", "Fasting Blood Glucose"),
    ("FBS", "Fasting Blood Glucose"),
    ("SGOT", "Aspartate Aminotransferase (SGOT/AST)"),
    ("Blood Urea Nitrogen", "Blood Urea Nitrogen (BUN)"),
    ("25 OH Vitamin D", "25-OH Vitamin D (Total)"),
    ("Total Leucocyte Count", "Total WBC Count"),
]

MUST_NOT_MERGE = [
    ("Albumin", "Albumin/Globulin Ratio (A/G)"),
    ("Bilirubin Total", "Bilirubin Indirect"),
    ("TSH", "T3"),
    ("TSH", "T3H"),
]


@pytest.mark.parametrize(("raw", "expected"), GOLDEN)
def test_golden_aliases(raw, expected):
    assert normalize_test_name(raw) == expected


@pytest.mark.parametrize(("alias", "canonical"), sorted(TEST_NAME_MAP.items()))
def test_every_alias_in_the_table_resolves(alias, canonical):
    assert normalize_test_name(alias) == canonical


@pytest.mark.parametrize("canonical", sorted(set(TEST_NAME_MAP.values())))
def test_canonical_names_are_fixed_points(canonical):
    """Normalising an already-canonical name must not move it."""
    assert normalize_test_name(canonical) == canonical


@pytest.mark.parametrize(("a", "b"), MUST_NOT_MERGE)
def test_must_not_merge_pairs_stay_apart(a, b):
    assert not are_same_test(a, b)


def _row(name, category="Other", date="2024-01-01", result="6.8"):
    return {"Test_Name": name, "Test_Category": category, "Test_Date": date, "Result": result, "Unit": "%"}


def test_synonyms_collapse_and_keep_their_aliases():
    rows = [_row("HbA1C", "Diabetes"), _row("haemoglobin a1c")]
    out = normalize_records(rows)
    assert len(out) == 1
    assert out[0]["Test_Name"] == "Glycated Haemoglobin (HbA1C)"
    assert {"HbA1C", "haemoglobin a1c", "Glycated Haemoglobin (HbA1C)"} <= set(out[0]["Aliases"])


def test_albumin_and_ag_ratio_stay_separate():
    out = normalize_records([_row("Albumin", "Liver Function", result="4.2"), _row("Albumin/Globulin Ratio (A/G)", "Liver Function", result="1.3")])
    assert sorted(r["Test_Name"] for r in out) == ["Albumin", "Albumin/Globulin Ratio (A/G)"]


def test_dedupe_does_not_compare_every_pair(monkeypatch):
    """The old clustering was O(k²) in are_same_test calls; 300 unknown names took 45k comparisons."""
    calls = {"n": 0}
    real = n.are_same_test

    def counting(a, b):
        calls["n"] += 1
        return real(a, b)

    monkeypatch.setattr(n, "are_same_test", counting)
    monkeypatch.setattr(n, "NORMALIZATION_ALIAS_WHITELIST_ONLY", True)
    rows = [_row(f"unmapped analyte {i}") for i in range(300)]
    out = normalize_records(rows)
    assert len(out) == 300
    # Category hinting compares each row against the hint table, which is linear.
    linear_bound = len(rows) * (len(n.TEST_CATEGORY_HINTS) + 1)
    quadratic = len(rows) * (len(rows) - 1) // 2
    assert calls["n"] <= linear_bound < quadratic, f"{calls['n']} comparisons"


def test_lookup_indexes_are_built_once():
    assert len(n._ALIASES_BY_NORMALIZED) > 0
    assert len(n._MUST_NOT_MERGE) == len(n.MUST_NOT_MERGE_RAW)
    assert n._canonical_from_lookup("ldl") == "LDL Cholesterol"
    assert n._canonical_from_lookup("l.d.l") == "LDL Cholesterol"
