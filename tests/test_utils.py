"""
Test suite for Credit Default Predictor utility functions.
Covers payload validation, feature alignment, and result formatting.
Dataset: Lending Club (141k rows) — binary default prediction.
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Src.utils import (
    validate_payload,
    align_features,
    format_result,
    FEATURES,
    GRADE_MAP,
    TERM_MAP,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_low_risk():
    """Grade A borrower — low risk profile."""
    return {
        "loan_amnt": 5000.0, "term": 36.0, "int_rate": 6.5,
        "installment": 150.0, "grade": 0, "sub_grade": 1,
        "emp_length": 10.0, "home_ownership": 3, "annual_inc": 120000.0,
        "verification_status": 2, "purpose": 0, "dti": 5.0,
        "delinq_2yrs": 0.0, "inq_last_6mths": 0.0, "open_acc": 10.0,
        "pub_rec": 0.0, "revol_bal": 2000.0, "revol_util": 10.0,
        "total_acc": 25.0,
    }

@pytest.fixture
def valid_high_risk():
    """Grade G borrower — high risk profile."""
    return {
        "loan_amnt": 35000.0, "term": 60.0, "int_rate": 28.0,
        "installment": 900.0, "grade": 6, "sub_grade": 34,
        "emp_length": 0.0, "home_ownership": 4, "annual_inc": 18000.0,
        "verification_status": 0, "purpose": 6, "dti": 38.0,
        "delinq_2yrs": 5.0, "inq_last_6mths": 8.0, "open_acc": 2.0,
        "pub_rec": 3.0, "revol_bal": 25000.0, "revol_util": 95.0,
        "total_acc": 3.0,
    }

# ── validate_payload ──────────────────────────────────────────────────────────

def test_validate_payload_valid_low_risk(valid_low_risk):
    is_valid, msg = validate_payload(valid_low_risk)
    assert is_valid is True
    assert "Valid" in msg

def test_validate_payload_valid_high_risk(valid_high_risk):
    is_valid, msg = validate_payload(valid_high_risk)
    assert is_valid is True

def test_validate_payload_missing_field(valid_low_risk):
    del valid_low_risk["loan_amnt"]
    is_valid, msg = validate_payload(valid_low_risk)
    assert is_valid is False
    assert "loan_amnt" in msg

def test_validate_payload_multiple_missing():
    is_valid, msg = validate_payload({"loan_amnt": 5000})
    assert is_valid is False
    assert "Missing required fields" in msg

def test_validate_payload_dti_out_of_range(valid_low_risk):
    valid_low_risk["dti"] = 999.0
    is_valid, msg = validate_payload(valid_low_risk)
    assert is_valid is False
    assert "dti" in msg

def test_validate_payload_negative_income(valid_low_risk):
    valid_low_risk["annual_inc"] = -1000.0
    is_valid, msg = validate_payload(valid_low_risk)
    assert is_valid is False
    assert "annual_inc" in msg

def test_validate_payload_non_numeric_field(valid_low_risk):
    valid_low_risk["int_rate"] = "high"
    is_valid, msg = validate_payload(valid_low_risk)
    assert is_valid is False
    assert "int_rate" in msg

# ── align_features ────────────────────────────────────────────────────────────

def test_align_features_returns_dataframe(valid_low_risk):
    df = align_features(valid_low_risk)
    assert isinstance(df, pd.DataFrame)

def test_align_features_correct_column_count(valid_low_risk):
    df = align_features(valid_low_risk)
    assert len(df.columns) == len(FEATURES)

def test_align_features_correct_column_order(valid_low_risk):
    df = align_features(valid_low_risk)
    assert list(df.columns) == FEATURES

def test_align_features_values_preserved(valid_low_risk):
    df = align_features(valid_low_risk)
    assert df["loan_amnt"].iloc[0] == 5000.0
    assert df["grade"].iloc[0] == 0

def test_align_features_missing_columns_filled_zero():
    """Partial payload — missing columns must be filled with 0."""
    partial = {"loan_amnt": 10000.0, "int_rate": 12.0}
    df = align_features(partial)
    assert df["dti"].iloc[0] == 0
    assert df["annual_inc"].iloc[0] == 0

def test_align_features_single_row(valid_high_risk):
    df = align_features(valid_high_risk)
    assert len(df) == 1

# ── format_result ─────────────────────────────────────────────────────────────

def test_format_result_no_default():
    result = format_result(0, 0.15)
    assert result["prediction"] == 0
    assert result["label"] == "No Default"
    assert result["risk_level"] == "LOW RISK"
    assert result["default_probability"] == 0.15

def test_format_result_default():
    result = format_result(1, 0.72)
    assert result["prediction"] == 1
    assert result["label"] == "Default"
    assert result["risk_level"] == "HIGH RISK"
    assert result["default_probability"] == 0.72

def test_format_result_probability_rounded():
    result = format_result(0, 0.123456789)
    assert result["default_probability"] == 0.1235

def test_format_result_keys_present():
    result = format_result(1, 0.55)
    assert all(k in result for k in
               ["prediction", "label", "risk_level", "default_probability"])

# ── Encoding maps ─────────────────────────────────────────────────────────────

def test_grade_map_complete():
    assert set(GRADE_MAP.keys()) == {"A", "B", "C", "D", "E", "F", "G"}

def test_grade_map_values_sequential():
    assert list(GRADE_MAP.values()) == [0, 1, 2, 3, 4, 5, 6]

def test_term_map_values():
    assert TERM_MAP["36 months"] == 36.0
    assert TERM_MAP["60 months"] == 60.0

def test_features_length():
    assert len(FEATURES) == 19

def test_features_no_duplicates():
    assert len(FEATURES) == len(set(FEATURES))
