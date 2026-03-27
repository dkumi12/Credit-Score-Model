"""
Shared utility functions for the Credit Default Predictor.
Covers feature validation, payload building, and result formatting
for the Lending Club binary default classification model.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

# ── Feature config (must match training exactly) ──────────────────────────────

FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc",
]

GRADE_MAP   = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
TERM_MAP    = {"36 months": 36.0, "60 months": 60.0}
HOME_MAP    = {"MORTGAGE": 0, "NONE": 1, "OTHER": 2, "OWN": 3, "RENT": 4}
VERIF_MAP   = {"Not Verified": 0, "Source Verified": 1, "Verified": 2}
PURPOSE_MAP = {
    "Debt Consolidation": 0, "Credit Card": 1, "Home Improvement": 2,
    "Other": 3, "Major Purchase": 4, "Medical": 5, "Small Business": 6,
    "Car": 7, "Vacation": 8, "Moving": 9, "House": 10,
    "Wedding": 11, "Educational": 12, "Renewable Energy": 13,
}

REQUIRED_FIELDS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc",
]

NUMERIC_RANGES = {
    "loan_amnt":       (500,    40000),
    "int_rate":        (1.0,    35.0),
    "dti":             (0.0,    60.0),
    "annual_inc":      (1000,   500000),
    "revol_util":      (0.0,    100.0),
    "emp_length":      (0,      10),
}

# ── Validation ────────────────────────────────────────────────────────────────

def validate_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate an incoming prediction payload against required fields
    and numeric range constraints.

    Returns:
        (is_valid, message)
    """
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"

    for field, (lo, hi) in NUMERIC_RANGES.items():
        val = payload.get(field)
        if val is not None:
            try:
                val = float(val)
            except (TypeError, ValueError):
                return False, f"Field '{field}' must be numeric, got: {val}"
            if not (lo <= val <= hi):
                return False, f"Field '{field}' out of range [{lo}, {hi}]: {val}"

    return True, "Valid payload"


# ── Feature alignment ─────────────────────────────────────────────────────────

def align_features(raw: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a single-row DataFrame with columns in the exact training order.
    Missing columns are filled with 0 (safe — all features are numeric).
    """
    df = pd.DataFrame([raw])
    df = df.reindex(columns=FEATURES, fill_value=0)
    return df


# ── Result formatting ─────────────────────────────────────────────────────────

def format_result(prediction: int, probability: float) -> Dict[str, Any]:
    """
    Format a raw model output into a human-readable result dict.
    """
    labels = {
        0: {"label": "No Default", "risk": "LOW RISK"},
        1: {"label": "Default",    "risk": "HIGH RISK"},
    }
    info = labels.get(prediction, {"label": "Unknown", "risk": "UNKNOWN"})
    return {
        "prediction":          prediction,
        "label":               info["label"],
        "risk_level":          info["risk"],
        "default_probability": round(probability, 4),
    }
