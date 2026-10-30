"""Validate the deliberately fictional classroom dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ["annual_income", "loan_amount", "term_months", "credit_score", "debt_to_income"]
CATEGORICAL = ["employment_type"]
FEATURES = NUMERIC + CATEGORICAL
EMPLOYMENT = ["salaried", "self_employed", "contract"]
BOUNDS = {"annual_income": (1, 10000000), "loan_amount": (1, 10000000), "term_months": (1, 480), "credit_score": (300, 850), "debt_to_income": (0, 1)}


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != set(FEATURES):
        raise ValueError("Features must be exactly: " + ", ".join(FEATURES))
    result = frame[FEATURES].copy()
    for name, (low, high) in BOUNDS.items():
        result[name] = pd.to_numeric(result[name], errors="raise")
        if not np.isfinite(result[name].to_numpy(dtype=float)).all() or not result[name].between(low, high).all():
            raise ValueError(f"{name} must contain finite values between {low} and {high}")
        if name in ("term_months", "credit_score") and (result[name] % 1 != 0).any():
            raise ValueError(f"{name} must be a whole number")
    result["employment_type"] = result.employment_type.map(lambda value: value.strip().lower() if isinstance(value, str) else value)
    if not result.employment_type.isin(EMPLOYMENT).all():
        raise ValueError("employment_type must be salaried, self_employed or contract")
    return result


def load_data(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if set(raw.columns) != set(["application_id", *FEATURES, "approved"]):
        raise ValueError("CSV must contain application_id, the six documented features and approved only")
    if len(raw) < 50:
        raise ValueError("Use at least 50 observations")
    identifiers = raw.application_id.astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError("application_id must be nonblank and unique")
    result = validate_features(raw[FEATURES])
    labels = pd.to_numeric(raw.approved, errors="raise")
    if not labels.isin([0, 1]).all() or set(labels.unique()) != {0, 1}:
        raise ValueError("approved must contain both classes 0 and 1")
    if labels.value_counts().min() < 10:
        raise ValueError("Each approval class needs at least 10 observations")
    result.insert(0, "application_id", identifiers)
    result["approved"] = labels.astype(int)
    return result
