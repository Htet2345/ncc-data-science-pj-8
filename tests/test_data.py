import numpy as np
import pandas as pd
import pytest

from loan_prediction.data import FEATURES, load_data, validate_features
from loan_prediction.model import DEFAULT_DATA
from scripts.generate_data import generate


def test_synthetic_data_and_reproducible_generator(tmp_path):
    frame = load_data(DEFAULT_DATA)
    assert frame.shape == (500, 8)
    assert set(frame.approved) == {0, 1}
    assert frame.application_id.str.startswith("DEMO").all()
    recreated = tmp_path / "loans.csv"
    generate(recreated)
    pd.testing.assert_frame_equal(load_data(recreated), frame)


@pytest.mark.parametrize("kind", ["missing_column", "extra_column", "few_rows", "missing_value", "infinity", "zero_income", "negative_amount", "fractional_term", "credit_low", "credit_high", "fractional_credit", "debt_low", "debt_high", "bad_employment", "duplicate_id", "blank_id", "bad_target", "one_class", "rare_class"])
def test_invalid_training_data(tmp_path, kind):
    frame = pd.read_csv(DEFAULT_DATA)
    if kind == "missing_column":
        frame = frame.drop(columns="annual_income")
    elif kind == "extra_column":
        frame["unused"] = 0
    elif kind == "few_rows":
        frame = frame.head(10)
    elif kind == "missing_value":
        frame["annual_income"] = frame.annual_income.astype(float)
        frame.loc[0, "annual_income"] = np.nan
    elif kind == "infinity":
        frame["annual_income"] = frame.annual_income.astype(float)
        frame.loc[0, "annual_income"] = np.inf
    elif kind == "zero_income":
        frame.loc[0, "annual_income"] = 0
    elif kind == "negative_amount":
        frame.loc[0, "loan_amount"] = -1
    elif kind == "fractional_term":
        frame["term_months"] = frame.term_months.astype(float)
        frame.loc[0, "term_months"] = 3.5
    elif kind == "credit_low":
        frame.loc[0, "credit_score"] = 299
    elif kind == "credit_high":
        frame.loc[0, "credit_score"] = 851
    elif kind == "fractional_credit":
        frame["credit_score"] = frame.credit_score.astype(float)
        frame.loc[0, "credit_score"] = 700.5
    elif kind == "debt_low":
        frame.loc[0, "debt_to_income"] = -0.1
    elif kind == "debt_high":
        frame.loc[0, "debt_to_income"] = 1.1
    elif kind == "bad_employment":
        frame.loc[0, "employment_type"] = "unknown"
    elif kind == "duplicate_id":
        frame.loc[1, "application_id"] = frame.loc[0, "application_id"]
    elif kind == "blank_id":
        frame.loc[0, "application_id"] = " "
    elif kind == "bad_target":
        frame.loc[0, "approved"] = 2
    elif kind == "one_class":
        frame["approved"] = 0
    elif kind == "rare_class":
        frame["approved"] = 0
        frame.loc[0, "approved"] = 1
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)


def test_feature_order_and_category_normalization():
    row = load_data(DEFAULT_DATA)[FEATURES].iloc[[0]][list(reversed(FEATURES))]
    row["employment_type"] = " Salaried "
    result = validate_features(row)
    assert list(result.columns) == FEATURES
    assert result.employment_type.iloc[0] == "salaried"
