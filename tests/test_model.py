import json

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageChops
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from loan_prediction.data import FEATURES, load_data
from loan_prediction.model import DEFAULT_DATA, make_pipeline, predict, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp("loan-model")
    return output, train(output_dir=output)


def test_disjoint_split_and_training_only_tree(trained):
    output, metrics = trained
    training, testing = metrics["train_application_ids"], metrics["test_application_ids"]
    assert len(training) == 400 and len(testing) == 100
    assert not set(training) & set(testing)
    frame = load_data(DEFAULT_DATA).set_index("application_id")
    assert set(training) | set(testing) == set(frame.index)
    assert abs(frame.loc[training, "approved"].mean() - frame.loc[testing, "approved"].mean()) < 0.02
    saved = joblib.load(output / "model.joblib")["pipeline"]
    reference = make_pipeline().fit(frame.loc[training, FEATURES], frame.loc[training, "approved"])
    assert np.allclose(saved.predict_proba(frame.loc[testing, FEATURES]), reference.predict_proba(frame.loc[testing, FEATURES]))
    tree = saved.named_steps["tree"]
    assert tree.get_depth() <= 5
    assert tree.tree_.n_node_samples[0] == 400
    leaves = tree.tree_.children_left == -1
    assert tree.tree_.n_node_samples[leaves].min() >= 8
    categories = saved.named_steps["encode"].named_transformers_["category"].categories_[0]
    assert set(categories) == set(frame.loc[training, "employment_type"])


def test_held_out_metrics(trained):
    output, metrics = trained
    rows = pd.read_csv(output / "test_predictions.csv")
    assert list(rows.application_id) == metrics["test_application_ids"]
    assert metrics["accuracy"] == pytest.approx(accuracy_score(rows.actual_approved, rows.predicted_approved))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(rows.actual_approved, rows.approval_probability))
    assert metrics["confusion_matrix"] == confusion_matrix(rows.actual_approved, rows.predicted_approved, labels=[0, 1]).tolist()
    assert sum(map(sum, metrics["confusion_matrix"])) == 100
    assert rows.approval_probability.between(0, 1).all()
    assert metrics["accuracy"] > metrics["baseline_accuracy"]
    assert json.loads((output / "metrics.json").read_text()) == metrics


def test_prediction_decision_path_and_exported_rules(trained):
    result = predict(60000, 120000, 120, 720, 0.25, "salaried", trained[0] / "model.joblib")
    assert result["predicted_synthetic_label"] in (0, 1)
    assert 0 <= result["synthetic_approval_probability"] <= 1
    assert 1 <= len(result["decision_path"]) <= 5
    for rule in result["decision_path"]:
        assert rule["feature"] in trained[1]["encoded_feature_names"]
        if rule["operator"] == "<=":
            assert rule["value"] <= rule["threshold"]
        else:
            assert rule["value"] > rule["threshold"]
    assert result["outside_training_range"] == []
    assert "class:" in (trained[0] / "tree_rules.txt").read_text()
    assert pd.read_csv(trained[0] / "feature_importances.csv").importance.sum() == pytest.approx(1)


@pytest.mark.parametrize("changes", [{"annual_income": 0}, {"annual_income": float("nan")}, {"loan_amount": -1}, {"term_months": 3.5}, {"credit_score": 900}, {"debt_to_income": 2}, {"employment_type": "unknown"}])
def test_invalid_profiles(trained, changes):
    values = {"annual_income": 60000, "loan_amount": 120000, "term_months": 120, "credit_score": 720, "debt_to_income": 0.25, "employment_type": "salaried"}
    values.update(changes)
    with pytest.raises(ValueError):
        predict(**values, model_path=trained[0] / "model.joblib")


def test_outside_training_range_flagged(trained):
    result = predict(1000000, 1000000, 480, 400, 0.9, "salaried", trained[0] / "model.joblib")
    assert "annual_income" in result["outside_training_range"]
    assert "credit_score" in result["outside_training_range"]


def test_chart_images(trained):
    for name in ("confusion_matrix.png", "feature_importances.png"):
        with Image.open(trained[0] / name) as image:
            assert image.format == "PNG" and image.size == (1600, 960)
            assert ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white")).getbbox()
