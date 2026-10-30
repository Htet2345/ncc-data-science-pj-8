"""Train an interpretable tree on a held-out synthetic-data split."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from loan_prediction.data import CATEGORICAL, FEATURES, NUMERIC, load_data, validate_features

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "sample_data" / "loans.csv"
DEFAULT_OUTPUT = ROOT / "artifacts"


def make_pipeline():
    encoder = ColumnTransformer([("numeric", "passthrough", NUMERIC), ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, random_state=42)
    return Pipeline([("encode", encoder), ("tree", tree)])


def train(data_path: Path = DEFAULT_DATA, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    frame = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(frame[FEATURES], frame.approved, test_size=0.2, random_state=42, stratify=frame.approved)
    pipeline = make_pipeline().fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    probability = pipeline.predict_proba(x_test)[:, 1]
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train).predict(x_test)
    tree = pipeline.named_steps["tree"]
    feature_names = list(pipeline.named_steps["encode"].get_feature_names_out())
    metrics = {
        "dataset": str(data_path.resolve()), "data_note": "Bundled records and labels are synthetic; this model is an educational demonstration.",
        "rows": len(frame), "train_rows": len(x_train), "test_rows": len(x_test),
        "random_state": 42, "test_fraction": 0.2, "stratified": True,
        "max_depth": 5, "min_samples_leaf": 8, "fitted_depth": tree.get_depth(),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "baseline_accuracy": float(accuracy_score(y_test, baseline)),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "positive_class": "approved = 1 (synthetic label)",
        "feature_columns": FEATURES,
        "encoded_feature_names": feature_names,
        "training_ranges": {name: [float(x_train[name].min()), float(x_train[name].max())] for name in NUMERIC},
        "train_application_ids": frame.loc[x_train.index, "application_id"].tolist(),
        "test_application_ids": frame.loc[x_test.index, "application_id"].tolist(),
        "classification_report": classification_report(y_test, predicted, labels=[0, 1], output_dict=True, zero_division=0),
        "scikit_learn_version": sklearn.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics, "schema_version": 1}, output_dir / "model.joblib")
    results = x_test.copy()
    results.insert(0, "application_id", frame.loc[x_test.index, "application_id"])
    results["actual_approved"] = y_test
    results["predicted_approved"] = predicted
    results["approval_probability"] = probability
    results.to_csv(output_dir / "test_predictions.csv", index=False)
    importances = pd.DataFrame({"feature": feature_names, "importance": tree.feature_importances_}).sort_values("importance", ascending=False)
    importances.to_csv(output_dir / "feature_importances.csv", index=False)
    (output_dir / "tree_rules.txt").write_text(export_text(tree, feature_names=feature_names), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from loan_prediction.plots import save_plots
    save_plots(metrics["confusion_matrix"], importances, output_dir)
    return metrics


def predict(annual_income: float, loan_amount: float, term_months: int, credit_score: int, debt_to_income: float, employment_type: str, model_path: Path = DEFAULT_OUTPUT / "model.joblib") -> dict:
    row = validate_features(pd.DataFrame([[annual_income, loan_amount, term_months, credit_score, debt_to_income, employment_type]], columns=FEATURES))
    # Load only your own/trusted model files; joblib uses pickle.
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported model artifact; retrain this project")
    pipeline = bundle["pipeline"]
    # Trees evaluate float32 inputs; use that precision for the explanation too.
    transformed = pipeline.named_steps["encode"].transform(row).astype(np.float32)
    tree = pipeline.named_steps["tree"]
    leaf = int(tree.apply(transformed)[0])
    path = tree.decision_path(transformed).indices
    rules = []
    for node in path:
        if node == leaf:
            continue
        feature_index = tree.tree_.feature[node]
        value = float(transformed[0, feature_index])
        threshold = float(tree.tree_.threshold[node])
        rules.append({"feature": bundle["metrics"]["encoded_feature_names"][feature_index], "value": value, "operator": "<=" if value <= threshold else ">", "threshold": threshold})
    outside = [name for name, (low, high) in bundle["metrics"]["training_ranges"].items() if not low <= float(row.iloc[0][name]) <= high]
    return {
        "predicted_synthetic_label": int(pipeline.predict(row)[0]),
        "synthetic_approval_probability": float(pipeline.predict_proba(row)[0, 1]),
        "decision_path": rules, "outside_training_range": outside,
        "note": "Classroom simulation only. Leaf class frequencies are not calibrated approval chances or an actual lending decision.",
    }
