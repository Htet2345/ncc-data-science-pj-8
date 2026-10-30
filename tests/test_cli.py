import json
import subprocess
import sys

from loan_prediction.model import ROOT


def run(*args):
    return subprocess.run([sys.executable, "-m", "loan_prediction", *args], cwd=ROOT, capture_output=True, text=True, timeout=90)


PROFILE = ["predict", "--income", "60000", "--amount", "120000", "--term", "120", "--credit-score", "720", "--debt-ratio", "0.25", "--employment", "salaried"]


def test_train_and_saved_prediction_in_separate_processes(tmp_path):
    result = run("train", "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rows"] == 500
    result = run(*PROFILE, "--model", str(tmp_path / "model.joblib"))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["predicted_synthetic_label"] in (0, 1)


def test_missing_model_error(tmp_path):
    result = run(*PROFILE, "--model", str(tmp_path / "missing.joblib"))
    assert result.returncode == 2
    assert "training first" in result.stderr and "Traceback" not in result.stderr


def test_invalid_profile_error():
    profile = PROFILE.copy()
    profile[profile.index("60000")] = "nan"
    result = run(*profile)
    assert result.returncode == 2
    assert "finite" in result.stderr


def test_missing_data_error(tmp_path):
    result = run("train", "--data", str(tmp_path / "missing.csv"))
    assert result.returncode == 2
    assert "File not found" in result.stderr
