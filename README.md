# Loan Approval Prediction - Data Science Project 8

An educational decision-tree classifier for **synthetic approval labels**.
The topic matches Module 5.3, videos **24-25: Loan Approval Prediction, Parts 1-2**.
Video 23 is the Decision Tree prerequisite.

This is an original implementation based on those lesson titles. The exact
teacher code and dataset have not been verified. All bundled records and
outcomes are fictional. The project demonstrates classification and tree
interpretation; its output is not an actual lending decision or advice.

## Workflow

1. Load and validate six features and the synthetic binary target.
2. Make a fixed stratified 80/20 split using random seed 42.
3. One-hot encode employment type using training rows only.
4. Fit a decision tree with `max_depth=5` and `min_samples_leaf=8`.
5. Measure held-out accuracy, precision, recall, F1 and ROC-AUC.
6. Compare accuracy with a majority-training-class baseline.
7. Save the pipeline, evaluation outputs and human-readable tree rules.
8. Classify a fictional example and show the exact path through the tree.

Numeric features pass through unchanged: decision trees do not need feature
scaling. The tree limits are fixed before evaluating the test split; no test
results are used to choose hyperparameters. The saved model is the same
training-fitted pipeline evaluated on held-out data, not a full-data refit.

## Data and validation

The bundled CSV has 500 fictional records and is reproducible using the
generator under `scripts/`. See `sample_data/README.md` for the exact invented
target-generation rule, random label noise and units.

| Feature | Meaning | Accepted values |
| --- | --- | --- |
| `annual_income` | Annual income, synthetic USD | 1-10,000,000 |
| `loan_amount` | Requested principal, synthetic USD | 1-10,000,000 |
| `term_months` | Loan term | Whole number 1-480 |
| `credit_score` | Fictional score | Whole number 300-850 |
| `debt_to_income` | Supplied debt-to-income fraction | 0-1 |
| `employment_type` | Fictional employment category | salaried, self_employed, contract |

All values are required and must be finite. Missing values are rejected rather
than silently filled. Employment categories are trimmed and lowercased.
`application_id` is a unique, nonblank audit identifier and is never a feature.
`approved` is the training target, 0 or 1. Both classes need at least 10 rows;
custom datasets need at least 50 rows in total and exactly the documented schema.

The broad validation bounds do not imply adequate training coverage. Numeric
inputs outside observed training ranges are reported in prediction output.
The example does not collect protected personal traits or use actual credit
records. Its limited features and synthetic labels do not validate deployment
for real people.

## Setup on Windows

Requires Python 3.10+; developed with Python 3.13.

Open this folder in VS Code and choose **Terminal > New Terminal**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The environment is already installed on the development machine. These
commands reproduce it on a fresh checkout. The lock file includes exact tested
runtime and test versions. `requirements.txt` lists direct runtime dependencies;
`requirements-dev.txt` adds pytest.

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe -m loan_prediction train
```

The default split has 400 training and 100 test records. Stratification keeps
approximately the same target-class balance in each split. Terminal output
shows the main metrics; `artifacts/metrics.json` includes the full report and
exact split IDs.

## Classify a fictional example

```powershell
.\.venv\Scripts\python.exe -m loan_prediction predict --income 60000 --amount 120000 --term 120 --credit-score 720 --debt-ratio 0.25 --employment salaried
```

The result includes a synthetic label, leaf class frequency, exact tree path
and any numeric fields outside training ranges. Each path rule contains a
feature name, value, comparison operator and threshold. One-hot category
features have values 0 or 1. A returned probability describes the proportion
of positive training labels in a leaf; it is not a calibrated real approval chance.

Alternative data/output paths:

```powershell
.\.venv\Scripts\python.exe -m loan_prediction train --data .\sample_data\loans.csv --output-dir .\artifacts
.\.venv\Scripts\python.exe -m loan_prediction predict --income 60000 --amount 120000 --term 120 --credit-score 720 --debt-ratio 0.25 --employment salaried --model .\artifacts\model.joblib
```

Run module commands from the project folder. Default paths are project-relative;
CLI-supplied relative paths follow the current shell directory. Training
replaces generated outputs. Only load artifacts you created or trust: joblib
uses pickle, which can execute code from an untrusted file. Use matching
recorded dependency versions when reloading the model.

## Outputs

| File under `artifacts/` | Contents |
| --- | --- |
| `model.joblib` | Encoder/tree pipeline and training metadata |
| `metrics.json` | Metrics, classification report and exact split IDs |
| `test_predictions.csv` | Held-out features, true/predicted labels and probabilities |
| `tree_rules.txt` | Full text representation of the learned tree |
| `feature_importances.csv` | Training-fitted impurity importances |
| `confusion_matrix.png` | Actual labels on rows, predictions on columns |
| `feature_importances.png` | Ranked impurity importances |

Class 1 is positive. Confusion-matrix label order [0,1] gives `[[TN,FP],[FN,TP]]`.
The baseline always predicts the most common training label. Accuracy alone
can be misleading when labels are imbalanced, so precision, recall, F1 and AUC
are also reported. Impurity importances can favour some feature types and
do not show causal effects or justify actual lending criteria.

The deliberately simple invented labels make this a learning exercise. Good
performance measures how well the tree approximates that simulation, not how
well it predicts real repayment, real approvals or fair lending outcomes.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Tests cover deterministic data generation, schema/input validation, disjoint
stratified splits, training-only encoding/fitting, tree depth and leaf limits,
held-out metrics, probability bounds, decision-path conditions, saved rules,
feature importances, chart decoding and separate-process CLI execution.

VS Code **F5** trains the model. **Terminal > Run Task** provides training,
fictional prediction and tests. No server, external account or PowerShell policy
changes are needed.

## Structure

```text
loan_prediction/
  data.py       CSV and feature validation
  model.py      Encoding, tree, evaluation and path explanations
  plots.py      Confusion matrix and feature-importance chart
  __main__.py   Command-line interface
sample_data/    Fictional CSV and its provenance
scripts/        Reproducible synthetic-data generator
tests/          Data, model and command-line checks
.vscode/        Interpreter, debugger and tasks
artifacts/      Generated outputs, ignored by Git
```

## GitHub submission

Include source, fictional dataset/generator, tests, README, requirements, pytest
and VS Code configuration. Generated artifacts, environment, caches and local
test report are ignored. No demo screenshots are included. Repository: https://github.com/Htet2345/ncc-data-science-pj-8

## References

- [scikit-learn DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [scikit-learn ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [scikit-learn export_text](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html)
