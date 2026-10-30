"""Train or classify fictional example records."""

import argparse
import json
from pathlib import Path

from loan_prediction.model import DEFAULT_DATA, DEFAULT_OUTPUT, predict, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Educational loan-label classification with a decision tree")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=DEFAULT_DATA)
    training.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--income", type=float, required=True, help="Annual income in synthetic USD")
    prediction.add_argument("--amount", type=float, required=True)
    prediction.add_argument("--term", type=int, required=True, help="Term in months")
    prediction.add_argument("--credit-score", type=int, required=True)
    prediction.add_argument("--debt-ratio", type=float, required=True)
    prediction.add_argument("--employment", required=True)
    prediction.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "model.joblib")
    args = parser.parse_args(argv)
    try:
        result = train(args.data, args.output_dir) if args.command == "train" else predict(args.income, args.amount, args.term, args.credit_score, args.debt_ratio, args.employment, args.model)
    except FileNotFoundError as error:
        parser.error(f"File not found: {error.filename}. Run training first if the model is missing.")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    excluded = {"train_application_ids", "test_application_ids", "classification_report"} if args.command == "train" else set()
    print(json.dumps({key: value for key, value in result.items() if key not in excluded}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
