"""Generate fictional records and labels; the rule is not a lending policy."""

import csv
import random
from pathlib import Path


def generate(destination: Path):
    rng = random.Random(2026)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["application_id", "annual_income", "loan_amount", "term_months", "credit_score", "debt_to_income", "employment_type", "approved"])
        for index in range(1, 501):
            income = rng.randrange(20000, 150001, 1000)
            amount = rng.randrange(10000, 350001, 1000)
            term = rng.choice([12, 24, 36, 60, 120, 180, 240, 360])
            credit = rng.randint(500, 850)
            debt_ratio = round(rng.uniform(0.05, 0.6), 3)
            employment = rng.choice(["salaried", "self_employed", "contract"])
            approved = int(credit >= 650 and debt_ratio <= 0.40 and income >= 30000 and amount <= income * 4)
            if rng.random() < 0.05:
                approved = 1 - approved
            writer.writerow([f"DEMO{index:04d}", income, amount, term, credit, debt_ratio, employment, approved])


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent.parent / "sample_data" / "loans.csv")
