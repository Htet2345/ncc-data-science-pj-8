# Synthetic loan-classification data

`loans.csv` contains 500 generated classroom records, not real applicants,
accounts or lending outcomes. The generator uses Python's standard library
and random seed 2026. IDs beginning with DEMO are arbitrary audit labels.

The synthetic target starts as 1 when all these invented conditions hold:

```text
credit_score >= 650
debt_to_income <= 0.40
annual_income >= 30000
loan_amount <= 4 * annual_income
```

Each label is then independently flipped with a 5% probability to add noise.
This is an invented data-generation rule, not a lender's policy or an
affordability recommendation. Employment type and loan term are included as
candidate features but are not part of the target-generation rule.

Income and loan amounts use synthetic USD units. Debt-to-income is a supplied
fraction from 0 to 1, not a percentage from 0 to 100. It is not recalculated
from loan amount/term; interest rates and payment schedules are not modelled.

Run `python -m scripts.generate_data` from the project folder to recreate the
same file. No account login or data download is required. The dataset has not
been confirmed as the one used in the teacher's videos.
