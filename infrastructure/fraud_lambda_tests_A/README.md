
# Fraud Lambdas — Test Suite (Structure A)

**Assumed repo layout (you said A):**
```
lambdas/
  accounts/accounts.py
  notify/notify.py
  transactions/transaction.py
  scores/score.py
tests/
  unit/   <-- put these tests here (two levels deeper than repo root)
```

These tests import like:
```python
from lambdas.accounts import accounts as accounts_mod
from lambdas.notify import notify as notify_mod
from lambdas.transactions import transaction as txn_mod
from lambdas.scores import score as score_mod
```

We add the repo root to `sys.path` using a helper in `conftest.py`:
`repo_root = Path(__file__).resolve().parents[2]`

## Quick start

```bash
# from your repo root
mkdir -p tests/unit
cp -R fraud_lambda_tests_A/* tests/unit/

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "moto[boto3]" boto3 botocore pytest twilio numpy pandas lightgbm joblib
pytest -q
```

## What’s mocked

- DynamoDB + SNS via `moto`
- Twilio client (`notify.py`) — no real SMS
- LightGBM model (`score.py`) via small mock
- `lambda_client.invoke` (`score.py`) — stubbed

## Notes

- `notify.py` constructs tables and validates Twilio env at **import time**.
  Tests create the tables + env **before** importing and then reload the module.
- `transaction.py` uses a placeholder SNS ARN. We stub `sns.publish`.
- `score.py` gets a mocked `joblib.load` returning a minimal model and feature list.
