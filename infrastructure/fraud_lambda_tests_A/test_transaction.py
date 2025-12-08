import os, json, importlib
from conftest import create_table_dynamodb


def test_transaction_lambda_stores_and_publishes(ddb_resource, monkeypatch):
    create_table_dynamodb(ddb_resource.meta.client, "FraudDetection-Transactions", "transaction_id")

    from lambdas.transaction import transaction as txn_mod
    importlib.reload(txn_mod)

    # Keep a reference to stdlib json.dumps to avoid recursion
    _orig_json_dumps = json.dumps

    # Allow Decimal to JSON serialize and accept any kwargs.
    def _safe_dumps(obj, *args, **kwargs):
        if "default" not in kwargs:
            kwargs["default"] = str
        return _orig_json_dumps(obj, *args, **kwargs)

    # Patch only inside the transaction module (SNS publish path)
    monkeypatch.setattr(txn_mod.json, "dumps", _safe_dumps, raising=True)

    published = {}

    def fake_publish(TopicArn, Message):
        published["TopicArn"] = TopicArn
        published["Message"] = Message
        return {"MessageId": "mid-123"}

    txn_mod.sns.publish = fake_publish

    # Invoke lambda
    body = {"account_id": "acc_1", "amount": 42.5, "merchant": "Coffee Shop"}
    event = {"body": json.dumps(body)}  # stdlib json is fine here
    res = txn_mod.lambda_handler(event, None)
    assert res["statusCode"] == 200

    # Validate lambda response
    payload = json.loads(res["body"])
    tx_id = payload["transaction_id"]
    assert payload["message"] == "Transaction added successfully"

    # Validate SNS message payload (what we attempted to publish)
    msg = json.loads(published["Message"])
    assert msg["transaction_id"] == tx_id
    assert msg["account_id"] == "acc_1"
    # amount may have become Decimal inside the lambda; compare numerically
    assert float(msg["amount"]) == 42.5
    assert msg["merchant"] == "Coffee Shop"
