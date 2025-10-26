import os, json, importlib, io
import numpy as np
from datetime import datetime
from decimal import Decimal
from conftest import create_table_dynamodb


class MockModel:
    def predict_proba(self, X):
        amt = float(X.iloc[0]["amount"]) if "amount" in X.columns else 0.0
        p1 = 1.0 if amt >= 500 else 0.1
        return np.array([[1 - p1, p1]])


def make_stream_insert_image(item):
    image = {}
    for k, v in item.items():
        if isinstance(v, (int, float, Decimal)):
            image[k] = {"N": str(v)}
        else:
            image[k] = {"S": str(v)}
    return image


def test_score_handler_updates_items_and_invokes_notify(ddb_resource, monkeypatch):
    # Create table
    create_table_dynamodb(ddb_resource.meta.client, "FraudDetection-Transactions", "transaction_id")
    tbl = ddb_resource.Table("FraudDetection-Transactions")

    # ✅ Use Decimal for DynamoDB numeric fields
    txn_low = {
        "transaction_id": "t_low",
        "user_id": "u1",
        "amount": Decimal("100.0"),
        "timestamp": datetime.utcnow().isoformat()
    }
    txn_hi = {
        "transaction_id": "t_hi",
        "user_id": "u2",
        "amount": Decimal("1000.0"),
        "timestamp": datetime.utcnow().isoformat()
    }
    tbl.put_item(Item=txn_low)
    tbl.put_item(Item=txn_hi)

    # Point score module at our table & model configuration
    monkeypatch.setenv("TRANSACTIONS_TABLE", "FraudDetection-Transactions")
    monkeypatch.setenv("FRAUD_THRESHOLD", "0.5")
    monkeypatch.setenv("MODEL_VERSION", "test-v1")

    # ✅ IMPORTANT: Force the S3 branch so no local file is opened
    monkeypatch.setenv("MODEL_BUCKET", "unit-test-bucket")

    from lambdas.score import score as score_mod
    importlib.reload(score_mod)

    # Fake S3 get_object -> returns a file-like body
    score_mod.s3.get_object = lambda Bucket, Key: {"Body": io.BytesIO(b"fake")}

    # Fake model loader returning our mock model + required feature list
    def fake_joblib_load(file_like):
        return {
            "model": MockModel(),
            "feature_names": ["amount", "trans_hour", "trans_day", "trans_month", "trans_year"],
            "threshold": 0.5,
        }
    score_mod.joblib.load = fake_joblib_load

    # ✅ Spy the async notify invoke
    calls = {"count": 0}
    def fake_invoke(FunctionName, InvocationType, Payload):
        assert InvocationType == "Event"
        calls["count"] += 1
        return {"StatusCode": 202}
    score_mod.lambda_client.invoke = fake_invoke

    event = {
        "Records": [
            {"eventName": "INSERT", "dynamodb": {"NewImage": make_stream_insert_image(txn_low)}},
            {"eventName": "INSERT", "dynamodb": {"NewImage": make_stream_insert_image(txn_hi)}},
        ]
    }

    res = score_mod.handler(event, None)
    assert res["processed"] == 2

    i_low = tbl.get_item(Key={"transaction_id": "t_low"})["Item"]
    i_hi = tbl.get_item(Key={"transaction_id": "t_hi"})["Item"]
    assert i_low["decision"] == "no_alert"
    assert i_hi["decision"] == "alert"
    assert calls["count"] == 1
