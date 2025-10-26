
import os, json, importlib
from conftest import create_table_dynamodb

def test_notify_lambda_updates_tx_and_builds_sms(ddb_resource, monkeypatch):
    # Create tables with names expected by the code
    create_table_dynamodb(ddb_resource.meta.client, "FraudDetection-Transactions", "transaction_id")
    create_table_dynamodb(ddb_resource.meta.client, "FraudDetection-Accounts", "user_id")
    tx = ddb_resource.Table("FraudDetection-Transactions")
    acct = ddb_resource.Table("FraudDetection-Accounts")

    acct.put_item(Item={"user_id": "user_001", "phone_number": "+15555550000"})
    tx.put_item(Item={"transaction_id": "txn_123"})

    # Twilio env
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACxxx")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok_xxx")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15555559999")

    from lambdas.notify import notify as notify_mod
    importlib.reload(notify_mod)

    class FakeMsg:
        def __init__(self): self.sid = "SM_fake_sid"
    class FakeTwilioClient:
        class _Messages:
            def create(self, to, from_, body):
                assert "FRAUD ALERT" in body
                assert "Transaction ID: txn_123" in body
                return FakeMsg()
        def __init__(self,*a,**kw):
            self.messages = self._Messages()
    notify_mod.twilio_client = FakeTwilioClient()

    event = {
        "transaction_id": "txn_123",
        "user_id": "user_001",
        "amount": 1500.0,
        "merchant": "Suspicious Store",
        "p_raw": 0.85,
        "decision": "alert"
    }
    res = notify_mod.lambda_handler(event, None)
    assert res["statusCode"] == 200

    # Ensure tx updated
    item = tx.get_item(Key={"transaction_id": "txn_123"})["Item"]
    assert item.get("notificationSent") is True
    assert "messageSid" in item
    assert "notificationAt" in item

def test_notify_user_missing(ddb_resource, monkeypatch):
    create_table_dynamodb(ddb_resource.meta.client, "FraudDetection-Transactions", "transaction_id")
    create_table_dynamodb(ddb_resource.meta.client, "FraudDetection-Accounts", "user_id")
    ddb_resource.Table("FraudDetection-Transactions").put_item(Item={"transaction_id": "txn_404"})

    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACxxx")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok_xxx")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15555559999")

    from lambdas.notify import notify as notify_mod
    importlib.reload(notify_mod)

    class FakeTwilioClient:
        class _Messages:
            def create(self, **kw): raise AssertionError("Should not send for 404")
        def __init__(self): self.messages = self._Messages()
    notify_mod.twilio_client = FakeTwilioClient()

    res = notify_mod.lambda_handler(
        {"transaction_id": "txn_404", "user_id": "missing"}, None
    )
    assert res["statusCode"] == 404
