
import os
import json
import importlib
from conftest import create_table_dynamodb

def test_accounts_lambda_happy_path(ddb_resource, monkeypatch):
    create_table_dynamodb(ddb_resource.meta.client, "FraudDetection-Accounts", "user_id")
    accounts_tbl = ddb_resource.Table("FraudDetection-Accounts")
    accounts_tbl.put_item(Item={
        "user_id": "user_123",
        "phone_number": "+15555550123",
        "email": "u123@example.com",
        "name": "Arul"
    })
    monkeypatch.setenv("ACCOUNTS_TABLE", "FraudDetection-Accounts")

    from lambdas.accounts import accounts as accounts_mod
    importlib.reload(accounts_mod)

    res = accounts_mod.lambda_handler({"user_id": "user_123"}, None)
    body = json.loads(res["body"])
    assert res["statusCode"] == 200
    assert body["phone_number"] == "+15555550123"

def test_accounts_lambda_user_not_found(ddb_resource, monkeypatch):
    create_table_dynamodb(ddb_resource.meta.client, "FraudDetection-Accounts", "user_id")
    monkeypatch.setenv("ACCOUNTS_TABLE", "FraudDetection-Accounts")

    from lambdas.accounts import accounts as accounts_mod
    importlib.reload(accounts_mod)

    res = accounts_mod.lambda_handler({"user_id": "missing"}, None)
    assert res["statusCode"] == 404

def test_accounts_lambda_missing_user_id(ddb_resource, monkeypatch):
    create_table_dynamodb(ddb_resource.meta.client, "FraudDetection-Accounts", "user_id")
    monkeypatch.setenv("ACCOUNTS_TABLE", "FraudDetection-Accounts")

    from lambdas.accounts import accounts as accounts_mod
    importlib.reload(accounts_mod)

    res = accounts_mod.lambda_handler({}, None)
    assert res["statusCode"] == 400

def test_get_phone_number_helper(ddb_resource, monkeypatch):
    create_table_dynamodb(ddb_resource.meta.client, "FraudDetection-Accounts", "user_id")
    ddb_resource.Table("FraudDetection-Accounts").put_item(Item={
        "user_id": "u1", "phone_number": "+12223334444"
    })
    monkeypatch.setenv("ACCOUNTS_TABLE", "FraudDetection-Accounts")

    from lambdas.accounts import accounts as accounts_mod
    importlib.reload(accounts_mod)

    assert accounts_mod.get_phone_number("u1") == "+12223334444"
    assert accounts_mod.get_phone_number("none") is None
