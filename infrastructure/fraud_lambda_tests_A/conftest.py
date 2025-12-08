
import sys
from pathlib import Path

# ---- Make repo root importable (tests live two dirs below repo root) ----
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import os
import json
import importlib
import pytest
from moto import mock_aws
import boto3

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

@pytest.fixture(scope="function")
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", AWS_REGION)
    monkeypatch.setenv("AWS_REGION", AWS_REGION)
    yield

@pytest.fixture(scope="function")
def moto_aws(aws_env):
    with mock_aws():
        yield

def create_table_dynamodb(dynamodb_client, name, hash_key, range_key=None):
    key_schema=[{"AttributeName": hash_key, "KeyType": "HASH"}]
    attr_defs=[{"AttributeName": hash_key, "AttributeType": "S"}]
    if range_key:
        key_schema.append({"AttributeName": range_key, "KeyType": "RANGE"})
        attr_defs.append({"AttributeName": range_key, "AttributeType": "S"})
    return dynamodb_client.create_table(
        TableName=name,
        KeySchema=key_schema,
        AttributeDefinitions=attr_defs,
        BillingMode="PAY_PER_REQUEST",
    )

@pytest.fixture()
def ddb_resource(moto_aws):
    return boto3.resource("dynamodb", region_name=AWS_REGION)

@pytest.fixture()
def ddb_client(moto_aws):
    return boto3.client("dynamodb", region_name=AWS_REGION)

@pytest.fixture()
def sns_client(moto_aws):
    return boto3.client("sns", region_name=AWS_REGION)

class Spy:
    def __init__(self):
        self.calls = []
    def __call__(self, *a, **kw):
        self.calls.append({"args": a, "kwargs": kw})
        return {"StatusCode": 202}

@pytest.fixture()
def spy():
    return Spy()
