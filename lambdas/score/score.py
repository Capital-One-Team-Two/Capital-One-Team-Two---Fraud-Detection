#arul
import os
import json
import base64
import logging
from datetime import datetime, timezone
import random

import boto3
from boto3.dynamodb.types import TypeDeserializer

log = logging.getLogger()
log.setLevel(logging.INFO)

dynamodb = boto3.client("dynamodb")
sagemaker_rt = boto3.client("sagemaker-runtime")

deser = TypeDeserializer()

REGION = os.environ.get("AWS_REGION", "us-east-1")
TRANSACTIONS_TABLE = os.environ["TRANSACTIONS_TABLE"]              
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.7"))  
MODEL_VERSION = os.environ.get("MODEL_VERSION", "mock-v0")

def _from_stream_image(image: dict) -> dict:
    return {k: deser.deserialize(v) for k, v in image.items()}

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def mock_score(txn: dict) -> float:
    """
    Replace this with a real model call later.
    For now: simple heuristic + randomness so tests are reproducible-ish.
    """
    base = 0.15
    amt = float(txn.get("amount", 0.0))
    base += min(0.6, amt / float(os.environ.get("HIGH_RISK_AMOUNT", "1000.0")) * 0.6)
    if str(txn.get("merchant", "")).lower().startswith("unknown"):
        base += 0.15
    score = max(0.0, min(1.0, base + random.uniform(-0.05, 0.05)))
    return score

def update_txn_scores(transaction_id: str, p_raw: float, decision: str):
    dynamodb.update_item(
        TableName=TRANSACTIONS_TABLE,
        Key={"transaction_id": {"S": transaction_id}},
        UpdateExpression="SET p_raw = :p, decision = :d, scored_at = :t, model_version = :m",
        ExpressionAttributeValues={
            ":p": {"N": f"{p_raw:.6f}"},
            ":d": {"S": decision},
            ":t": {"S": _iso_now()},
            ":m": {"S": MODEL_VERSION},
        },
    )

def handler(event, context):
    processed = 0
    for rec in event.get("Records", []):
        if rec.get("eventName") not in ("INSERT",):
            continue
        new_img = rec.get("dynamodb", {}).get("NewImage")
        if not new_img:
            continue

        txn = _from_stream_image(new_img)
        txn_id = txn.get("transaction_id")
        user_id = txn.get("user_id")
        if not txn_id or not user_id:
            log.warning("Skipping record missing transaction_id/user_id: %s", txn)
            continue

        try:
            p_raw = mock_score(txn)
            decision = "alert" if p_raw >= FRAUD_THRESHOLD else "no_alert"
            update_txn_scores(txn_id, p_raw, decision)
            log.info("Scored txn=%s user=%s p_raw=%.4f thr=%.2f decision=%s",
                     txn_id, user_id, p_raw, FRAUD_THRESHOLD, decision)
            processed += 1
        except Exception as e:
            log.exception("Failed to score transaction %s: %s", txn_id, e)

    return {"processed": processed}