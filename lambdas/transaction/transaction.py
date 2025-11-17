import json
import boto3
import uuid
import datetime
from decimal import Decimal

db = boto3.resource('dynamodb')
table = db.Table('FraudDetection-Transactions')


def decimal_to_native(x):
    if isinstance(x, Decimal):
        if x % 1 == 0:
            return int(x)
        return float(x)
    if isinstance(x, dict):
        return {k: decimal_to_native(v) for k, v in x.items()}
    if isinstance(x, list):
        return [decimal_to_native(i) for i in x]
    return x


def lambda_handler(event, context):

    body_str = event.get("body")
    if not body_str:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing body"})
        }

    body = json.loads(body_str, parse_float=Decimal)

    if "transaction_id" not in body:
        body["transaction_id"] = str(uuid.uuid4())

    if "timestamp" not in body:
        body["timestamp"] = datetime.datetime.utcnow().isoformat()

    table.put_item(Item=body)

    # ---- Remove these fields from return ONLY ----
    filtered = {k: v for k, v in body.items()
                if k not in ("p_raw", "decision", "is_fraud")}

    filtered = decimal_to_native(filtered)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Transaction created successfully",
            "transaction": filtered
        })
    }
