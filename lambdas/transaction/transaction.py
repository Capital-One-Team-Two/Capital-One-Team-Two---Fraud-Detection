# arav
import json
import boto3
import uuid
import datetime
from decimal import Decimal

db = boto3.resource('dynamodb')
table = db.Table('FraudDetection-Transactions')

def decimal_to_float(obj):
    """Convert Decimal to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def lambda_handler(event, context):

    # ----- Validate body -----
    body_str = event.get("body")
    if body_str is None:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing body in request"})
        }

    body = json.loads(body_str, parse_float=Decimal)

    # Required fields
    required = ["user_id", "amount", "merchant"]
    for field in required:
        if field not in body:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Missing required field: {field}"})
            }

    # Generate transaction ID
    tx_id = str(uuid.uuid4())

    # Build transaction object
    transaction = {
        "transaction_id": tx_id,
        "user_id": body["user_id"],
        "amount": body["amount"],
        "merchant": body["merchant"],
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    # Store in DynamoDB
    table.put_item(Item=transaction)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Transaction added successfully",
            "transaction_id": tx_id
        })
    }
