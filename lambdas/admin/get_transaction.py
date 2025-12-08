# lambdas/admin/get_transaction.py
import json, boto3, os
table = boto3.resource('dynamodb').Table(os.environ['TRANSACTIONS_TABLE'])

def lambda_handler(event, context):
    txn_id = event["pathParameters"].get("transaction_id")
    response = table.get_item(Key={"transaction_id": txn_id})
    if "Item" not in response:
        return {"statusCode": 404, "body": json.dumps({"error": "Not found"})}
    return {"statusCode": 200, "body": json.dumps(response["Item"], default=str)}
