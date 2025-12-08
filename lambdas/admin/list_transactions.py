import json, boto3, os
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TRANSACTIONS_TABLE'])

def lambda_handler(event, context):
    params = event.get("queryStringParameters") or {}
    user_id = params.get("user_id")
    status = params.get("status")   # e.g. "flagged", "verified"
    limit = int(params.get("limit", 20))
    # Build scan filter expression only if filters present
    filter_expr = None
    attr_vals = {}
    if user_id:
        from boto3.dynamodb.conditions import Attr
        filter_expr = Attr("user_id").eq(user_id)
    if status:
        from boto3.dynamodb.conditions import Attr
        expr = Attr("decision").eq(status)
        filter_expr = expr if not filter_expr else filter_expr & expr

    if filter_expr:
        response = table.scan(
            FilterExpression=filter_expr,
            Limit=limit
        )
    else:
        response = table.scan(Limit=limit)
    items = response.get("Items", [])
    return {
        "statusCode": 200,
        "body": json.dumps({"transactions": items}, default=str)
    }
