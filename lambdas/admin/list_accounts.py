import json, boto3, os
accounts_table = boto3.resource('dynamodb').Table(os.environ['ACCOUNTS_TABLE'])

def lambda_handler(event, context):
    params = event.get("queryStringParameters") or {}
    filter_expr = None
    attr_vals = {}
    # build filters as needed, similar to transactions
    response = accounts_table.scan()
    return {"statusCode": 200, "body": json.dumps({"accounts": response["Items"]}, default=str)}
