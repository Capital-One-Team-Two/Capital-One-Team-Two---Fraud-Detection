import json
import boto3
import os
import datetime

dynamodb = boto3.resource('dynamodb')
accounts_table = dynamodb.Table("FraudDetection-Accounts")

def lambda_handler(event, context):
    """
    Update or create an account record.
    URL pattern: POST /admin/accounts/{user_id}
    Body can include: phone_number, email, name.
    """
    try:
        user_id = event['pathParameters'].get('user_id')
        if not user_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'user_id is required'})
            }

        if not event.get('body'):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing request body'})
            }

        body = json.loads(event['body'])
        # Build the update expression dynamically
        update_expr = []
        expr_attr_values = {}
        for field in ['phone_number', 'email', 'name']:
            if field in body:
                update_expr.append(f"{field} = :{field}")
                expr_attr_values[f":{field}"] = body[field]
        # Always update last_updated timestamp
        update_expr.append("last_updated = :lu")
        expr_attr_values[":lu"] = datetime.datetime.utcnow().isoformat()

        if not update_expr:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No valid fields provided'})
            }

        accounts_table.update_item(
            Key={'user_id': user_id},
            UpdateExpression="SET " + ", ".join(update_expr),
            ExpressionAttributeValues=expr_attr_values
        )

        return {
            'statusCode': 200,
            'body': json.dumps({'ok': True, 'user_id': user_id})
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error', 'message': str(e)})
        }
