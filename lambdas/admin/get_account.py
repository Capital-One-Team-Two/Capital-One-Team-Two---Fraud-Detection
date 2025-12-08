import json
import boto3
import os

dynamodb = boto3.resource('dynamodb')
accounts_table = dynamodb.Table(os.environ['ACCOUNTS_TABLE'])

def lambda_handler(event, context):
    """
    Retrieve a single account by user_id.
    URL pattern: GET /admin/accounts/{user_id}
    """
    try:
        user_id = event['pathParameters'].get('user_id')
        if not user_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'user_id is required'})
            }

        resp = accounts_table.get_item(Key={'user_id': user_id})
        if 'Item' not in resp:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Account not found'})
            }

        return {
            'statusCode': 200,
            'body': json.dumps(resp['Item'], default=str)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error', 'message': str(e)})
        }
