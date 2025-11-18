import json
import boto3
import os
import datetime

dynamodb = boto3.resource('dynamodb')
accounts_table = dynamodb.Table('FraudDetection-Accounts')

def lambda_handler(event, context):

    body_str = event.get('body')
    if not body_str:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'message': 'Missing body'
            })
        }

    body = json.loads(body_str)

    if 'user_id' not in body:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'message': 'Missing user_id'
            })
        }

    if 'phone_number' not in body:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'message': 'Missing phone_number'
            })
        }

    if 'name' not in body:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'message': 'Missing name'
            })
        }

    if 'email' not in body:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'message': 'Missing email'
            })
        }

    now_iso = datetime.datetime.utcnow().isoformat()

    item = {
        'user_id': body['user_id'],
        'phone_number': body['phone_number'],
        'name': body['name'],
        'email': body['email'],
        'last_updated': now_iso
    }

    accounts_table.put_item(Item=item)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Account created/updated"
        })
    }