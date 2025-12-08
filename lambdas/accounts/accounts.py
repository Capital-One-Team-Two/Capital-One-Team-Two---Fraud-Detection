import json
import boto3
import os

dynamodb = boto3.resource('dynamodb')
accounts_table = dynamodb.Table(os.environ['ACCOUNTS_TABLE'])
def lambda_handler(event, context):
    
    try:
        # Extract user_id from event
        # This Lambda is triggered by the Transaction Lambda via DynamoDB Stream
        user_id = event.get('user_id')
        
        if not user_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'user_id required'})
            }
        
        # Query DynamoDB for user account information
        response = accounts_table.get_item(
            Key={'user_id': user_id}
        )
        
        if 'Item' not in response:
            print(f"User {user_id} not found")
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'User not found'})
            }
        
        user_data = response['Item']
        phone_number = user_data.get('phone_number')
        email = user_data.get('email')
        name = user_data.get('name')
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'user_id': user_id,
                'phone_number': phone_number,
                'email': email,
                'last_updated': last_updated,
                'name': name
            })
        }
        
    except Exception as e:
        print(f"Error retrieving account: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error'})
        }


def get_phone_number(user_id):
    """
    Helper function to get phone number for a user
    Used by other Lambda functions
    """
    response = accounts_table.get_item(Key={'user_id': user_id})
    
    if 'Item' in response:
        return response['Item'].get('phone_number')
    return None
