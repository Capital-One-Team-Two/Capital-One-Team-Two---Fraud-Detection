#zx
import json
import boto3
import os

dynamodb = boto3.resource('dynamodb')
transactions_table = dynamodb.Table(os.environ['TRANSACTIONS_TABLE'])

def lambda_handler(event, context):
    
    try:
        # Get transaction_id from path parameters
        transaction_id = event['pathParameters'].get('transaction_id')
        
        if not transaction_id:
            return {
                'statusCode': 400,
                'headers': get_cors_headers(),
                'body': json.dumps({'error': 'transaction_id required'})
            }
        
        # Retrieve transaction from DynamoDB
        response = transactions_table.get_item(
            Key={'transaction_id': transaction_id}
        )
        
        if 'Item' not in response:
            return {
                'statusCode': 404,
                'headers': get_cors_headers(),
                'body': json.dumps({'error': 'Transaction not found'})
            }
        
        transaction = response['Item']
        
        # Check if scoring is complete
        if transaction.get('status') == 'pending':
            return {
                'statusCode': 202,
                'headers': get_cors_headers(),
                'body': json.dumps({
                    'message': 'Transaction is being processed',
                    'status': 'pending'
                })
            }
        
        # Prepare response
        result = {
            'transaction_id': transaction['transaction_id'],
            'status': transaction.get('status', 'unknown'),
            'fraud_score': float(transaction.get('fraud_score', 0)),
            'is_fraud': transaction.get('is_fraud', False),
            'amount': float(transaction['amount']),
            'merchant': transaction['merchant'],
            'timestamp': transaction.get('timestamp')
        }
        
        # Add appropriate message based on fraud detection
        if result['is_fraud']:
            result['message'] = (
                'This transaction has been flagged as potentially fraudulent. '
                'We have sent you an SMS notification. Please verify this transaction.'
            )
            result['action_required'] = True
        else:
            result['message'] = 'Transaction approved'
            result['action_required'] = False
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': json.dumps(result, default=str)
        }
        
    except Exception as e:
        print(f"Error in user response: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': 'Internal server error'})
        }


def get_cors_headers():
    """Return CORS headers for API Gateway"""
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
    }
