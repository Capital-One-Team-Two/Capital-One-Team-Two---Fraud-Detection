import json
import boto3
import os
from boto3.dynamodb.conditions import Attr

dynamodb = boto3.resource('dynamodb')
transactions_table = dynamodb.Table(os.environ['TRANSACTIONS_TABLE'])

def lambda_handler(event, context):
    """
    Return high-level metrics about transactions.
    URL pattern: GET /admin/metrics
    Metrics returned:
      total_transactions      – total count
      flagged_transactions    – where decision is 'alert'
      sms_sent                – where notificationSent = True
      user_confirmed          – where userDecision is 'confirmed'
      user_denied             – where userDecision is 'denied'
      fraud_rate              – flagged/total
    """
    try:
        # Scan entire table (for small datasets; for large tables, consider pre-aggregated stats)
        resp = transactions_table.scan()
        items = resp.get('Items', [])

        total = len(items)
        flagged = sum(1 for i in items if i.get('decision') == 'alert')
        sms_sent = sum(1 for i in items if i.get('notificationSent'))
        user_confirmed = sum(1 for i in items if i.get('userDecision') == 'confirmed')
        user_denied = sum(1 for i in items if i.get('userDecision') == 'denied')

        metrics = {
            'total_transactions': total,
            'flagged_transactions': flagged,
            'sms_sent': sms_sent,
            'user_confirmed': user_confirmed,
            'user_denied': user_denied,
            'fraud_rate': (flagged / total) if total > 0 else 0.0
        }

        return {
            'statusCode': 200,
            'body': json.dumps(metrics, default=str)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error', 'message': str(e)})
        }
