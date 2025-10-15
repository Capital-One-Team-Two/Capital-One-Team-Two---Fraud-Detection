#arav
import json
import boto3
import uuid
import datetime
from decimal import Decimal


# boto is the AWS SDK for Python  to interact with dynamo...
sns = boto3.client('sns')

db = boto3.resource('dynamodb')

table = db.Table('FraudDetection-Transactions')

def lambda_handler(event, context):

    body = json.loads(event['body'], parse_float=Decimal)
    
    # Generate a unique ID for this transaction
    tx_id = str(uuid.uuid4())

    transaction = {
        'transaction_id': tx_id,
        'account_id': body['account_id'],   
        'amount': body['amount'],           
        'merchant': body['merchant'],       
        'timestamp': datetime.datetime.utcnow().isoformat(),  
        'card_number': None,                                
    }

    #adding stuff to the db
    table.put_item(Item=transaction)


    sns.publish(
        #needs to be replaced with correct sns arn
        TopicArn='arn:aws:sns:REPLACE:ScoreTopic',
        Message=json.dumps(transaction)
    )
    return {'statusCode': 200, 'body': json.dumps(
        {'message': 'Transaction added successfully',
            'transaction_id': tx_id
        })
    }


