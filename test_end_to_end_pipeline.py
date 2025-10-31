#!/usr/bin/env python3
"""
End-to-end test script for fraud detection pipeline
This script adds transactions to DynamoDB and automatically triggers the scoring lambda
"""

import sys
import json
import boto3
import uuid
import datetime
import pandas as pd
from decimal import Decimal
from types import SimpleNamespace

# Add paths for Lambda functions
sys.path.insert(0, 'lambdas/score')
sys.path.insert(0, 'lambdas/transaction')

def load_test_transactions():
    """Load one fraudulent and one non-fraudulent transaction from fraudTest.csv"""
    print("📚 Loading transactions from fraudTest.csv...")
    
    df = pd.read_csv('infrastructure/fraudTest.csv')
    
    # Get one fraudulent transaction
    fraud_df = df[df['is_fraud'] == 1].iloc[0]
    
    # Get one non-fraudulent transaction
    non_fraud_df = df[df['is_fraud'] == 0].iloc[0]
    
    return fraud_df, non_fraud_df

def convert_csv_to_transaction(row):
    """Convert CSV row to transaction dictionary"""
    transaction = {
        'transaction_id': str(uuid.uuid4()),  # Generate new ID
        'user_id': f"user_{str(abs(hash(str(row.get('cc_num', 0)))))[:6]}",
        'account_id': f"acc_{str(abs(hash(str(row.get('cc_num', 0)))))[:6]}",
        'trans_date_trans_time': str(row.get('trans_date_trans_time', datetime.datetime.utcnow().isoformat())),
        'cc_num': int(row.get('cc_num', 0)),
        'merchant': str(row.get('merchant', 'Unknown')),
        'category': str(row.get('category', 'unknown')),
        'amt': Decimal(str(row.get('amt', 0.0))),  # Use 'amt' not 'amount' - this is what the model expects!
        'amount': Decimal(str(row.get('amt', 0.0))),  # Keep 'amount' for display purposes
        'first': str(row.get('first', 'Unknown')),
        'last': str(row.get('last', 'Unknown')),
        'gender': str(row.get('gender', 'M')),
        'street': str(row.get('street', '')),
        'city': str(row.get('city', 'Unknown')),
        'state': str(row.get('state', '')),
        'zip': str(row.get('zip', '00000')),
        'lat': Decimal(str(row.get('lat', 0.0))),
        'long': Decimal(str(row.get('long', 0.0))),
        'city_pop': int(row.get('city_pop', 0)),
        'job': str(row.get('job', 'Unknown')),
        'dob': str(row.get('dob', '1970-01-01')),
        'trans_num': str(row.get('trans_num', str(uuid.uuid4()))),
        'unix_time': int(row.get('unix_time', int(datetime.datetime.utcnow().timestamp()))),
        'merch_lat': Decimal(str(row.get('merch_lat', 0.0))),
        'merch_long': Decimal(str(row.get('merch_long', 0.0))),
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'is_fraud': int(row.get('is_fraud', 0))
    }
    
    return transaction

def add_transaction_to_dynamodb(transaction, table):
    """Add transaction directly to DynamoDB"""
    print(f"\n📝 Adding transaction {transaction['transaction_id']}...")
    print(f"   Amount: ${transaction['amount']}")
    print(f"   Merchant: {transaction['merchant']}")
    print(f"   Expected Fraud: {transaction['is_fraud']}")
    
    # Convert to DynamoDB format
    item = {}
    for key, value in transaction.items():
        if isinstance(value, Decimal):
            item[key] = value
        elif isinstance(value, int):
            item[key] = Decimal(str(value))
        elif isinstance(value, float):
            item[key] = Decimal(str(value))
        else:
            item[key] = value
    
    table.put_item(Item=item)
    print(f"   ✅ Transaction added to DynamoDB")
    
    return item

def convert_to_stream_event(transaction):
    """Convert transaction to DynamoDB Stream event format"""
    # Helper to convert Python types to DynamoDB format
    def to_dynamodb_format(value):
        if isinstance(value, Decimal):
            return {"N": str(value)}
        elif isinstance(value, str):
            return {"S": value}
        elif isinstance(value, int):
            return {"N": str(value)}
        elif isinstance(value, float):
            return {"N": str(value)}
        else:
            return {"S": str(value)}
    
    # Convert transaction to NewImage format
    new_image = {k: to_dynamodb_format(v) for k, v in transaction.items()}
    
    return {
        "Records": [
            {
                "eventName": "INSERT",
                "dynamodb": {
                    "NewImage": new_image
                }
            }
        ]
    }

def trigger_score_lambda(event):
    """Trigger the score lambda with the DynamoDB stream event"""
    print("   🔄 Triggering score lambda...")
    
    from score import handler
    
    result = handler(event, None)
    
    print(f"   ✅ Score lambda processed: {result.get('processed', 0)} transaction(s)")
    return result

def clear_dynamodb_tables(table):
    """Clear all items from the DynamoDB table"""
    print("\n🧹 Clearing DynamoDB tables...")
    
    try:
        # Scan the table and delete all items
        response = table.scan()
        items = response.get('Items', [])
        
        if not items:
            print("   ℹ️  Table is already empty")
            return
        
        print(f"   📋 Found {len(items)} item(s) to delete")
        
        # Delete items in batches
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={'transaction_id': item['transaction_id']})
        
        print(f"   ✅ Cleared {len(items)} item(s) from the table")
        
    except Exception as e:
        print(f"   ⚠️  Error clearing table: {e}")

def main():
    """Run the end-to-end pipeline test"""
    print("=" * 70)
    print("🚀 FRAUD DETECTION END-TO-END PIPELINE TEST")
    print("=" * 70)
    
    # Initialize DynamoDB
    dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
    table = dynamodb.Table('FraudDetection-Transactions')
    
    # Clear tables before starting
    clear_dynamodb_tables(table)
    
    # Load transactions from fraudTest.csv
    fraud_row, legit_row = load_test_transactions()
    
    # Convert to transaction dictionaries
    fraud_transaction = convert_csv_to_transaction(fraud_row)
    legit_transaction = convert_csv_to_transaction(legit_row)
    
    # Test 1: Add fraudulent transaction
    print("\n" + "=" * 70)
    print("TEST 1: FRAUDULENT TRANSACTION (from fraudTest.csv)")
    print("=" * 70)
    
    fraud_item = add_transaction_to_dynamodb(fraud_transaction, table)
    fraud_event = convert_to_stream_event(fraud_item)
    trigger_score_lambda(fraud_event)
    
    # Wait a moment for processing
    import time
    time.sleep(2)
    
    # Test 2: Add legitimate transaction
    print("\n" + "=" * 70)
    print("TEST 2: LEGITIMATE TRANSACTION (from fraudTest.csv)")
    print("=" * 70)
    
    legit_item = add_transaction_to_dynamodb(legit_transaction, table)
    legit_event = convert_to_stream_event(legit_item)
    trigger_score_lambda(legit_event)
    
    # Wait for processing
    time.sleep(2)
    
    # Verify results
    print("\n" + "=" * 70)
    print("📊 VERIFYING RESULTS")
    print("=" * 70)
    
    # Get the fraud transaction
    fraud_id = fraud_transaction['transaction_id']
    response = table.get_item(Key={'transaction_id': fraud_id})
    if 'Item' in response:
        fraud_result = response['Item']
        print(f"\n✅ Fraudulent Transaction Results:")
        print(f"   Transaction ID: {fraud_result.get('transaction_id')}")
        print(f"   Amount: ${fraud_result.get('amount')}")
        print(f"   Merchant: {fraud_result.get('merchant')}")
        print(f"   P_raw (fraud probability): {fraud_result.get('p_raw', 'Not scored yet')}")
        print(f"   Decision: {fraud_result.get('decision', 'Not scored yet')}")
    
    # Get the legitimate transaction
    legit_id = legit_transaction['transaction_id']
    response = table.get_item(Key={'transaction_id': legit_id})
    if 'Item' in response:
        legit_result = response['Item']
        print(f"\n✅ Legitimate Transaction Results:")
        print(f"   Transaction ID: {legit_result.get('transaction_id')}")
        print(f"   Amount: ${legit_result.get('amount')}")
        print(f"   Merchant: {legit_result.get('merchant')}")
        print(f"   P_raw (fraud probability): {legit_result.get('p_raw', 'Not scored yet')}")
        print(f"   Decision: {legit_result.get('decision', 'Not scored yet')}")
    
    print("\n" + "=" * 70)
    print("✅ END-TO-END PIPELINE TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

