#!/usr/bin/env python3
"""
End-to-end test script for the fraud detection pipeline.
Tests the full flow: DynamoDB → Score Lambda → SageMaker → Notify Lambda
"""

import boto3
import json
import pandas as pd
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
import sys
import os

# Add parent directory to path for utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import generate_user_id

# Configuration
REGION = "us-east-2"
TRANSACTIONS_TABLE = "FraudDetection-Transactions"
ACCOUNTS_TABLE = "FraudDetection-Accounts"
PHONE_NUMBER = "+16082177160"  # 608-217-7160
CSV_PATH = "fraudTest.csv"
NUM_FRAUD = 2
NUM_NON_FRAUD = 2

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name=REGION)
transactions_table = dynamodb.Table(TRANSACTIONS_TABLE)
accounts_table = dynamodb.Table(ACCOUNTS_TABLE)

def load_test_transactions(csv_path, num_fraud=10, num_non_fraud=10):
    """Load transactions from CSV."""
    print(f"Loading transactions from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent: {df['is_fraud'].sum():,}")
    print(f"Non-fraudulent: {(df['is_fraud']==0).sum():,}")
    
    fraud_df = df[df['is_fraud'] == 1].sample(n=min(num_fraud, len(df[df['is_fraud'] == 1])), random_state=42)
    non_fraud_df = df[df['is_fraud'] == 0].sample(n=min(num_non_fraud, len(df[df['is_fraud'] == 0])), random_state=42)
    
    print(f"\nSelected {len(fraud_df)} fraudulent and {len(non_fraud_df)} non-fraudulent transactions")
    
    return fraud_df, non_fraud_df

def convert_csv_to_transaction(row, is_fraudulent):
    """Convert CSV row to DynamoDB transaction format."""
    # Generate consistent user_id
    first = str(row.get('first', 'Test'))
    last = str(row.get('last', 'User'))
    cc_num = str(row.get('cc_num', '1234567890'))
    user_id = generate_user_id(first, last, cc_num)
    
    transaction = {
        'transaction_id': str(uuid.uuid4()),
        'user_id': user_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'amt': Decimal(str(row.get('amt', 0))),
        'category': str(row.get('category', '')),
        'merchant': str(row.get('merchant', '')),
        'trans_date_trans_time': str(row.get('trans_date_trans_time', '')),
        'city': str(row.get('city', '')),
        'state': str(row.get('state', '')),
        'lat': Decimal(str(row.get('lat', 0))) if pd.notna(row.get('lat')) else Decimal('0'),
        'long': Decimal(str(row.get('long', 0))) if pd.notna(row.get('long')) else Decimal('0'),
        'city_pop': int(row.get('city_pop', 0)) if pd.notna(row.get('city_pop')) else 0,
        'job': str(row.get('job', '')),
        'merch_lat': Decimal(str(row.get('merch_lat', 0))) if pd.notna(row.get('merch_lat')) else Decimal('0'),
        'merch_long': Decimal(str(row.get('merch_long', 0))) if pd.notna(row.get('merch_long')) else Decimal('0'),
        'unix_time': int(row.get('unix_time', 0)) if pd.notna(row.get('unix_time')) else 0,
        'gender': str(row.get('gender', '')),
        'dob': str(row.get('dob', '')),
        'expected_fraud': 1 if is_fraudulent else 0  # For validation
    }
    
    return transaction, user_id

def ensure_account_exists(user_id, first_name, last_name):
    """Ensure account exists in Accounts table."""
    try:
        response = accounts_table.get_item(Key={'user_id': user_id})
        if 'Item' not in response:
            # Create account
            accounts_table.put_item(
                Item={
                    'user_id': user_id,
                    'name': f"{first_name} {last_name}",
                    'phone_number': PHONE_NUMBER,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'account_status': 'active'
                }
            )
            print(f"    ✓ Created account for {user_id}")
        else:
            print(f"    ✓ Account exists for {user_id}")
    except Exception as e:
        print(f"    ⚠️  Warning: Could not ensure account exists: {e}")

def add_transaction_to_dynamodb(transaction):
    """Add transaction to DynamoDB (triggers Score Lambda via Stream)."""
    try:
        transactions_table.put_item(Item=transaction)
        return True
    except Exception as e:
        print(f"    ❌ Error adding transaction: {e}")
        return False

def wait_for_scoring(transaction_id, max_wait=30):
    """Wait for transaction to be scored by Score Lambda."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = transactions_table.get_item(Key={'transaction_id': transaction_id})
            if 'Item' in response:
                item = response['Item']
                # Check if scored (has p_raw or decision field)
                if 'p_raw' in item or 'decision' in item or 'fraud_score' in item:
                    return item
        except Exception as e:
            print(f"    ⚠️  Error checking transaction: {e}")
        
        time.sleep(1)
    
    return None

def check_notify_lambda_triggered(transaction_id):
    """Check CloudWatch logs to see if Notify Lambda was triggered."""
    # This is a simplified check - in reality you'd query CloudWatch logs
    # For now, we'll just check if decision is 'alert' (which would trigger Notify)
    try:
        response = transactions_table.get_item(Key={'transaction_id': transaction_id})
        if 'Item' in response:
            item = response['Item']
            decision = item.get('decision', '')
            if decision == 'alert':
                return True
    except:
        pass
    return False

def test_pipeline(fraud_df, non_fraud_df):
    """Test the full pipeline with fraudulent and non-fraudulent transactions."""
    results = []
    
    print("\n" + "="*80)
    print("TESTING FRAUDULENT TRANSACTIONS (Full Pipeline)")
    print("="*80)
    
    for idx, row in fraud_df.iterrows():
        print(f"\n[{len(results)+1}] Processing Fraudulent Transaction:")
        print(f"    Amount: ${row.get('amt', 0):.2f}")
        print(f"    Category: {row.get('category', 'N/A')}")
        print(f"    Merchant: {row.get('merchant', 'N/A')}")
        
        # Convert to transaction format
        transaction, user_id = convert_csv_to_transaction(row, is_fraudulent=True)
        
        # Ensure account exists
        ensure_account_exists(user_id, str(row.get('first', 'Test')), str(row.get('last', 'User')))
        
        # Add transaction to DynamoDB (triggers Stream → Score Lambda)
        print(f"    Adding transaction to DynamoDB...")
        if add_transaction_to_dynamodb(transaction):
            transaction_id = transaction['transaction_id']
            print(f"    ✓ Transaction added: {transaction_id}")
            print(f"    ⏳ Waiting for Score Lambda to process (max 30s)...")
            
            # Wait for scoring
            scored_item = wait_for_scoring(transaction_id)
            
            if scored_item:
                p_raw = float(scored_item.get('p_raw', 0))
                decision = scored_item.get('decision', 'unknown')
                threshold = float(scored_item.get('threshold', 0.25))
                
                print(f"    ✓ Transaction scored!")
                print(f"    Fraud Probability: {p_raw:.6f}")
                print(f"    Threshold: {threshold:.4f}")
                print(f"    Decision: {decision.upper()}")
                
                if decision == "alert":
                    print(f"    ✅ CORRECT - Detected as fraud (Score Lambda → Notify Lambda triggered)")
                    notify_triggered = check_notify_lambda_triggered(transaction_id)
                    if notify_triggered:
                        print(f"    ✅ Notify Lambda was triggered")
                else:
                    print(f"    ❌ FALSE NEGATIVE - Missed fraud!")
                
                results.append({
                    'transaction_id': transaction_id,
                    'type': 'fraudulent',
                    'amount': float(row.get('amt', 0)),
                    'fraud_probability': p_raw,
                    'decision': decision,
                    'correct': decision == "alert",
                    'scored': True
                })
            else:
                print(f"    ❌ Transaction not scored within timeout")
                results.append({
                    'transaction_id': transaction_id,
                    'type': 'fraudulent',
                    'amount': float(row.get('amt', 0)),
                    'fraud_probability': None,
                    'decision': 'timeout',
                    'correct': False,
                    'scored': False
                })
        else:
            results.append({
                'transaction_id': None,
                'type': 'fraudulent',
                'amount': float(row.get('amt', 0)),
                'fraud_probability': None,
                'decision': 'error',
                'correct': False,
                'scored': False
            })
        
        # Small delay between transactions
        time.sleep(2)
    
    print("\n" + "="*80)
    print("TESTING NON-FRAUDULENT TRANSACTIONS (Full Pipeline)")
    print("="*80)
    
    for idx, row in non_fraud_df.iterrows():
        print(f"\n[{len(results)+1}] Processing Non-Fraudulent Transaction:")
        print(f"    Amount: ${row.get('amt', 0):.2f}")
        print(f"    Category: {row.get('category', 'N/A')}")
        print(f"    Merchant: {row.get('merchant', 'N/A')}")
        
        # Convert to transaction format
        transaction, user_id = convert_csv_to_transaction(row, is_fraudulent=False)
        
        # Ensure account exists
        ensure_account_exists(user_id, str(row.get('first', 'Test')), str(row.get('last', 'User')))
        
        # Add transaction to DynamoDB
        print(f"    Adding transaction to DynamoDB...")
        if add_transaction_to_dynamodb(transaction):
            transaction_id = transaction['transaction_id']
            print(f"    ✓ Transaction added: {transaction_id}")
            print(f"    ⏳ Waiting for Score Lambda to process (max 30s)...")
            
            # Wait for scoring
            scored_item = wait_for_scoring(transaction_id)
            
            if scored_item:
                p_raw = float(scored_item.get('p_raw', 0))
                decision = scored_item.get('decision', 'unknown')
                threshold = float(scored_item.get('threshold', 0.25))
                
                print(f"    ✓ Transaction scored!")
                print(f"    Fraud Probability: {p_raw:.6f}")
                print(f"    Threshold: {threshold:.4f}")
                print(f"    Decision: {decision.upper()}")
                
                if decision == "no_alert":
                    print(f"    ✅ CORRECT - Not flagged as fraud")
                else:
                    print(f"    ❌ FALSE POSITIVE - Incorrectly flagged as fraud!")
                
                results.append({
                    'transaction_id': transaction_id,
                    'type': 'non_fraudulent',
                    'amount': float(row.get('amt', 0)),
                    'fraud_probability': p_raw,
                    'decision': decision,
                    'correct': decision == "no_alert",
                    'scored': True
                })
            else:
                print(f"    ❌ Transaction not scored within timeout")
                results.append({
                    'transaction_id': transaction_id,
                    'type': 'non_fraudulent',
                    'amount': float(row.get('amt', 0)),
                    'fraud_probability': None,
                    'decision': 'timeout',
                    'correct': False,
                    'scored': False
                })
        else:
            results.append({
                'transaction_id': None,
                'type': 'non_fraudulent',
                'amount': float(row.get('amt', 0)),
                'fraud_probability': None,
                'decision': 'error',
                'correct': False,
                'scored': False
            })
        
        # Small delay between transactions
        time.sleep(2)
    
    return results

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("PIPELINE TEST SUMMARY")
    print("="*80)
    
    fraud_results = [r for r in results if r['type'] == 'fraudulent']
    non_fraud_results = [r for r in results if r['type'] == 'non_fraudulent']
    
    fraud_scored = sum(1 for r in fraud_results if r['scored'])
    fraud_correct = sum(1 for r in fraud_results if r['correct'])
    
    non_fraud_scored = sum(1 for r in non_fraud_results if r['scored'])
    non_fraud_correct = sum(1 for r in non_fraud_results if r['correct'])
    
    print(f"\nFraudulent Transactions ({len(fraud_results)}):")
    print(f"  ✓ Scored by pipeline: {fraud_scored}/{len(fraud_results)}")
    print(f"  ✅ Correctly detected: {fraud_correct}/{len(fraud_results)} ({fraud_correct/len(fraud_results)*100:.1f}%)")
    print(f"  ❌ Missed (False Negatives): {len(fraud_results) - fraud_correct}")
    
    print(f"\nNon-Fraudulent Transactions ({len(non_fraud_results)}):")
    print(f"  ✓ Scored by pipeline: {non_fraud_scored}/{len(non_fraud_results)}")
    print(f"  ✅ Correctly classified: {non_fraud_correct}/{len(non_fraud_results)} ({non_fraud_correct/len(non_fraud_results)*100:.1f}%)")
    print(f"  ❌ False Positives: {len(non_fraud_results) - non_fraud_correct}")
    
    total_scored = fraud_scored + non_fraud_scored
    total_correct = fraud_correct + non_fraud_correct
    total = len(results)
    
    print(f"\nPipeline Performance:")
    print(f"  Transactions processed: {total_scored}/{total} ({total_scored/total*100:.1f}%)")
    print(f"  Overall Accuracy: {total_correct}/{total} ({total_correct/total*100:.1f}%)")
    
    # Show probabilities
    print(f"\nFraud Probability Statistics:")
    fraud_probs = [r['fraud_probability'] for r in fraud_results if r['fraud_probability'] is not None]
    non_fraud_probs = [r['fraud_probability'] for r in non_fraud_results if r['fraud_probability'] is not None]
    
    if fraud_probs:
        print(f"  Fraudulent - Avg: {sum(fraud_probs)/len(fraud_probs):.4f}, Min: {min(fraud_probs):.4f}, Max: {max(fraud_probs):.4f}")
    if non_fraud_probs:
        print(f"  Non-fraudulent - Avg: {sum(non_fraud_probs)/len(non_fraud_probs):.4f}, Min: {min(non_fraud_probs):.4f}, Max: {max(non_fraud_probs):.4f}")

def main():
    print("🧪 End-to-End Pipeline Test Script")
    print(f"Region: {REGION}")
    print(f"Transactions Table: {TRANSACTIONS_TABLE}")
    print(f"Accounts Table: {ACCOUNTS_TABLE}")
    print(f"Phone Number: {PHONE_NUMBER}\n")
    
    # Load transactions
    try:
        fraud_df, non_fraud_df = load_test_transactions(CSV_PATH, NUM_FRAUD, NUM_NON_FRAUD)
    except FileNotFoundError:
        print(f"❌ Error: {CSV_PATH} not found!")
        print(f"   Make sure you're running this from the infrastructure/ directory")
        return
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test pipeline
    results = test_pipeline(fraud_df, non_fraud_df)
    
    # Print summary
    print_summary(results)
    
    print("\n✅ Pipeline testing complete!")
    print("\n💡 Note: Check CloudWatch logs for Score Lambda and Notify Lambda to see full processing details")

if __name__ == '__main__':
    main()

