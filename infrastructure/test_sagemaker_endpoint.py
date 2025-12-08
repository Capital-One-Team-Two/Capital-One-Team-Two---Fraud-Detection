#!/usr/bin/env python3
"""
Test script for SageMaker fraud detection endpoint.
Tests 10 fraudulent and 10 non-fraudulent transactions.
"""

import boto3
import json
import pandas as pd
import random
from datetime import datetime

# Configuration
ENDPOINT_NAME = "fraud-detection-endpoint-v2"
REGION = "us-east-2"
PHONE_NUMBER = "+16082177160"  # 608-217-7160
CSV_PATH = "fraudTest.csv"

# Initialize SageMaker runtime client
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION)

def load_transactions(csv_path, num_fraud=10, num_non_fraud=10):
    """Load transactions from CSV, selecting equal numbers of fraud/non-fraud."""
    print(f"Loading transactions from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent: {df['is_fraud'].sum():,}")
    print(f"Non-fraudulent: {(df['is_fraud']==0).sum():,}")
    
    # Sample fraudulent transactions
    fraud_df = df[df['is_fraud'] == 1].sample(n=min(num_fraud, len(df[df['is_fraud'] == 1])), random_state=42)
    
    # Sample non-fraudulent transactions
    non_fraud_df = df[df['is_fraud'] == 0].sample(n=min(num_non_fraud, len(df[df['is_fraud'] == 0])), random_state=42)
    
    print(f"\nSelected {len(fraud_df)} fraudulent and {len(non_fraud_df)} non-fraudulent transactions")
    
    return fraud_df, non_fraud_df

def format_transaction_for_endpoint(row):
    """Format a transaction row for SageMaker endpoint input."""
    # Convert to dict matching the CSV column names (inference.py expects these)
    txn = {
        "amt": float(row.get('amt', 0)),
        "trans_date_trans_time": str(row.get('trans_date_trans_time', '')),
        "category": str(row.get('category', '')),
        "city": str(row.get('city', '')),
        "state": str(row.get('state', '')),
        "lat": float(row.get('lat', 0)) if pd.notna(row.get('lat')) else 0.0,
        "long": float(row.get('long', 0)) if pd.notna(row.get('long')) else 0.0,
        "city_pop": int(row.get('city_pop', 0)) if pd.notna(row.get('city_pop')) else 0,
        "job": str(row.get('job', '')),
        "merch_lat": float(row.get('merch_lat', 0)) if pd.notna(row.get('merch_lat')) else 0.0,
        "merch_long": float(row.get('merch_long', 0)) if pd.notna(row.get('merch_long')) else 0.0,
        "unix_time": int(row.get('unix_time', 0)) if pd.notna(row.get('unix_time')) else 0,
        "gender": str(row.get('gender', '')),
        "dob": str(row.get('dob', ''))
    }
    return txn

def invoke_endpoint(transaction_data):
    """Invoke the SageMaker endpoint with transaction data."""
    try:
        # Serialize to JSON
        payload = json.dumps(transaction_data)
        
        # Invoke endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result_body = response['Body'].read().decode('utf-8')
        result = json.loads(result_body)
        
        return result
    except Exception as e:
        return {'error': str(e)}

def test_transactions(fraud_df, non_fraud_df):
    """Test both fraudulent and non-fraudulent transactions."""
    results = []
    
    print("\n" + "="*80)
    print("TESTING FRAUDULENT TRANSACTIONS")
    print("="*80)
    
    for idx, row in fraud_df.iterrows():
        print(f"\n[{len(results)+1}] Fraudulent Transaction:")
        print(f"    Amount: ${row.get('amt', 0):.2f}")
        print(f"    Category: {row.get('category', 'N/A')}")
        print(f"    Merchant: {row.get('merchant', 'N/A')}")
        print(f"    Location: {row.get('city', 'N/A')}, {row.get('state', 'N/A')}")
        
        txn_data = format_transaction_for_endpoint(row)
        result = invoke_endpoint(txn_data)
        
        if 'error' in result:
            print(f"    ❌ ERROR: {result['error']}")
            decision = "ERROR"
            prob = None
        else:
            prob = result.get('fraud_probability', 0)
            threshold = result.get('threshold', 0.25)
            decision = result.get('decision', 'unknown')
            
            print(f"    Fraud Probability: {prob:.6f}")
            print(f"    Threshold: {threshold:.4f}")
            print(f"    Decision: {decision.upper()}")
            
            if decision == "alert":
                print(f"    ✅ CORRECT - Detected as fraud")
            else:
                print(f"    ❌ FALSE NEGATIVE - Missed fraud!")
        
        results.append({
            'transaction_num': len(results)+1,
            'type': 'fraudulent',
            'amount': float(row.get('amt', 0)),
            'category': str(row.get('category', '')),
            'fraud_probability': prob,
            'decision': decision,
            'correct': decision == "alert" if 'error' not in result else False
        })
    
    print("\n" + "="*80)
    print("TESTING NON-FRAUDULENT TRANSACTIONS")
    print("="*80)
    
    for idx, row in non_fraud_df.iterrows():
        print(f"\n[{len(results)+1}] Non-Fraudulent Transaction:")
        print(f"    Amount: ${row.get('amt', 0):.2f}")
        print(f"    Category: {row.get('category', 'N/A')}")
        print(f"    Merchant: {row.get('merchant', 'N/A')}")
        print(f"    Location: {row.get('city', 'N/A')}, {row.get('state', 'N/A')}")
        
        txn_data = format_transaction_for_endpoint(row)
        result = invoke_endpoint(txn_data)
        
        if 'error' in result:
            print(f"    ❌ ERROR: {result['error']}")
            decision = "ERROR"
            prob = None
        else:
            prob = result.get('fraud_probability', 0)
            threshold = result.get('threshold', 0.25)
            decision = result.get('decision', 'unknown')
            
            print(f"    Fraud Probability: {prob:.6f}")
            print(f"    Threshold: {threshold:.4f}")
            print(f"    Decision: {decision.upper()}")
            
            if decision == "no_alert":
                print(f"    ✅ CORRECT - Not flagged as fraud")
            else:
                print(f"    ❌ FALSE POSITIVE - Incorrectly flagged as fraud!")
        
        results.append({
            'transaction_num': len(results)+1,
            'type': 'non_fraudulent',
            'amount': float(row.get('amt', 0)),
            'category': str(row.get('category', '')),
            'fraud_probability': prob,
            'decision': decision,
            'correct': decision == "no_alert" if 'error' not in result else False
        })
    
    return results

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    fraud_results = [r for r in results if r['type'] == 'fraudulent']
    non_fraud_results = [r for r in results if r['type'] == 'non_fraudulent']
    
    fraud_correct = sum(1 for r in fraud_results if r['correct'])
    non_fraud_correct = sum(1 for r in non_fraud_results if r['correct'])
    
    print(f"\nFraudulent Transactions ({len(fraud_results)}):")
    print(f"  ✅ Correctly detected: {fraud_correct}/{len(fraud_results)} ({fraud_correct/len(fraud_results)*100:.1f}%)")
    print(f"  ❌ Missed (False Negatives): {len(fraud_results) - fraud_correct}")
    
    print(f"\nNon-Fraudulent Transactions ({len(non_fraud_results)}):")
    print(f"  ✅ Correctly classified: {non_fraud_correct}/{len(non_fraud_results)} ({non_fraud_correct/len(non_fraud_results)*100:.1f}%)")
    print(f"  ❌ False Positives: {len(non_fraud_results) - non_fraud_correct}")
    
    total_correct = fraud_correct + non_fraud_correct
    total = len(results)
    print(f"\nOverall Accuracy: {total_correct}/{total} ({total_correct/total*100:.1f}%)")
    
    # Show probabilities
    print(f"\nFraud Probability Statistics:")
    fraud_probs = [r['fraud_probability'] for r in results if r['fraud_probability'] is not None and r['type'] == 'fraudulent']
    non_fraud_probs = [r['fraud_probability'] for r in results if r['fraud_probability'] is not None and r['type'] == 'non_fraudulent']
    
    if fraud_probs:
        print(f"  Fraudulent transactions - Avg: {sum(fraud_probs)/len(fraud_probs):.4f}, Min: {min(fraud_probs):.4f}, Max: {max(fraud_probs):.4f}")
    if non_fraud_probs:
        print(f"  Non-fraudulent transactions - Avg: {sum(non_fraud_probs)/len(non_fraud_probs):.4f}, Min: {min(non_fraud_probs):.4f}, Max: {max(non_fraud_probs):.4f}")

def main():
    print("🧪 SageMaker Endpoint Test Script")
    print(f"Endpoint: {ENDPOINT_NAME}")
    print(f"Region: {REGION}")
    print(f"Phone Number: {PHONE_NUMBER}\n")
    
    # Load transactions
    try:
        fraud_df, non_fraud_df = load_transactions(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ Error: {CSV_PATH} not found!")
        print(f"   Make sure you're running this from the infrastructure/ directory")
        return
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Test transactions
    results = test_transactions(fraud_df, non_fraud_df)
    
    # Print summary
    print_summary(results)
    
    print("\n✅ Testing complete!")

if __name__ == '__main__':
    main()

