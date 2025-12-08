#!/usr/bin/env python3
"""
Test the full fraud detection pipeline:
1. Create user with phone number
2. Create high-risk transaction using real fraud data
3. Watch DynamoDB Stream trigger Score Lambda
4. Score Lambda calls SageMaker
5. If fraud detected, Notify Lambda sends SMS
"""
import boto3
import json
import uuid
import pandas as pd
import os
from datetime import datetime, timezone
from decimal import Decimal

# Configuration
REGION = "us-east-2"
TRANSACTIONS_TABLE = "FraudDetection-Transactions"
ACCOUNTS_TABLE = "FraudDetection-Accounts"
TEST_PHONE = "+16082177160"  # Your test phone number

# Find the CSV file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "fraudTest.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(SCRIPT_DIR, "fraudTrain.csv")

dynamodb = boto3.resource("dynamodb", region_name=REGION)
tx_table = dynamodb.Table(TRANSACTIONS_TABLE)
acct_table = dynamodb.Table(ACCOUNTS_TABLE)

def create_test_user(user_id="test_user", phone=TEST_PHONE):
    """Create or update test user with phone number."""
    print(f"👤 Creating/updating user: {user_id}")
    print(f"   Phone: {phone}")
    
    try:
        acct_table.put_item(
            Item={
                "user_id": user_id,
                "phone_number": phone,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )
        print(f"✅ User created/updated successfully")
        return user_id
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        return None

def load_fraud_transaction():
    """Load a real fraud transaction from the CSV file."""
    print(f"📚 Loading fraud transaction from: {CSV_PATH}")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
    # Get a fraud transaction
    fraud_df = df[df['is_fraud'] == 1]
    if len(fraud_df) == 0:
        raise ValueError("No fraud transactions found in CSV")
    
    # Get the first fraud transaction
    fraud_row = fraud_df.iloc[0]
    
    print(f"   Found fraud transaction: ${fraud_row['amt']:.2f} at {fraud_row['merchant']}")
    return fraud_row

def convert_csv_to_transaction(row, user_id="test_user"):
    """Convert CSV row to transaction dictionary with all required fields."""
    now = datetime.now(timezone.utc)
    
    transaction = {
        'transaction_id': f"test-{uuid.uuid4().hex[:8]}",
        'user_id': user_id,
        'account_id': user_id,
        'trans_date_trans_time': str(row.get('trans_date_trans_time', now.isoformat())),
        'cc_num': int(row.get('cc_num', 0)) if pd.notna(row.get('cc_num')) else 0,
        'merchant': str(row.get('merchant', 'Unknown')),
        'category': str(row.get('category', 'unknown')),
        'amt': Decimal(str(row.get('amt', 0.0))),  # Model expects 'amt'
        'amount': Decimal(str(row.get('amt', 0.0))),  # Keep 'amount' for display
        'first': str(row.get('first', 'Test')),
        'last': str(row.get('last', 'User')),
        'gender': str(row.get('gender', 'M')),
        'street': str(row.get('street', '123 Test St')),
        'city': str(row.get('city', 'Test City')),
        'state': str(row.get('state', 'CA')),
        'zip': str(int(row.get('zip', 12345))) if pd.notna(row.get('zip')) else '12345',
        'lat': Decimal(str(row.get('lat', 0.0))) if pd.notna(row.get('lat')) else Decimal('0.0'),
        'long': Decimal(str(row.get('long', 0.0))) if pd.notna(row.get('long')) else Decimal('0.0'),
        'city_pop': int(row.get('city_pop', 1000)) if pd.notna(row.get('city_pop')) else 1000,
        'job': str(row.get('job', 'Unknown')),
        'dob': str(row.get('dob', '1980-01-01')),
        'trans_num': str(row.get('trans_num', str(uuid.uuid4()))),
        'unix_time': int(row.get('unix_time', int(now.timestamp()))) if pd.notna(row.get('unix_time')) else int(now.timestamp()),
        'merch_lat': Decimal(str(row.get('merch_lat', 0.0))) if pd.notna(row.get('merch_lat')) else Decimal('0.0'),
        'merch_long': Decimal(str(row.get('merch_long', 0.0))) if pd.notna(row.get('merch_long')) else Decimal('0.0'),
        'timestamp': now.isoformat(),
    }
    
    return transaction

def create_high_risk_transaction(user_id="test_user"):
    """Create a high-risk transaction using real fraud data from CSV."""
    print(f"\n💳 Loading real fraud transaction...")
    
    try:
        # Load a real fraud transaction
        fraud_row = load_fraud_transaction()
        
        # Convert to transaction format
        transaction = convert_csv_to_transaction(fraud_row, user_id)
        txn_id = transaction['transaction_id']
        
        print(f"   Transaction ID: {txn_id}")
        print(f"   Amount: ${transaction['amount']:.2f}")
        print(f"   Merchant: {transaction['merchant']}")
        print(f"   Category: {transaction['category']}")
        
        # Convert to DynamoDB format (ensure all Decimals are properly handled)
        item = {}
        for key, value in transaction.items():
            if isinstance(value, Decimal):
                item[key] = value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                item[key] = Decimal(str(value))
            else:
                item[key] = value
        
        tx_table.put_item(Item=item)
        print(f"✅ Transaction created successfully")
        print(f"   This will trigger DynamoDB Stream → Score Lambda → SageMaker → Notify Lambda")
        return txn_id
    except Exception as e:
        print(f"❌ Error creating transaction: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_transaction_status(txn_id, max_wait=60):
    """Check if transaction was scored and SMS was sent."""
    import time
    
    print(f"\n⏳ Waiting for pipeline to process (max {max_wait}s)...")
    print(f"   (Checking every 2 seconds)")
    
    last_status = None
    for i in range(max_wait // 2):
        try:
            response = tx_table.get_item(Key={"transaction_id": txn_id})
            if "Item" in response:
                item = response["Item"]
                
                # Check if scored
                if "p_raw" in item:
                    p_raw = float(item.get("p_raw", 0))
                    decision = item.get("decision", "unknown")
                    
                    # Check for notificationSent - DynamoDB stores it as boolean or string
                    # Check all possible field names and formats
                    notification_sent = False
                    if "notificationSent" in item:
                        val = item["notificationSent"]
                        if isinstance(val, bool):
                            notification_sent = val
                        elif isinstance(val, str):
                            notification_sent = val.lower() in ["true", "1", "yes"]
                        else:
                            notification_sent = bool(val)
                    
                    # Also check for messageSid as an indicator
                    has_message_sid = "messageSid" in item and item.get("messageSid")
                    if has_message_sid and not notification_sent:
                        notification_sent = True  # If messageSid exists, notification was sent
                    
                    # Status message
                    status_msg = f"   [{i*2}s] Scored: p={p_raw:.4f}, decision={decision}"
                    if status_msg != last_status:
                        print(status_msg)
                        last_status = status_msg
                    
                    # Debug: Show what fields we found
                    if decision == "alert":
                        print(f"   Debug - Has notificationSent: {'notificationSent' in item}")
                        print(f"   Debug - Has messageSid: {has_message_sid}")
                        if "notificationSent" in item:
                            print(f"   Debug - notificationSent value: {item['notificationSent']} (type: {type(item['notificationSent']).__name__})")
                    
                    # If we have all info, show final results
                    if decision in ["alert", "no_alert"]:
                        print(f"\n✅ Transaction processed!")
                        print(f"   Fraud Probability: {p_raw:.4f} ({p_raw*100:.2f}%)")
                        print(f"   Decision: {decision}")
                        
                        # Determine if SMS was sent (use messageSid as authoritative check)
                        sms_actually_sent = notification_sent or has_message_sid
                        print(f"   SMS Sent: {'✅ Yes' if sms_actually_sent else '❌ No'}")
                        
                        if sms_actually_sent:
                            msg_sid = item.get('messageSid', 'N/A')
                            print(f"   Message SID: {msg_sid}")
                            print(f"\n📱 Check your phone ({TEST_PHONE}) for SMS!")
                        elif decision == "alert":
                            print(f"\n⚠️  Fraud detected but SMS not sent!")
                            print(f"   Check Notify Lambda logs for errors")
                        else:
                            print(f"\nℹ️  Transaction below fraud threshold - no SMS sent")
                        
                        return True
                else:
                    # Still waiting for scoring
                    if i % 5 == 0:  # Print every 10 seconds
                        print(f"   [{i*2}s] Waiting for scoring...")
            time.sleep(2)
        except Exception as e:
            print(f"   Error checking status: {e}")
            time.sleep(2)
    
    print(f"\n⚠️  Transaction not processed within {max_wait} seconds")
    print(f"   Check CloudWatch logs:")
    print(f"   - /aws/lambda/FraudDetection-Score")
    print(f"   - /aws/lambda/FraudDetection-Notify")
    return False

def main():
    print("=" * 60)
    print("🧪 FULL PIPELINE TEST")
    print("=" * 60)
    print(f"Test Phone: {TEST_PHONE}")
    print(f"Region: {REGION}")
    print("")
    
    # Step 1: Create test user
    print("STEP 1: Create Test User")
    print("-" * 60)
    user_id = create_test_user("test_user", TEST_PHONE)
    if not user_id:
        print("❌ Failed to create user. Exiting.")
        return
    
    # Step 2: Create high-risk transaction using real fraud data
    print("\nSTEP 2: Create High-Risk Transaction (from real fraud data)")
    print("-" * 60)
    txn_id = create_high_risk_transaction(user_id)
    if not txn_id:
        print("❌ Failed to create transaction. Exiting.")
        return
    
    # Step 3: Wait and check status
    print("\nSTEP 3: Monitor Pipeline")
    print("-" * 60)
    check_transaction_status(txn_id, max_wait=30)
    
    print("\n" + "=" * 60)
    print("✅ Test Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check your phone for SMS at: " + TEST_PHONE)
    print("2. Check CloudWatch logs:")
    print("   - /aws/lambda/FraudDetection-Score")
    print("   - /aws/lambda/FraudDetection-Notify")
    print("3. Check DynamoDB table for updated transaction")

if __name__ == "__main__":
    main()

