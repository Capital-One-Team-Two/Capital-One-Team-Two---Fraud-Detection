#!/usr/bin/env python3
"""
Test script for Twilio SMS functionality
"""

import os
import json
from pathlib import Path
from twilio.rest import Client

def load_env_file(env_path='.env'):
    """Load variables from .env file."""
    env_file = Path(env_path)
    
    if env_file.exists():
        print(f"📄 Loading environment variables from {env_path}...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ[key] = value
        return True
    return False

def test_twilio_sms():
    """Test sending SMS via Twilio"""
    
    # Try to load from .env file first
    env_loaded = False
    for env_path in ['.env', '../.env', os.path.join(os.path.dirname(__file__), '.env')]:
        if load_env_file(env_path):
            env_loaded = True
            break
    
    # Get credentials from environment
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    from_number = os.environ.get('TWILIO_FROM_NUMBER')
    to_number = os.environ.get('TEST_PHONE_NUMBER')
    
    # Check which variables are missing
    missing = []
    if not account_sid:
        missing.append("TWILIO_ACCOUNT_SID")
    if not auth_token:
        missing.append("TWILIO_AUTH_TOKEN")
    if not from_number:
        missing.append("TWILIO_FROM_NUMBER")
    if not to_number:
        missing.append("TEST_PHONE_NUMBER")
    
    if missing:
        print("❌ Missing environment variables!")
        print(f"   Missing: {', '.join(missing)}")
        print("")
        if not env_loaded:
            print("💡 Tip: The script tried to load from .env file but couldn't find it.")
            print("   Make sure your .env file is in the project root and contains:")
        else:
            print("💡 Tip: Add to your .env file:")
        print("   TWILIO_ACCOUNT_SID=AC...")
        print("   TWILIO_AUTH_TOKEN=...")
        print("   TWILIO_FROM_NUMBER=+1...")
        print("   TEST_PHONE_NUMBER=+1...  # Your personal phone number")
        print("")
        print("Or export them manually:")
        print("   export $(cat .env | grep -v '^#' | xargs)")
        return False
    
    try:
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Create test message (shortened for Twilio trial account limits)
        # Trial accounts: max 160 characters
        message_body = "FRAUD TEST: Twilio working! Reply YES."
        
        print(f"📱 Sending test SMS...")
        print(f"   From: {from_number}")
        print(f"   To: {to_number}")
        print(f"   Message: {message_body}")
        
        # Send SMS
        message = client.messages.create(
            to=to_number,
            from_=from_number,
            body=message_body
        )
        
        print(f"✅ SMS sent successfully!")
        print(f"   Message SID: {message.sid}")
        print(f"   Status: {message.status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error sending SMS: {str(e)}")
        return False

def test_notify_lambda():
    """Test the Notify Lambda function locally"""
    
    # Import the notify lambda
    import sys
    sys.path.append('lambdas/notify')
    from notify import lambda_handler
    
    # Create test event
    test_event = {
        "transaction_id": "test_txn_123",
        "user_id": "user_001",
        "amount": 1500.00,
        "merchant": "Test Suspicious Store",
        "p_raw": 0.85,
        "decision": "alert"
    }
    
    print("🧪 Testing Notify Lambda...")
    print(f"   Event: {json.dumps(test_event, indent=2)}")
    
    try:
        result = lambda_handler(test_event, None)
        print(f"✅ Notify Lambda result: {json.dumps(result, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Notify Lambda error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Twilio Integration\n")
    
    # Test 1: Direct Twilio SMS
    print("=" * 50)
    print("TEST 1: Direct Twilio SMS")
    print("=" * 50)
    sms_success = test_twilio_sms()
    
    print("\n" + "=" * 50)
    print("TEST 2: Notify Lambda Function")
    print("=" * 50)
    lambda_success = test_notify_lambda()
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Direct SMS: {'✅ PASS' if sms_success else '❌ FAIL'}")
    print(f"Notify Lambda: {'✅ PASS' if lambda_success else '❌ FAIL'}")
    
    if sms_success and lambda_success:
        print("\n🎉 All tests passed! Your Twilio integration is working!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
