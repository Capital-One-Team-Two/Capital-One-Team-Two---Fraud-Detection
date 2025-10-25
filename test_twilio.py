#!/usr/bin/env python3
"""
Test script for Twilio SMS functionality
"""

import os
import json
from twilio.rest import Client

def test_twilio_sms():
    """Test sending SMS via Twilio"""
    
    # Get credentials from environment
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    from_number = os.environ.get('TWILIO_FROM_NUMBER')
    to_number = os.environ.get('TEST_PHONE_NUMBER')
    
    if not all([account_sid, auth_token, from_number, to_number]):
        print("❌ Missing environment variables!")
        print("Please set:")
        print("  TWILIO_ACCOUNT_SID")
        print("  TWILIO_AUTH_TOKEN") 
        print("  TWILIO_FROM_NUMBER")
        print("  TEST_PHONE_NUMBER")
        return False
    
    try:
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Create test message
        message_body = (
            "🧪 FRAUD DETECTION TEST 🧪\n\n"
            "This is a test message from your fraud detection system.\n"
            "If you receive this, Twilio is working correctly!\n\n"
            "Reply YES to confirm you received this message."
        )
        
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
