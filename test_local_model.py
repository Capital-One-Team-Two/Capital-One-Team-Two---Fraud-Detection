#!/usr/bin/env python3
"""
Test script for local fraud detection model
"""

import sys
import os
sys.path.append('lambdas/score')

def test_model_loading():
    """Test if the local model loads correctly"""
    try:
        from score import _load_model_if_needed, _MODEL, _THRESHOLD, _FEATURE_NAMES
        
        print("🧪 Testing Local Model Loading...")
        print("=" * 50)
        
        # Load the model
        _load_model_if_needed()
        
        if _MODEL is not None:
            print("✅ Model loaded successfully!")
            print(f"   Model type: {type(_MODEL)}")
            print(f"   Threshold: {_THRESHOLD}")
            print(f"   Features: {len(_FEATURE_NAMES) if _FEATURE_NAMES else 'Unknown'}")
            return True
        else:
            print("❌ Model failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def test_score_lambda():
    """Test the score lambda function"""
    try:
        from score import handler
        
        print("\n🧪 Testing Score Lambda...")
        print("=" * 50)
        
        # Create a test event (simulating DynamoDB stream)
        test_event = {
            "Records": [
                {
                    "eventName": "INSERT",
                    "dynamodb": {
                        "NewImage": {
                            "transaction_id": {"S": "test_txn_123"},
                            "user_id": {"S": "user_001"},
                            "amount": {"N": "1500.00"},
                            "merchant": {"S": "Suspicious Store"},
                            "category": {"S": "misc_net"},
                            "trans_date_trans_time": {"S": "2024-10-25 15:30:00"},
                            "first": {"S": "John"},
                            "last": {"S": "Doe"},
                            "gender": {"S": "M"},
                            "street": {"S": "123 Main St"},
                            "city": {"S": "New York"},
                            "state": {"S": "NY"},
                            "zip": {"S": "10001"},
                            "lat": {"N": "40.7128"},
                            "long": {"N": "-74.0060"},
                            "city_pop": {"N": "8336817"},
                            "job": {"S": "Software Engineer"},
                            "dob": {"S": "1990-01-01"},
                            "trans_num": {"S": "abc123"},
                            "unix_time": {"N": "1698255000"},
                            "merch_lat": {"N": "40.7000"},
                            "merch_long": {"N": "-74.0000"},
                            "is_fraud": {"N": "0"}
                        }
                    }
                }
            ]
        }
        
        # Test the handler
        result = handler(test_event, None)
        
        print(f"✅ Score Lambda executed successfully!")
        print(f"   Processed: {result.get('processed', 0)} transactions")
        return True
        
    except Exception as e:
        print(f"❌ Error in Score Lambda: {str(e)}")
        return False

def test_notify_lambda():
    """Test the notify lambda function"""
    try:
        sys.path.append('lambdas/notify')
        from notify import lambda_handler
        
        print("\n🧪 Testing Notify Lambda...")
        print("=" * 50)
        
        # Create a test event
        test_event = {
            "transaction_id": "test_txn_123",
            "user_id": "user_001",
            "amount": 1500.00,
            "merchant": "Suspicious Store",
            "p_raw": 0.85,
            "decision": "alert"
        }
        
        # Test the handler
        result = lambda_handler(test_event, None)
        
        print(f"✅ Notify Lambda executed!")
        print(f"   Status Code: {result.get('statusCode', 'Unknown')}")
        print(f"   Response: {result.get('body', 'No body')}")
        return True
        
    except Exception as e:
        print(f"❌ Error in Notify Lambda: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Local Fraud Detection System")
    print("=" * 60)
    
    # Test 1: Model Loading
    model_success = test_model_loading()
    
    # Test 2: Score Lambda
    score_success = test_score_lambda()
    
    # Test 3: Notify Lambda
    notify_success = test_notify_lambda()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Model Loading:     {'✅ PASS' if model_success else '❌ FAIL'}")
    print(f"Score Lambda:      {'✅ PASS' if score_success else '❌ FAIL'}")
    print(f"Notify Lambda:     {'✅ PASS' if notify_success else '❌ FAIL'}")
    
    if all([model_success, score_success, notify_success]):
        print("\n🎉 All tests passed! Your local fraud detection system is ready!")
        print("\nNext steps:")
        print("1. Complete Twilio verification")
        print("2. Deploy Lambda functions to AWS")
        print("3. Set up DynamoDB tables")
        print("4. Test the complete flow")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
