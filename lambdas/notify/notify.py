import os
import json
import boto3
import logging
from twilio.rest import Client
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
dynamodb = boto3.resource("dynamodb")
tx_table = dynamodb.Table("FraudDetection-Transactions")
acct_table = dynamodb.Table("FraudDetection-Accounts")

# Twilio configuration
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM_NUMBER")

if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM]):
    logger.error("Missing Twilio environment variables")
    raise ValueError("Missing Twilio configuration")

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

def lambda_handler(event, context):
    """
    Notify Lambda - sends SMS alerts for suspicious transactions
    Expected event format:
    {
        "transaction_id": "txn_123",
        "user_id": "user_001",  # Changed from account_id to user_id
        "amount": 1500.00,
        "merchant": "Suspicious Store",
        "p_raw": 0.85,
        "decision": "alert"
    }
    """
    try:
        logger.info(f"Notify Lambda invoked with event: {event}")
        
        # Parse event (handle both direct invocation and API Gateway)
        if "body" in event:
            event = json.loads(event["body"])
        
        # Extract required fields
        txn_id = event.get("transaction_id")
        user_id = event.get("user_id")  # Changed from account_id
        amount = event.get("amount", 0)
        merchant = event.get("merchant", "Unknown")
        p_raw = event.get("p_raw", 0)
        
        if not txn_id or not user_id:
            logger.error("Missing required fields: transaction_id or user_id")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing transaction_id or user_id"})
            }
        
        # Look up user phone number
        logger.info(f"Looking up phone for user_id: {user_id}")
        resp = acct_table.get_item(Key={"user_id": user_id})
        
        if "Item" not in resp:
            logger.error(f"User {user_id} not found in accounts table")
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "User not found"})
            }
        
        phone = resp["Item"].get("phone_number")  # Changed from "phone" to "phone_number"
        if not phone:
            logger.error(f"No phone number found for user {user_id}")
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "No phone number found for user"})
            }
        
        # Create SMS message
        body = (
            f"🚨 FRAUD ALERT 🚨\n\n"
            f"Transaction ID: {txn_id}\n"
            f"Amount: ${amount:.2f}\n"
            f"Merchant: {merchant}\n"
            f"Risk Score: {p_raw:.2f}\n\n"
            f"Reply YES if you made this transaction.\n"
            f"Reply NO to block this transaction."
        )
        
        # Send SMS
        logger.info(f"Sending SMS to {phone}")
        msg = twilio_client.messages.create(
            to=phone, 
            from_=TWILIO_FROM, 
            body=body
        )
        
        # Update transaction record
        tx_table.update_item(
            Key={"transaction_id": txn_id},
            UpdateExpression="SET notificationSent = :ns, messageSid = :ms, notificationAt = :na",
            ExpressionAttributeValues={
                ":ns": True,
                ":ms": msg.sid,
                ":na": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Successfully sent SMS. Message SID: {msg.sid}")
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "ok": True, 
                "messageSid": msg.sid,
                "phone": phone
            })
        }
        
    except Exception as e:
        logger.exception(f"Error in notify lambda: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"})
        }
