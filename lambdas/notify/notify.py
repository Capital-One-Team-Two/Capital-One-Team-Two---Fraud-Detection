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

# Twilio configuration (optional for testing)
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM_NUMBER")

# Check if Twilio is configured
TWILIO_CONFIGURED = all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM])
if TWILIO_CONFIGURED:
    twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
else:
    logger.warning("Twilio not configured. Running in mock mode.")
    twilio_client = None

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
        
        # ⚠️ CRITICAL: Check if notification already sent to prevent duplicates
        tx_resp = tx_table.get_item(Key={"transaction_id": txn_id})
        if "Item" in tx_resp:
            existing_notification = tx_resp["Item"].get("notificationSent", False)
            if existing_notification:
                logger.info(f"⏭️  Notification already sent for transaction {txn_id}. Skipping duplicate.")
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "ok": True,
                        "message": "Notification already sent",
                        "skipped": True
                    })
                }
        
        # Clean up merchant name (remove common prefixes)
        merchant_clean = merchant
        if merchant_clean.startswith("fraud_"):
            merchant_clean = merchant_clean[6:]  # Remove "fraud_" prefix
        # Clean up other common prefixes
        merchant_clean = merchant_clean.replace("_", " ").strip()
        
        # Format merchant name (limit length but be smarter about it)
        if len(merchant_clean) > 20:
            merchant_clean = merchant_clean[:20].rsplit(" ", 1)[0]  # Cut at word boundary
        
        # Create well-formatted SMS message
        # Trial accounts: max 160 chars per segment
        risk_pct = int(p_raw * 100)
        body = (
            f"🚨 FRAUD ALERT 🚨\n"
            f"Amount: ${amount:.2f}\n"
            f"Merchant: {merchant_clean}\n"
            f"Risk: {risk_pct}%\n"
            f"Txn: {txn_id[-8:]}\n"
            f"Reply YES or NO"
        )
        
        # Ensure message is under 160 characters
        if len(body) > 160:
            # Shorter version if needed
            body = (
                f"🚨 FRAUD ALERT\n"
                f"${amount:.2f} at {merchant_clean[:15]}\n"
                f"Risk: {risk_pct}%\n"
                f"Reply YES/NO"
            )
        
        # Final safety check
        if len(body) > 160:
            body = f"FRAUD: ${amount:.2f} {merchant_clean[:12]}\nRisk: {risk_pct}%\nReply YES/NO"
        
        # Send SMS (mock mode if Twilio not configured)
        if TWILIO_CONFIGURED:
            logger.info(f"Sending SMS to {phone}")
            msg = twilio_client.messages.create(
                to=phone, 
                from_=TWILIO_FROM, 
                body=body
            )
            message_sid = msg.sid
        else:
            logger.info(f"📱 MOCK MODE: Would send SMS to {phone}")
            logger.info(f"   Message: {body[:100]}...")
            message_sid = "MOCK-" + txn_id
        
        # Update transaction record (with error handling)
        try:
            tx_table.update_item(
                Key={"transaction_id": txn_id},
                UpdateExpression="SET notificationSent = :ns, messageSid = :ms, notificationAt = :na",
                ExpressionAttributeValues={
                    ":ns": True,
                    ":ms": message_sid,
                    ":na": datetime.utcnow().isoformat()
                }
            )
            logger.info(f"✅ Successfully updated transaction record with notification status")
        except Exception as db_error:
            logger.warning(f"⚠️  Failed to update DynamoDB (but SMS was sent): {db_error}")
            # Don't fail the whole Lambda if DynamoDB update fails - SMS was already sent
        
        logger.info(f"✅ Successfully processed notification. Message SID: {message_sid}")
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "ok": True, 
                "messageSid": message_sid,
                "phone": phone
            })
        }
        
    except Exception as e:
        logger.exception(f"Error in notify lambda: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"})
        }
