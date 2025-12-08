import os
import json
import boto3
import logging
from urllib.parse import parse_qs
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
dynamodb = boto3.resource("dynamodb")
transactions_table = dynamodb.Table("FraudDetection-Transactions")

def lambda_handler(event, context):
    """
    Twilio Webhook Handler - processes SMS responses from users
    Expected Twilio webhook format:
    {
        "Body": "YES",  # or "NO"
        "From": "+15551234567",
        "To": "+15559876543",
        "MessageSid": "SM1234567890abcdef"
    }
    """
    try:
        logger.info(f"Twilio webhook received: {event}")
        
        # Parse the webhook data
        if event.get("isBase64Encoded"):
            # If using API Gateway with binary media types
            body = event.get("body", "")
        else:
            # Parse form data from Twilio
            body = event.get("body", "")
            if isinstance(body, str):
                # Parse URL-encoded form data
                parsed_data = parse_qs(body)
                sms_body = parsed_data.get("Body", [""])[0]
                from_number = parsed_data.get("From", [""])[0]
                to_number = parsed_data.get("To", [""])[0]
                message_sid = parsed_data.get("MessageSid", [""])[0]
            else:
                # Direct invocation
                sms_body = event.get("Body", "")
                from_number = event.get("From", "")
                to_number = event.get("To", "")
                message_sid = event.get("MessageSid", "")
        
        logger.info(f"Parsed SMS: Body='{sms_body}', From='{from_number}', MessageSid='{message_sid}'")
        
        # Find transaction by message SID
        if not message_sid:
            logger.error("No MessageSid found in webhook")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No MessageSid provided"})
            }
        
        # Scan transactions table for the message SID
        response = transactions_table.scan(
            FilterExpression="messageSid = :msg_sid",
            ExpressionAttributeValues={":msg_sid": message_sid}
        )
        
        if not response.get("Items"):
            logger.error(f"No transaction found with messageSid: {message_sid}")
            return {
                "statusCode": 404,
                "body": json.dumps({"error": "Transaction not found"})
            }
        
        transaction = response["Items"][0]
        transaction_id = transaction["transaction_id"]
        
        # Parse user response
        response_text = sms_body.upper().strip()
        user_confirmed = response_text in ["YES", "Y", "CONFIRM", "CONFIRMED"]
        user_denied = response_text in ["NO", "N", "DENY", "DENIED", "BLOCK"]
        
        if not (user_confirmed or user_denied):
            logger.warning(f"Unrecognized response: '{sms_body}'")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "Response recorded but not recognized"})
            }
        
        # Update transaction with user response
        user_decision = "confirmed" if user_confirmed else "denied"
        
        transactions_table.update_item(
            Key={"transaction_id": transaction_id},
            UpdateExpression="SET userResponse = :ur, userResponseAt = :ura, userDecision = :ud",
            ExpressionAttributeValues={
                ":ur": sms_body,
                ":ura": datetime.utcnow().isoformat(),
                ":ud": user_decision
            }
        )
        
        logger.info(f"Updated transaction {transaction_id} with user response: {user_decision}")
        
        # Send confirmation SMS back to user
        if user_confirmed:
            confirmation_msg = f"✅ Transaction {transaction_id} confirmed. Thank you!"
        else:
            confirmation_msg = f"🚫 Transaction {transaction_id} blocked. Your card has been secured."
        
        # Note: In a real implementation, you'd send this via Twilio
        # For now, we'll just log it
        logger.info(f"Confirmation message for {from_number}: {confirmation_msg}")
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Response processed successfully",
                "transaction_id": transaction_id,
                "user_decision": user_decision
            })
        }
        
    except Exception as e:
        logger.exception(f"Error processing Twilio webhook: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"})
        }

def get_cors_headers():
    """Return CORS headers for API Gateway"""
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST,OPTIONS'
    }
