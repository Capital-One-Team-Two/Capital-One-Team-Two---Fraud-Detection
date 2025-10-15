#aaushi
import os, json, boto3
from twilio.rest import Client
from datetime import datetime

dynamodb = boto3.resource("dynamodb")
tx_table = dynamodb.Table("FraudDetection-Transactions")
acct_table = dynamodb.Table("FraudDetection-Accounts")

TWILIO_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_FROM = os.environ["TWILIO_FROM_NUMBER"]
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

def lambda_handler(event, context):
    # if coming from API Gateway
    if "body" in event:
        event = json.loads(event["body"])

    txn_id = event["transaction_id"]
    acct_id = event["account_id"]

    # lookup phone
    resp = acct_table.get_item(Key={"account_id": acct_id})
    phone = resp.get("Item", {}).get("phone")
    if not phone:
        return {"ok": False, "error": "no phone found"}

    # make sms text
    body = f"Transaction {txn_id} for ${event['amount']} at {event['merchant']} looks suspicious. Reply YES if you made it."

    # send SMS
    msg = twilio_client.messages.create(to=phone, from_=TWILIO_FROM, body=body)

    # record status
    tx_table.update_item(
        Key={"transaction_id": txn_id},
        UpdateExpression="SET notificationSent=:t, messageSid=:m, notificationAt=:n",
        ExpressionAttributeValues={
            ":t": True,
            ":m": msg.sid,
            ":n": datetime.utcnow().isoformat()
        },
    )

    return {"ok": True, "messageSid": msg.sid}
