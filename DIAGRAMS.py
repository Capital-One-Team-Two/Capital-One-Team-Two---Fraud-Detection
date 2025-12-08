"""
Visual ASCII diagram of the fraud detection system flow
"""

SYSTEM_FLOW = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                   FRAUD DETECTION SYSTEM FLOW                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌──────────────┐
│  End User    │  Initiates credit card transaction
│  (Mobile/Web)│
└──────┬───────┘
       │ HTTPS Request
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY (REST API)                          │
│  Endpoints: POST /transaction, GET /transaction/{id}                    │
│  Features: Rate limiting, CORS, Request validation                      │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
      ┌───────────────┴────────────────┐
      │                                │
      ▼                                ▼
┌──────────────┐              ┌──────────────┐
│ TRANSACTION  │              │USER RESPONSE │
│   LAMBDA     │              │   LAMBDA     │
│              │              │              │
│ • Validate   │              │ • Query DB   │
│ • Store data │              │ • Return     │
│              │              │   status     │
└──────┬───────┘              └──────┬───────┘
       │                             │
       │ putItem                     │ getItem
       ▼                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      DYNAMODB - TRANSACTIONS TABLE                     │
│  Schema: transaction_id, user_id, amount, merchant, fraud_score, ...  │
│  Stream: ENABLED (NEW_AND_OLD_IMAGES)                                 │
└────────────────────┬───────────────────────────────────────────────────┘
                     │
                     │ DynamoDB Stream Event
                     ▼
           ┌──────────────────┐
           │  SCORE LAMBDA    │
           │                  │
           │ • Read stream    │
           │ • Prepare        │
           │   features       │
           └────────┬─────────┘
                    │
                    │ Invoke endpoint
                    ▼
           ┌──────────────────────────────┐
           │   SAGEMAKER ML MODEL         │
           │                              │
           │ Model: Random Forest         │
           │ Input: Transaction features  │
           │ Output: Fraud probability    │
           │ Threshold: 0.7               │
           └────────┬─────────────────────┘
                    │
                    │ Return fraud_score
                    ▼
           ┌──────────────────┐
           │  Update Score    │
           │  in DynamoDB     │
           └────────┬─────────┘
                    │
                    │ If fraud_score > 0.7
                    ▼
           ┌──────────────────┐
           │  NOTIFY LAMBDA   │
           │                  │
           │ • Detect fraud   │
           │ • Get phone #    │
           │ • Send SMS       │
           └────────┬─────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
  ┌─────────┐  ┌─────────┐  ┌──────────┐
  │ACCOUNTS │  │ TWILIO  │  │ END USER │
  │  TABLE  │  │   API   │  │  PHONE   │
  │         │  │         │  │          │
  │Get phone│  │Send SMS │  │Receives  │
  │ number  │  │ alert   │  │fraud     │
  │         │  │         │  │alert     │
  └─────────┘  └─────────┘  └──────────┘

═══════════════════════════════════════════════════════════════════════════

DETAILED COMPONENT INTERACTIONS:

1. TRANSACTION FLOW
   User → API Gateway → Transaction Lambda → DynamoDB
   
2. FRAUD SCORING FLOW
   DynamoDB Stream → Score Lambda → SageMaker → Update DynamoDB
   
3. NOTIFICATION FLOW
   DynamoDB Stream → Notify Lambda → Accounts Table → Twilio → User SMS
   
4. STATUS CHECK FLOW
   User → API Gateway → User Response Lambda → DynamoDB → User

═══════════════════════════════════════════════════════════════════════════

DATA FLOW EXAMPLE:

Transaction Input:
{
  "transaction_id": "txn_001",
  "user_id": "user_001",
  "amount": 2500.00,
  "merchant": "Unknown Vendor",
  "location": "Lagos, Nigeria"
}

↓ Stored in DynamoDB

↓ Processed by ML Model

Fraud Score Output: 0.92 (HIGH RISK)

↓ SMS Notification

"FRAUD ALERT: Suspicious transaction detected!
Amount: $2500.00
Merchant: Unknown Vendor
Transaction ID: txn_001
Risk Score: 0.92
If this wasn't you, reply BLOCK to freeze your card."

═══════════════════════════════════════════════════════════════════════════
"""

ARCHITECTURE_DIAGRAM = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                      SYSTEM ARCHITECTURE LAYERS                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────────────────┐
│ PRESENTATION LAYER                                                         │
│ • Mobile Apps   • Web Apps   • Third-party Integrations                   │
└───────────────────────────────────────────────────────────────────────────┘
                                    ↕ HTTPS
┌───────────────────────────────────────────────────────────────────────────┐
│ API LAYER                                                                  │
│ • API Gateway (REST)                                                       │
│ • Authentication & Authorization                                           │
│ • Rate Limiting & Throttling                                              │
└───────────────────────────────────────────────────────────────────────────┘
                                    ↕
┌───────────────────────────────────────────────────────────────────────────┐
│ BUSINESS LOGIC LAYER                                                       │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐             │
│ │Transaction │ │   Score    │ │  Notify    │ │  Response  │             │
│ │   Lambda   │ │   Lambda   │ │  Lambda    │ │   Lambda   │             │
│ └────────────┘ └────────────┘ └────────────┘ └────────────┘             │
└───────────────────────────────────────────────────────────────────────────┘
                                    ↕
┌───────────────────────────────────────────────────────────────────────────┐
│ DATA LAYER                                                                 │
│ ┌─────────────────────┐         ┌─────────────────────┐                  │
│ │   Transactions      │         │      Accounts       │                  │
│ │   DynamoDB Table    │         │   DynamoDB Table    │                  │
│ │   (with Streams)    │         │                     │                  │
│ └─────────────────────┘         └─────────────────────┘                  │
└───────────────────────────────────────────────────────────────────────────┘
                                    ↕
┌───────────────────────────────────────────────────────────────────────────┐
│ ML LAYER                                                                   │
│ • SageMaker Endpoint                                                       │
│ • Random Forest Model                                                      │
│ • Real-time Inference                                                      │
└───────────────────────────────────────────────────────────────────────────┘
                                    ↕
┌───────────────────────────────────────────────────────────────────────────┐
│ INTEGRATION LAYER                                                          │
│ • Twilio SMS API                                                           │
│ • Email Services (Future)                                                  │
│ • Push Notifications (Future)                                              │
└───────────────────────────────────────────────────────────────────────────┘
                                    ↕
┌───────────────────────────────────────────────────────────────────────────┐
│ MONITORING & LOGGING LAYER                                                 │
│ • CloudWatch Logs                                                          │
│ • CloudWatch Metrics                                                       │
│ • X-Ray Tracing                                                            │
│ • CloudTrail Audit Logs                                                    │
└───────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == '__main__':
    print(SYSTEM_FLOW)
    print("\n\n")
    print(ARCHITECTURE_DIAGRAM)
