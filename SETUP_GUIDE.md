# Fraud Detection System Setup Guide

This guide will help you set up the complete fraud detection pipeline with local ML model and Twilio notifications.

## 🏗️ Architecture Overview

```
User Transaction → API Gateway → Transaction Lambda → DynamoDB
                                                      ↓
DynamoDB Stream → Score Lambda (Local Model) → Notify Lambda → Twilio → User
                                                      ↓
User Response → API Gateway → Webhook Lambda → DynamoDB
```

## 📋 Prerequisites

1. **AWS Account** with appropriate permissions
2. **Twilio Account** with SMS capabilities
3. **Python 3.9+** environment
4. **AWS CLI** configured
5. **Terraform** (optional, for infrastructure)

## 🚀 Step-by-Step Setup

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. AWS Configuration

Set up your AWS environment variables:

```bash
export AWS_REGION="us-east-2"
export SAGEMAKER_ROLE_ARN="arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
export S3_BUCKET_NAME="your-fraud-detection-models"
export TRANSACTIONS_TABLE="FraudDetection-Transactions"
export ACCOUNTS_TABLE="FraudDetection-Accounts"
```

### 3. Twilio Setup

1. **Create Twilio Account**: Sign up at [twilio.com](https://twilio.com)
2. **Get Credentials**:
   ```bash
   export TWILIO_ACCOUNT_SID="your_account_sid"
   export TWILIO_AUTH_TOKEN="your_auth_token"
   export TWILIO_FROM_NUMBER="+1234567890"  # Your Twilio phone number
   ```

3. **Configure Webhook URL**:
   - In Twilio Console, set webhook URL to: `https://your-api-gateway-url/webhook/twilio`

### 4. Train and Deploy ML Model

#### Option A: Use Existing Model
If you have a trained model file (`fraud_lgbm_balanced.pkl`):

```bash
# Deploy to SageMaker
cd infrastructure
python deploy_sagemaker.py
```

#### Option B: Train New Model
```bash
# Train the model
cd lambdas/score
python prepare_data.py

# Deploy to SageMaker
cd ../../infrastructure
python deploy_sagemaker.py
```

### 5. Deploy Lambda Functions

#### Using AWS CLI:

```bash
# Package and deploy each Lambda function
cd lambdas/transaction
zip -r transaction.zip .
aws lambda create-function \
    --function-name FraudDetection-Transaction \
    --runtime python3.9 \
    --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
    --handler transaction.lambda_handler \
    --zip-file fileb://transaction.zip

# Repeat for other functions:
# - lambdas/score/score_sagemaker.py → FraudDetection-Score
# - lambdas/notify/notify.py → FraudDetection-Notify
# - lambdas/webhook/twilio_webhook.py → FraudDetection-Webhook
# - lambdas/accounts/accounts.py → FraudDetection-Accounts
# - lambdas/user_response/user_response.py → FraudDetection-UserResponse
```

#### Using Terraform (Recommended):
```bash
# Create terraform configuration
terraform init
terraform plan
terraform apply
```

### 6. Set Up DynamoDB Tables

```bash
# Create Transactions table
aws dynamodb create-table \
    --table-name FraudDetection-Transactions \
    --attribute-definitions \
        AttributeName=transaction_id,AttributeType=S \
    --key-schema \
        AttributeName=transaction_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST

# Create Accounts table
aws dynamodb create-table \
    --table-name FraudDetection-Accounts \
    --attribute-definitions \
        AttributeName=user_id,AttributeType=S \
    --key-schema \
        AttributeName=user_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST

# Enable DynamoDB Stream on Transactions table
aws dynamodb update-table \
    --table-name FraudDetection-Transactions \
    --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES
```

### 7. Configure API Gateway

1. **Create REST API** in AWS Console
2. **Create Resources**:
   - `POST /transaction` → Transaction Lambda
   - `GET /transaction/{id}` → User Response Lambda
   - `POST /webhook/twilio` → Webhook Lambda
3. **Enable CORS** for all endpoints
4. **Deploy API** and note the endpoint URL

### 8. Set Up DynamoDB Stream Trigger

```bash
# Create event source mapping for Score Lambda
aws lambda create-event-source-mapping \
    --function-name FraudDetection-Score \
    --event-source-arn arn:aws:dynamodb:us-east-2:YOUR_ACCOUNT:table/FraudDetection-Transactions/stream/2024-01-01T00:00:00.000 \
    --starting-position LATEST
```

### 9. Seed Test Data

```bash
# Add sample accounts and transactions
cd infrastructure
python seed_db.py all
```

### 10. Test the System

#### Test Transaction Flow:
```bash
# Create a test transaction
curl -X POST https://your-api-gateway-url/transaction \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "amount": 1500.00,
    "merchant": "Suspicious Store",
    "location": "Unknown Location"
  }'
```

#### Check Transaction Status:
```bash
# Get transaction status
curl https://your-api-gateway-url/transaction/txn_123
```

## 🔧 Configuration Files

### Environment Variables for Lambda Functions:

**Score Lambda:**
```bash
SAGEMAKER_ENDPOINT_NAME=fraud-detection-endpoint
FRAUD_THRESHOLD=0.5
TRANSACTIONS_TABLE=FraudDetection-Transactions
NOTIFY_LAMBDA_NAME=FraudDetection-Notify
```

**Notify Lambda:**
```bash
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
```

**Webhook Lambda:**
```bash
TRANSACTIONS_TABLE=FraudDetection-Transactions
```

## 🧪 Testing

### 1. Test Fraud Detection
Create transactions with high amounts or suspicious merchants to trigger fraud alerts.

### 2. Test SMS Notifications
Check that SMS messages are sent when fraud is detected.

### 3. Test User Responses
Reply to SMS messages and verify responses are recorded in DynamoDB.

## 📊 Monitoring

### CloudWatch Logs
- Monitor Lambda function logs
- Set up alarms for errors
- Track performance metrics

### DynamoDB Metrics
- Monitor read/write capacity
- Track throttling events
- Monitor item count

### SageMaker Metrics
- Monitor endpoint health
- Track inference latency
- Monitor model performance

## 🚨 Troubleshooting

### Common Issues:

1. **SageMaker Endpoint Not Responding**
   - Check endpoint status in SageMaker console
   - Verify IAM permissions
   - Check CloudWatch logs

2. **Twilio SMS Not Sending**
   - Verify Twilio credentials
   - Check phone number format
   - Verify account balance

3. **DynamoDB Stream Not Triggering**
   - Check stream configuration
   - Verify Lambda permissions
   - Check event source mapping

4. **Feature Engineering Errors**
   - Ensure training and inference use same preprocessing
   - Check for missing columns
   - Verify data types

## 🔒 Security Considerations

1. **IAM Roles**: Use least privilege principle
2. **Environment Variables**: Store secrets in AWS Secrets Manager
3. **API Gateway**: Enable authentication if needed
4. **DynamoDB**: Enable encryption at rest
5. **SageMaker**: Use VPC endpoints for private access

## 📈 Scaling Considerations

1. **Lambda Concurrency**: Set appropriate limits
2. **DynamoDB Capacity**: Use auto-scaling
3. **SageMaker Endpoints**: Use multiple instances for high availability
4. **API Gateway**: Enable caching for better performance

## 🎯 Next Steps

1. **Model Improvement**: Implement model retraining pipeline
2. **Advanced Features**: Add email notifications, push notifications
3. **Analytics**: Add fraud detection analytics dashboard
4. **A/B Testing**: Implement model A/B testing
5. **Compliance**: Add audit logging and compliance features

## 📞 Support

For issues or questions:
1. Check CloudWatch logs
2. Review this setup guide
3. Check AWS documentation
4. Contact your system administrator
