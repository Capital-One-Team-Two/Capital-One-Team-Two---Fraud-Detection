#!/bin/bash
# Script to grant Lambda permission to invoke SageMaker endpoint

set -e

# Configuration
LAMBDA_FUNCTION_NAME="${1:-FraudDetection-Score}"
SAGEMAKER_ENDPOINT_NAME="${2:-fraud-detection-endpoint}"
REGION="${3:-us-east-2}"

echo "🔐 Granting Lambda permission to invoke SageMaker endpoint"
echo "   Lambda Function: $LAMBDA_FUNCTION_NAME"
echo "   SageMaker Endpoint: $SAGEMAKER_ENDPOINT_NAME"
echo "   Region: $REGION"
echo ""

# Step 1: Get Lambda execution role ARN
echo "📋 Step 1: Getting Lambda execution role..."
LAMBDA_ROLE_ARN=$(aws lambda get-function-configuration \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --region "$REGION" \
    --query 'Role' \
    --output text 2>/dev/null)

if [ -z "$LAMBDA_ROLE_ARN" ]; then
    echo "❌ Error: Could not find Lambda function: $LAMBDA_FUNCTION_NAME"
    echo "   Make sure the function name is correct and you have permissions."
    exit 1
fi

LAMBDA_ROLE_NAME=$(echo "$LAMBDA_ROLE_ARN" | awk -F'/' '{print $NF}')
echo "✅ Found Lambda role: $LAMBDA_ROLE_NAME"
echo "   Role ARN: $LAMBDA_ROLE_ARN"
echo ""

# Step 2: Get AWS Account ID from the role ARN
ACCOUNT_ID=$(echo "$LAMBDA_ROLE_ARN" | awk -F':' '{print $5}')
echo "📋 Step 2: Detected AWS Account ID: $ACCOUNT_ID"
echo ""

# Step 3: Create IAM policy document
echo "📋 Step 3: Creating IAM policy..."
POLICY_DOC=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker-runtime:InvokeEndpoint"
            ],
            "Resource": "arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:endpoint/${SAGEMAKER_ENDPOINT_NAME}"
        }
    ]
}
EOF
)

# Save policy to temp file
POLICY_FILE=$(mktemp)
echo "$POLICY_DOC" > "$POLICY_FILE"
echo "✅ Created policy document"
echo ""

# Step 4: Attach policy to Lambda role
echo "📋 Step 4: Attaching policy to Lambda role..."
POLICY_NAME="SageMakerInvokeEndpoint-${SAGEMAKER_ENDPOINT_NAME}"

aws iam put-role-policy \
    --role-name "$LAMBDA_ROLE_NAME" \
    --policy-name "$POLICY_NAME" \
    --policy-document "file://$POLICY_FILE" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Successfully attached policy: $POLICY_NAME"
else
    echo "❌ Error: Failed to attach policy"
    echo "   Make sure you have permissions to update IAM policies"
    rm -f "$POLICY_FILE"
    exit 1
fi

# Clean up
rm -f "$POLICY_FILE"

echo ""
echo "✅ Permission granted successfully!"
echo ""
echo "📝 Summary:"
echo "   - Lambda Function: $LAMBDA_FUNCTION_NAME"
echo "   - IAM Role: $LAMBDA_ROLE_NAME"
echo "   - Policy: $POLICY_NAME"
echo "   - Permission: sagemaker-runtime:InvokeEndpoint"
echo "   - Resource: arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:endpoint/${SAGEMAKER_ENDPOINT_NAME}"
echo ""
echo "🔍 Verify the policy:"
echo "   aws iam get-role-policy --role-name $LAMBDA_ROLE_NAME --policy-name $POLICY_NAME"
echo ""
echo "📝 Next step: Update Lambda environment variables:"
echo "   ./update_lambda_env.sh $LAMBDA_FUNCTION_NAME $SAGEMAKER_ENDPOINT_NAME true"


