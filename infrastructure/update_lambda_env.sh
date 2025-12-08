#!/bin/bash
# Script to update Lambda function environment variables for SageMaker integration

LAMBDA_FUNCTION_NAME="${1:-FraudDetection-Score}"
SAGEMAKER_ENDPOINT_NAME="${2:-fraud-detection-endpoint}"
USE_SAGEMAKER="${3:-true}"

echo "🔧 Updating Lambda function: $LAMBDA_FUNCTION_NAME"
echo "   Endpoint: $SAGEMAKER_ENDPOINT_NAME"
echo "   Use SageMaker: $USE_SAGEMAKER"

# Get current environment variables
CURRENT_ENV_JSON=$(aws lambda get-function-configuration \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --query 'Environment.Variables' \
    --output json 2>/dev/null)

if [ -z "$CURRENT_ENV_JSON" ] || [ "$CURRENT_ENV_JSON" = "null" ]; then
    CURRENT_ENV_JSON="{}"
fi

# Use Python to merge environment variables (works without jq)
UPDATED_ENV=$(python3 << EOF
import json
import sys

current = json.loads('''$CURRENT_ENV_JSON''')
current['SAGEMAKER_ENDPOINT_NAME'] = '$SAGEMAKER_ENDPOINT_NAME'
current['USE_SAGEMAKER'] = '$USE_SAGEMAKER'

print(json.dumps(current))
EOF
)

# Update Lambda function
echo "   Updating environment variables..."
aws lambda update-function-configuration \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --environment "Variables=$UPDATED_ENV" \
    --output json > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Updated Lambda environment variables"
    echo ""
    echo "Verifying update..."
    aws lambda get-function-configuration \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --query 'Environment.Variables.{SAGEMAKER_ENDPOINT_NAME:SAGEMAKER_ENDPOINT_NAME,USE_SAGEMAKER:USE_SAGEMAKER}' \
        --output table 2>/dev/null || echo "   (Could not verify, but update completed)"
    echo ""
    echo "📝 Note: Make sure your Lambda execution role has permission to invoke the SageMaker endpoint:"
    echo "   Action: sagemaker-runtime:InvokeEndpoint"
    echo "   Resource: arn:aws:sagemaker:*:*:endpoint/$SAGEMAKER_ENDPOINT_NAME"
else
    echo "❌ Error: Failed to update Lambda environment variables"
    echo "   Make sure you have permissions to update the Lambda function"
    exit 1
fi

