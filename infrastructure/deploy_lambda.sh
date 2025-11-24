#!/bin/bash
# Script to deploy Lambda function code

set -e

LAMBDA_FUNCTION_NAME="${1:-FraudDetection-Score}"
LAMBDA_DIR="${2:-../lambdas/score}"
REGION="${3:-us-east-2}"

echo "🚀 Deploying Lambda function: $LAMBDA_FUNCTION_NAME"
echo "   Source directory: $LAMBDA_DIR"
echo "   Region: $REGION"
echo ""

# Change to Lambda directory
cd "$(dirname "$0")/$LAMBDA_DIR" || {
    echo "❌ Error: Could not find Lambda directory: $LAMBDA_DIR"
    exit 1
}

# Create temporary directory for packaging
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "📦 Packaging Lambda function..."

# Copy Lambda code files
cp *.py "$TEMP_DIR/" 2>/dev/null || echo "   Warning: No .py files found to copy"

# Check if function exists
if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$REGION" > /dev/null 2>&1; then
    echo "✅ Lambda function exists, updating code..."
    
    # Create zip file
    cd "$TEMP_DIR"
    zip -r function.zip . > /dev/null
    
    # Update function code
    aws lambda update-function-code \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --zip-file "fileb://function.zip" \
        --region "$REGION" \
        --output json > /dev/null
    
    echo "✅ Lambda function code updated successfully!"
else
    echo "❌ Error: Lambda function '$LAMBDA_FUNCTION_NAME' does not exist"
    echo "   Create it first using AWS Console or:"
    echo "   aws lambda create-function --function-name $LAMBDA_FUNCTION_NAME ..."
    exit 1
fi

# Wait for update to complete
echo "⏳ Waiting for update to complete..."
aws lambda wait function-updated \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --region "$REGION"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📝 Next steps:"
echo "   - Test the Lambda function"
echo "   - Check CloudWatch logs for '🎯 Using SageMaker endpoint'"
echo "   - Verify predictions are working"

