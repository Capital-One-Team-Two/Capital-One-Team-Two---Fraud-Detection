"""
Deploy LightGBM fraud detection model to SageMaker using SKLearn container.
This script deploys the model WITHOUT requiring a custom Docker image.

The SKLearn container will automatically install packages from requirements.txt
and use the custom inference.py script for handling predictions.
"""

import os
import sys
import json
import boto3
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker import get_execution_role, image_uris
from sagemaker.model import Model
from sagemaker.predictor import Predictor
import argparse
import tempfile
import shutil
import tarfile

# Add parent directory to path to import utils if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deploy_model(
    model_file_path: str,
    s3_bucket: str,
    role_arn: str = None,
    endpoint_name: str = "fraud-detection-endpoint",
    instance_type: str = "ml.t2.medium",
    region: str = "us-east-2"
):
    """
    Deploy LightGBM model to SageMaker using SKLearn container.
    
    This method uses SageMaker's SKLearn container with custom inference code.
    The container will automatically install LightGBM from requirements.txt.
    
    Args:
        model_file_path: Path to the model .pkl file
        s3_bucket: S3 bucket name for storing model artifacts
        role_arn: IAM role ARN for SageMaker (if None, will try to get execution role)
        endpoint_name: Name for the SageMaker endpoint
        instance_type: Instance type for the endpoint
        region: AWS region
    """
    print(f"🚀 Deploying LightGBM model to SageMaker...")
    print(f"   Model file: {model_file_path}")
    print(f"   S3 Bucket: {s3_bucket}")
    print(f"   Endpoint: {endpoint_name}")
    print(f"   Instance: {instance_type}")
    print(f"   Region: {region}")
    
    # Initialize SageMaker session
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Get or use provided IAM role
    if role_arn:
        role = role_arn
    else:
        try:
            role = get_execution_role()
            print(f"   Using execution role: {role}")
        except Exception as e:
            print(f"❌ Error getting execution role: {e}")
            print("   Please provide role_arn explicitly or set up SageMaker execution role")
            return None
    
    # Find inference script
    inference_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'lambdas', 'score', 'inference.py'
    )
    if not os.path.exists(inference_script):
        print(f"❌ Inference script not found at: {inference_script}")
        return None
    
    # Prepare model artifact and source code
    print(f"\n📦 Preparing model artifact and source code...")
    
    s3_client = boto3.client('s3', region_name=region)
    
    # Create a temporary directory for preparing artifacts
    tmpdir = tempfile.mkdtemp()
    try:
        # Step 1: Create model.tar.gz containing the model file(s)
        model_tar_path = os.path.join(tmpdir, 'model.tar.gz')
        
        # Check if we're using native format (.txt) or pickle (.pkl)
        model_dir = os.path.dirname(model_file_path) or '.'
        model_base = os.path.basename(model_file_path)
        
        with tarfile.open(model_tar_path, 'w:gz') as tar:
            # Add model file
            tar.add(model_file_path, arcname=model_base)
            
            # If using native format, also add metadata JSON
            if model_file_path.endswith('.txt'):
                metadata_file = model_file_path.replace('.txt', '_metadata.json')
                if os.path.exists(metadata_file):
                    tar.add(metadata_file, arcname=os.path.basename(metadata_file))
                    print(f"   ✅ Including metadata: {os.path.basename(metadata_file)}")
        
        # Upload model.tar.gz to S3
        model_s3_key = "models/fraud-detection/model.tar.gz"
        model_s3_path = f"s3://{s3_bucket}/{model_s3_key}"
        print(f"   Uploading model to {model_s3_path}...")
        s3_client.upload_file(model_tar_path, s3_bucket, model_s3_key)
        print(f"   ✅ Model uploaded")
        
        # Step 2: Prepare source code directory with inference.py and requirements.txt
        source_dir = os.path.join(tmpdir, 'source')
        os.makedirs(source_dir, exist_ok=True)
        
        # Copy inference script
        shutil.copy2(inference_script, os.path.join(source_dir, 'inference.py'))
        
        # Create requirements.txt with LightGBM and dependencies
        # Note: SKLearn container has numpy 1.x (Python 3.8 doesn't support numpy 2.0+)
        # We'll use the container's numpy version (1.x) - model needs to be saved with numpy 1.x
        requirements_path = os.path.join(source_dir, 'requirements.txt')
        with open(requirements_path, 'w') as f:
            # Don't upgrade numpy - use container's version (1.x for Python 3.8)
            # Only install LightGBM
            f.write("lightgbm==3.3.5\n")  # Use 3.3.5 which has pre-built wheels (no compilation needed)
            # Don't specify pandas, numpy, scikit-learn, joblib - container already has them
        
        print(f"   ✅ Source code prepared")
        
        # Step 3: Create SageMaker Model using Model class
        print(f"\n🔧 Creating SageMaker model...")
        
        # Get the SKLearn container image URI
        # Container uses Python 3.8 (py3) which has numpy 1.x
        container_image = image_uris.retrieve(
            framework='sklearn',
            region=region,
            version='1.2-1',  # Use newer version if available
            py_version='py3',  # Python 3.8 (numpy 1.x)
            instance_type=instance_type
        )
        print(f"   Using container: {container_image}")
        
        # Create Model with custom inference code and model data
        # The Model class accepts local source_dir and will upload it automatically
        sklearn_model = Model(
            image_uri=container_image,
            model_data=model_s3_path,  # Location of model.tar.gz
            role=role,
            entry_point='inference.py',
            source_dir=source_dir,  # Local directory - SageMaker will upload it
            sagemaker_session=sagemaker_session,
        )
    finally:
        # Clean up temp directory after model is created (but before deployment)
        # Note: We could keep it, but it's not needed after Model creation
        pass  # Keep it for now in case we need to reference it
    
    # Deploy the model (this creates an endpoint)
    print(f"\n🚀 Deploying endpoint...")
    print(f"   This may take 5-10 minutes...")
    print(f"   Note: The container will install LightGBM from requirements.txt automatically")
    
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        wait=True
    )
    
    print(f"\n✅ Model deployed successfully!")
    print(f"   Endpoint Name: {endpoint_name}")
    if hasattr(predictor, 'endpoint'):
        print(f"   Endpoint ARN: {predictor.endpoint}")
    else:
        print(f"   Endpoint is ready for use")
    
    # Test the endpoint with a sample prediction
    print(f"\n🧪 Testing endpoint...")
    test_data = {
        "amount": 100.0,
        "trans_date_trans_time": "2024-01-15 14:30:00",
        "category": "grocery",
        "city": "New York",
        "state": "NY",
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8175133,
        "job": "Engineer",
        "merch_lat": 40.7580,
        "merch_long": -73.9855,
    }
    
    try:
        if predictor is None:
            # Create predictor manually if deploy didn't return one
            from sagemaker.predictor import Predictor
            predictor = Predictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session)
        
        # Serialize test data to JSON string
        test_data_json = json.dumps(test_data)
        response = predictor.predict(test_data_json, initial_args={'ContentType': 'application/json'})
        
        # Parse response if it's a string
        if isinstance(response, (bytes, bytearray)):
            response = response.decode('utf-8')
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except:
                pass
        
        print(f"   ✅ Test prediction successful!")
        print(f"   Response: {json.dumps(response, indent=2)}")
    except Exception as e:
        print(f"   ⚠️ Test prediction failed: {e}")
        print(f"   Endpoint is deployed and InService - you can test it manually")
        print(f"   Use the endpoint name: {endpoint_name}")
        print(f"   Example: Use boto3 sagemaker-runtime client to invoke the endpoint")
    
    return predictor


def delete_endpoint(endpoint_name: str, region: str = "us-east-2"):
    """Delete a SageMaker endpoint."""
    print(f"🗑️  Deleting endpoint: {endpoint_name}")
    
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"   ✅ Endpoint deletion initiated")
        
        # Also delete endpoint config and model
        try:
            endpoint_config_name = sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )['EndpointConfigName']
            sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
            print(f"   ✅ Endpoint config deleted")
        except Exception as e:
            print(f"   ⚠️ Could not delete endpoint config: {e}")
            
    except Exception as e:
        print(f"   ❌ Error deleting endpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description='Deploy LightGBM model to SageMaker')
    parser.add_argument('--model-file', type=str, 
                       default='../infrastructure/fraud_lgbm_balanced.pkl',
                       help='Path to the model .pkl file')
    parser.add_argument('--s3-bucket', type=str, required=True,
                       help='S3 bucket name for storing model artifacts')
    parser.add_argument('--role-arn', type=str, default=None,
                       help='IAM role ARN for SageMaker (optional)')
    parser.add_argument('--endpoint-name', type=str, default='fraud-detection-endpoint-v2',
                       help='Name for the SageMaker endpoint (default: fraud-detection-endpoint-v2)')
    parser.add_argument('--instance-type', type=str, default='ml.t2.medium',
                       help='Instance type for the endpoint')
    parser.add_argument('--region', type=str, default='us-east-2',
                       help='AWS region')
    parser.add_argument('--delete', action='store_true',
                       help='Delete the endpoint instead of deploying')
    
    args = parser.parse_args()
    
    if args.delete:
        delete_endpoint(args.endpoint_name, args.region)
    else:
        # Resolve model file path
        model_file = os.path.abspath(args.model_file)
        if not os.path.exists(model_file):
            # Try relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_file = os.path.join(script_dir, 'fraud_lgbm_balanced.pkl')
            if not os.path.exists(model_file):
                print(f"❌ Model file not found: {args.model_file}")
                print(f"   Also tried: {model_file}")
                sys.exit(1)
        
        deploy_model(
            model_file_path=model_file,
            s3_bucket=args.s3_bucket,
            role_arn=args.role_arn,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            region=args.region
        )


if __name__ == '__main__':
    main()

