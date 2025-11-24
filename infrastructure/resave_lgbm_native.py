#!/usr/bin/env python3
"""
Resave LightGBM model using native format for better cross-version compatibility.
This saves the booster in LightGBM's native format and reconstructs during load.
"""

import joblib
import numpy as np
import lightgbm as lgb
import sys
import tempfile
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def resave_model_native(input_path, output_path):
    """Resave model using LightGBM native format for compatibility"""
    
    print(f"Loading model from: {input_path}")
    print(f"Current LightGBM version: {lgb.__version__}")
    
    # Load the artifact
    artifact = joblib.load(input_path)
    model = artifact.get('model')
    
    print(f"\nModel loaded: {type(model)}")
    
    # Extract base model if it's CalibratedClassifierCV
    if 'CalibratedClassifierCV' in type(model).__name__:
        print(f"Extracting base estimator from CalibratedClassifierCV...")
        if hasattr(model, 'estimator'):
            model = model.estimator
            print(f"  Extracted: {type(model)}")
        artifact['model'] = model
    
    if 'LGBMClassifier' not in type(model).__name__:
        print(f"⚠️  Model is not LGBMClassifier, saving as-is")
        joblib.dump(artifact, output_path)
        return output_path
    
    print(f"\n🔧 Saving LightGBM model in native format for compatibility...")
    
    try:
        # Get model parameters
        params = model.get_params()
        print(f"  Model parameters extracted")
        
        # Save booster in native format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_booster_path = tmp_file.name
        
        model.booster_.save_model(tmp_booster_path)
        print(f"  Booster saved to native format: {tmp_booster_path}")
        
        # Read the native format file
        with open(tmp_booster_path, 'rb') as f:
            booster_data = f.read()
        
        # Clean up temp file
        os.unlink(tmp_booster_path)
        
        # Create a new artifact with the native format data
        new_artifact = {
            'model_params': params,
            'booster_data': booster_data,  # Native format string
            'threshold': artifact.get('threshold', 0.5),
            'feature_names': artifact.get('feature_names', [])
        }
        
        # Save the new artifact
        joblib.dump(new_artifact, output_path)
        print(f"✅ Model saved in native format!")
        print(f"   The inference script will reconstruct the model during load")
        
    except Exception as e:
        print(f"❌ Error saving in native format: {e}")
        import traceback
        traceback.print_exc()
        print(f"   Falling back to standard save...")
        joblib.dump(artifact, output_path)
    
    return output_path


if __name__ == '__main__':
    input_file = 'fraud_lgbm_balanced_compatible_v2.pkl'
    output_file = 'fraud_lgbm_balanced_native.pkl'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    resave_model_native(input_file, output_file)

