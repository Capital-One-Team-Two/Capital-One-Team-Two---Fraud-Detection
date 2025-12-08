"""
SageMaker inference script for LightGBM fraud detection model.
This script works with SageMaker's SKLearn container without needing a custom Docker image.
"""

import json
import logging
import os
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global variables for model caching
_model = None
_threshold = 0.5
_feature_names = None


def model_fn(model_dir):
    """
    Load the model from the model directory.
    This function is called once when the SageMaker endpoint is started.
    
    Args:
        model_dir: Path to the directory containing model files (/opt/ml/model/)
        
    Returns:
        Loaded model artifact dict
    """
    global _model, _threshold, _feature_names
    
    logger.info(f"Loading model from {model_dir}")
    logger.info(f"Contents of model_dir: {os.listdir(model_dir) if os.path.exists(model_dir) else 'Directory does not exist'}")
    
    # SageMaker extracts model.tar.gz to /opt/ml/model/
    # Try native LightGBM format (.txt) first, then .pkl
    model_path = None
    metadata_path = None
    
    # Check for native format first (no numpy pickle issues!)
    native_files = ['fraud_lgbm_balanced.txt', 'model.txt']
    for filename in native_files:
        potential_path = os.path.join(model_dir, filename)
        if os.path.exists(potential_path):
            model_path = potential_path
            # Look for metadata JSON
            metadata_name = filename.replace('.txt', '_metadata.json')
            metadata_path = os.path.join(model_dir, metadata_name)
            if not os.path.exists(metadata_path):
                metadata_path = os.path.join(model_dir, 'fraud_lgbm_balanced_metadata.json')
            logger.info(f"Found native model at: {model_path}")
            break
    
    # Fall back to pickle format
    if not model_path:
        pkl_files = ['fraud_lgbm_balanced.pkl', 'model.pkl']
        for filename in pkl_files:
            potential_path = os.path.join(model_dir, filename)
            if os.path.exists(potential_path):
                model_path = potential_path
                logger.info(f"Found pickle model at: {model_path}")
                break
        
        # If not found, search for any .pkl file
        if not model_path:
            try:
                files = os.listdir(model_dir)
                pkl_files_list = [f for f in files if f.endswith('.pkl')]
                if pkl_files_list:
                    model_path = os.path.join(model_dir, pkl_files_list[0])
                    logger.info(f"Found model file: {model_path}")
            except Exception as e:
                logger.error(f"Error listing directory: {e}")
    
    if not model_path:
        raise ValueError(f"No model file found in {model_dir}. Available files: {os.listdir(model_dir) if os.path.exists(model_dir) else 'N/A'}")
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Check if this is a native LightGBM format (.txt file)
        if model_path.endswith('.txt'):
            logger.info("Loading native LightGBM format (no pickle dependencies)")
            import lightgbm as lgb
            
            # Load model from native format
            booster = lgb.Booster(model_file=model_path)
            
            # Create LGBMClassifier wrapper
            model_obj = lgb.LGBMClassifier()
            model_obj._Booster = booster
            model_obj.fitted_ = True
            model_obj._n_features = booster.num_feature()
            model_obj._n_classes = 2  # Binary classification
            
            # Load metadata from JSON if available
            if metadata_path and os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    _threshold = float(metadata.get('threshold', 0.5))
                    _feature_names = metadata.get('feature_names', [])
            else:
                # Use defaults if no metadata
                _threshold = 0.5
                _feature_names = []
                logger.warning("No metadata file found, using defaults")
            
            logger.info(f"Model loaded from native format. Features: {len(_feature_names)}, Threshold: {_threshold}")
            
            return {
                'model': model_obj,
                'threshold': _threshold,
                'feature_names': _feature_names
            }
        
        # Otherwise load from pickle format
        # Load the artifact (which contains model, threshold, and feature_names)
        with open(model_path, 'rb') as f:
            artifact = joblib.load(f)
        
        # Extract components
        if isinstance(artifact, dict):
            model_obj = artifact.get('model')
            
            # Check if this is a native format (has booster_data)
            if 'booster_data' in artifact and 'model_params' in artifact:
                logger.info("Detected native LightGBM format - reconstructing model...")
                import lightgbm as lgb
                import io
                
                # Reconstruct LGBMClassifier from native format
                booster_data = artifact['booster_data']
                params = artifact['model_params']
                
                # Load booster from native format string
                # The booster data might be bytes or string
                if isinstance(booster_data, bytes):
                    booster_str = booster_data.decode('utf-8')
                else:
                    booster_str = booster_data
                
                # Create Booster from native format string
                booster = lgb.Booster(model_str=booster_str)
                
                # Create LGBMClassifier with the booster
                model_obj = lgb.LGBMClassifier(**params)
                model_obj._Booster = booster
                model_obj._n_features = len(artifact.get('feature_names', []))
                
                # Mark the model as fitted
                # LightGBM uses fitted_ attribute (not _is_fitted)
                model_obj.fitted_ = True
                
                # Try to determine number of classes from booster (LightGBM 3.x uses private attribute)
                num_classes = 2  # Default to binary classification
                try:
                    # LightGBM 3.x uses _Booster__num_class private attribute
                    # Note: _Booster__num_class=1 means binary classification (2 classes)
                    if hasattr(booster, '_Booster__num_class'):
                        try:
                            num_classes_val = booster._Booster__num_class
                            # In LightGBM, num_class=1 means binary classification (2 classes)
                            num_classes = 2 if num_classes_val == 1 else num_classes_val
                            logger.info(f"Detected {num_classes} classes from _Booster__num_class (raw value: {num_classes_val})")
                        except Exception as e:
                            logger.warning(f"Could not read _Booster__num_class: {e}, defaulting to binary")
                            num_classes = 2
                    else:
                        # Fallback: assume binary classification
                        logger.info("Could not find _Booster__num_class attribute, defaulting to binary classification (2 classes)")
                        num_classes = 2
                except Exception as e:
                    logger.info(f"Error determining number of classes: {e}, using default binary classification (2 classes)")
                    num_classes = 2
                
                # Set _n_classes which is used by LightGBM internally
                # classes_ is a property that will be computed from _n_classes and _Booster
                try:
                    model_obj._n_classes = num_classes
                    logger.info(f"Model initialized with {num_classes} classes (_n_classes={num_classes})")
                except Exception as e:
                    logger.warning(f"Could not set _n_classes: {e}, but model should still work")
                
                logger.info("Model reconstructed from native format successfully")
            
            _model = model_obj
            _threshold = float(artifact.get('threshold', 0.5))
            _feature_names = artifact.get('feature_names', [])
        else:
            # If artifact is just the model
            _model = artifact
            _threshold = 0.5
            _feature_names = []
        
        # If model is LGBMClassifier and booster has issues, try to fix it
        if hasattr(_model, 'booster_'):
            try:
                booster = _model.booster_
                # Check if booster needs fixing
                if not hasattr(booster, 'handle') and hasattr(booster, 'model_from_string'):
                    logger.warning("Booster missing 'handle' attribute - attempting to fix...")
                    # Try to reload the booster from its string representation
                    try:
                        model_str = booster.model_to_string()
                        import lightgbm as lgb
                        new_booster = lgb.Booster(model_str=model_str)
                        _model._Booster = new_booster
                        logger.info("Booster reloaded successfully")
                    except Exception as fix_e:
                        logger.error(f"Could not fix booster: {fix_e}")
            except Exception as e:
                logger.warning(f"Could not check/fix booster: {e}")
        
        logger.info(f"Model loaded successfully. Model type: {type(_model)}, Features: {len(_feature_names)}, Threshold: {_threshold}")
        
        return {
            'model': _model,
            'threshold': _threshold,
            'feature_names': _feature_names
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    
    Args:
        request_body: The body of the request
        request_content_type: The content type of the request
        
    Returns:
        Deserialized input ready for prediction
    """
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    elif request_content_type == 'text/csv':
        # If CSV format, parse it
        lines = request_body.strip().split('\n')
        # Convert CSV to dict (assumes first line is header)
        header = lines[0].split(',')
        values = lines[1].split(',') if len(lines) > 1 else []
        input_data = dict(zip(header, values))
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_artifact):
    """
    Perform prediction on the deserialized input data.
    
    Args:
        input_data: The deserialized input data
        model_artifact: The loaded model artifact
        
    Returns:
        Prediction results (fraud probability and decision)
    """
    global _model, _threshold, _feature_names
    
    logger.info("Performing prediction...")
    
    # Ensure model is loaded
    if _model is None:
        _model = model_artifact['model']
        _threshold = model_artifact['threshold']
        _feature_names = model_artifact['feature_names']
    
    # Feature engineering (must match training preprocessing)
    df = pd.DataFrame([input_data])
    
    # Extract datetime features
    if 'trans_date_trans_time' in df.columns:
        dt_series = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    elif 'timestamp' in df.columns:
        ts = df['timestamp'].astype(str).str.replace('Z', '+00:00', regex=False)
        dt_series = pd.to_datetime(ts, errors='coerce')
    else:
        dt_series = pd.to_datetime(pd.Series([None]), errors='coerce')
    
    # Extract datetime features (handle NaN values)
    df['trans_hour'] = dt_series.dt.hour.fillna(0).astype(int)
    df['trans_day'] = dt_series.dt.day.fillna(1).astype(int)
    df['trans_month'] = dt_series.dt.month.fillna(1).astype(int)
    df['trans_year'] = dt_series.dt.year.fillna(2024).astype(int)
    
    # Drop non-feature columns (must match training)
    drop_cols = [
        'is_fraud', 'trans_date_trans_time', 'cc_num', 'trans_num', 'Unnamed: 0',
        'first', 'last', 'street', 'city', 'zip', 'merchant', 'job'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=False)
    
    # Sanitize column names (must match training)
    df = _sanitize_columns(df)
    
    # Align to training feature names
    df = _align_to_columns(df, _feature_names)
    
    # Ensure all values are numeric and finite
    try:
        df_numeric = df.astype(float)
        if not np.isfinite(df_numeric.values).all():
            logger.warning("Non-finite values detected, filling with 0")
            df_numeric = df_numeric.fillna(0).replace([np.inf, -np.inf], 0)
        df = df_numeric
    except Exception as e:
        logger.warning(f"Could not convert all columns to float: {e}, continuing anyway")
    
    # Get prediction
    try:
        proba = _model.predict_proba(df)[0, 1]
        decision = "alert" if proba >= _threshold else "no_alert"
        
        logger.info(f"Prediction: p_raw={proba:.6f}, threshold={_threshold}, decision={decision}")
        
        return {
            'fraud_probability': float(proba),
            'threshold': float(_threshold),
            'decision': decision
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result.
    
    Args:
        prediction: The prediction result
        response_content_type: The desired response content type
        
    Returns:
        Serialized prediction
    """
    logger.info(f"Serializing prediction with content type: {response_content_type}")
    
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    elif response_content_type == 'text/csv':
        # Return as CSV
        return f"{prediction['fraud_probability']},{prediction['decision']}"
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names JSON-safe for LightGBM and unique."""
    df = df.copy()
    cols = (
        df.columns.astype(str).str.strip()
          .str.replace(r'[\\"/\b\f\n\r\t]', '_', regex=True)
          .str.replace(r'[^0-9A-Za-z_]', '_', regex=True)
    )
    seen = {}
    safe = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            safe.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0
            safe.append(c)
    df.columns = safe
    return df


def _align_to_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Ensure df has exactly the specified columns (missing as 0, extras dropped)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        df = df.copy()
        for c in missing:
            df[c] = 0
    return df[cols]

