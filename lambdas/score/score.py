import os
import io
import json
import logging
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.types import TypeDeserializer
from decimal import Decimal
from boto3.dynamodb.types import TypeSerializer
from types import SimpleNamespace

# Conditional imports for local model (only needed if not using SageMaker)
# These will be imported lazily when needed
_np = None
_pd = None
_joblib = None
_lgb = None

def _import_ml_libraries():
    """Lazy import of ML libraries - only needed for local model."""
    global _np, _pd, _joblib, _lgb
    if _np is None:
        import numpy as _np
        import pandas as _pd
        import joblib as _joblib
        import lightgbm as _lgb
    return _np, _pd, _joblib, _lgb
# ----------------- Logging -----------------
log = logging.getLogger()
log.setLevel(logging.INFO)
# Ensure logs show locally (Lambda uses CloudWatch automatically)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ----------------- Config -----------------
REGION = os.environ.get("AWS_REGION", "us-east-2")
TRANSACTIONS_TABLE = os.environ.get("TRANSACTIONS_TABLE", "FraudDetection-Transactions")

# SageMaker configuration (optional - if set, will use SageMaker endpoint instead of local model)
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME")  # e.g., "fraud-detection-endpoint"
USE_SAGEMAKER = os.environ.get("USE_SAGEMAKER", "false").lower() == "true"

# Local model configuration (used if SageMaker not configured)
MODEL_BUCKET = os.environ.get("MODEL_BUCKET")  # if provided, load from S3
MODEL_KEY = os.environ.get("MODEL_KEY", "fraud_lgbm_balanced.pkl")
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", f"./{MODEL_KEY}")  # used if no bucket

ENV_FRAUD_THRESHOLD = os.environ.get("FRAUD_THRESHOLD")  # optional override of saved threshold
MODEL_VERSION = os.environ.get("MODEL_VERSION", "lgbm-v1")
NOTIFY_LAMBDA_NAME = os.environ.get("NOTIFY_LAMBDA_NAME", "FraudDetection-Notify")

# ----------------- AWS Clients -----------------
dynamodb = boto3.client("dynamodb", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)

def get_sagemaker_runtime():
    """Get or create SageMaker runtime client."""
    if SAGEMAKER_ENDPOINT_NAME or USE_SAGEMAKER:
        return boto3.client("sagemaker-runtime", region_name=REGION)
    return None

deser = TypeDeserializer()

# ----------------- Model Cache -----------------
_MODEL = None
_THRESHOLD = 0.5
_FEATURE_NAMES = None

# ----------------- Utilities -----------------
def to_dynamodb_compatible(obj):
    """Recursively convert floats to Decimal for DynamoDB serialization."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: to_dynamodb_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dynamodb_compatible(v) for v in obj]
    return obj

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _from_stream_image(image: dict) -> dict:
    """Convert DynamoDB Stream image to plain Python types."""
    return {k: deser.deserialize(v) for k, v in image.items()}

# ---------- DynamoDB key handling ----------
_TABLE_KEY_SCHEMA = None
def _get_table_key_schema():
    global _TABLE_KEY_SCHEMA
    if _TABLE_KEY_SCHEMA is None:
        desc = dynamodb.describe_table(TableName=TRANSACTIONS_TABLE)
        _TABLE_KEY_SCHEMA = desc["Table"]["KeySchema"]  # list of {AttributeName, KeyType}
        log.info("Loaded KeySchema: %s", _TABLE_KEY_SCHEMA)
    return _TABLE_KEY_SCHEMA

def _looks_numeric(val: str) -> bool:
    try:
        float(str(val))
        return True
    except Exception:
        return False

def _build_key_from_txn(txn: dict) -> dict:
    """
    Build the DynamoDB Key dict based on the table's KeySchema and values in the transaction.
    Supports HASH-only or HASH+RANGE tables.
    """
    key = {}
    for ks in _get_table_key_schema():
        name = ks["AttributeName"]
        if name not in txn:
            raise KeyError(f"Missing key attribute '{name}' in transaction: present={list(txn.keys())}")
        val = txn[name]
        if _looks_numeric(val):
            key[name] = {"N": str(val)}
        else:
            key[name] = {"S": str(val)}
    return key

# ---------- Feature engineering (must match training) ----------
def _sanitize_columns(df):
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

def _align_to_columns(df, cols: list):
    """Ensure df has exactly the specified columns (missing as 0, extras dropped)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        df = df.copy()
        for c in missing:
            df[c] = 0
    return df[cols]

def _decimal_to_native(x):
    """Convert Decimal to float for ML features."""
    if isinstance(x, Decimal):
        return float(x)
    return x
def _feature_engineer_single(txn: dict):
    """
    Apply the SAME preprocessing as training:
      - Convert Decimals to floats
      - Datetime → hour/day/month/year
      - Drop ID/leaky fields
      - One-hot encode categoricals
      - Sanitize column names
      - Align to training feature_names
    """
    # Lazy import - only needed for local model
    _, pd, _, _ = _import_ml_libraries()
    
    # Convert Decimals first
    txn = {k: _decimal_to_native(v) for k, v in txn.items()}
    df = pd.DataFrame([txn])

    # ---------- Safe datetime extraction ----------
    dt_series = None

    if "trans_date_trans_time" in df.columns:
        dt_series = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")

    if dt_series is None or dt_series.isna().all():
        if "timestamp" in df.columns:
            ts = df["timestamp"].astype(str).str.replace("Z", "+00:00", regex=False)
            dt_series = pd.to_datetime(ts, errors="coerce")

    if dt_series is None:
        dt_series = pd.to_datetime(pd.Series([None]), errors="coerce")

    df["trans_hour"]  = dt_series.dt.hour
    df["trans_day"]   = dt_series.dt.day
    df["trans_month"] = dt_series.dt.month
    df["trans_year"]  = dt_series.dt.year

    # ---------- Drop leaky/non-feature columns ----------
    drop_cols = [
        'is_fraud', 'trans_date_trans_time', 'cc_num', 'trans_num', 'Unnamed: 0',
        'first', 'last', 'street', 'city', 'zip', 'merchant', 'job'
        # include or exclude 'lat', 'long' to match training
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # ---------- One-hot encode categoricals ----------
    df = pd.get_dummies(df, drop_first=False)

    # ---------- Sanitize and align ----------
    df = _sanitize_columns(df)
    df = _align_to_columns(df, _FEATURE_NAMES)

    # ---------- Safety check ----------
    np, _, _, _ = _import_ml_libraries()
    arr = df.to_numpy()
    # Convert to float to handle mixed data types
    try:
        arr_float = arr.astype(float)
        if not np.isfinite(arr_float).all():
            raise ValueError("Non-finite values detected after feature engineering.")
    except (ValueError, TypeError):
        # If conversion fails, just log a warning and continue
        log.warning("Could not convert all values to float for finite check, continuing...")

    return df
# ---------- SageMaker prediction ----------
def _predict_with_sagemaker(txn: dict) -> tuple[float, str]:
    """
    Call SageMaker endpoint for prediction.
    
    Args:
        txn: Transaction dictionary (raw, before feature engineering)
        
    Returns:
        Tuple of (fraud_probability, decision)
    """
    endpoint_name = SAGEMAKER_ENDPOINT_NAME
    if not endpoint_name:
        raise ValueError("SAGEMAKER_ENDPOINT_NAME not set, cannot use SageMaker")
    
    # Convert Decimal to native types for JSON serialization
    # DynamoDB returns Decimal types which aren't JSON serializable
    def convert_decimals(obj):
        """Recursively convert Decimal to float/int for JSON serialization."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_decimals(item) for item in obj]
        return obj
    
    # Prepare request payload - send raw transaction dict
    # SageMaker inference.py will do feature engineering
    txn_serializable = convert_decimals(txn)
    payload = json.dumps(txn_serializable)
    
    sagemaker_client = get_sagemaker_runtime()
    if not sagemaker_client:
        raise ValueError("SageMaker runtime client not available")
    
    try:
        log.info("🔹 Calling SageMaker endpoint: %s", endpoint_name)
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result_body = response['Body'].read().decode('utf-8')
        result = json.loads(result_body)
        
        fraud_prob = float(result.get('fraud_probability', 0.0))
        threshold = float(result.get('threshold', 0.5))
        decision = result.get('decision', 'no_alert')
        
        # Use threshold from response (SageMaker has the correct threshold)
        global _THRESHOLD
        _THRESHOLD = threshold
        
        log.info("🔹 SageMaker prediction: p_raw=%.6f thr=%.4f => decision=%s", 
                 fraud_prob, threshold, decision)
        
        return fraud_prob, decision
        
    except Exception as e:
        log.exception("❌ SageMaker invocation failed: %s", e)
        raise

# ---------- Model loading (for local model) ----------
def _load_model_if_needed():
    """Load LightGBM artifact once per container (from S3 or local layer)."""
    global _MODEL, _THRESHOLD, _FEATURE_NAMES
    
    # Skip loading if using SageMaker
    if USE_SAGEMAKER or SAGEMAKER_ENDPOINT_NAME:
        log.info("Using SageMaker endpoint, skipping local model load")
        return
    
    # Lazy import - only needed for local model
    _, _, joblib, _ = _import_ml_libraries()
    
    if _MODEL is not None:
        return

    if MODEL_BUCKET:
        log.info("Loading model from s3://%s/%s ...", MODEL_BUCKET, MODEL_KEY)
        obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
        byts = obj["Body"].read()
        artifact = joblib.load(io.BytesIO(byts))
    else:
        log.info("Loading model from local path: %s", MODEL_LOCAL_PATH)
        with open(MODEL_LOCAL_PATH, "rb") as f:
            artifact = joblib.load(f)

    _MODEL = artifact["model"]
    _FEATURE_NAMES = artifact["feature_names"]
    saved_thr = float(artifact.get("threshold", 0.5))
    _THRESHOLD = float(ENV_FRAUD_THRESHOLD) if ENV_FRAUD_THRESHOLD is not None else saved_thr

    log.info("✅ Model loaded. features=%d threshold=%.4f version=%s",
             len(_FEATURE_NAMES), _THRESHOLD, MODEL_VERSION)

# ---------- Update DynamoDB & confirm ----------
def update_txn_scores(txn: dict, p_raw: float, decision: str, model_version: str):
    """
    Update the item and return the ALL_NEW record for confirmation logs.
    IMPORTANT: This UPDATES an existing item - it does NOT create new items.
    """
    try:
        key = _build_key_from_txn(txn)
        log.info("🔑 Building update key: %s", key)
        log.info("🔑 Transaction has fields: %s", list(txn.keys())[:10])  # Log first 10 fields
        
        resp = dynamodb.update_item(
            TableName=TRANSACTIONS_TABLE,
            Key=key,
            UpdateExpression="SET p_raw = :p, decision = :d, scored_at = :t, model_version = :m",
            ExpressionAttributeValues={
                ":p": {"N": f"{p_raw:.6f}"},
                ":d": {"S": decision},
                ":t": {"S": _iso_now()},
                ":m": {"S": model_version},
            },
            ReturnValues="ALL_NEW",
            # Add condition to ensure item exists - prevents accidental creation behavior
            ConditionExpression="attribute_exists(transaction_id)",
        )
        updated_item = resp.get("Attributes")
        log.info("✅ Successfully updated existing item with key: %s", key)
        return updated_item
    except dynamodb.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'ConditionalCheckFailedException':
            log.error("❌ Item does not exist with key: %s. Cannot update non-existent item.", key)
            log.error("   This means the key doesn't match any existing transaction.")
            log.error("   Transaction fields: %s", list(txn.keys())[:10])
            raise ValueError(f"Transaction with key {key} does not exist - cannot update") from e
        else:
            log.exception("❌ DynamoDB update failed with error: %s", e)
            raise

# ---------- Trigger Notify Lambda ----------
def _trigger_notify_lambda(txn: dict, p_raw: float, decision: str):
    """Trigger the Notify Lambda for fraud alerts."""
    if decision != "alert":
        return
    
    try:
        # Prepare notification payload
        # Handle both "amount" and "amt" field names (amt is used in DynamoDB)
        amount = txn.get("amount") or txn.get("amt") or 0
        notify_payload = {
            "transaction_id": txn.get("transaction_id"),
            "user_id": txn.get("user_id"),  # Make sure this field exists in your transaction
            "amount": float(amount),
            "merchant": txn.get("merchant", "Unknown"),
            "p_raw": p_raw,
            "decision": decision
        }
        
        log.info("Triggering Notify Lambda for fraud alert: %s", notify_payload)
        
        # Invoke Notify Lambda asynchronously
        response = lambda_client.invoke(
            FunctionName=NOTIFY_LAMBDA_NAME,
            InvocationType='Event',  # Asynchronous invocation
            Payload=json.dumps(notify_payload)
        )
        
        log.info("Notify Lambda triggered successfully. StatusCode: %s", response['StatusCode'])
        
    except Exception as e:
        log.exception("Failed to trigger Notify Lambda: %s", e)

# ----------------- Lambda Handler -----------------
def handler(event, context):
    log.info("Lambda invoked. Region=%s Table=%s ModelVersion=%s", REGION, TRANSACTIONS_TABLE, MODEL_VERSION)

    processed = 0
    _get_table_key_schema()       # confirm key schema up front
    
    # Log which mode we're using
    if USE_SAGEMAKER or SAGEMAKER_ENDPOINT_NAME:
        log.info("🎯 Using SageMaker endpoint: %s", SAGEMAKER_ENDPOINT_NAME or "not set")
    else:
        log.info("🎯 Using local model (loading if needed)")
        _load_model_if_needed()   # warm model cache
    
    log.info("Starting to process %d record(s)...", len(event.get("Records", [])))

    for i, rec in enumerate(event.get("Records", []), start=1):
        log.info("---- Record %d/%d ----", i, len(event["Records"]))
        ev = rec.get("eventName")
        if ev not in ("INSERT", "MODIFY"):   # add MODIFY if you want rescoring on updates
            log.info("Skipping eventName=%s (only handling INSERT/MODIFY).", ev)
            continue

        new_img = rec.get("dynamodb", {}).get("NewImage")
        if not new_img:
            log.warning("No NewImage found; skipping.")
            continue

        txn = _from_stream_image(new_img)

        # Log key fields for traceability
        try:
            key_fields = {ks['AttributeName']: txn.get(ks['AttributeName']) for ks in _TABLE_KEY_SCHEMA}
            log.info("Scoring item keys: %s", key_fields)
        except Exception:
            log.info("Scoring item (unable to log keys; missing?): %s", list(txn.keys()))

        # ⚠️ CRITICAL: Skip items that already have scores to prevent recursive loops
        # When we update DynamoDB with p_raw, it triggers a MODIFY stream event
        # Without this check, we'd score it again → update → score again → infinite loop!
        # Check both the dict keys and values (p_raw could be stored in different formats)
        has_score = ("p_raw" in txn) or (txn.get("p_raw") is not None)
        if has_score:
            log.info("⏭️  Skipping item - already scored (has p_raw). This prevents recursive scoring loops.")
            log.info("   Event: %s, Transaction ID: %s, p_raw value: %s", ev, txn.get("transaction_id", "unknown"), txn.get("p_raw"))
            log.info("   To rescore, remove p_raw field from the transaction first.")
            continue

        try:
            # Prediction (either SageMaker or local model)
            if USE_SAGEMAKER or SAGEMAKER_ENDPOINT_NAME:
                # Use SageMaker endpoint (does feature engineering in SageMaker)
                log.info("🔹 Using SageMaker endpoint for prediction")
                p_raw, decision = _predict_with_sagemaker(txn)
            else:
                # Use local model (do feature engineering here)
                log.info("🔹 Feature engineering start")
                X = _feature_engineer_single(txn)
                log.info("🔹 Feature engineering done (%d features)", X.shape[1])
                
                p_raw = float(_MODEL.predict_proba(X)[0, 1])
                decision = "alert" if p_raw >= _THRESHOLD else "no_alert"
                log.info("🔹 Prediction p_raw=%.6f thr=%.4f => decision=%s", p_raw, _THRESHOLD, decision)

            # Update DynamoDB and confirm
            updated = update_txn_scores(txn, p_raw, decision, MODEL_VERSION)
            log.info("✅ DynamoDB updated (ALL_NEW): %s", updated)
            
            # Trigger notification if fraud detected
            _trigger_notify_lambda(txn, p_raw, decision)
            
            processed += 1

        except Exception as e:
            log.exception("❌ Failed to process item: %s", e)

    log.info("✅ Handler finished. processed=%d", processed)
    return {"processed": processed}
def score_entire_table(batch_size: int = 100):
    """
    Loop through all transactions in the table, score each one, and update it in place.
    """
    log.info("Starting full-table scoring for table: %s", TRANSACTIONS_TABLE)
    
    # Load model if not using SageMaker
    if not (USE_SAGEMAKER or SAGEMAKER_ENDPOINT_NAME):
        _load_model_if_needed()
    
    _get_table_key_schema()

    last_evaluated_key = None
    total_processed = 0

    while True:
        # 1. Scan a batch of items
        scan_kwargs = {"TableName": TRANSACTIONS_TABLE, "Limit": batch_size}
        if last_evaluated_key:
            scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

        resp = dynamodb.scan(**scan_kwargs)
        items = resp.get("Items", [])
        log.info("Fetched %d items in this batch", len(items))

        if not items:
            break

        # 2. Score and update each item
        for item in items:
            txn = _from_stream_image(item)
            try:
                if USE_SAGEMAKER or SAGEMAKER_ENDPOINT_NAME:
                    p_raw, decision = _predict_with_sagemaker(txn)
                else:
                    X = _feature_engineer_single(txn)
                    p_raw = float(_MODEL.predict_proba(X)[0, 1])
                    decision = "alert" if p_raw >= _THRESHOLD else "no_alert"
                
                update_txn_scores(txn, p_raw, decision, MODEL_VERSION)
                total_processed += 1
                log.info("✅ Updated txn %s | p_raw=%.4f | decision=%s",
                         txn.get("transaction_id"), p_raw, decision)
            except Exception as e:
                log.warning("⚠️ Skipping txn %s: %s", txn.get("transaction_id"), e)

        # 3. Continue if paginated
        last_evaluated_key = resp.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

    log.info("✅ Finished scoring entire table. Total processed: %d", total_processed)

if __name__ == "__main__":
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    score_entire_table(batch_size=100)
