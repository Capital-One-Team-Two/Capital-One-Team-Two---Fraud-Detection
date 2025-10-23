import os
import io
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import boto3
import joblib
import lightgbm as lgb
from boto3.dynamodb.types import TypeDeserializer
from decimal import Decimal
from boto3.dynamodb.types import TypeSerializer
from types import SimpleNamespace
# ----------------- Logging -----------------
log = logging.getLogger()
log.setLevel(logging.INFO)
# Ensure logs show locally (Lambda uses CloudWatch automatically)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ----------------- Config -----------------
REGION = os.environ.get("AWS_REGION", "us-east-2")
TRANSACTIONS_TABLE = os.environ.get("TRANSACTIONS_TABLE", "FraudDetection-Transactions")

MODEL_BUCKET = os.environ.get("MODEL_BUCKET")  # if provided, load from S3
MODEL_KEY = os.environ.get("MODEL_KEY", "fraud_lgbm_balanced.pkl")
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", f"./{MODEL_KEY}")  # used if no bucket

ENV_FRAUD_THRESHOLD = os.environ.get("FRAUD_THRESHOLD")  # optional override of saved threshold
MODEL_VERSION = os.environ.get("MODEL_VERSION", "lgbm-v1")

# ----------------- AWS Clients -----------------
dynamodb = boto3.client("dynamodb", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

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

def _decimal_to_native(x):
    """Convert Decimal to float for ML features."""
    if isinstance(x, Decimal):
        return float(x)
    return x
def _feature_engineer_single(txn: dict) -> pd.DataFrame:
    """
    Apply the SAME preprocessing as training:
      - Convert Decimals to floats
      - Datetime → hour/day/month/year
      - Drop ID/leaky fields
      - One-hot encode categoricals
      - Sanitize column names
      - Align to training feature_names
    """
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
    arr = df.to_numpy()
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values detected after feature engineering.")

    return df
# ---------- Model loading ----------
def _load_model_if_needed():
    """Load LightGBM artifact once per container (from S3 or local layer)."""
    global _MODEL, _THRESHOLD, _FEATURE_NAMES
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
    """
    key = _build_key_from_txn(txn)
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
    )
    return resp.get("Attributes")

# ----------------- Lambda Handler -----------------
def handler(event, context):
    log.info("Lambda invoked. Region=%s Table=%s ModelVersion=%s", REGION, TRANSACTIONS_TABLE, MODEL_VERSION)

    processed = 0
    _get_table_key_schema()       # confirm key schema up front
    _load_model_if_needed()       # warm model cache
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

        try:
            # Feature engineering + prediction
            log.info("🔹 Feature engineering start")
            X = _feature_engineer_single(txn)
            log.info("🔹 Feature engineering done (%d features)", X.shape[1])

            p_raw = float(_MODEL.predict_proba(X)[0, 1])
            decision = "alert" if p_raw >= _THRESHOLD else "no_alert"
            log.info("🔹 Prediction p_raw=%.6f thr=%.4f => decision=%s", p_raw, _THRESHOLD, decision)

            # Update DynamoDB and confirm
            updated = update_txn_scores(txn, p_raw, decision, MODEL_VERSION)
            log.info("✅ DynamoDB updated (ALL_NEW): %s", updated)
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
