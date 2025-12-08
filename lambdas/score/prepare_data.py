# train_lightgbm_fraud.py
# Improved training script with SMOTE for handling extreme class imbalance
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve,
    f1_score, recall_score, precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import joblib

# =========================
# Config
# =========================
CSV_PATH = '../../infrastructure/fraudTrain.csv'  # <-- update path as needed

# Data Sampling - Reduce dataset size for faster training
# Set to 1.0 to use all data, or 0.1-0.5 for faster training (10-50% of data)
# Recommended: 0.2-0.3 for faster training, 0.5-1.0 for best model
TRAINING_DATA_FRACTION = 0.3  # Balanced: 30% for reasonable speed/quality (adjust: 0.1-1.0)
# Note: Stratified sampling maintains fraud rate, so you'll still have enough fraud cases

# Undersample non-fraud cases AFTER SMOTE to keep training set smaller
# SMOTE creates synthetic fraud, which can make dataset larger. This downsamples non-fraud.
# Set to None to keep all non-fraud cases, or 0.1-0.5 to keep only 10-50% of non-fraud
# Recommended: 0.2-0.3 for faster training, 0.4-0.5 for better quality
UNDERSAMPLE_NON_FRAUD_AFTER_SMOTE = 0.3  # Keep 30% of non-fraud after SMOTE (None = keep all)

# SMOTE Configuration - CRITICAL for handling extreme imbalance (0.39% fraud)
USE_SMOTE = True              # Enable SMOTE resampling
SMOTE_STRATEGY = 0.05         # Target 5% fraud after resampling (adjust: 0.01-0.10)
SMOTE_K_NEIGHBORS = 3         # Lower k for very rare classes (default 5)

# Class Weight Configuration
# For LightGBM, use scale_pos_weight instead of class_weight
USE_SCALE_POS_WEIGHT = True   # Use scale_pos_weight (better for LightGBM)
SCALE_POS_WEIGHT_MULTIPLIER = 15  # Increased from 10 to 15 for better fraud detection (try 10-25)

# Probability calibration can reduce overconfident scores
CALIBRATE = True              # set False to skip isotonic calibration

# Threshold selection strategy
USE_PRECISION_TARGET = True   # True: target precision; False: target FPR or F1
TARGET_PRECISION = 0.55       # Minimum precision to maintain (adjust: 0.50-0.70)
TARGET_FPR_CAP = 0.001        # e.g., <= 0.1% false-positive rate on validation set
MIN_RECALL = 0.75             # Target 75% recall to catch more fraud (adjust: 0.70-0.80)
MIN_PRECISION = 0.50          # Minimum precision floor - don't go below this (adjust: 0.45-0.60)

RANDOM_STATE = 42

# =========================
# Load & basic cleaning
# =========================
print("=" * 60)
print("Fraud Detection Model Training - Improved for Imbalanced Data")
print("=" * 60)
print("\nLoading data...")
data = pd.read_csv(CSV_PATH)
data = data.dropna().reset_index(drop=True)

print(f"Original data shape: {data.shape}")
print(f"Fraud rate: {data['is_fraud'].mean():.4f} ({data['is_fraud'].mean()*100:.2f}%)")
print(f"Fraud samples: {data['is_fraud'].sum():,}")
print(f"Non-fraud samples: {(data['is_fraud']==0).sum():,}")
print(f"Imbalance ratio: {(data['is_fraud']==0).sum() / data['is_fraud'].sum():.1f}:1")

# =========================
# Sample data for faster training (optional)
# =========================
if TRAINING_DATA_FRACTION < 1.0:
    print(f"\n{'='*60}")
    print(f"Sampling {TRAINING_DATA_FRACTION*100:.0f}% of data for faster training...")
    print(f"{'='*60}")
    print(f"Before sampling: {len(data):,} rows")
    
    # Use stratified sampling to maintain fraud rate
    # Note: train_test_split is already imported at the top
    _, data, _, _ = train_test_split(
        data, data['is_fraud'],
        test_size=TRAINING_DATA_FRACTION,
        random_state=RANDOM_STATE,
        stratify=data['is_fraud']
    )
    data = data.reset_index(drop=True)
    
    print(f"After sampling: {len(data):,} rows ({TRAINING_DATA_FRACTION*100:.0f}% of original)")
    print(f"Fraud samples: {data['is_fraud'].sum():,}")
    print(f"Non-fraud samples: {(data['is_fraud']==0).sum():,}")
    print(f"Fraud rate maintained: {data['is_fraud'].mean():.4f}")
else:
    print("\nUsing full dataset for training")

# =========================
# Feature engineering
# =========================
print("\nFeature engineering...")
if 'trans_date_trans_time' in data.columns:
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['trans_hour']  = data['trans_date_trans_time'].dt.hour
    data['trans_day']   = data['trans_date_trans_time'].dt.day
    data['trans_month'] = data['trans_date_trans_time'].dt.month
    data['trans_year']  = data['trans_date_trans_time'].dt.year
    
    # Additional temporal features for better fraud detection
    data['is_weekend'] = (data['trans_day'].isin([5, 6])).astype(int)
    data['is_night'] = ((data['trans_hour'] >= 22) | (data['trans_hour'] <= 6)).astype(int)
    data['is_business_hours'] = ((data['trans_hour'] >= 9) & (data['trans_hour'] <= 17)).astype(int)

# Amount-based features
if 'amt' in data.columns:
    data['amount_log'] = np.log1p(data['amt'])
    data['amount_sqrt'] = np.sqrt(data['amt'])
    data['amount_squared'] = data['amt'] ** 2
    # High-value transaction flag (fraud often involves unusual amounts)
    data['is_high_value'] = (data['amt'] > data['amt'].quantile(0.95)).astype(int)
    data['is_low_value'] = (data['amt'] < data['amt'].quantile(0.05)).astype(int)

# Distance-based features (if lat/long available) - BEFORE dropping columns
if 'lat' in data.columns and 'long' in data.columns and 'merch_lat' in data.columns and 'merch_long' in data.columns:
    # Calculate distance between cardholder and merchant using Haversine formula
    from math import radians, sin, cos, sqrt, atan2
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points on Earth in km."""
        R = 6371  # Earth radius in km
        try:
            lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c
        except (ValueError, TypeError):
            return np.nan
    
    print("Calculating distance features...")
    data['distance_to_merchant'] = data.apply(
        lambda row: haversine_distance(row['lat'], row['long'], row['merch_lat'], row['merch_long']),
        axis=1
    )
    data['distance_log'] = np.log1p(data['distance_to_merchant'].fillna(0))
    data['is_far_transaction'] = (data['distance_to_merchant'] > 100).astype(int)  # >100km

cols_to_drop = [
    'is_fraud', 'trans_date_trans_time', 'cc_num', 'trans_num', 'Unnamed: 0',
    # Often leaky or too specific to generalize:
    'first', 'last', 'street', 'city', 'zip', 'merchant', 'job', 'lat', 'long',
    'merch_lat', 'merch_long',  # Drop after calculating distance
]
present_to_drop = [c for c in cols_to_drop if c in data.columns]

y = data['is_fraud'].astype(int)
X = data.drop(columns=present_to_drop, errors='ignore')

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=False)

# Sanitize column names for LightGBM JSON feature name safety
def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
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

X = sanitize_columns(X)

# =========================
# Split (stratified)
# =========================
print("\nSplitting data (stratified)...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print("Fraud rate: train={:.4f}, val={:.4f}, test={:.4f}".format(
    y_train.mean(), y_val.mean(), y_test.mean()
))
print(f"Train fraud samples: {y_train.sum()}, non-fraud: {(y_train==0).sum()}")

# =========================
# SMOTE Resampling - CRITICAL for extreme imbalance
# =========================
if USE_SMOTE:
    print(f"\n{'='*60}")
    print("Applying SMOTE to handle class imbalance...")
    print(f"{'='*60}")
    print(f"Before SMOTE - Train shape: {X_train.shape}")
    print(f"  Fraud rate: {y_train.mean():.4f} ({y_train.sum()} fraud, {(y_train==0).sum()} non-fraud)")
    
    try:
        smote = SMOTE(
            sampling_strategy=SMOTE_STRATEGY,
            random_state=RANDOM_STATE,
            k_neighbors=min(SMOTE_K_NEIGHBORS, y_train.sum() - 1)  # Ensure k < fraud samples
        )
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Train shape: {X_train_resampled.shape}")
        print(f"  Fraud rate: {y_train_resampled.mean():.4f} ({y_train_resampled.sum()} fraud, {(y_train_resampled==0).sum()} non-fraud)")
        print(f"  Resampling ratio: {len(X_train_resampled) / len(X_train):.2f}x")
        
        # Optional: Undersample non-fraud to reduce training set size
        if UNDERSAMPLE_NON_FRAUD_AFTER_SMOTE is not None and UNDERSAMPLE_NON_FRAUD_AFTER_SMOTE < 1.0:
            print(f"\nUndersampling non-fraud cases to {UNDERSAMPLE_NON_FRAUD_AFTER_SMOTE*100:.0f}% for faster training...")
            print(f"Before undersampling: {X_train_resampled.shape[0]:,} rows")
            
            # Use RandomUnderSampler to keep only a fraction of non-fraud cases
            # We want to maintain the fraud rate, so we undersample non-fraud proportionally
            # Calculate target: if we have F fraud and want to keep U% of non-fraud
            # New total = F + (non_fraud * U)
            fraud_count = y_train_resampled.sum()
            non_fraud_count = (y_train_resampled == 0).sum()
            target_non_fraud = int(non_fraud_count * UNDERSAMPLE_NON_FRAUD_AFTER_SMOTE)
            
            # Create sampling strategy: keep all fraud, but only U% of non-fraud
            sampling_strategy = {0: target_non_fraud, 1: fraud_count}
            
            undersampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=RANDOM_STATE
            )
            X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_resampled, y_train_resampled)
            
            print(f"After undersampling: {X_train_resampled.shape[0]:,} rows")
            print(f"  Fraud rate: {y_train_resampled.mean():.4f} ({y_train_resampled.sum()} fraud, {(y_train_resampled==0).sum()} non-fraud)")
            print(f"  Size reduction: {len(X_train_resampled) / len(X_train):.2f}x original size")
    except Exception as e:
        print(f"⚠️  SMOTE failed: {e}")
        print("   Falling back to original training data")
        X_train_resampled, y_train_resampled = X_train, y_train
else:
    X_train_resampled, y_train_resampled = X_train, y_train
    print("\n⚠️  SMOTE disabled - model may struggle with extreme imbalance")

# =========================
# Calculate Class Weights
# =========================
if USE_SCALE_POS_WEIGHT:
    # LightGBM's scale_pos_weight is more effective than class_weight
    base_ratio = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()
    scale_pos_weight = base_ratio * SCALE_POS_WEIGHT_MULTIPLIER
    print(f"\nClass weighting:")
    print(f"  Base ratio (non-fraud/fraud): {base_ratio:.2f}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f} (multiplier: {SCALE_POS_WEIGHT_MULTIPLIER})")
else:
    scale_pos_weight = None
    print(f"\nUsing default class weighting")

# =========================
# Model & grid search
# =========================
print(f"\n{'='*60}")
print("Training LightGBM model...")
print(f"{'='*60}")

base = lgb.LGBMClassifier(
    n_estimators=600,  # Balanced: more than 500 but less than 1000
    learning_rate=0.04,  # Slightly lower than 0.05 for stability
    max_depth=9,  # Slightly deeper than 8 but less than 10
    scale_pos_weight=scale_pos_weight,  # Better than class_weight for LightGBM
    subsample=0.85,  # Slightly lower for regularization
    colsample_bytree=0.85,  # Slightly lower for regularization
    reg_alpha=0.1,  # L1 regularization to prevent overfitting
    reg_lambda=0.1,  # L2 regularization to prevent overfitting
    min_split_gain=0.0,  # Minimum gain for split
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

# Reduced parameter grid for faster tuning (still good coverage)
param_grid = {
    'num_leaves': [31, 63, 127],  # Removed 255 to reduce combinations
    'min_child_samples': [20, 50, 100],  # Removed 10 to reduce combinations
    'max_bin': [255, 511],
    'reg_alpha': [0.0, 0.1],  # Reduced from 3 to 2 options
    'reg_lambda': [0.0, 0.1],  # Reduced from 3 to 2 options
}

print("Running grid search with PR-AUC scoring (best for imbalanced data)...")
grid = GridSearchCV(
    base,
    param_grid=param_grid,
    scoring='average_precision',   # PR-AUC is best for rare positives
    cv=3,
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train_resampled, y_train_resampled)

best = grid.best_estimator_
print("Best params:", grid.best_params_)

# Refit with early stopping using the validation set
print("\nRefitting with early stopping on validation set...")
best.set_params(n_estimators=600)  # Balanced ceiling; early stopping will cut it
best.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_val, y_val)],
    eval_metric='average_precision',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),  # Standard patience
        lgb.log_evaluation(period=100)  # Log progress every 100 iterations (less verbose)
    ]
)

# =========================
# Optional: Calibrate probabilities (isotonic)
# =========================
model_for_thresholds = best
if CALIBRATE:
    cal = CalibratedClassifierCV(best, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    model_for_thresholds = cal
    print("Applied isotonic calibration on validation set.")

# =========================
# Threshold selection helpers
# =========================
def threshold_for_precision(y_true, proba, p_floor=0.90, min_recall=0.70, min_precision=0.50):
    """Find threshold that balances recall and precision for fraud detection."""
    p, r, t = precision_recall_curve(y_true, proba)
    
    # For fraud detection, we want high recall BUT also reasonable precision
    # Strategy: Find threshold that meets recall target while maintaining minimum precision
    
    # First, try to find thresholds that meet BOTH precision and recall requirements
    precision_mask = p[:-1] >= p_floor
    recall_mask = r[:-1] >= min_recall
    both_mask = precision_mask & recall_mask
    
    if both_mask.any():
        # Great! We have thresholds that meet both - pick the one with best F1 (balanced)
        idx_both = np.where(both_mask)[0]
        f1_scores = 2 * (p[:-1][idx_both] * r[:-1][idx_both]) / (p[:-1][idx_both] + r[:-1][idx_both] + 1e-10)
        j = idx_both[np.argmax(f1_scores)]
    else:
        # No threshold meets both - find best balance
        # Filter by minimum precision floor to avoid too many false alarms
        min_prec_mask = p[:-1] >= min_precision
        recall_idx = np.where(recall_mask & min_prec_mask)[0]
        
        if len(recall_idx) > 0:
            # Among thresholds meeting recall AND minimum precision, pick best F1
            f1_scores = 2 * (p[:-1][recall_idx] * r[:-1][recall_idx]) / (p[:-1][recall_idx] + r[:-1][recall_idx] + 1e-10)
            j = recall_idx[np.argmax(f1_scores)]
        else:
            # Can't meet both - find best compromise
            # Try to get as close to recall target as possible while maintaining min precision
            min_prec_idx = np.where(min_prec_mask)[0]
            if len(min_prec_idx) > 0:
                # Among thresholds with min precision, find one closest to recall target
                recall_diff = np.abs(r[:-1][min_prec_idx] - min_recall)
                j = min_prec_idx[np.argmin(recall_diff)]
            else:
                # Last resort: maximize F1 score overall
                f1_scores = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-10)
                j = np.argmax(f1_scores)
    
    f1 = 2 * p[j] * r[j] / (p[j] + r[j] + 1e-10)
    return t[j], {"precision": float(p[j]), "recall": float(r[j]), "f1": float(f1)}

def threshold_for_fpr(y_true, proba, fpr_cap=0.001):
    """Find threshold that meets FPR cap."""
    fpr, tpr, t = roc_curve(y_true, proba)
    idx = np.where(fpr <= fpr_cap)[0]
    if len(idx) == 0:
        j = np.argmin(fpr)
        return t[j], {"fpr": float(fpr[j]), "tpr": float(tpr[j])}
    j = idx[-1]  # highest threshold while still meeting the cap
    return t[j], {"fpr": float(fpr[j]), "tpr": float(tpr[j])}

def threshold_for_f1(y_true, proba):
    """Find threshold that maximizes F1 score."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1_scores = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    return t[optimal_idx], {
        "precision": float(p[optimal_idx]),
        "recall": float(r[optimal_idx]),
        "f1": float(f1_scores[optimal_idx])
    }

# =========================
# Pick threshold on validation
# =========================
print(f"\n{'='*60}")
print("Selecting optimal threshold on validation set...")
print(f"{'='*60}")
val_proba = model_for_thresholds.predict_proba(X_val)[:, 1]

if USE_PRECISION_TARGET:
    opt_thr, stats = threshold_for_precision(y_val, val_proba, p_floor=TARGET_PRECISION, min_recall=MIN_RECALL, min_precision=MIN_PRECISION)
    print(f"Strategy: Precision ≥ {TARGET_PRECISION:.2f} (min {MIN_PRECISION:.2f}) with recall ≥ {MIN_RECALL:.2f}")
    print(f"Chosen threshold: {opt_thr:.4f}")
    print(f"  Precision: {stats['precision']:.3f}")
    print(f"  Recall: {stats['recall']:.3f}")
    print(f"  F1 Score: {stats.get('f1', 0):.3f}")
else:
    opt_thr, stats = threshold_for_fpr(y_val, val_proba, fpr_cap=TARGET_FPR_CAP)
    print(f"Strategy: FPR ≤ {TARGET_FPR_CAP:.4f}")
    print(f"Chosen threshold: {opt_thr:.4f}")
    print(f"  FPR: {stats['fpr']:.5f}")
    print(f"  TPR (Recall): {stats['tpr']:.3f}")

# =========================
# Evaluate on test
# =========================
print(f"\n{'='*60}")
print("Evaluating on test set...")
print(f"{'='*60}")
test_proba = model_for_thresholds.predict_proba(X_test)[:, 1]
test_pred  = (test_proba >= opt_thr).astype(int)

# Calculate comprehensive metrics
roc_auc = roc_auc_score(y_test, test_proba)
pr_auc = average_precision_score(y_test, test_proba)
f1 = f1_score(y_test, test_pred)
precision = precision_score(y_test, test_pred)
recall = recall_score(y_test, test_pred)
cm = confusion_matrix(y_test, test_pred)

print(f"\n📊 Test Set Metrics (threshold: {opt_thr:.4f})")
print(f"{'='*60}")
print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"PR-AUC (AP):       {pr_auc:.4f}  (better metric for imbalanced data)")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1 Score:          {f1:.4f}")
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]:,}")
print(f"  False Positives: {cm[0,1]:,}")
print(f"  False Negatives: {cm[1,0]:,}  ⚠️  (missed fraud cases)")
print(f"  True Positives:  {cm[1,1]:,}")
print(f"\nConfusion Matrix (array):\n{cm}")
print(f"\nDetailed Classification Report:\n{classification_report(y_test, test_pred, digits=3)}")

# Additional insights
print(f"\n💡 Key Insights:")
print(f"  - Fraud detection rate (Recall): {recall*100:.1f}%")
print(f"  - Of predicted frauds, {precision*100:.1f}% are actually fraud")
print(f"  - Missed {cm[1,0]} fraud cases out of {y_test.sum()} total fraud cases")
if cm[1,0] > 0:
    missed_rate = cm[1,0] / y_test.sum()
    print(f"  - ⚠️  Missing {missed_rate*100:.1f}% of fraud cases - consider lowering threshold")

# =========================
# Save artifact (model or calibrated model) & threshold
# =========================
print(f"\n{'='*60}")
print("Saving model artifact...")
print(f"{'='*60}")
final_model = model_for_thresholds
artifact = {
    'model': final_model,
    'threshold': float(opt_thr),
    'feature_names': X.columns.tolist(),
    'training_config': {
        'used_smote': USE_SMOTE,
        'smote_strategy': SMOTE_STRATEGY if USE_SMOTE else None,
        'scale_pos_weight': scale_pos_weight,
        'calibrated': CALIBRATE,
    }
}
output_file = 'fraud_lgbm_balanced.pkl'
joblib.dump(artifact, output_file)
print(f"\n✅ Saved: {output_file}")
print(f"   - Model: {type(final_model).__name__}")
print(f"   - Threshold: {opt_thr:.4f}")
print(f"   - Features: {len(X.columns)}")
print(f"   - Test Recall: {recall:.3f}")
print(f"   - Test Precision: {precision:.3f}")
print(f"   - Test F1: {f1:.3f}")
print(f"\n🎉 Training complete!")
