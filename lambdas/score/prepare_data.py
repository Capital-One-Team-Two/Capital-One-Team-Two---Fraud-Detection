# train_lightgbm_fraud.py
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import joblib

# =========================
# Config
# =========================
CSV_PATH = '../../infrastructure/fraudTest.csv'  # <-- update path as needed

# Probability calibration can reduce overconfident scores
CALIBRATE = True           # set False to skip isotonic calibration

# Pick ONE operating-point rule:
USE_PRECISION_TARGET = True    # True: target precision; False: target FPR
TARGET_PRECISION     = 0.90    # e.g., >= 90% precision on validation set
TARGET_FPR_CAP       = 0.001   # e.g., <= 0.1% false-positive rate on validation set

# Optional: nudge class weights if you still get too many alerts
# Set to None to keep 'balanced'
CLASS_WEIGHT = 'balanced'      # or try {0:1.0, 1:0.7} / {0:1.0, 1:0.5}

RANDOM_STATE = 42

# =========================
# Load & basic cleaning
# =========================
data = pd.read_csv(CSV_PATH)
data = data.dropna().reset_index(drop=True)

# =========================
# Feature engineering
# =========================
if 'trans_date_trans_time' in data.columns:
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['trans_hour']  = data['trans_date_trans_time'].dt.hour
    data['trans_day']   = data['trans_date_trans_time'].dt.day
    data['trans_month'] = data['trans_date_trans_time'].dt.month
    data['trans_year']  = data['trans_date_trans_time'].dt.year

cols_to_drop = [
    'is_fraud', 'trans_date_trans_time', 'cc_num', 'trans_num', 'Unnamed: 0',
    # Often leaky or too specific to generalize:
    'first', 'last', 'street', 'city', 'zip', 'merchant', 'job', 'lat', 'long',
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

# =========================
# Model & grid search
# =========================
base = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    class_weight=CLASS_WEIGHT,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

param_grid = {
    'num_leaves': [31, 63, 127],
    'min_child_samples': [20, 50, 100],
    'max_bin': [255, 511],
}

grid = GridSearchCV(
    base,
    param_grid=param_grid,
    scoring='average_precision',   # PR-AUC is best for rare positives
    cv=3,
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train, y_train)

best = grid.best_estimator_
print("Best params:", grid.best_params_)

# Refit with early stopping using the validation set
best.set_params(n_estimators=500)  # ceiling; early stopping will cut it
best.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='average_precision',
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
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
def threshold_for_precision(y_true, proba, p_floor=0.90):
    p, r, t = precision_recall_curve(y_true, proba)
    idx = np.where(p[:-1] >= p_floor)[0]
    if len(idx) == 0:
        j = np.argmax(p[:-1])
        return t[j], {"precision": float(p[j]), "recall": float(r[j])}
    # among thresholds reaching the precision floor, pick the one with highest recall
    j = idx[np.argmax(r[:-1][idx])]
    return t[j], {"precision": float(p[j]), "recall": float(r[j])}

def threshold_for_fpr(y_true, proba, fpr_cap=0.001):
    fpr, tpr, t = roc_curve(y_true, proba)
    idx = np.where(fpr <= fpr_cap)[0]
    if len(idx) == 0:
        j = np.argmin(fpr)
        return t[j], {"fpr": float(fpr[j]), "tpr": float(tpr[j])}
    j = idx[-1]  # highest threshold while still meeting the cap
    return t[j], {"fpr": float(fpr[j]), "tpr": float(tpr[j])}

# =========================
# Pick threshold on validation
# =========================
val_proba = model_for_thresholds.predict_proba(X_val)[:, 1]

if USE_PRECISION_TARGET:
    opt_thr, stats = threshold_for_precision(y_val, val_proba, p_floor=TARGET_PRECISION)
    print(f"Chosen threshold for precision ≥ {TARGET_PRECISION:.2f}: {opt_thr:.4f} | "
          f"P={stats['precision']:.3f}, R={stats['recall']:.3f}")
else:
    opt_thr, stats = threshold_for_fpr(y_val, val_proba, fpr_cap=TARGET_FPR_CAP)
    print(f"Chosen threshold for FPR ≤ {TARGET_FPR_CAP:.4f}: {opt_thr:.4f} | "
          f"FPR={stats['fpr']:.5f}, TPR={stats['tpr']:.3f}")

# =========================
# Evaluate on test
# =========================
test_proba = model_for_thresholds.predict_proba(X_test)[:, 1]
test_pred  = (test_proba >= opt_thr).astype(int)

print("\n=== Test Metrics (threshold tuned on val) ===")
print("ROC-AUC:", roc_auc_score(y_test, test_proba))
print("PR-AUC (Average Precision):", average_precision_score(y_test, test_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))
print("\nClassification Report:\n", classification_report(y_test, test_pred, digits=3))

# =========================
# Save artifact (model or calibrated model) & threshold
# =========================
final_model = model_for_thresholds
artifact = {
    'model': final_model,
    'threshold': float(opt_thr),
    'feature_names': X.columns.tolist(),
}
joblib.dump(artifact, 'fraud_lgbm_balanced.pkl')
print("\nSaved: fraud_lgbm_balanced.pkl (includes model, threshold, and feature names)")
