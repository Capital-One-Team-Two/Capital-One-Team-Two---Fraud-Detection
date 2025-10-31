# fraud_model_eval_sample.py
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

CSV_PATH = "../../infrastructure/fraudTest.csv"   # adjust if needed
ARTIFACT_PATH = "fraud_lgbm_balanced.pkl"
RANDOM_STATE = 42
N_SAMPLES = 10

# ======== Helper functions ========

def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["trans_hour"] = df["trans_date_trans_time"].dt.hour
        df["trans_day"] = df["trans_date_trans_time"].dt.day
        df["trans_month"] = df["trans_date_trans_time"].dt.month
        df["trans_year"] = df["trans_date_trans_time"].dt.year
    return df

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        df.columns.astype(str).str.strip()
        .str.replace(r'[\\"/\b\f\n\r\t]', "_", regex=True)
        .str.replace(r"[^0-9A-Za-z_]", "_", regex=True)
    )
    seen, safe = {}, []
    for c in cols:
        if c in seen:
            seen[c] += 1
            safe.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0
            safe.append(c)
    df.columns = safe
    return df

def build_features(df: pd.DataFrame):
    df = add_time_parts(df)
    drop_cols = [
        "is_fraud", "trans_date_trans_time", "cc_num", "trans_num", "Unnamed: 0",
        "first", "last", "street", "city", "zip", "merchant", "job", "lat", "long",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return X

def align_dummies(X: pd.DataFrame, feature_names):
    Xd = pd.get_dummies(X, drop_first=False)
    Xd = sanitize_columns(Xd)
    for c in feature_names:
        if c not in Xd:
            Xd[c] = 0
    Xd = Xd[feature_names]
    return Xd

# ======== Main test ========

def main():
    print("Loading model artifact...")
    art = joblib.load(ARTIFACT_PATH)
    model = art["model"]
    threshold = float(art["threshold"])
    feature_names = art.get("feature_names", None)
    preprocessor = art.get("preprocessor", None)

    print("Loading sample data...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna().reset_index(drop=True)

    # Choose 10 fraud + 10 non-fraud
    fraud_df = df[df["is_fraud"] == 1].sample(n=N_SAMPLES, random_state=RANDOM_STATE)
    nonfraud_df = df[df["is_fraud"] == 0].sample(n=N_SAMPLES, random_state=RANDOM_STATE)
    sample_df = pd.concat([fraud_df, nonfraud_df]).reset_index(drop=True)

    X_raw = build_features(sample_df)
    y_true = sample_df["is_fraud"].astype(int)

    # Transform features
    if preprocessor:
        X = preprocessor.transform(X_raw)
    else:
        X = align_dummies(X_raw, feature_names)

    # Predict
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    acc = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)

    print(f"\n✅ Evaluated {len(y_true)} transactions")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix [ [TN FP]; [FN TP] ]:\n", cm)

    # Show predictions
    show_cols = [c for c in ["amount", "merchant", "category"] if c in sample_df.columns]
    result_df = sample_df[show_cols + ["is_fraud"]].copy()
    result_df["pred_prob"] = proba
    result_df["pred_label"] = preds

    print("\n=== 10 REAL FRAUD TRANSACTIONS ===")
    print(result_df[result_df["is_fraud"] == 1].head(10))

    print("\n=== 10 REAL NON-FRAUD TRANSACTIONS ===")
    print(result_df[result_df["is_fraud"] == 0].head(10))

if __name__ == "__main__":
    main()
