"""
Step 1 — Data Preprocessing
-----------------------------
• Parse & cast all columns to correct types
• Filter invalid AQI (< 0)
• Map pollutant NaN → 'None', clean labels, one-hot encode
• Complete-case deletion on numeric features
• Temporal 70 / 15 / 15 split (no shuffling → no leakage)
• StandardScaler fit on train only, transform val & test
• Save preprocessed arrays + metadata DataFrames for downstream steps
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PATH, RESULTS_DIR, MODELS_DIR,
    NUMERIC_FEATURES, TARGET, CAT_FEATURE,
    POLLUTANT_MAP, POLLUTANT_NONE_LABEL,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
)


def load_and_clean(path: str) -> pd.DataFrame:
    print("Loading dataset …")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # Parse date with mixed formats
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['date'] = pd.to_datetime(df['date'], format='mixed')

    # Cast numeric feature columns (some loaded as str)
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove invalid AQI
    before = len(df)
    df = df[df[TARGET] >= 0]
    print(f"  Dropped {before - len(df):,} rows with AQI < 0")

    # Map pollutant categories → clean labels; NaN → 'None'
    df[CAT_FEATURE] = df[CAT_FEATURE].map(
        lambda x: POLLUTANT_MAP.get(x, POLLUTANT_NONE_LABEL) if pd.notna(x) else POLLUTANT_NONE_LABEL
    )

    return df


def encode_and_split(df: pd.DataFrame):
    # One-hot encode pollutant (drop_first=False — keep all 7 columns for SHAP interpretability)
    df = pd.get_dummies(df, columns=[CAT_FEATURE], prefix='pol', dtype=float)
    pol_cols = [c for c in df.columns if c.startswith('pol_')]
    print(f"  OHE pollutant columns ({len(pol_cols)}): {pol_cols}")

    all_features = NUMERIC_FEATURES + pol_cols

    # Complete-case deletion on features + target
    before = len(df)
    df = df.dropna(subset=all_features + [TARGET]).reset_index(drop=True)
    print(f"  Dropped {before - len(df):,} rows with any NaN in features/target")
    print(f"  Clean shape: {df.shape}")

    # Temporal sort
    df = df.sort_values('date').reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    print(f"\n  Temporal split:")
    print(f"    Train : {len(train_df):>7,}  ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"    Val   : {len(val_df):>7,}  ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"    Test  : {len(test_df):>7,}  ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # Scale — fit ONLY on train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[all_features].values)
    X_val   = scaler.transform(val_df[all_features].values)
    X_test  = scaler.transform(test_df[all_features].values)

    y_train = train_df[TARGET].values
    y_val   = val_df[TARGET].values
    y_test  = test_df[TARGET].values

    # Metadata DataFrames — keep date + siteid for LSTM sequence building
    meta_cols = ['date', 'siteid'] + all_features + [TARGET]
    # Attach scaled features back to metadata frame (needed for LSTM)
    for i, feat in enumerate(all_features):
        train_df[feat] = X_train[:, i]
        val_df[feat]   = X_val[:, i]
        test_df[feat]  = X_test[:, i]

    return (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        train_df, val_df, test_df,
        all_features, pol_cols, scaler,
    )


def main():
    df = load_and_clean(DATA_PATH)

    # Keep raw lat/lon/county for EDA map — save separately before dropping
    map_df = df[['county', 'latitude', 'longitude', TARGET]].dropna().copy()
    map_df[TARGET] = pd.to_numeric(map_df[TARGET], errors='coerce')
    map_df.to_pickle(os.path.join(RESULTS_DIR, 'map_data.pkl'))
    print(f"\n  Saved map data: {map_df.shape}")

    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     train_df, val_df, test_df,
     all_features, pol_cols, scaler) = encode_and_split(df)

    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    # Save everything in a single bundle
    bundle = dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_names=all_features,
        pol_cols=pol_cols,
        n_features=len(all_features),
    )
    out_path = os.path.join(RESULTS_DIR, 'preprocessed.pkl')
    joblib.dump(bundle, out_path)
    print(f"\n  Saved preprocessed bundle → {out_path}")
    print(f"  Feature count: {len(all_features)}")
    print("Preprocessing complete.\n")


if __name__ == '__main__':
    main()
