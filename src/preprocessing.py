"""
preprocessing.py – Data loading, validation, cleaning, splitting, and scaling.

Run directly:
    python src/preprocessing.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make src imports available when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
)


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load CSV and run basic validation checks."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # Check expected columns
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Unexpected columns
    extra_cols = [c for c in df.columns if c not in FEATURE_COLUMNS + [TARGET_COLUMN]]
    if extra_cols:
        print(f"[WARN] Unexpected columns (will be ignored): {extra_cols}")

    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print dataset summary."""
    print("\n=== DATASET INSPECTION ===")
    print(f"Shape         : {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicates    : {df.duplicated().sum()}")
    print(f"\nTarget ({TARGET_COLUMN}):")
    print(f"  Min  : {df[TARGET_COLUMN].min()}")
    print(f"  Max  : {df[TARGET_COLUMN].max()}")
    print(f"  Mean : {df[TARGET_COLUMN].mean():.4f}")
    print(f"  Std  : {df[TARGET_COLUMN].std():.4f}")

    print("\nFeature min/max (actual vs expected):")
    for feat in FEATURE_COLUMNS:
        actual_min = df[feat].min()
        actual_max = df[feat].max()
        exp_min = FEATURE_RANGES[feat]["min"]
        exp_max = FEATURE_RANGES[feat]["max"]
        flag = "" if (actual_min >= exp_min and actual_max <= exp_max) else " ⚠ RANGE MISMATCH"
        print(f"  {feat:<36} actual=[{actual_min},{actual_max}] expected=[{exp_min},{exp_max}]{flag}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicates."""
    initial_rows = len(df)

    # Drop duplicates
    df = df.drop_duplicates()
    dropped_dupes = initial_rows - len(df)
    if dropped_dupes:
        print(f"[INFO] Dropped {dropped_dupes} duplicate rows")

    # Fill missing numeric values with median
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[INFO] Filled {col} missing values with median={median_val}")

    print(f"[INFO] Dataset after cleaning: {df.shape[0]} rows")
    return df


def preprocess(test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Ensure output directories exist
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and validate
    df = load_and_validate(RAW_DATA_PATH)
    inspect_dataset(df)
    df = clean_dataset(df)

    # Keep only expected columns in defined order
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]

    # Feature / target split
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    print(f"\n[INFO] Features shape : {X.shape}")
    print(f"[INFO] Target shape   : {y.shape}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train samples  : {X_train.shape[0]}")
    print(f"[INFO] Test samples   : {X_test.shape[0]}")

    # Scaling — fit ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Scaler saved   : {SCALER_PATH}")

    # Save processed dataset
    df_processed = pd.DataFrame(
        np.vstack([X_train, X_test]),
        columns=FEATURE_COLUMNS,
    )
    df_processed[TARGET_COLUMN] = np.concatenate([y_train, y_test])
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[INFO] Processed data : {PROCESSED_DATA_PATH}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    preprocess()
    print("\n[DONE] Preprocessing complete.")
