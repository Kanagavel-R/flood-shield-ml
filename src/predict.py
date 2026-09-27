"""
predict.py – Prediction pipeline for Flood Shield.
Loads saved scaler and final model, never retrains.

Usage:
    python src/predict.py
"""

import math
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    FEATURE_COLUMNS,
    FINAL_MODEL_PATH,
    SCALER_PATH,
    clamp_prediction,
    get_risk_level,
    get_safety_recommendation,
    pct_to_raw,
    validate_pct_inputs,
)


def load_artifacts():
    """Load scaler and final model from disk. Raises clear errors if missing."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run preprocessing first."
        )
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {FINAL_MODEL_PATH}. Run optimization first."
        )
    scaler = joblib.load(SCALER_PATH)
    model  = joblib.load(FINAL_MODEL_PATH)
    return scaler, model


def predict_from_raw(raw_values: dict) -> dict:
    """
    Predict FloodProbability from raw dataset-scale feature values.

    Args:
        raw_values: dict mapping each feature name to its raw value.

    Returns:
        dict with probability, probability_pct, risk_level, recommendation.

    Raises:
        ValueError: if any expected feature is missing.
        RuntimeError: if the model produces a non-finite output.
    """
    scaler, model = load_artifacts()

    # Build feature array in exact training order
    missing = [f for f in FEATURE_COLUMNS if f not in raw_values]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = np.array([[raw_values[f] for f in FEATURE_COLUMNS]])
    X_scaled = scaler.transform(X)

    raw_prob = float(model.predict(X_scaled)[0])

    # Guard against NaN / infinity (should not occur with a trained MLP,
    # but defensive check ensures the pipeline never silently returns garbage)
    if not math.isfinite(raw_prob):
        raise RuntimeError(
            f"Model returned a non-finite prediction ({raw_prob}). "
            "Check model integrity and input values."
        )

    probability    = clamp_prediction(raw_prob)
    risk_level     = get_risk_level(probability)
    recommendation = get_safety_recommendation(risk_level)

    return {
        "probability":     probability,
        "probability_pct": round(probability * 100, 1),
        "risk_level":      risk_level,
        "recommendation":  recommendation,
    }


def predict_from_percentages(pct_values: dict) -> dict:
    """
    Predict from 0–100 UI percentage values.
    Validates inputs, converts each percentage to original dataset scale,
    then delegates to predict_from_raw.

    Args:
        pct_values: dict mapping feature name to 0–100 percentage.

    Returns:
        Same dict as predict_from_raw.

    Raises:
        ValueError: if any input fails validation.
    """
    errors = validate_pct_inputs(pct_values)
    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(errors))

    raw_values = {feat: pct_to_raw(feat, pct) for feat, pct in pct_values.items()}
    return predict_from_raw(raw_values)


if __name__ == "__main__":
    # Self-test with a high-risk scenario (all features at ~80% severity)
    print("=== PREDICTION PIPELINE TEST ===")
    test_pct = {feat: 80.0 for feat in FEATURE_COLUMNS}
    result   = predict_from_percentages(test_pct)
    print(f"Input      : all features at 80% severity")
    print(f"Probability: {result['probability_pct']}%")
    print(f"Risk Level : {result['risk_level']}")
    print(f"Recommend  : {result['recommendation']}")

    print("\n--- Low-risk scenario (all features at 20% severity) ---")
    test_low = {feat: 20.0 for feat in FEATURE_COLUMNS}
    result2  = predict_from_percentages(test_low)
    print(f"Probability: {result2['probability_pct']}%")
    print(f"Risk Level : {result2['risk_level']}")
