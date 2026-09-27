"""
ann.py – Artificial Neural Network using sklearn MLPRegressor.

Run directly:
    python src/ann.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import GRAPHS_DIR, MODELS_DIR, REPORTS_DIR
from preprocessing import preprocess


def train_ann():
    """Train MLPRegressor and evaluate."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, _ = preprocess()

    print("\n=== ANN (MLPRegressor) TRAINING ===")
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_train, y_train)
    print(f"[INFO] Training stopped at iteration: {mlp.n_iter_}")

    y_pred = mlp.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  MLPRegressor Results:")
    print(f"    MAE  : {mae:.6f}")
    print(f"    MSE  : {mse:.6f}")
    print(f"    RMSE : {rmse:.6f}")
    print(f"    R²   : {r2:.6f}")

    # Save model
    joblib.dump(mlp, MODELS_DIR / "MLPRegressor.pkl")
    print(f"[INFO] Saved: models/MLPRegressor.pkl")

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(mlp.loss_curve_, label="Training Loss", color="steelblue")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None:
        # MLPRegressor stores best_validation_score, not full curve — skip
        pass
    plt.title("ANN Training Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "ann_loss_curve.png", dpi=150)
    plt.close()
    print(f"[INFO] Loss curve saved: outputs/graphs/ann_loss_curve.png")

    # Save report
    report = pd.DataFrame([{
        "model": "MLPRegressor",
        "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2
    }])
    report.to_csv(REPORTS_DIR / "ann_results.csv", index=False)

    return {"model": "MLPRegressor", "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    train_ann()
    print("\n[DONE] ANN training complete.")
