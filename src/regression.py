"""
regression.py – Supervised learning: Linear Regression, Decision Tree, Random Forest.

Run directly:
    python src/regression.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FEATURE_COLUMNS, GRAPHS_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COLUMN
from preprocessing import preprocess


def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute and print regression metrics."""
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    print(f"\n  {name}")
    print(f"    MAE  : {mae:.6f}")
    print(f"    MSE  : {mse:.6f}")
    print(f"    RMSE : {rmse:.6f}")
    print(f"    R²   : {r2:.6f}")
    return {"model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def save_comparison_chart(results: list, save_path: Path) -> None:
    """Bar chart comparing model metrics."""
    names = [r["model"] for r in results]
    r2s   = [r["R2"]  for r in results]
    rmses = [r["RMSE"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(names, r2s, color=["steelblue", "coral", "seagreen"])
    axes[0].set_title("R² Score (higher is better)")
    axes[0].set_ylabel("R²")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    axes[1].bar(names, rmses, color=["steelblue", "coral", "seagreen"])
    axes[1].set_title("RMSE (lower is better)")
    axes[1].set_ylabel("RMSE")
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    plt.suptitle("Supervised Model Comparison", fontsize=13)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Comparison chart saved: {save_path}")


def train_supervised():
    """Train all three regressors and save models + results."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, _ = preprocess()

    models = {
        "LinearRegression":    LinearRegression(),
        "DecisionTree":        DecisionTreeRegressor(random_state=42),
        "RandomForest":        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = []
    print("\n=== SUPERVISED LEARNING EVALUATION ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
        print(f"  Saved: models/{name}.pkl")

    # Save comparison
    save_comparison_chart(results, GRAPHS_DIR / "supervised_comparison.png")

    # Save report
    df_results = pd.DataFrame(results)
    df_results.to_csv(REPORTS_DIR / "supervised_results.csv", index=False)
    print(f"[INFO] Report saved: outputs/reports/supervised_results.csv")

    # Best model by R²
    best = max(results, key=lambda r: r["R2"])
    print(f"\n[INFO] Best supervised model: {best['model']} (R²={best['R2']:.6f})")

    return results


if __name__ == "__main__":
    train_supervised()
    print("\n[DONE] Supervised learning complete.")
