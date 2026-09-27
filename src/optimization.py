"""
optimization.py – GridSearchCV, cross-validation, and final model selection.

Dataset note:
  FloodProbability is a near-perfect linear combination of the 20 features,
  so LinearRegression achieves R²=1.0 on test data. However, it extrapolates
  beyond the training target range [0.285, 0.725] when features take
  simultaneous extreme values not seen together in the dataset.
  MLPRegressor (R²=0.9999) is selected as the final model: it captures the
  same near-perfect relationship with a safer, bounded output profile.

Run directly:
    python src/optimization.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import FEATURE_COLUMNS, GRAPHS_DIR, MODELS_DIR, REPORTS_DIR, FINAL_MODEL_PATH
from preprocessing import preprocess


def run_optimization():
    """GridSearchCV + cross-validation + final model selection."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, _ = preprocess()

    # ── Evaluate all trained base models ─────────────────────────────────────
    all_results  = []
    loaded_models = {}

    base_models = {
        "LinearRegression": "LinearRegression.pkl",
        "DecisionTree":     "DecisionTree.pkl",
        "RandomForest":     "RandomForest.pkl",
        "MLPRegressor":     "MLPRegressor.pkl",
    }

    print("\n=== BASE MODEL PERFORMANCE ===")
    for name, fname in base_models.items():
        pkl_path = MODELS_DIR / fname
        if not pkl_path.exists():
            print(f"  [SKIP] {name} — not found, run regression.py / ann.py first")
            continue
        m      = joblib.load(pkl_path)
        y_pred = m.predict(X_test)
        r2     = r2_score(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        mae    = mean_absolute_error(y_test, y_pred)
        mse    = mean_squared_error(y_test, y_pred)
        print(f"  {name:<22} R²={r2:.6f}  RMSE={rmse:.8f}")
        all_results.append({
            "model": name, "type": "base",
            "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
        })
        loaded_models[name] = m

    # ── GridSearchCV — Decision Tree ──────────────────────────────────────────
    print("\n=== GRIDSEARCHCV — Decision Tree ===")
    param_grid_dt = {
        "max_depth":         [5, 10, None],
        "min_samples_split": [2, 5],
    }
    dt_cv = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid_dt,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    dt_cv.fit(X_train, y_train)
    print(f"  Best params : {dt_cv.best_params_}")
    print(f"  Best CV R²  : {dt_cv.best_score_:.6f}")

    y_pred_dt = dt_cv.best_estimator_.predict(X_test)
    dt_r2   = r2_score(y_test, y_pred_dt)
    dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    dt_mae  = mean_absolute_error(y_test, y_pred_dt)
    dt_mse  = mean_squared_error(y_test, y_pred_dt)
    print(f"  Test R²={dt_r2:.6f}  RMSE={dt_rmse:.6f}")

    joblib.dump(dt_cv.best_estimator_, MODELS_DIR / "DT_GridSearch.pkl")
    all_results.append({
        "model": "DT_GridSearch", "type": "optimized",
        "MAE": dt_mae, "MSE": dt_mse, "RMSE": dt_rmse, "R2": dt_r2,
    })
    loaded_models["DT_GridSearch"] = dt_cv.best_estimator_

    # ── Cross-validation on MLPRegressor ─────────────────────────────────────
    print("\n=== CROSS-VALIDATION — MLPRegressor (5-fold) ===")
    mlp_model = loaded_models.get("MLPRegressor")
    if mlp_model:
        cv_scores = cross_val_score(
            mlp_model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
        )
        print(f"  CV R² scores : {np.round(cv_scores, 6)}")
        print(f"  Mean R²      : {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

    # ── Final model selection ─────────────────────────────────────────────────
    # MLPRegressor is selected over LinearRegression (R²=1.0) because:
    # - The dataset target is a near-perfect linear combination of features.
    # - LinearRegression extrapolates beyond [0.285, 0.725] for extreme inputs.
    # - MLPRegressor (R²=0.9999) captures the same relationship more safely.
    # - With output clamping, MLPRegressor gives the most reliable predictions.

    FINAL_MODEL_NAME = "MLPRegressor"

    df_results = pd.DataFrame(all_results)
    print(f"\n=== ALL MODEL RESULTS ===")
    print(df_results[["model", "type", "R2", "RMSE", "MAE"]].to_string(index=False))

    final_row = df_results[df_results["model"] == FINAL_MODEL_NAME].iloc[0]
    print(f"\n=== FINAL MODEL SELECTED ===")
    print(f"  Model  : {FINAL_MODEL_NAME}")
    print(f"  R²     : {final_row['R2']:.6f}")
    print(f"  RMSE   : {final_row['RMSE']:.8f}")
    print(f"  Reason : Best generalising model; LR extrapolates outside training range.")

    final_model = loaded_models[FINAL_MODEL_NAME]
    joblib.dump(final_model, FINAL_MODEL_PATH)
    print(f"[INFO] Final model saved: {FINAL_MODEL_PATH}")

    # ── Feature importance chart (DT GridSearch) ──────────────────────────────
    dt_best = loaded_models["DT_GridSearch"]
    importances = pd.Series(
        dt_best.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values()
    plt.figure(figsize=(9, 6))
    importances.plot(kind="barh", color="teal")
    plt.title("Feature Importances — Decision Tree (GridSearch)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("[INFO] Feature importance saved: outputs/graphs/feature_importance.png")

    # ── All-models R² comparison chart ───────────────────────────────────────
    df_plot = df_results.sort_values("R2", ascending=True)
    colors  = ["#2ecc71" if m == FINAL_MODEL_NAME else "steelblue"
               for m in df_plot["model"]]
    plt.figure(figsize=(10, 5))
    bars = plt.barh(df_plot["model"], df_plot["R2"], color=colors)
    plt.title("All Models — R² Comparison  (green = selected final model)")
    plt.xlabel("R²")
    plt.xlim(0, 1.08)
    for bar, val in zip(bars, df_plot["R2"]):
        plt.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "all_models_comparison.png", dpi=150)
    plt.close()
    print("[INFO] Comparison chart saved: outputs/graphs/all_models_comparison.png")

    # ── Save report ───────────────────────────────────────────────────────────
    df_results.to_csv(REPORTS_DIR / "optimization_results.csv", index=False)
    print("[INFO] Report saved: outputs/reports/optimization_results.csv")

    return FINAL_MODEL_NAME, final_model


if __name__ == "__main__":
    run_optimization()
    print("\n[DONE] Optimization complete.")
