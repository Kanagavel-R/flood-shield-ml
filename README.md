# 🌊 Flood Shield — Intelligent Flood Risk Prediction System

A Machine Learning based flood risk prediction application built as a B.Tech IT academic project.  
The system accepts 20 environmental and infrastructure-related input factors and predicts **Flood Probability**, a corresponding **Risk Level**, and **Personalized Preventive Safety Suggestions**.

---

## Problem Statement

Floods are among the most damaging natural disasters.  Early awareness of flood risk based on environmental and infrastructure conditions can help communities take timely preventive action.  
This system uses a trained ML model to estimate flood risk from 20 measurable risk factors.

---

## Solution

A Streamlit web application where users input 20 severity scores (0–100%).  
The inputs are converted to dataset-scale values, passed through a trained StandardScaler and MLPRegressor, and the resulting FloodProbability is displayed with a risk level and personalized preventive recommendations.

---

## Dataset

| Property | Value |
|---|---|
| File | `data/raw/flood.csv` |
| Records | 50,000 |
| Input features | 20 |
| Target | FloodProbability (continuous, range 0.285–0.725) |
| Missing values | None |
| Duplicates | None |

> **Dataset note:** FloodProbability in this dataset is synthetically generated as `sum(all_features) × 0.005`.  
> This explains the extremely high R² scores (LR: 1.0, MLP: ~0.9999).  
> The high R² is a property of the dataset, not a sign of overfitting or data leakage.  
> The MLP is retained as the final model because LinearRegression extrapolates dangerously beyond the training target range [0.285, 0.725] when input combinations fall outside the training distribution.

---

## 20 Input Features

| # | Feature | Dataset Range |
|---|---|---|
| 1 | MonsoonIntensity | 0 – 16 |
| 2 | TopographyDrainage | 0 – 18 |
| 3 | RiverManagement | 0 – 16 |
| 4 | Deforestation | 0 – 17 |
| 5 | Urbanization | 0 – 17 |
| 6 | ClimateChange | 0 – 17 |
| 7 | DamsQuality | 0 – 16 |
| 8 | Siltation | 0 – 16 |
| 9 | AgriculturalPractices | 0 – 16 |
| 10 | Encroachments | 0 – 18 |
| 11 | IneffectiveDisasterPreparedness | 0 – 16 |
| 12 | DrainageSystems | 0 – 17 |
| 13 | CoastalVulnerability | 0 – 17 |
| 14 | Landslides | 0 – 16 |
| 15 | Watersheds | 0 – 16 |
| 16 | DeterioratingInfrastructure | 0 – 17 |
| 17 | PopulationScore | 0 – 19 |
| 18 | WetlandLoss | 0 – 22 |
| 19 | InadequatePlanning | 0 – 16 |
| 20 | PoliticalFactors | 0 – 16 |

---

## ML Approach

| Stage | Details |
|---|---|
| Supervised Learning | Linear Regression, Decision Tree, Random Forest |
| Neural Network | MLPRegressor — hidden layers (128, 64, 32), ReLU, Adam, early stopping |
| Unsupervised Learning | K-Means Clustering + PCA (academic/internal only) |
| Hyperparameter Tuning | GridSearchCV (Decision Tree), 5-fold cross-validation |
| Final Model | MLPRegressor — selected for safe, bounded output behaviour |
| Evaluation | MAE, MSE, RMSE, R² on held-out 20% test set |

---

## Prediction Workflow

```
User Input (20 sliders, 0–100% severity)
        ↓
Input Validation (range check: 0–100%)
        ↓
Convert UI % → original dataset scale (per-feature min/max)
        ↓
Apply saved StandardScaler (fitted on training data only)
        ↓
MLPRegressor (final_model.pkl)
        ↓
Clamp output to [0.285, 0.725]
        ↓
FloodProbability (displayed as %)
        ↓
Risk Level + Risk Interpretation
        ↓
Personalized Preventive Safety Suggestions
```

---

## Risk Level Thresholds

| FloodProbability | Risk Level |
|---|---|
| < 0.45 | 🟢 LOW RISK |
| 0.45 – 0.55 | 🟠 MODERATE RISK |
| > 0.55 | 🔴 HIGH RISK |

*Based on actual target distribution terciles (range: 0.285–0.725, mean: 0.499, std: 0.050).*

---

## Preventive Safety Suggestions

After prediction, the app displays personalized suggestions based on which input factors the user rated as high severity (≥ 60%).  
Up to 5 of the most relevant suggestions are shown.

> These preventive actions can help reduce the potential impact of flooding.  
> They do not replace official government warnings or emergency services.

---

## Project Structure

```
Flood_Shield/
├── app.py                  # Streamlit frontend
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
│
├── src/
│   ├── utils.py            # Shared constants, paths, helpers, validation
│   ├── preprocessing.py    # Data loading, cleaning, train/test split, scaling
│   ├── regression.py       # Supervised learning (LR, DT, RF)
│   ├── ann.py              # MLPRegressor (ANN)
│   ├── clustering.py       # K-Means + PCA (academic use)
│   ├── optimization.py     # GridSearchCV + final model selection
│   └── predict.py          # Prediction pipeline (used by app.py)
│
├── data/
│   ├── raw/flood.csv       # Original dataset (do not modify)
│   └── processed/          # Cleaned/split data saved here after preprocessing
│
├── models/
│   ├── scaler.pkl          # StandardScaler fitted on training data only
│   ├── final_model.pkl     # Final selected MLPRegressor
│   └── *.pkl               # All individual trained model files
│
├── outputs/
│   ├── graphs/             # Training curves, comparison charts, PCA plots
│   └── reports/            # CSV metric reports for all models
│
├── docs/                   # Additional documentation
└── screenshots/            # App screenshots
```

---

## Installation

```bash
git clone https://github.com/your-username/flood-shield.git
cd flood-shield
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Train all models (run once, in order)

```bash
python src/regression.py
python src/ann.py
python src/clustering.py
python src/optimization.py
```

### Step 2 — Launch the Streamlit app

```bash
streamlit run app.py
```

---

## Data Leakage Audit

- Train/test split (80/20) is performed **before** fitting the StandardScaler.
- The scaler is fitted **only on training data** (`fit_transform` on train, `transform` on test).
- GridSearchCV uses cross-validation on training data only.
- Final evaluation is performed on the held-out test set.
- **No data leakage exists** in the current pipeline.

---

## Limitations

- Uses a **static, synthetic dataset**.  Does not use live weather data, river sensors, or real-time flood monitoring APIs.
- FloodProbability is mathematically derived from the sum of input features (`sum × 0.005`), not from real-world observations.
- Feature values are abstract severity scores, not physical measurements.
- The model output is clamped to [0.285, 0.725] — the range observed in the training data.  Extreme input combinations outside the training distribution will hit this ceiling.
- This application does **not** replace official government flood warnings or emergency services.

---

## Future Improvements

- Integrate real-time weather and river-level API data
- Replace synthetic dataset with real observational flood data
- Add SHAP-based feature explainability
- Deploy on Streamlit Cloud / AWS / GCP
- Add model retraining pipeline with periodic data updates
