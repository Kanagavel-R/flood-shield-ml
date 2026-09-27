"""
utils.py – Shared constants, paths, and helper functions for Flood Shield.
"""

from pathlib import Path
from typing import Dict, List

# ── Project root (one level above src/) ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = ROOT / "data" / "raw" / "flood.csv"
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "flood_processed.csv"
MODELS_DIR          = ROOT / "models"
GRAPHS_DIR          = ROOT / "outputs" / "graphs"
REPORTS_DIR         = ROOT / "outputs" / "reports"

SCALER_PATH         = MODELS_DIR / "scaler.pkl"
FINAL_MODEL_PATH    = MODELS_DIR / "final_model.pkl"

# ── Feature definitions ──────────────────────────────────────────────────────
# 20 input features — exact column order used throughout the project
# (training, scaling, and prediction must all use this same order)
FEATURE_COLUMNS = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]

TARGET_COLUMN = "FloodProbability"

# Actual min/max per feature — verified from dataset inspection
FEATURE_RANGES = {
    "MonsoonIntensity":               {"min": 0, "max": 16},
    "TopographyDrainage":             {"min": 0, "max": 18},
    "RiverManagement":                {"min": 0, "max": 16},
    "Deforestation":                  {"min": 0, "max": 17},
    "Urbanization":                   {"min": 0, "max": 17},
    "ClimateChange":                  {"min": 0, "max": 17},
    "DamsQuality":                    {"min": 0, "max": 16},
    "Siltation":                      {"min": 0, "max": 16},
    "AgriculturalPractices":          {"min": 0, "max": 16},
    "Encroachments":                  {"min": 0, "max": 18},
    "IneffectiveDisasterPreparedness":{"min": 0, "max": 16},
    "DrainageSystems":                {"min": 0, "max": 17},
    "CoastalVulnerability":           {"min": 0, "max": 17},
    "Landslides":                     {"min": 0, "max": 16},
    "Watersheds":                     {"min": 0, "max": 16},
    "DeterioratingInfrastructure":    {"min": 0, "max": 17},
    "PopulationScore":                {"min": 0, "max": 19},
    "WetlandLoss":                    {"min": 0, "max": 22},
    "InadequatePlanning":             {"min": 0, "max": 16},
    "PoliticalFactors":               {"min": 0, "max": 16},
}

# UI display labels for each feature
FEATURE_LABELS = {
    "MonsoonIntensity":               "Monsoon Intensity",
    "TopographyDrainage":             "Topography Drainage",
    "RiverManagement":                "River Management",
    "Deforestation":                  "Deforestation",
    "Urbanization":                   "Urbanization",
    "ClimateChange":                  "Climate Change",
    "DamsQuality":                    "Dams Quality",
    "Siltation":                      "Siltation",
    "AgriculturalPractices":          "Agricultural Practices",
    "Encroachments":                  "Encroachments",
    "IneffectiveDisasterPreparedness":"Ineffective Disaster Preparedness",
    "DrainageSystems":                "Drainage Systems",
    "CoastalVulnerability":           "Coastal Vulnerability",
    "Landslides":                     "Landslides",
    "Watersheds":                     "Watersheds",
    "DeterioratingInfrastructure":    "Deteriorating Infrastructure",
    "PopulationScore":                "Population Score",
    "WetlandLoss":                    "Wetland Loss",
    "InadequatePlanning":             "Inadequate Planning",
    "PoliticalFactors":               "Political Factors",
}

# ── Conversion helpers ───────────────────────────────────────────────────────

def raw_to_pct(feature: str, raw_value: float) -> float:
    """Convert a raw dataset value to a 0–100 UI percentage."""
    fmin = FEATURE_RANGES[feature]["min"]
    fmax = FEATURE_RANGES[feature]["max"]
    if fmax == fmin:
        return 0.0
    return round(((raw_value - fmin) / (fmax - fmin)) * 100, 2)


def pct_to_raw(feature: str, pct: float) -> float:
    """Convert a 0–100 UI percentage back to the original dataset scale."""
    fmin = FEATURE_RANGES[feature]["min"]
    fmax = FEATURE_RANGES[feature]["max"]
    return fmin + (pct / 100.0) * (fmax - fmin)


# ── Target range (verified from dataset) ─────────────────────────────────────
# The dataset target is EXACTLY:  FloodProbability = sum(all_features) * 0.005
# This is a synthetically generated dataset; the target is a deterministic
# linear combination of the 20 features.  This explains R²=1.0 for Linear
# Regression and R²≈0.9999 for MLPRegressor.
#
# Important implication for the UI:
#   - Each feature ranges from 0–16/17/18/19/22 in the dataset.
#   - The training data has feature sums in [57, 145], giving targets [0.285, 0.725].
#   - When a user sets all sliders to 50%, the combined raw sum (~169.5) is
#     OUTSIDE the training distribution, so the model extrapolates beyond 0.725.
#   - The clamp below is essential for safe, bounded output.
TARGET_MIN = 0.285
TARGET_MAX = 0.725


# ── Risk classification thresholds ───────────────────────────────────────────
# Defined once here; imported everywhere — never duplicated.
# Based on actual target distribution terciles (mean=0.499, std=0.050):
#   Low      < 0.45
#   Moderate 0.45 – 0.55
#   High     > 0.55
RISK_LOW_THRESHOLD      = 0.45
RISK_HIGH_THRESHOLD     = 0.55


def clamp_prediction(probability: float) -> float:
    """
    Clamp the model output to the actual target range seen in the dataset.

    MLP and linear models extrapolate beyond [TARGET_MIN, TARGET_MAX] when
    feature combinations fall outside the training distribution (e.g. all
    sliders set to 50% produces a feature sum of ~169, but the training data
    only contains sums in [57, 145]).  Clamping is the correct and safe fix.
    """
    return max(TARGET_MIN, min(TARGET_MAX, probability))


# ── Risk classification ───────────────────────────────────────────────────────

def get_risk_level(probability: float) -> str:
    """Return risk level string based on predicted FloodProbability."""
    if probability < RISK_LOW_THRESHOLD:
        return "LOW RISK"
    elif probability <= RISK_HIGH_THRESHOLD:
        return "MODERATE RISK"
    else:
        return "HIGH RISK"


def get_safety_recommendation(risk_level: str) -> str:
    """Return safety recommendation based on risk level."""
    recommendations = {
        "LOW RISK": (
            "Current conditions indicate a relatively low flood risk. "
            "Continue monitoring local weather and water conditions."
        ),
        "MODERATE RISK": (
            "Moderate flood risk detected. Stay alert and monitor local "
            "conditions and official advisories."
        ),
        "HIGH RISK": (
            "High flood risk detected. Follow local authority warnings "
            "and move to a safe area if evacuation is advised."
        ),
    }
    return recommendations.get(risk_level, "")


# ── Input validation ──────────────────────────────────────────────────────────

def validate_pct_inputs(pct_values: Dict[str, float]) -> List[str]:
    """
    Validate that all 20 feature inputs are present and within [0, 100].

    Args:
        pct_values: dict mapping feature name to 0–100 percentage value.

    Returns:
        List of error message strings.  Empty list means all inputs are valid.
    """
    errors: List[str] = []

    # Check all expected features are present
    missing = [f for f in FEATURE_COLUMNS if f not in pct_values]
    if missing:
        errors.append(f"Missing input features: {', '.join(missing)}")
        return errors  # cannot range-check missing features

    for feat in FEATURE_COLUMNS:
        val = pct_values[feat]
        try:
            val = float(val)
        except (TypeError, ValueError):
            errors.append(f"{FEATURE_LABELS[feat]}: value must be numeric (got {val!r})")
            continue
        if not (0.0 <= val <= 100.0):
            errors.append(
                f"{FEATURE_LABELS[feat]}: value {val} is outside the valid range [0, 100]"
            )

    return errors
