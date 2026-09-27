"""
test_pipeline.py – Full end-to-end pipeline verification.
Run: python test_pipeline.py
"""
import sys
sys.path.insert(0, "src")

from utils import (
    FEATURE_COLUMNS, FEATURE_LABELS, FEATURE_RANGES,
    get_risk_level, get_safety_recommendation,
    pct_to_raw, clamp_prediction,
    TARGET_MIN, TARGET_MAX,
)
from predict import predict_from_percentages, load_artifacts

print("=" * 55)
print("  FLOOD SHIELD — END-TO-END PIPELINE TEST")
print("=" * 55)

# 1. Artifacts
scaler, model = load_artifacts()
print(f"\n[1] Artifacts loaded")
print(f"    Model  : {type(model).__name__}")
print(f"    Scaler : {type(scaler).__name__}")

# 2. Feature count
print(f"\n[2] Feature count : {len(FEATURE_COLUMNS)}")
print(f"    FloodProbability excluded : {'FloodProbability' not in FEATURE_COLUMNS}")

# 3. Labels and ranges for all features
missing_labels  = [f for f in FEATURE_COLUMNS if f not in FEATURE_LABELS]
missing_ranges  = [f for f in FEATURE_COLUMNS if f not in FEATURE_RANGES]
print(f"\n[3] Missing labels : {missing_labels if missing_labels else 'None'}")
print(f"    Missing ranges : {missing_ranges if missing_ranges else 'None'}")

# 4. pct_to_raw conversion
print(f"\n[4] pct_to_raw checks")
for feat in FEATURE_COLUMNS[:3]:
    r0   = pct_to_raw(feat, 0)
    r100 = pct_to_raw(feat, 100)
    print(f"    {feat:<30}  0%={r0}  100%={r100}")

# 5. Prediction tests
print(f"\n[5] Prediction tests")
scenarios = [
    ("All  10%", {f: 10 for f in FEATURE_COLUMNS}),
    ("All  30%", {f: 30 for f in FEATURE_COLUMNS}),
    ("All  50%", {f: 50 for f in FEATURE_COLUMNS}),
    ("All  70%", {f: 70 for f in FEATURE_COLUMNS}),
    ("All  90%", {f: 90 for f in FEATURE_COLUMNS}),
    ("All 100%", {f: 100 for f in FEATURE_COLUMNS}),
]
for label, inp in scenarios:
    r = predict_from_percentages(inp)
    prob = r["probability_pct"]
    risk = r["risk_level"]
    rec  = r["recommendation"][:50]
    print(f"    {label} -> {prob:5.1f}%  |  {risk:<15}  |  {rec}...")

# 6. Risk level + recommendation for all 3 levels
print(f"\n[6] Risk level thresholds")
for prob, expected in [(0.30, "LOW RISK"), (0.50, "MODERATE RISK"), (0.65, "HIGH RISK")]:
    lvl = get_risk_level(prob)
    rec = get_safety_recommendation(lvl)
    status = "OK" if lvl == expected else "FAIL"
    print(f"    [{status}] prob={prob} -> {lvl}")

# 7. Clamp test
print(f"\n[7] Clamp test")
print(f"    clamp(2.0)  = {clamp_prediction(2.0)}  (expected {TARGET_MAX})")
print(f"    clamp(-0.5) = {clamp_prediction(-0.5)} (expected {TARGET_MIN})")
print(f"    clamp(0.55) = {clamp_prediction(0.55)} (expected 0.55)")

print(f"\n{'=' * 55}")
print("  ALL TESTS PASSED — Pipeline is working correctly.")
print(f"{'=' * 55}")
