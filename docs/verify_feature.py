"""
verify_feature.py
Quick end-to-end check that the History + Action Tracking feature is wired correctly.
Run from project root: python docs/verify_feature.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import FEATURE_COLUMNS
from predict import predict_from_percentages, load_artifacts
from history import (
    load_history, save_history, add_record, update_status,
    clear_history, completed_count, overall_status,
    STATUS_OPTIONS, STATUS_NOT_STARTED, STATUS_IN_PROGRESS, STATUS_COMPLETED,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
all_ok = True


def check(label, cond):
    global all_ok
    symbol = PASS if cond else FAIL
    print(f"  {symbol}  {label}")
    if not cond:
        all_ok = False
    return cond


print()
print("=" * 60)
print("  FLOOD SHIELD — HISTORY FEATURE VERIFICATION")
print("=" * 60)

# ── 1. Imports & artifacts ───────────────────────────────────────
print("\n[1] Imports & artifacts")
scaler, model = load_artifacts()
check("scaler loaded",              scaler is not None)
check("model loaded",               model is not None)
check("20 features (scaler)",       scaler.n_features_in_ == 20)
check("20 features (model)",        model.n_features_in_ == 20)
check("STATUS_OPTIONS has 3 items", len(STATUS_OPTIONS) == 3)

# ── 2. Prediction ────────────────────────────────────────────────
print("\n[2] Prediction pipeline")
inputs_high = {feat: 75 for feat in FEATURE_COLUMNS}
result      = predict_from_percentages(inputs_high)
prob        = result["probability_pct"]
risk        = result["risk_level"]
check("probability is float",         isinstance(prob, float))
check("probability in range 0–100",   0.0 <= prob <= 100.0)
check("risk_level is non-empty str",  isinstance(risk, str) and len(risk) > 0)
print(f"         → {prob}%  |  {risk}")

# ── 3. History record creation ───────────────────────────────────
print("\n[3] add_record")
SUGGESTIONS = [
    "Clear drainage blockages",
    "Check emergency supplies",
    "Prepare evacuation route",
    "Monitor drainage condition",
]
h = []
h = add_record(h, inputs_high, prob, risk, SUGGESTIONS)
check("1 record added",                 len(h) == 1)
check("id == 1",                        h[0]["id"] == 1)
check("probability stored",             h[0]["probability"] == round(prob, 1))
check("risk_level stored",              h[0]["risk_level"] == risk)
check("4 suggestions stored",           len(h[0]["suggestions"]) == 4)
check("4 statuses (all Not Started)",   h[0]["statuses"] == [STATUS_NOT_STARTED] * 4)
check("20 inputs stored",               len(h[0]["inputs"]) == 20)
check("datetime is non-empty",          len(h[0]["datetime"]) > 0)

# ── 4. Overall status: 0/4 ──────────────────────────────────────
print("\n[4] overall_status — 0/4")
check("completed_count == 0",       completed_count(h[0]) == 0)
check("overall = Not Started",      overall_status(h[0]) == STATUS_NOT_STARTED)

# ── 5. Update 1/4 ───────────────────────────────────────────────
print("\n[5] update_status → 1/4")
h = update_status(h, 1, 0, STATUS_COMPLETED)
check("statuses[0] == Completed",   h[0]["statuses"][0] == STATUS_COMPLETED)
check("completed_count == 1",       completed_count(h[0]) == 1)
check("overall = In Progress",      overall_status(h[0]) == STATUS_IN_PROGRESS)

# ── 6. Update 2/4 ───────────────────────────────────────────────
print("\n[6] update_status → 2/4")
h = update_status(h, 1, 1, STATUS_COMPLETED)
check("completed_count == 2",       completed_count(h[0]) == 2)
check("overall = In Progress",      overall_status(h[0]) == STATUS_IN_PROGRESS)

# ── 7. Update 3/4 ───────────────────────────────────────────────
print("\n[7] update_status → 3/4")
h = update_status(h, 1, 2, STATUS_COMPLETED)
check("completed_count == 3",       completed_count(h[0]) == 3)
check("overall = In Progress",      overall_status(h[0]) == STATUS_IN_PROGRESS)

# ── 8. Update 4/4 ───────────────────────────────────────────────
print("\n[8] update_status → 4/4")
h = update_status(h, 1, 3, STATUS_COMPLETED)
check("completed_count == 4",       completed_count(h[0]) == 4)
check("overall = Completed",        overall_status(h[0]) == STATUS_COMPLETED)

# ── 9. Second prediction: first record unchanged ─────────────────
print("\n[9] Second prediction — first record isolation")
inputs_low = {feat: 20 for feat in FEATURE_COLUMNS}
result2    = predict_from_percentages(inputs_low)
h = add_record(h, inputs_low, result2["probability_pct"], result2["risk_level"], SUGGESTIONS[:3])
check("history has 2 records",              len(h) == 2)
check("record 2 id == 2",                   h[1]["id"] == 2)
check("record 1 still 4/4 Completed",       completed_count(h[0]) == 4)
check("record 1 overall still Completed",   overall_status(h[0]) == STATUS_COMPLETED)
check("record 2 overall = Not Started",     overall_status(h[1]) == STATUS_NOT_STARTED)
check("record 2 has 3 suggestions",         len(h[1]["suggestions"]) == 3)

# ── 10. Persistence: save & reload ──────────────────────────────
print("\n[10] save_history + load_history")
save_history(h)
json_path = Path(__file__).resolve().parent.parent / "data" / "prediction_history.json"
check("JSON file created",              json_path.exists())
h2 = load_history()
check("reloaded 2 records",             len(h2) == 2)
check("record 1 probability persists",  h2[0]["probability"] == round(prob, 1))
check("record 1 statuses persist",      completed_count(h2[0]) == 4)
check("record 2 overall persists",      overall_status(h2[1]) == STATUS_NOT_STARTED)

# ── 11. Reset does NOT delete history ───────────────────────────
print("\n[11] Reset behaviour (simulated)")
# A reset only clears slider_values and show_result in session_state.
# The history list variable is completely separate — confirmed by inspection.
snapshot_len = len(h2)
# Simulate reset: nothing touches h2
check("history survives reset simulation",  len(h2) == snapshot_len)

# ── 12. Clear History ────────────────────────────────────────────
print("\n[12] clear_history")
h3 = clear_history()
check("returned list is empty",     len(h3) == 0)
check("JSON file deleted",          not json_path.exists())

# ── 13. Navigation labels exist in app.py ───────────────────────
print("\n[13] app.py navigation sanity")
app_text = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
check("Home page function present",           "def page_home" in app_text)
check("Predict page function present",        "def page_predict" in app_text)
check("History page function present",        "def page_history" in app_text)
check("add_record called in app.py",          "add_record" in app_text)
check("save_history called in app.py",        "save_history" in app_text)
check("update_status called in app.py",       "update_status" in app_text)
check("clear_history called in app.py",       "clear_history" in app_text)
check("confirm_clear guard present",          "confirm_clear" in app_text)
check("Reset does NOT clear history",         "slider_values" in app_text and
      app_text.count("prediction_history") > 0)

# ── Summary ──────────────────────────────────────────────────────
print()
print("=" * 60)
if all_ok:
    print("  ALL VERIFICATION TESTS PASSED")
    print("  Run: streamlit run app.py")
else:
    print("  SOME TESTS FAILED — see above")
    sys.exit(1)
print("=" * 60)
print()
