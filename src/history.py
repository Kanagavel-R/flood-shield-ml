"""
history.py – Prediction History persistence for Flood Shield.

Storage strategy:
  • Session-level  → st.session_state["prediction_history"]  (list of dicts)
  • Cross-session  → data/prediction_history.json            (auto-save on change)

Each history record schema:
{
    "id":          int,               # sequential, 1-based
    "datetime":    str,               # "23 Sep 2026, 05:30 PM"
    "inputs":      dict,              # {feature: pct_value, ...}  (20 items)
    "probability": float,             # e.g. 68.4
    "risk_level":  str,               # "HIGH RISK" | "MODERATE RISK" | "LOW RISK"
    "suggestions": list[str],         # preventive suggestion strings
    "statuses":    list[str],         # per-suggestion: "Not Started"|"In Progress"|"Completed"
}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# ── Storage path ──────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / "data" / "prediction_history.json"

# ── Allowed statuses ──────────────────────────────────────────────────────────
STATUS_OPTIONS    = ["⏳ Not Started", "🔄 In Progress", "✅ Completed"]
STATUS_NOT_STARTED = "⏳ Not Started"
STATUS_IN_PROGRESS = "🔄 In Progress"
STATUS_COMPLETED   = "✅ Completed"


# ── JSON persistence helpers ──────────────────────────────────────────────────

def load_history() -> List[Dict[str, Any]]:
    """
    Load prediction history from the JSON file.
    Returns an empty list if the file does not exist or is corrupt.
    """
    if not HISTORY_PATH.exists():
        return []
    try:
        text = HISTORY_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def save_history(history: List[Dict[str, Any]]) -> None:
    """
    Persist prediction history list to the JSON file.
    Creates parent directory if needed. Silently ignores write errors
    (session state remains the source of truth while the app is running).
    """
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


# ── Record helpers ────────────────────────────────────────────────────────────

def build_record(
    history:     List[Dict[str, Any]],
    inputs:      Dict[str, float],
    probability: float,
    risk_level:  str,
    suggestions: List[str],
) -> Dict[str, Any]:
    """
    Create a new prediction history record.

    Args:
        history:     Existing history list (used to derive next sequential ID).
        inputs:      Dict of {feature: pct_value} (20 items).
        probability: Flood probability percentage (0–100 float).
        risk_level:  Risk level string.
        suggestions: List of preventive suggestion strings.

    Returns:
        A new record dict (not yet appended to history).
    """
    next_id   = max((r["id"] for r in history), default=0) + 1
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

    return {
        "id":          next_id,
        "datetime":    timestamp,
        "inputs":      dict(inputs),          # shallow copy – all values are scalars
        "probability": round(float(probability), 1),
        "risk_level":  risk_level,
        "suggestions": list(suggestions),
        "statuses":    [STATUS_NOT_STARTED] * len(suggestions),
    }


def add_record(
    history:     List[Dict[str, Any]],
    inputs:      Dict[str, float],
    probability: float,
    risk_level:  str,
    suggestions: List[str],
) -> List[Dict[str, Any]]:
    """
    Add a new prediction record to history and return the updated list.
    The caller is responsible for updating session_state and calling save_history().
    """
    record = build_record(history, inputs, probability, risk_level, suggestions)
    return history + [record]


# ── Action-status helpers ─────────────────────────────────────────────────────

def completed_count(record: Dict[str, Any]) -> int:
    """Return the number of suggestions marked as Completed."""
    return sum(1 for s in record["statuses"] if s == STATUS_COMPLETED)


def overall_status(record: Dict[str, Any]) -> str:
    """
    Calculate overall action status from individual suggestion statuses.

    Rule:
      0 completed           → Not Started
      1..N-1 completed      → In Progress
      all completed         → Completed
    """
    total     = len(record["statuses"])
    completed = completed_count(record)

    if total == 0:
        return STATUS_NOT_STARTED
    if completed == 0:
        return STATUS_NOT_STARTED
    if completed == total:
        return STATUS_COMPLETED
    return STATUS_IN_PROGRESS


def update_status(
    history:    List[Dict[str, Any]],
    record_id:  int,
    action_idx: int,
    new_status: str,
) -> List[Dict[str, Any]]:
    """
    Return a new history list with the status of one action updated.
    Does NOT mutate the original list.

    Args:
        history:    Current history list.
        record_id:  The `id` field of the target record.
        action_idx: 0-based index into the record's `statuses` list.
        new_status: One of STATUS_OPTIONS.

    Returns:
        Updated history list.
    """
    updated = []
    for rec in history:
        if rec["id"] == record_id:
            new_rec           = dict(rec)
            new_rec["statuses"] = list(rec["statuses"])
            if 0 <= action_idx < len(new_rec["statuses"]):
                new_rec["statuses"][action_idx] = new_status
            updated.append(new_rec)
        else:
            updated.append(rec)
    return updated


def clear_history() -> List[Dict[str, Any]]:
    """
    Clear all prediction history and remove the JSON file.
    Returns an empty list.
    """
    try:
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
    except OSError:
        pass
    return []
