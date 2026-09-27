"""
app.py – Flood Shield: Intelligent Flood Risk Prediction System
Run: streamlit run app.py

Pages:
  🏠 Home              – welcome screen
  🔮 Predict Risk      – 20-input prediction (unchanged logic)
  📋 Prediction History – history + action tracking
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils import FEATURE_COLUMNS, FEATURE_LABELS, FEATURE_RANGES, validate_pct_inputs
from predict import predict_from_percentages, load_artifacts
from history import (
    load_history,
    save_history,
    add_record,
    update_status,
    clear_history,
    completed_count,
    overall_status,
    STATUS_OPTIONS,
    STATUS_NOT_STARTED,
    STATUS_COMPLETED,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Shield",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  .stApp { background-color: #f0f4f8; }

  /* ── Brand header ── */
  .fs-header { text-align:center; padding:2rem 1rem 0.5rem 1rem; }
  .fs-title  { font-size:2.6rem; font-weight:800; color:#1a3c5e; letter-spacing:1px; margin-bottom:0; }
  .fs-subtitle { font-size:1.05rem; color:#4a6fa5; font-weight:500; margin-top:0.2rem; }

  /* ── Divider ── */
  .fs-divider { border:none; border-top:2px solid #d0dce8; margin:0.8rem 0; }

  /* ── Nav pills ── */
  .nav-container {
    display:flex; gap:0.5rem; justify-content:center;
    padding:0.5rem 0 1.2rem 0;
  }
  .nav-pill {
    display:inline-block; padding:0.45rem 1.2rem;
    border-radius:20px; font-size:0.9rem; font-weight:600;
    cursor:pointer; text-decoration:none; border:2px solid #d0dce8;
    background:#ffffff; color:#4a6fa5; transition:all 0.15s ease;
  }
  .nav-pill:hover  { background:#e8f0fb; border-color:#4a6fa5; }
  .nav-pill.active { background:#1a3c5e; color:#ffffff; border-color:#1a3c5e; }

  /* ── Feature sliders ── */
  .feat-label { font-size:0.78rem; font-weight:700; color:#1a3c5e;
                letter-spacing:0.8px; text-transform:uppercase; margin-bottom:0; }
  .feat-hint  { font-size:0.72rem; color:#8a9bb0; margin-top:-4px; margin-bottom:2px; }

  /* ── Result box ── */
  .result-box { border-radius:12px; padding:2rem; text-align:center; margin:1.5rem 0; }
  .result-title { font-size:0.85rem; font-weight:700; letter-spacing:1px;
                  text-transform:uppercase; margin-bottom:0.3rem; }
  .result-prob  { font-size:3.2rem; font-weight:900; margin:0.2rem 0; }
  .result-risk  { font-size:1.4rem; font-weight:800; letter-spacing:1px; margin-top:0.2rem; }

  /* ── History cards ── */
  .hist-card {
    background:#ffffff; border:1.5px solid #d0dce8; border-radius:12px;
    padding:1.2rem 1.4rem; margin-bottom:1rem;
    box-shadow:0 1px 4px rgba(0,0,0,0.06);
  }
  .hist-card-id   { font-size:0.82rem; font-weight:700; color:#6b7c93;
                    letter-spacing:0.5px; text-transform:uppercase; }
  .hist-card-prob { font-size:1.8rem; font-weight:900; }
  .hist-card-risk { font-size:1rem; font-weight:700; }
  .hist-card-date { font-size:0.82rem; color:#8a9bb0; margin-bottom:0.6rem; }
  .hist-card-prog { font-size:0.88rem; color:#4a6fa5; font-weight:600; margin-top:0.4rem; }

  /* ── Progress badge ── */
  .badge-ns   { background:#f0f4f8; color:#6b7c93;  border:1px solid #d0dce8; }
  .badge-ip   { background:#fff3cd; color:#856404;  border:1px solid #f0c040; }
  .badge-done { background:#d4edda; color:#155724;  border:1px solid #b2dfbb; }
  .status-badge {
    display:inline-block; padding:0.2rem 0.7rem; border-radius:12px;
    font-size:0.82rem; font-weight:700; margin-top:0.3rem;
  }

  /* ── Action item rows ── */
  .action-row {
    background:#f7faff; border:1px solid #d0dce8; border-radius:8px;
    padding:0.7rem 1rem; margin-bottom:0.5rem;
  }
  .action-text { font-size:0.92rem; color:#2d3e50; font-weight:500; margin-bottom:0.3rem; }

  /* ── Input rows ── */
  .input-row { display:flex; justify-content:space-between;
               font-size:0.82rem; color:#4a6fa5; margin:2px 0; }
  .input-label { font-weight:600; }
  .input-value { color:#1a3c5e; font-weight:700; }

  /* ── Section heading ── */
  .sec-heading {
    font-size:0.78rem; font-weight:700; letter-spacing:1px;
    text-transform:uppercase; color:#1a3c5e; margin-bottom:0.6rem;
    border-bottom:2px solid #d0dce8; padding-bottom:0.3rem;
  }

  /* ── Buttons ── */
  div.stButton > button {
    width:100%; padding:0.65rem 1rem; font-size:0.95rem;
    font-weight:700; border-radius:8px;
  }

  /* ── Home hero ── */
  .home-hero {
    text-align:center; padding:2rem 1rem;
    background:linear-gradient(135deg,#1a3c5e 0%,#2e6da4 100%);
    border-radius:16px; color:#ffffff; margin:1rem 0 1.5rem 0;
  }
  .home-hero h2 { font-size:1.5rem; font-weight:800; margin-bottom:0.4rem; }
  .home-hero p  { font-size:0.95rem; opacity:0.88; margin:0; line-height:1.6; }
  .home-step {
    background:#ffffff; border:1.5px solid #d0dce8; border-radius:10px;
    padding:1rem 1.2rem; text-align:center;
    box-shadow:0 1px 4px rgba(0,0,0,0.06);
  }
  .home-step-num  { font-size:1.4rem; font-weight:900; color:#1a3c5e; }
  .home-step-text { font-size:0.88rem; color:#4a6fa5; font-weight:600; margin-top:0.2rem; }
</style>
""", unsafe_allow_html=True)


# ── Load ML artifacts (cached) ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_artifacts():
    try:
        scaler, model = load_artifacts()
        return scaler, model, None
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Unexpected error loading model: {e}"


scaler, model, load_error = get_artifacts()

if load_error:
    st.error(f"⚠️ Model not ready: {load_error}")
    st.info("Please run the training pipeline first:\n```\npython src/regression.py\npython src/ann.py\npython src/optimization.py\n```")
    st.stop()

if scaler is not None and model is not None:
    expected = len(FEATURE_COLUMNS)
    if scaler.n_features_in_ != expected or model.n_features_in_ != expected:
        st.error(
            f"⚠️ Feature count mismatch: FEATURE_COLUMNS has {expected} features, "
            f"but scaler expects {scaler.n_features_in_} and model expects "
            f"{model.n_features_in_}. Retrain the pipeline."
        )
        st.stop()


# ── Session-state initialisation ─────────────────────────────────────────────
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "result" not in st.session_state:
    st.session_state.result = None
if "slider_values" not in st.session_state:
    st.session_state.slider_values = {feat: 50 for feat in FEATURE_COLUMNS}
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = load_history()
if "history_detail_id" not in st.session_state:
    st.session_state.history_detail_id = None
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False


# ── Preventive suggestions ────────────────────────────────────────────────────
SUGGESTIONS_MAP = {
    "MonsoonIntensity": [
        "Monitor heavy rainfall forecasts and official weather alerts regularly.",
        "Keep emergency supplies (water, food, torch, medicines) ready before severe rainfall seasons.",
        "Secure loose outdoor items and check roof drainage before monsoon season.",
    ],
    "TopographyDrainage": [
        "Identify low-lying areas near your location and plan evacuation routes in advance.",
        "Avoid settling or storing valuables in flood-prone low-lying zones.",
    ],
    "RiverManagement": [
        "Stay informed about local river water levels and flood warnings from authorities.",
        "Maintain and strengthen river embankments and flood-control infrastructure.",
        "Report damaged river barriers or embankments to local authorities promptly.",
    ],
    "Deforestation": [
        "Promote tree planting and vegetation restoration in vulnerable and upstream areas.",
        "Support local reforestation initiatives to improve natural water absorption.",
    ],
    "Urbanization": [
        "Ensure paved and concrete surfaces around your property have adequate drainage outlets.",
        "Advocate for green spaces and permeable surfaces in urban planning.",
    ],
    "ClimateChange": [
        "Stay updated on long-term climate and flood-risk advisories for your region.",
        "Support and follow community-level climate resilience and flood preparedness plans.",
    ],
    "DamsQuality": [
        "Stay informed about the maintenance status of nearby dams and reservoirs.",
        "Follow evacuation instructions immediately if dam safety warnings are issued.",
    ],
    "Siltation": [
        "Ensure local water channels and rivers are not blocked by silt or debris.",
        "Support regular dredging and desilting of water bodies in your area.",
    ],
    "AgriculturalPractices": [
        "Use contour farming and bunding techniques to slow surface water runoff.",
        "Avoid excessive irrigation that may contribute to waterlogging and runoff.",
    ],
    "Encroachments": [
        "Avoid construction or encroachment on natural drainage channels and floodplains.",
        "Report illegal encroachments on waterways to local municipal authorities.",
    ],
    "IneffectiveDisasterPreparedness": [
        "Prepare a household emergency plan including evacuation routes and contact numbers.",
        "Participate in local flood drills and community disaster preparedness programs.",
        "Keep emergency documents (ID, insurance) in a waterproof bag.",
    ],
    "DrainageSystems": [
        "Keep drainage channels, gutters, and stormwater paths free of waste and blockages.",
        "Report blocked or overflowing drains to municipal authorities before the rainy season.",
        "Ensure your property's drainage is connected properly to the main stormwater system.",
    ],
    "CoastalVulnerability": [
        "Monitor coastal flood and storm surge warnings from meteorological departments.",
        "Prepare for possible evacuation if you live in low-lying coastal areas.",
        "Reinforce coastal structures and check for erosion around your property.",
    ],
    "Landslides": [
        "Avoid construction on steep slopes and unstable hillside areas.",
        "Watch for warning signs of landslides: cracks in ground, tilting trees, unusual water flow.",
        "Follow evacuation orders immediately if landslide risk warnings are issued.",
    ],
    "Watersheds": [
        "Protect upstream watershed areas from deforestation and overgrazing.",
        "Support watershed management programs that maintain natural water flow regulation.",
    ],
    "DeterioratingInfrastructure": [
        "Inspect and maintain flood walls, culverts, and drainage infrastructure regularly.",
        "Report deteriorating roads, bridges, or embankments to relevant authorities.",
        "Prioritize essential repairs in flood-prone locations before the rainy season.",
    ],
    "PopulationScore": [
        "Ensure emergency evacuation plans account for the local population density.",
        "Coordinate with community leaders on shelter and relief resource availability.",
    ],
    "WetlandLoss": [
        "Protect and restore wetlands and natural water-retention areas in your region.",
        "Avoid draining or filling wetlands as they act as natural flood buffers.",
    ],
    "InadequatePlanning": [
        "Prepare and rehearse household evacuation routes and emergency contact plans.",
        "Work with local authorities to ensure flood-risk zoning is respected in development.",
        "Keep a go-bag ready with essentials in case of sudden evacuation.",
    ],
    "PoliticalFactors": [
        "Engage with local government bodies to prioritize flood preparedness and infrastructure.",
        "Stay informed about flood management policies and disaster response plans in your area.",
    ],
}

HIGH_THRESHOLD = 60


def get_preventive_suggestions(slider_values: dict, max_suggestions: int = 5) -> list:
    high_features = sorted(
        [(feat, pct) for feat, pct in slider_values.items() if pct >= HIGH_THRESHOLD],
        key=lambda x: x[1],
        reverse=True,
    )
    suggestions = []
    seen = set()
    for feat, _ in high_features:
        for suggestion in SUGGESTIONS_MAP.get(feat, []):
            if suggestion not in seen:
                seen.add(suggestion)
                suggestions.append(suggestion)
            if len(suggestions) >= max_suggestions:
                return suggestions
    if not suggestions:
        suggestions = [
            "Continue monitoring local weather and flood advisories.",
            "Keep emergency supplies accessible as a precaution.",
            "Ensure drainage around your property is clear and functional.",
        ]
    return suggestions[:max_suggestions]


# ── Shared brand header ───────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="fs-header">
        <div class="fs-title">🌊 FLOOD SHIELD</div>
        <div class="fs-subtitle">Intelligent Flood Risk Prediction System</div>
    </div>
    """, unsafe_allow_html=True)


# ── Navigation bar ────────────────────────────────────────────────────────────
def render_nav():
    pages = [
        ("🏠 Home",               "Home"),
        ("🔮 Predict Risk",        "Predict Risk"),
        ("📋 Prediction History",  "Prediction History"),
    ]
    cols = st.columns(len(pages))
    for col, (label, key) in zip(cols, pages):
        with col:
            is_active = st.session_state.active_page == key
            btn_type  = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{key}", use_container_width=True, type=btn_type):
                st.session_state.active_page      = key
                st.session_state.history_detail_id = None
                st.session_state.confirm_clear     = False
                st.rerun()
    st.markdown("<hr class='fs-divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class="home-hero">
        <h2>Predict. Prepare. Protect.</h2>
        <p>
            Flood Shield uses machine learning to assess flood risk from<br>
            20 environmental and infrastructure factors — instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1", "Enter 20 risk factors"),
        ("2", "Get flood probability"),
        ("3", "Review suggestions"),
        ("4", "Track your actions"),
    ]
    for col, (num, text) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class="home-step">
                <div class="home-step-num">{num}</div>
                <div class="home-step-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        if st.button("🔮 Start Prediction", use_container_width=True, type="primary"):
            st.session_state.active_page = "Predict Risk"
            st.rerun()
    with col_r:
        history = st.session_state.prediction_history
        n       = len(history)
        label   = f"📋 View History  ({n} record{'s' if n != 1 else ''})"
        if st.button(label, use_container_width=True):
            st.session_state.active_page      = "Prediction History"
            st.session_state.history_detail_id = None
            st.rerun()

    if len(history) > 0:
        st.markdown("<hr class='fs-divider'>", unsafe_allow_html=True)
        st.markdown("#### Recent Prediction")
        last = history[-1]
        _render_mini_card(last)


def _render_mini_card(rec):
    """Small summary card used on the Home page."""
    s     = _risk_style(rec["risk_level"])
    done  = completed_count(rec)
    total = len(rec["suggestions"])
    ov    = overall_status(rec)
    st.markdown(f"""
    <div class="hist-card">
        <div class="hist-card-id">Prediction #{rec['id']:02d}</div>
        <div class="hist-card-date">📅 {rec['datetime']}</div>
        <div class="hist-card-prob" style="color:{s['text']};">{rec['probability']}%</div>
        <div class="hist-card-risk" style="color:{s['text']};">{s['icon']} {rec['risk_level']}</div>
        <div class="hist-card-prog">Preventive Actions: ✅ {done} / {total} Completed &nbsp;|&nbsp; {ov}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT RISK  (original logic preserved 100%)
# ─────────────────────────────────────────────────────────────────────────────
def page_predict():
    if not st.session_state.show_result:
        _render_input_form()
    else:
        _render_result()


def _render_input_form():
    st.markdown("#### Input Risk Factors")
    st.caption("Adjust each slider to reflect the current severity of each factor (0% = minimal, 100% = maximum).")
    st.markdown("")

    pct_inputs = {}
    for feat in FEATURE_COLUMNS:
        label = FEATURE_LABELS[feat].upper()
        st.markdown(f'<div class="feat-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="feat-hint">Normalized factor severity</div>', unsafe_allow_html=True)

        col_l, col_s, col_r = st.columns([1, 10, 1])
        with col_l:
            st.markdown(
                "<div style='text-align:right;font-size:0.75rem;color:#8a9bb0;padding-top:8px'>Low</div>",
                unsafe_allow_html=True,
            )
        with col_s:
            val = st.slider(
                label=feat,
                min_value=0,
                max_value=100,
                value=st.session_state.slider_values[feat],
                step=1,
                label_visibility="collapsed",
                key=f"slider_{feat}",
            )
            pct_inputs[feat] = val
        with col_r:
            st.markdown(
                "<div style='font-size:0.75rem;color:#8a9bb0;padding-top:8px'>High</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='text-align:center;font-size:0.85rem;color:#4a6fa5;font-weight:600;"
            f"margin-top:-8px;margin-bottom:12px'>{val}%</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='fs-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        predict_btn = st.button("🔍 PREDICT FLOOD RISK", use_container_width=True, type="primary")
    with col2:
        reset_btn = st.button("↺ RESET", use_container_width=True)

    # RESET: clears inputs only — does NOT touch history
    if reset_btn:
        st.session_state.slider_values = {feat: 50 for feat in FEATURE_COLUMNS}
        st.session_state.show_result   = False
        st.session_state.result        = None
        st.rerun()

    if predict_btn:
        errors = validate_pct_inputs(pct_inputs)
        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")
        else:
            try:
                result = predict_from_percentages(pct_inputs)
                st.session_state.result        = result
                st.session_state.slider_values = dict(pct_inputs)
                st.session_state.show_result   = True

                # ── Save to prediction history ────────────────────────────
                suggestions = get_preventive_suggestions(pct_inputs)
                history     = st.session_state.prediction_history
                new_history = add_record(
                    history,
                    inputs      = pct_inputs,
                    probability = result["probability_pct"],
                    risk_level  = result["risk_level"],
                    suggestions = suggestions,
                )
                st.session_state.prediction_history = new_history
                save_history(new_history)
                # ─────────────────────────────────────────────────────────

                st.rerun()
            except Exception as e:
                st.error(f"Prediction failed: {e}")


def _render_result():
    result      = st.session_state.result
    slider_vals = st.session_state.slider_values
    prob        = result["probability_pct"]
    risk        = result["risk_level"]
    rec         = result["recommendation"]

    styles = {
        "LOW RISK":      {"bg": "#e8f8ee", "border": "#2ecc71", "text": "#1a7a40", "icon": "🟢"},
        "MODERATE RISK": {"bg": "#fff8e6", "border": "#f39c12", "text": "#9a6200", "icon": "🟠"},
        "HIGH RISK":     {"bg": "#fde8e8", "border": "#e74c3c", "text": "#8b1a1a", "icon": "🔴"},
    }
    s = styles.get(risk, styles["MODERATE RISK"])

    st.markdown(f"""
    <div class="result-box" style="background:{s['bg']};border:2px solid {s['border']};">
        <div class="result-title" style="color:{s['text']};">FLOOD RISK RESULT</div>
        <div style="font-size:0.82rem;color:#6b7c93;margin-bottom:4px;">Flood Probability</div>
        <div class="result-prob" style="color:{s['text']};">{prob}%</div>
        <div class="result-risk" style="color:{s['text']};">{s['icon']} {risk}</div>
        <div style="color:#4a4a4a;margin-top:0.8rem;font-size:0.92rem;line-height:1.5;">{rec}</div>
    </div>
    """, unsafe_allow_html=True)

    # Suggestions panel
    suggestions   = get_preventive_suggestions(slider_vals)
    bullets_html  = "".join(
        f"<li style='margin-bottom:0.5rem;line-height:1.5;'>{sg}</li>"
        for sg in suggestions
    )
    st.markdown(f"""
    <div style="background:#f7faff;border:1.5px solid #b8cfe8;border-radius:10px;
                padding:1.5rem 1.8rem;margin:1rem 0;">
        <div style="font-size:0.82rem;font-weight:700;letter-spacing:1px;
                    text-transform:uppercase;color:#1a3c5e;margin-bottom:0.8rem;">
            🛡️ Preventive Flood Safety Suggestions
        </div>
        <div style="font-size:0.8rem;color:#4a6fa5;margin-bottom:0.9rem;">
            These preventive actions can help reduce the potential impact of flooding.
        </div>
        <ul style="padding-left:1.2rem;margin:0;color:#2d3e50;font-size:0.91rem;">
            {bullets_html}
        </ul>
        <div style="margin-top:1rem;font-size:0.73rem;color:#8a9bb0;
                    border-top:1px solid #d0dce8;padding-top:0.6rem;">
            ⚠️ These are preventive recommendations and do not replace official emergency guidance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Find the record just saved and offer a direct link to its history entry
    history = st.session_state.prediction_history
    if history:
        last_id = history[-1]["id"]
        st.markdown("<br>", unsafe_allow_html=True)
        col_again, col_hist = st.columns(2)
        with col_again:
            if st.button("🔄 PREDICT AGAIN", use_container_width=True, type="primary"):
                st.session_state.show_result = False
                st.rerun()
        with col_hist:
            if st.button("📋 View in History", use_container_width=True):
                st.session_state.active_page       = "Prediction History"
                st.session_state.history_detail_id = last_id
                st.session_state.show_result       = False
                st.rerun()
    else:
        if st.button("🔄 PREDICT AGAIN", use_container_width=True, type="primary"):
            st.session_state.show_result = False
            st.rerun()

    st.markdown("""
    <p style="font-size:0.75rem;color:#9aa8b5;text-align:center;margin-top:0.5rem;">
    This prediction is based on a machine learning model trained on historical flood data.
    It does not replace official emergency services or government flood warnings.
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def page_history():
    history = st.session_state.prediction_history

    # ── Detail view ───────────────────────────────────────────────────────────
    detail_id = st.session_state.history_detail_id
    if detail_id is not None:
        rec = next((r for r in history if r["id"] == detail_id), None)
        if rec is not None:
            _render_detail(rec)
            return
        # Record no longer exists (was cleared)
        st.session_state.history_detail_id = None

    # ── List view ─────────────────────────────────────────────────────────────
    st.markdown("#### Prediction History")

    if not history:
        st.info("No predictions yet. Go to **🔮 Predict Risk** to generate your first prediction.")
        return

    # Summary counts
    total    = len(history)
    done_all = sum(1 for r in history if overall_status(r) == STATUS_COMPLETED)
    st.caption(f"{total} prediction{'s' if total != 1 else ''}  •  {done_all} fully completed")
    st.markdown("")

    # Cards — newest first
    for rec in reversed(history):
        _render_history_card(rec)

    # ── Clear History ─────────────────────────────────────────────────────────
    st.markdown("<hr class='fs-divider'>", unsafe_allow_html=True)

    if not st.session_state.confirm_clear:
        if st.button("🗑️ Clear All History", use_container_width=False):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("⚠️ This will permanently delete all prediction history. Are you sure?")
        c_yes, c_no = st.columns(2)
        with c_yes:
            if st.button("✅ Yes, Clear History", use_container_width=True, type="primary"):
                st.session_state.prediction_history = clear_history()
                st.session_state.history_detail_id  = None
                st.session_state.confirm_clear      = False
                st.success("History cleared.")
                st.rerun()
        with c_no:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()


def _risk_style(risk_level: str) -> dict:
    return {
        "LOW RISK":      {"text": "#1a7a40", "icon": "🟢", "border": "#2ecc71"},
        "MODERATE RISK": {"text": "#9a6200", "icon": "🟠", "border": "#f39c12"},
        "HIGH RISK":     {"text": "#8b1a1a", "icon": "🔴", "border": "#e74c3c"},
    }.get(risk_level, {"text": "#4a6fa5", "icon": "🔵", "border": "#4a6fa5"})


def _overall_badge_class(ov_status: str) -> str:
    if ov_status == STATUS_COMPLETED:
        return "badge-done"
    if "In Progress" in ov_status:
        return "badge-ip"
    return "badge-ns"


def _render_history_card(rec: dict):
    """Render a summary card with a View Details button."""
    s      = _risk_style(rec["risk_level"])
    done   = completed_count(rec)
    total  = len(rec["suggestions"])
    ov     = overall_status(rec)
    badge  = _overall_badge_class(ov)

    # Card HTML (display-only portion)
    st.markdown(f"""
    <div class="hist-card" style="border-left:4px solid {s['border']};">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <div class="hist-card-id">Prediction #{rec['id']:02d}</div>
                <div class="hist-card-date">📅 {rec['datetime']}</div>
            </div>
            <div style="text-align:right;">
                <div class="hist-card-prob" style="color:{s['text']};">{rec['probability']}%</div>
                <div class="hist-card-risk" style="color:{s['text']};">{s['icon']} {rec['risk_level']}</div>
            </div>
        </div>
        <div style="margin-top:0.6rem;display:flex;align-items:center;gap:0.8rem;flex-wrap:wrap;">
            <span style="font-size:0.88rem;color:#4a6fa5;font-weight:600;">
                🛡️ Preventive Actions: ✅ {done} / {total} Completed
            </span>
            <span class="status-badge {badge}">{ov}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(f"🔍 View Details  →  Prediction #{rec['id']:02d}", key=f"view_{rec['id']}"):
        st.session_state.history_detail_id = rec["id"]
        st.session_state.confirm_clear     = False
        st.rerun()


def _render_detail(rec: dict):
    """Full detail view for a single prediction record with live action tracking."""
    s     = _risk_style(rec["risk_level"])
    done  = completed_count(rec)
    total = len(rec["suggestions"])
    ov    = overall_status(rec)
    badge = _overall_badge_class(ov)

    # Back button
    if st.button("← Back to History"):
        st.session_state.history_detail_id = None
        st.rerun()

    st.markdown(f"#### Prediction #{rec['id']:02d}")
    st.markdown(f"<div style='font-size:0.88rem;color:#8a9bb0;margin-bottom:1rem;'>📅 {rec['datetime']}</div>",
                unsafe_allow_html=True)

    # ── Risk summary ──────────────────────────────────────────────────────────
    col_p, col_r = st.columns(2)
    with col_p:
        st.markdown(f"""
        <div style="background:#f7faff;border:1.5px solid #b8cfe8;border-radius:10px;
                    padding:1rem;text-align:center;">
            <div style="font-size:0.75rem;font-weight:700;letter-spacing:1px;
                        text-transform:uppercase;color:#6b7c93;">Flood Probability</div>
            <div style="font-size:2.4rem;font-weight:900;color:{s['text']};">{rec['probability']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        st.markdown(f"""
        <div style="background:#f7faff;border:1.5px solid #b8cfe8;border-radius:10px;
                    padding:1rem;text-align:center;">
            <div style="font-size:0.75rem;font-weight:700;letter-spacing:1px;
                        text-transform:uppercase;color:#6b7c93;">Risk Level</div>
            <div style="font-size:1.3rem;font-weight:800;color:{s['text']};margin-top:0.3rem;">
                {s['icon']} {rec['risk_level']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Action progress summary ───────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#ffffff;border:1.5px solid #d0dce8;border-radius:10px;
                padding:0.9rem 1.2rem;margin-bottom:1rem;
                display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
        <span style="font-size:1rem;font-weight:700;color:#1a3c5e;">
            🛡️ Preventive Actions: ✅ {done} / {total} Completed
        </span>
        <span class="status-badge {badge}">{ov}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Preventive Actions with status selectors ──────────────────────────────
    st.markdown('<div class="sec-heading">Preventive Actions</div>', unsafe_allow_html=True)

    history = st.session_state.prediction_history
    for idx, (suggestion, status) in enumerate(zip(rec["suggestions"], rec["statuses"])):
        icon = "✅" if status == STATUS_COMPLETED else ("🔄" if "In Progress" in status else "⏳")
        with st.container():
            st.markdown(f"""
            <div class="action-row">
                <div class="action-text">{idx + 1}. {suggestion}</div>
            </div>
            """, unsafe_allow_html=True)

            new_status = st.selectbox(
                label        = f"Status for action {idx + 1}",
                options      = STATUS_OPTIONS,
                index        = STATUS_OPTIONS.index(status),
                key          = f"status_{rec['id']}_{idx}",
                label_visibility = "collapsed",
            )

            if new_status != status:
                updated = update_status(history, rec["id"], idx, new_status)
                st.session_state.prediction_history = updated
                save_history(updated)
                st.rerun()

    # ── 20 Input values ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 View Input Values Used for This Prediction"):
        st.markdown('<div class="sec-heading">20 Input Risk Factors</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        items = list(rec["inputs"].items())
        mid   = len(items) // 2
        for col, chunk in zip([col_a, col_b], [items[:mid], items[mid:]]):
            with col:
                for feat, val in chunk:
                    label = FEATURE_LABELS.get(feat, feat)
                    st.markdown(
                        f"<div class='input-row'>"
                        f"<span class='input-label'>{label}</span>"
                        f"<span class='input-value'>{val}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER LOOP
# ─────────────────────────────────────────────────────────────────────────────
render_header()
render_nav()

page = st.session_state.active_page
if page == "Home":
    page_home()
elif page == "Predict Risk":
    page_predict()
elif page == "Prediction History":
    page_history()
