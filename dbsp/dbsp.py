"""
        Module: MIS41420 Sports & Performance Analytics

            Dead Ball Success Predictor 

This is analytical tool directed at helping coaches, analyst and players to turn any planned dead-ball routine into:
  1. Expected Goals (xG) : Done using Engine A which is the regressor
  2. Second-ball retention probability(%) : Done using which is the classifier
Additionally there is a concise second-ball plan for the attacking team and model explainability charts.

Further, it Supports eight dead-ball situations (corner, free kicks, penalty, throw-in, goal kick, kick-off, play restart) and multiple set of attacking triggers.

The application opens on a landing page. Here, the user clicks "Launch the
predictor" to enter the tool. A 'Home' button in the sidebar returns to the opening page.

Steps to Run :-

  1. (for the first time) pip install -r requirements.txt
  2. (for the first time) python src/train_models.py
  3. streamlit run app.py
  
The application opens at http://localhost:8501
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the src/ folder importable no matter where the app is launched from.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from feature_engineering import build_feature_matrix
from data_ingestion import SET_PIECE_TYPES, TRIGGER_COLS
from scenario import (
    random_scenario, routine_label, second_ball_drop_zone,
    retention_levers, retention_advice, pretty,
    DELIVERY_TYPES, DELIVERY_ZONES, DEFENSIVE_SCHEMES,
)

MODELS_DIR = ROOT / "models"

# ----- Colour scheme: Green & Blue on a light pitch 
GREEN = "#0B6E10"; BLUE = "#3ba9e8"; SLATE = "#34445a"; INK = "#36331c"
MUTED = "#6b888c"; LINE = "#E2E8F0"; SOFT = "#F5F7FA"; CARD = "#FFFFFF"

st.set_page_config(page_title="Dead Ball Success Predictor", page_icon="dart", layout="wide")

# ----- Global theme CSS (same template for the opening page AND the tool)
st.markdown(f"""
<style>
.stApp {{ background:{SOFT}; }}
.block-container {{ padding-top: 1.4rem; }}
.kpi {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px;
        padding:18px 20px; box-shadow:0 1px 3px rgba(20,40,60,.06); }}
.kpi .v {{ font-size:40px; font-weight:800; line-height:1; }}
.kpi .l {{ font-size:12px; color:{MUTED}; text-transform:uppercase;
           letter-spacing:1px; margin-top:8px; }}
.section {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px;
           padding:18px 22px; box-shadow:0 1px 3px rgba(20,40,60,.06); }}
.lever-on  {{ color:{GREEN}; font-weight:700; }}
.lever-off {{ color:#B6C0CA; }}
h1, h2, h3 {{ color:{INK}; }}
/* ---- Opening (landing) page styling, same green & blue template ---- */
.hero {{ background:{CARD}; border:1px solid {LINE}; border-radius:20px;
         padding:42px 46px; box-shadow:0 2px 10px rgba(20,40,60,.07);
         border-top:6px solid {GREEN}; }}
.hero h1 {{ font-size:40px; font-weight:800; margin:0 0 6px 0; }}
.hero .accent {{ color:{GREEN}; }}
.hero p {{ font-size:16px; color:{MUTED}; margin:0; }}
.feature {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px;
            padding:18px 20px; height:100%;
            box-shadow:0 1px 3px rgba(20,40,60,.06); }}
.feature .ico {{ font-size:22px; }}
.feature .ttl {{ font-weight:700; color:{INK}; margin:6px 0 4px 0; font-size:15px; }}
.feature .txt {{ color:{MUTED}; font-size:13px; }}
.pipeline {{ color:{SLATE}; font-size:14px; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    return (joblib.load(MODELS_DIR / "engine_a_xg.joblib"),
            joblib.load(MODELS_DIR / "engine_b_retention.joblib"),
            joblib.load(MODELS_DIR / "feature_columns.joblib"),
            json.loads((MODELS_DIR / "metrics.json").read_text()))

try:
    engine_a, engine_b, FEATURE_COLS, METRICS = load_models()
except FileNotFoundError:
    st.error("Models not found. Run **python src/train_models.py** once, then refresh.")
    st.stop()


# ---------------------------------------------------------------------------
# Navigation + scenario helpers
# ---------------------------------------------------------------------------
SELECT_KEYS = ["set_piece_type", "delivery_type", "delivery_zone", "defensive_scheme"]
SLIDER_KEYS = ["num_attackers_box", "num_defenders_box", "aerial_threat", "second_ball_runners"]

def roll_scenario():
    """Populate every control with a fresh, realistic random routine."""
    sc = random_scenario()
    for k in SELECT_KEYS + SLIDER_KEYS:
        st.session_state[k] = sc[k]
    for t in TRIGGER_COLS:
        st.session_state[t] = bool(sc[t])

def ensure_scenario():
    """Guarantee the controls have values before the widgets are created."""
    if "set_piece_type" not in st.session_state:
        roll_scenario()

def enter_app():
    """Opening page -> tool. (Runs before the rerun, so widgets get values.)"""
    ensure_scenario()
    st.session_state.view = "app"

def go_home():
    """Tool -> opening page."""
    st.session_state.view = "landing"

# First visit starts on the opening page.
if "view" not in st.session_state:
    st.session_state.view = "landing"


# ===========================================================================
" LANDING PAGE "
# ===========================================================================
if st.session_state.view == "landing":
    # Hide the (empty) sidebar so the opening page is clean.
    st.markdown("<style>[data-testid='stSidebar']{display:none;}</style>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hero">
      <h1>Dead Ball <span class="accent">Success Predictor</span></h1>
      <p>Prescriptive analytical tool for support staff &mdash;
      instant pre-match xG and second-ball retention forecasts for any planned dead ball routine.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")  # spacer

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""<div class="feature"><div class="ico">💯</div>
        <div class="ttl">8 dead-ball situations</div>
        <div class="txt">All dead-ball situations - corners, free kicks, penalties, throw-ins, goal kicks,
        kick-offs &amp; restarts &mdash; all in one tool.</div></div>""",
        unsafe_allow_html=True)
    with f2:
        st.markdown(f"""<div class="feature"><div class="ico">🤔</div>
        <div class="ttl">Dual Machine Learning Models</div>
        <div class="txt"> Predicts xG and second-ball retention probability.</div></div>""",
        unsafe_allow_html=True)
    with f3:
        st.markdown(f"""<div class="feature"><div class="ico">🔃</div>
        <div class="ttl">Second-ball Retention Plan</div>
        <div class="txt">Likely drop zone, expected second-ball xG, active
        attacking and precise coach recommendations.</div></div>""",
        unsafe_allow_html=True)

    st.markdown("")  # spacer
    st.markdown(f"<p class='pipeline'><b></b> Ingested Data "
                f"&rarr; Tactical Features &rarr; Dual ML Models &rarr; "
                f"Interactive Dashboard.</p>", unsafe_allow_html=True)

    # Centred launch button (green via the theme's primaryColor).
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.button("☄️ Predictor Launch", type="primary",
                  on_click=enter_app, use_container_width=True)


    st.stop()  


# ===========================================================================
"      ANALYTICAL TOOL  "
# ===========================================================================
ensure_scenario()

# --- Header ---
left, right = st.columns([3, 1])
with left:
    st.markdown("<h1 style='margin-bottom:2px'>&#127919; Dead Ball Success Predictor</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<span style='color:{MUTED}'>This is prescriptive set-piece analytics tool for "
                f"coaches &amp; assistant coaches &amp; analysts for all dead ball situations &mdash; corners, free kicks, penalties, "
                f"throw-ins, goal kicks, kick-offs &amp; restarts.</span>",
                unsafe_allow_html=True)
with right:
    st.button("🧩 Randomise scenario", on_click=roll_scenario, use_container_width=True)


# --- Sidebar control panel ---
sb = st.sidebar
sb.button("🔙 Back", on_click=go_home, use_container_width=True)
sb.header("Console")
sb.button("🎮 New scenario", on_click=roll_scenario, use_container_width=True)

sb.subheader("Dead Ball Situation")
sb.selectbox("Dead Ball Routine", SET_PIECE_TYPES, format_func=pretty, key="set_piece_type")
sb.selectbox("Ball Delivery Type", DELIVERY_TYPES, format_func=pretty, key="delivery_type")
sb.selectbox("Targetted Delivery Zone", DELIVERY_ZONES, format_func=pretty, key="delivery_zone")
sb.selectbox("Opponent Defensive Tactic", DEFENSIVE_SCHEMES, key="defensive_scheme")

sb.subheader("Players in the Box")
sb.slider("Attackers", 1, 7, key="num_attackers_box")
sb.slider("Defenders", 3, 11, key="num_defenders_box")
sb.slider("Attacking Aerial Threat", 1, 10, key="aerial_threat")
sb.slider("Second-ball Runners at Box-Edge", 0, 4, key="second_ball_runners",
          help="Players you station around the box edge to win loose / cleared balls.")

sb.subheader("Tactical Situations")
TRIGGER_LABELS = {
    "near_post_decoy": "Near-Post Decoy Run",
    "gk_screen": "Goalkeeper Screening",
    "blocker_action": "Off-the-ball Blocker ",
    "taker_foot_matches": "Foot Matches Ball Swing",
    "runner_from_deep": "Runner From Deep",
    "overload_far_post": "Far-Post Overload ",
    "second_wave_runner": "Designated Second-Ball Runner",
    "quick_delivery": "Quick Delivery ",
    "dummy_run": "Dummy Run ",
}
for t in TRIGGER_COLS:
    sb.checkbox(TRIGGER_LABELS[t], key=t)


# --- Assemble scenario + predict ---
scenario = {k: st.session_state[k] for k in SELECT_KEYS + SLIDER_KEYS}
for t in TRIGGER_COLS:
    scenario[t] = int(st.session_state[t])

X = build_feature_matrix(pd.DataFrame([scenario])).reindex(columns=FEATURE_COLS, fill_value=0)
pred_xg = float(engine_a.predict(X)[0])
pred_ret = float(engine_b.predict_proba(X)[0][1])
is_penalty = scenario["set_piece_type"] == "penalty"

st.markdown(f"#### Current routine: <span style='color:{GREEN}'>{routine_label(scenario)}</span>",
            unsafe_allow_html=True)


# --- KPI cards ---
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='kpi'><div class='v' style='color:{GREEN}'>{pred_xg:.3f}</div>"
                f"<div class='l'>Expected goals (xG)</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='kpi'><div class='v' style='color:{BLUE}'>{pred_ret*100:.0f}%</div>"
                f"<div class='l'>Second-ball retention</div></div>", unsafe_allow_html=True)
with c3:
    rating = min(100, int((pred_xg / 0.25) * 60 + pred_ret * 40))
    rc = GREEN if rating >= 60 else (BLUE if rating >= 35 else "#C0392B")
    st.markdown(f"<div class='kpi'><div class='v' style='color:{rc}'>{rating}</div>"
                f"<div class='l'>Routine rating / 100</div></div>", unsafe_allow_html=True)

if is_penalty:
    st.info("Penalty selected: xG is the estimated penalty conversion rate, and "
            " second-ball retention is the rebound off a save/post.")

st.markdown("")


# --- Gauge + xG drivers ---
def importance_chart(model, title):
    imp = (pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
           .sort_values("importance").tail(10))
    imp["feature"] = imp["feature"].str.replace("_", " ")
    fig = px.bar(imp, x="importance", y="feature", orientation="h", title=title,
                 template="plotly_white")
    fig.update_traces(marker_color=GREEN)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=46, b=10),
                      yaxis_title="", xaxis_title="Importance", font_color=INK)
    return fig

g, i = st.columns([1, 1])
with g:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pred_ret * 100, number={"suffix": "%"},
        title={"text": "Second-ball retention probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": BLUE},
               "steps": [{"range": [0, 35], "color": "#F3D9CC"},
                         {"range": [35, 60], "color": "#FBEBDD"},
                         {"range": [60, 100], "color": "#D8ECEC"}]}))
    gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10),
                        template="plotly_white", font_color=INK)
    st.plotly_chart(gauge, use_container_width=True)
with i:
    st.plotly_chart(importance_chart(engine_a, "Why this xG? (Engine A drivers)"),
                    use_container_width=True)


#       SECOND-BALL RETENTION PLAN 
st.markdown("### Second-ball Retention Tactic ")
st.markdown(f"<span style='color:{MUTED}'>This tells the likelihood of the attacking team "
            f"keeping the ball and sustaining the attack.</span>",
            unsafe_allow_html=True)

drop_zone = second_ball_drop_zone(scenario["delivery_zone"])
second_phase_xg = round(pred_ret * 0.06, 3)
levers = retention_levers(scenario)
active = sum(l["active"] for l in levers)

m1, m2, m3 = st.columns(3)
m1.metric("Estimated Drop zone", drop_zone)
m2.metric("Expected 2nd time xG", f"{second_phase_xg:.3f}",
          help="Retention probability x a typical second-phase chance quality.")
m3.metric("Active second-ball takers", f"{active} / {len(levers)}")

lcol, rcol = st.columns([1, 1])
with lcol:
    st.markdown("**Second-ball attackers in this routine**")
    for l in levers:
        mark = "[x]" if l["active"] else "[ ]"
        cls = "lever-on" if l["active"] else "lever-off"
        st.markdown(f"`{mark}` <span class='{cls}'>{l['label']}</span> "
                    f"<span style='color:{MUTED};font-size:12px'>&mdash; {l['detail']}</span>",
                    unsafe_allow_html=True)
with rcol:
    st.markdown("**Coaching Recommendations**")
    for tip in retention_advice(pred_ret, scenario):
        st.markdown(f"- {tip}")

st.plotly_chart(importance_chart(engine_b, "Second-ball retention Logic"),
                use_container_width=True)


# --- Model performance footer ---
with st.expander(" Performance of Model and Data"):
    st.json(METRICS)
    st.caption(f" First situation is scored with MSE/RMSE/R2 while the second ball situation is done using ROC-AUC. "
               f"Data source: {METRICS.get('data_source')} - {METRICS.get('n_rows')} "
               f"routines - {METRICS.get('n_features')} features. Note: aggregate xG R2 is "
               f"high partly because penalties are near-deterministic.")
