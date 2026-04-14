"""
DISTILLATION COLUMN ETHANOL PURITY PREDICTOR — REDESIGNED UI V2.0
================================================================================
Author: Mr. Krishna Narayan Singh  |  Enhanced UI by Claude
Features: 21 Physics-Augmented Features | AI Chatbot | Modern Design
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import os
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. OPTIONAL DEPENDENCIES (graceful fallback if not installed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()          # loads .env into os.environ
except ImportError:
    pass                   # python-dotenv not installed — use env vars directly

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EthanolIQ · Purity Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. GLOBAL CSS — Industrial-Refined Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:          #0F1117;
    --surface:     #161B27;
    --surface2:    #1E2636;
    --border:      #2A3450;
    --accent:      #00D4AA;
    --accent2:     #3B8BEB;
    --warn:        #F5A623;
    --danger:      #E8445A;
    --text:        #E2E8F4;
    --muted:       #7A8BA6;
    --mono:        'Space Mono', monospace;
    --sans:        'DM Sans', sans-serif;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1280px; }

/* ── Custom header bar ── */
.app-header {
    display: flex; align-items: center; gap: 16px;
    padding: 20px 0 28px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.app-header .logo {
    font-family: var(--mono);
    font-size: 2rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.app-header .sub {
    color: var(--muted);
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Section labels ── */
.section-label {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: border-color .2s;
}
.card:hover { border-color: var(--accent2); }

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 16px 0; }
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-card .val {
    font-family: var(--mono);
    font-size: 1.45rem;
    color: var(--accent);
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Prediction result ── */
.pred-box {
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 2px solid;
    margin: 8px 0 20px;
}
.pred-box.good   { background: #0A1F1A; border-color: var(--accent); }
.pred-box.warn   { background: #1F1608; border-color: var(--warn);   }
.pred-box.danger { background: #1F080C; border-color: var(--danger); }

.pred-value {
    font-family: var(--mono);
    font-size: 4.5rem;
    line-height: 1;
    letter-spacing: -2px;
    margin: 8px 0;
}
.pred-box.good   .pred-value { color: var(--accent); }
.pred-box.warn   .pred-value { color: var(--warn);   }
.pred-box.danger .pred-value { color: var(--danger); }

.pred-status {
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    padding: 5px 16px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 6px;
}
.pred-box.good   .pred-status { background: rgba(0,212,170,.15); color: var(--accent); }
.pred-box.warn   .pred-status { background: rgba(245,166,35,.15); color: var(--warn);   }
.pred-box.danger .pred-status { background: rgba(232,68,90,.15);  color: var(--danger); }

/* ── Progress bar ── */
.purity-bar-wrap { margin: 16px 0 8px; }
.purity-bar-bg {
    background: var(--surface2);
    border-radius: 6px; height: 10px;
    border: 1px solid var(--border);
    overflow: hidden;
}
.purity-bar-fill {
    height: 100%; border-radius: 6px;
    transition: width .8s cubic-bezier(.4,0,.2,1);
}

/* ── Streamlit widget overrides ── */
.stSlider [data-baseweb="slider"] { padding: 4px 0; }
div[data-testid="stMetricValue"] { font-family: var(--mono) !important; color: var(--accent) !important; }
div[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .8rem !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00A882);
    color: #000 !important;
    font-family: var(--mono);
    font-weight: 700;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 10px;
    padding: 14px 28px;
    font-size: 0.9rem;
    transition: opacity .2s, transform .15s;
}
.stButton > button:hover { opacity: .88; transform: translateY(-1px); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    /* ✅ NO fixed min/max-width — lets Streamlit's collapse button work freely */
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem 2rem;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--accent);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* Collapse/expand toggle button — make it visible & styled */
button[data-testid="collapsedControl"],
button[kind="header"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent) !important;
    border-radius: 0 8px 8px 0 !important;
}

/* ── Sidebar custom metric cards ── */
.sb-metric {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.sb-metric .sb-val {
    font-family: var(--mono);
    font-size: 1.25rem;
    color: var(--accent);
    line-height: 1.1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sb-metric .sb-lbl {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 2px;
}
.sb-metric-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 0;
}
.sb-metric-row .sb-metric { margin-bottom: 0; }

/* ── Chatbot ── */
.chat-wrap { max-height: 440px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; padding: 4px 0; }
.chat-bubble {
    border-radius: 14px;
    padding: 12px 16px;
    line-height: 1.6;
    font-size: 0.9rem;
    max-width: 92%;
}
.chat-user {
    background: var(--accent2);
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
}
.chat-ai {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    align-self: flex-start;
    border-bottom-left-radius: 4px;
}
.chat-ai .ai-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
div[data-testid="stTextInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
}

/* ── Number inputs ── */
input[type="number"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent !important; }
.stTabs [data-baseweb="tab"] {
    background: var(--surface2) !important;
    color: var(--muted) !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    border: 1px solid var(--border) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent) !important;
    border-bottom-color: var(--surface) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* ── Expander ── */
details { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
summary { color: var(--text) !important; font-family: var(--mono) !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. AI CHATBOT INTEGRATION  (Groq / OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

# ── Helper to get env vars with sidebar fallback ──
def _get_api_cfg():
    """Return (api_key, base_url, model_name) from env or session state."""
    key   = os.getenv("GROQ_API_KEY",    st.session_state.get("_api_key", ""))
    url   = os.getenv("GROQ_BASE_URL",   "https://api.groq.com/openai/v1")
    model = os.getenv("MODEL_NAME",      "llama-3.3-70b-versatile")
    return key, url, model


def build_context_prompt(params: dict, prediction: float | None = None) -> str:
    """Build a rich context message for the LLM about the current column state."""
    pred_line = f"Predicted ethanol purity: {prediction:.4f} ({prediction*100:.2f} mol%)" \
                if prediction is not None else "No prediction run yet."
    return f"""You are DistilBot, a helpful process-engineering assistant embedded inside an
ethanol distillation monitoring app. Speak in a friendly, expert but approachable tone.
Use short paragraphs. Never use bullet points unless asked. Avoid markdown headers.

Current column operating state:
  Pressure          : {params.get('pressure', 'N/A')} bar
  Top temperature   : {params.get('t1', 'N/A')} °C
  Reflux flow L     : {params.get('L', 'N/A')} kmol/hr
  Vapor flow V      : {params.get('V', 'N/A')} kmol/hr
  Distillate D      : {params.get('D', 'N/A')} kmol/hr
  Bottoms B         : {params.get('B', 'N/A')} kmol/hr
  Feed F            : {params.get('F', 'N/A')} kmol/hr
  Reflux ratio      : {params.get('reflux_ratio', 'N/A'):.3f}
  Reboiler intensity: {params.get('reboiler', 'N/A'):.3f}
  {pred_line}

Answer the user's question below based on this context."""


def get_ai_response(user_message: str, params: dict, prediction: float | None,
                    history: list) -> str:
    """
    Call Groq Cloud (or any OpenAI-compatible) API and return assistant reply.
    Falls back to a rule-based response if no API key is configured.
    """
    api_key, base_url, model_name = _get_api_cfg()

    # ── Rule-based fallback (no API key) ──
    if not api_key:
        return _rule_based_response(user_message, params, prediction)

    # ── Live API call ──
    try:
        from openai import OpenAI     # pip install openai>=1.0
        client = OpenAI(api_key=api_key, base_url=base_url)

        messages = [{"role": "system", "content": build_context_prompt(params, prediction)}]
        for h in history[-6:]:        # keep last 3 turns for context
            messages.append(h)
        messages.append({"role": "user", "content": user_message})

        resp = client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=500, temperature=0.7
        )
        return resp.choices[0].message.content.strip()

    except ImportError:
        return ("⚠️ The `openai` package is not installed. "
                "Run `pip install openai` then restart the app.\n\n"
                + _rule_based_response(user_message, params, prediction))
    except Exception as e:
        return f"⚠️ API error: {e}\n\n" + _rule_based_response(user_message, params, prediction)


def _rule_based_response(msg: str, params: dict, pred: float | None) -> str:
    """Fallback advisor when no LLM is available."""
    msg_l = msg.lower()
    rr = params.get("reflux_ratio", 0)
    t1 = params.get("t1", 78)

    if pred is None:
        return ("Run the simulation first by clicking 'Run Simulation', then ask me "
                "about the result and I'll give you a detailed breakdown!")

    pct = pred * 100

    if any(w in msg_l for w in ["good", "bad", "result", "purity", "explain"]):
        quality = "excellent" if pred >= 0.85 else ("acceptable" if pred >= 0.75 else "below spec")
        tip = ""
        if pred < 0.75:
            tip = (f" Your reflux ratio is {rr:.2f} — try increasing it above 1.5. "
                   f"Also, top temperature of {t1}°C looks high; "
                   "reducing steam to the reboiler can help bring it down.")
        elif pred < 0.85:
            tip = (f" A small increase in reflux (currently {rr:.2f}) toward 1.8–2.0 "
                   "could push you into the optimal zone.")
        return (f"The predicted purity of {pct:.2f} mol% is {quality}.{tip} "
                "Want me to walk you through which parameter has the biggest lever?")

    if any(w in msg_l for w in ["improve", "increase", "higher"]):
        return (f"To raise purity from {pct:.1f}%, focus on three levers: "
                "1️⃣ raise the reflux ratio (increase L while keeping V steady), "
                "2️⃣ ensure top temperature stays near 78 °C (ethanol boiling point at ~1 bar), "
                "3️⃣ avoid overloading the feed — keep F below 620 kmol/hr for best separation.")

    if any(w in msg_l for w in ["reflux", "ratio"]):
        return (f"Your reflux ratio is {rr:.3f}. A typical target for high-purity ethanol is 1.5–2.2. "
                f"{'This looks healthy!' if 1.5 <= rr <= 2.5 else 'Consider adjusting it into the 1.5–2.2 band.'}")

    if any(w in msg_l for w in ["temp", "temperature"]):
        return (f"Top tray temperature is {t1} °C. At ~1 bar, pure ethanol boils at 78.4 °C. "
                f"{'You are right on target — great sign.' if 77 <= t1 <= 80 else 'Values above 82°C suggest water vapour in the distillate, which hurts purity.'}")

    return ("I'm DistilBot — ask me anything about your column! For example: "
            "'Is this purity good?', 'How do I improve it?', or 'What does reflux ratio mean?'")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD ML ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open("model.pkl",         "rb") as f: model    = pickle.load(f)
        with open("scaler.pkl",        "rb") as f: scaler   = pickle.load(f)
        with open("features_names.pkl","rb") as f: features = pickle.load(f)
        return model, scaler, features, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, feature_names, load_error = load_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# 5. SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "last_pred"     not in st.session_state: st.session_state.last_pred     = None
if "last_params"   not in st.session_state: st.session_state.last_params   = {}
if "chat_input"    not in st.session_state: st.session_state.chat_input    = ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand
    st.markdown(
        "<div style='padding:4px 0 16px'>"
        "<div style='font-family:\"Space Mono\",monospace;font-size:1.1rem;"
        "background:linear-gradient(135deg,#00D4AA,#3B8BEB);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "font-weight:700;letter-spacing:-0.5px'>⚗️ EthanolIQ</div>"
        "<div style='font-size:0.72rem;color:#7A8BA6;margin-top:2px;"
        "letter-spacing:0.06em;text-transform:uppercase'>"
        "Distillation Intelligence Platform</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1px;background:#2A3450;margin-bottom:16px'></div>",
                unsafe_allow_html=True)

    # Model Performance — pure HTML cards (no st.metric, no truncation)
    st.markdown(
        "<div style='font-family:\"Space Mono\",monospace;font-size:0.72rem;color:#00D4AA;"
        "letter-spacing:0.14em;text-transform:uppercase;margin-bottom:10px'>"
        "📊 Model Performance</div>"
        "<div class='sb-metric-row'>"
        "<div class='sb-metric'><div class='sb-val'>≥&nbsp;0.98</div><div class='sb-lbl'>R² Score</div></div>"
        "<div class='sb-metric'><div class='sb-val'>0.0155</div><div class='sb-lbl'>RMSE</div></div>"
        "</div>"
        "<div style='height:8px'></div>"
        "<div class='sb-metric-row'>"
        "<div class='sb-metric'><div class='sb-val'>0.0122</div><div class='sb-lbl'>MAE</div></div>"
        "<div class='sb-metric'><div class='sb-val'>±1.55%</div><div class='sb-lbl'>Accuracy</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1px;background:#2A3450;margin:16px 0'></div>",
                unsafe_allow_html=True)

    # Quick Tips
    st.markdown(
        "<div style='font-family:\"Space Mono\",monospace;font-size:0.72rem;color:#00D4AA;"
        "letter-spacing:0.14em;text-transform:uppercase;margin-bottom:10px'>"
        "💡 Quick Tips</div>",
        unsafe_allow_html=True,
    )
    tips = [
        ("🔼", "Higher reflux → Higher purity"),
        ("⚖️", "D + B ≈ F  (mass balance)"),
        ("🌡️", "T1 near 78 °C is ideal"),
        ("🚫", "Flood risk if V &gt; 1200"),
        ("📐", "Reflux ratio: target 1.5–2.2"),
    ]
    tips_html = "".join(
        f"<div style='font-size:.81rem;color:#7A8BA6;margin-bottom:7px;"
        f"display:flex;gap:8px;align-items:flex-start'>"
        f"<span>{icon}</span><span>{tip}</span></div>"
        for icon, tip in tips
    )
    st.markdown(tips_html, unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#2A3450;margin:16px 0'></div>",
                unsafe_allow_html=True)

    # AI Config expander
    with st.expander("🔑 AI Assistant Config"):
        api_key_input = st.text_input(
            "Groq / OpenAI API Key",
            value=st.session_state.get("_api_key", ""),
            type="password",
            help="Set GROQ_API_KEY in .env or paste here",
        )
        if api_key_input:
            st.session_state["_api_key"] = api_key_input
        st.caption("Without a key, DistilBot uses rule-based responses.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. HEADER
# ─────────────────────────────────────────────────────────────────────────────
if load_error:
    st.error(f"🚨 Could not load model files: `{load_error}`")
    st.info("Place `model.pkl`, `scaler.pkl`, `features_names.pkl` in the app folder and restart.")
    st.stop()

st.markdown("""
<div class="app-header">
  <div>
    <div class="logo">⚗️ EthanolIQ</div>
    <div class="sub">Distillation Column · Purity Prediction Engine · v2.0</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='color:#7A8BA6;margin:-16px 0 28px;font-size:.88rem'>"
    "Let's tune your distillation process — enter your column parameters below and hit <strong style='color:#00D4AA'>Run Simulation</strong>.</p>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 8. INPUT PANELS
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:

    # ── Core Parameters ──
    st.markdown('<div class="section-label">01 · Core Parameters</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pressure = st.slider("🔵 Pressure (bar)", 0.5, 3.0, 1.01, 0.01,
                             help="Column operating pressure. Most ethanol columns run near 1 bar.")
        t1_celsius = st.slider("🌡️ T1 Top Tray Temp (°C)", 60.0, 120.0, 78.0, 0.5,
                               help="Temperature at the top tray. Near 78°C = ethanol-rich vapour.")
        l_input = st.slider("💧 Reflux Flow L (kmol/hr)", 300.0, 1200.0, 780.0, 10.0,
                            help="Liquid returned to the column. Higher = better separation, but costs energy.")
    with c2:
        v_input = st.slider("♨️ Vapor Flow V (kmol/hr)", 600.0, 1500.0, 1040.0, 10.0,
                            help="Vapour rising from the reboiler. Drives separation upward.")
        d_input = st.slider("🏭 Distillate D (kmol/hr)", 100.0, 500.0, 260.0, 10.0,
                            help="Product withdrawn from the top. Should be ethanol-rich.")
        b_input = st.slider("🪣 Bottoms B (kmol/hr)", 100.0, 500.0, 340.0, 10.0,
                            help="Liquid drawn from the bottom. Should be mostly water.")

    st.markdown('<div class="section-label" style="margin-top:20px">02 · Feed Stream</div>',
                unsafe_allow_html=True)
    f_input = st.slider("⚗️ Feed Flow F (kmol/hr)", 350.0, 700.0, 580.0, 10.0,
                        help="Total feed entering the column. Mass balance: D + B ≈ F.")

    # ── Mass balance check ──
    mb_err = abs((d_input + b_input) - f_input)
    if mb_err > 30:
        st.warning(f"⚠️ Mass balance gap: |D+B−F| = {mb_err:.0f} kmol/hr. Check your flow rates.")
    else:
        st.success(f"✅ Mass balance OK — gap: {mb_err:.0f} kmol/hr")


# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
epsilon           = 1e-6
v_safe            = max(v_input, epsilon)
f_safe            = max(f_input, epsilon)

temp_top          = t1_celsius
temp_bottom       = t1_celsius + 20.0
temp_diff         = temp_bottom - temp_top

MEAN_FEED_FLOW    = 545.23
feed_normalized   = f_input / MEAN_FEED_FLOW

reflux_ratio      = l_input / v_safe
reboiler_intensity= v_input / f_safe
condenser_load    = l_input / f_safe
distillate_w      = d_input / f_safe
bottoms_w         = b_input / f_safe
column_load       = (l_input + v_input) / f_safe

reflux_x_temp_top   = reflux_ratio * temp_top
reflux_x_temp_diff  = reflux_ratio * temp_diff
reboiler_x_temp_bot = reboiler_intensity * temp_bottom
feed_x_reflux       = feed_normalized * reflux_ratio
feed_x_reboiler     = feed_normalized * reboiler_intensity

separation_duty  = reflux_ratio * reboiler_intensity
column_efficiency= reflux_ratio * column_load


# ─────────────────────────────────────────────────────────────────────────────
# 10. DERIVED METRICS PANEL (right column)
# ─────────────────────────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="section-label">03 · Derived Metrics (live)</div>',
                unsafe_allow_html=True)

    metrics = [
        ("Reflux Ratio",       f"{reflux_ratio:.3f}",       "L/V — target 1.5–2.2"),
        ("Reboiler Intensity", f"{reboiler_intensity:.3f}",  "V/F — heat per feed"),
        ("Condenser Load",     f"{condenser_load:.3f}",      "L/F — cooling demand"),
        ("Column Load",        f"{column_load:.3f}",         "(L+V)/F — hydraulic load"),
        ("Separation Duty",    f"{separation_duty:.3f}",     "Reflux × Reboiler"),
        ("Column Efficiency",  f"{column_efficiency:.3f}",   "Reflux × Load"),
    ]

    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    for lbl, val, hint in metrics:
        st.markdown(f"""
        <div class="metric-card" title="{hint}">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Reflux ratio gauge bar ──
    rr_pct  = min(reflux_ratio / 3.0, 1.0) * 100
    rr_color= "#00D4AA" if 1.5 <= reflux_ratio <= 2.5 else ("#F5A623" if reflux_ratio < 1.5 else "#E8445A")
    st.markdown(f"""
    <div style="margin-top:12px">
      <div style="font-size:.72rem;color:#7A8BA6;margin-bottom:4px;font-family:'Space Mono'">
        REFLUX RATIO GAUGE &nbsp; ({reflux_ratio:.3f})
      </div>
      <div class="purity-bar-bg">
        <div class="purity-bar-fill" style="width:{rr_pct:.1f}%;background:{rr_color}"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:.68rem;color:#7A8BA6;margin-top:2px">
        <span>0</span><span>1.5 optimal</span><span>3.0</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Run button ──
    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("🔮  Run Simulation", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# 11. PREDICTION OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">04 · Simulation Result</div>', unsafe_allow_html=True)

if run_clicked:
    input_vector = np.array([[
        pressure, l_input, v_input, d_input, b_input, f_input,
        temp_bottom, reflux_ratio, reboiler_intensity, condenser_load,
        feed_normalized, distillate_w, bottoms_w, column_load,
        reflux_x_temp_top, reflux_x_temp_diff, reboiler_x_temp_bot,
        feed_x_reflux, feed_x_reboiler, separation_duty, column_efficiency
    ]])

    try:
        with st.spinner("Running physics simulation…"):
            time.sleep(0.4)                        # brief dramatic pause ✨
            scaled  = scaler.transform(input_vector)
            pred    = float(model.predict(scaled)[0])
            pct     = pred * 100

        # Store for chatbot
        st.session_state.last_pred   = pred
        st.session_state.last_params = {
            "pressure": pressure, "t1": t1_celsius,
            "L": l_input, "V": v_input, "D": d_input,
            "B": b_input, "F": f_input,
            "reflux_ratio": reflux_ratio,
            "reboiler": reboiler_intensity,
        }

        # Visual class
        if pred >= 0.82:  cls, status, emoji = "good",   "OPTIMAL",        "✅"
        elif pred >= 0.75: cls, status, emoji = "warn",   "ACCEPTABLE",     "⚠️"
        else:             cls, status, emoji = "danger", "BELOW SPEC",     "🚨"

        bar_color = {"good": "#00D4AA", "warn": "#F5A623", "danger": "#E8445A"}[cls]

        col_pred, col_bar = st.columns([1, 1])
        with col_pred:
            st.markdown(f"""
            <div class="pred-box {cls}">
              <div style="font-size:.72rem;font-family:'Space Mono';color:#7A8BA6;letter-spacing:.18em">
                PREDICTED ETHANOL PURITY
              </div>
              <div class="pred-value">{pred:.4f}</div>
              <div style="font-size:1.4rem;color:#7A8BA6;margin-top:4px">{pct:.2f} mol%</div>
              <div class="pred-status">{emoji} {status}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_bar:
            bar_pct = min(pct, 100)
            st.markdown(f"""
            <div style="padding:20px 0">
              <div style="font-size:.72rem;font-family:'Space Mono';color:#7A8BA6;margin-bottom:12px;letter-spacing:.1em">
                PURITY SCALE (MOL%)
              </div>
              <div class="purity-bar-bg" style="height:20px;border-radius:10px">
                <div class="purity-bar-fill" style="width:{bar_pct:.1f}%;background:{bar_color};height:100%;border-radius:10px"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:.72rem;color:#7A8BA6;margin-top:6px;font-family:'Space Mono'">
                <span>0%</span><span>75% spec</span><span>82% optimal</span><span>100%</span>
              </div>

              <div style="margin-top:24px;background:#1E2636;border-radius:10px;padding:14px 16px">
                <div style="font-size:.72rem;font-family:'Space Mono';color:#7A8BA6;margin-bottom:8px">PARAMETER SNAPSHOT</div>
                <table style="width:100%;font-size:.82rem;border-collapse:collapse">
                  <tr><td style="color:#7A8BA6;padding:2px 0">Reflux Ratio</td><td style="text-align:right;font-family:'Space Mono';color:#00D4AA">{reflux_ratio:.3f}</td></tr>
                  <tr><td style="color:#7A8BA6;padding:2px 0">Reboiler Intensity</td><td style="text-align:right;font-family:'Space Mono';color:#00D4AA">{reboiler_intensity:.3f}</td></tr>
                  <tr><td style="color:#7A8BA6;padding:2px 0">Top Temperature</td><td style="text-align:right;font-family:'Space Mono';color:#00D4AA">{t1_celsius}°C</td></tr>
                  <tr><td style="color:#7A8BA6;padding:2px 0">Column Efficiency</td><td style="text-align:right;font-family:'Space Mono';color:#00D4AA">{column_efficiency:.3f}</td></tr>
                </table>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Auto-inject a context message into chat
        auto_msg = f"I just ran a simulation. Purity = {pct:.2f} mol%. Can you explain this result?"
        st.session_state.chat_history.append({"role": "user", "content": auto_msg})
        ai_reply = get_ai_response(auto_msg, st.session_state.last_params,
                                   pred, st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.markdown("""
    <div style="background:#161B27;border:1px dashed #2A3450;border-radius:12px;
                padding:40px;text-align:center;color:#7A8BA6">
      <div style="font-size:2rem;margin-bottom:8px">⚗️</div>
      <div style="font-family:'Space Mono';font-size:.82rem;letter-spacing:.1em">
        AWAITING SIMULATION
      </div>
      <div style="font-size:.8rem;margin-top:8px">
        Adjust your parameters and click <strong style="color:#00D4AA">Run Simulation</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 12. AI CHATBOT PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">05 · DistilBot — AI Assistant</div>',
            unsafe_allow_html=True)

chat_col, info_col = st.columns([3, 2], gap="large")

with chat_col:
    # Chat history display
    if st.session_state.chat_history:
        chat_html = '<div class="chat-wrap">'
        for msg in st.session_state.chat_history[-12:]:
            if msg["role"] == "user":
                chat_html += f'<div class="chat-bubble chat-user">{msg["content"]}</div>'
            else:
                chat_html += (f'<div class="chat-bubble chat-ai">'
                              f'<div class="ai-label">⚗️ DISTILBOT</div>{msg["content"]}</div>')
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-bubble chat-ai" style="max-width:100%">
          <div class="ai-label">⚗️ DISTILBOT</div>
          Hey! I'm DistilBot — your column advisor. Run a simulation and I'll explain the result,
          or ask me anything: "What is reflux ratio?", "How do I increase purity?", etc.
        </div>
        """, unsafe_allow_html=True)

    # Input row
    inp_col, btn_col = st.columns([5, 1])
    with inp_col:
        user_text = st.text_input("Message DistilBot…", key="chat_input",
                                  label_visibility="collapsed",
                                  placeholder="Ask about your column, the result, or anything…")
    with btn_col:
        send_clicked = st.button("Send", use_container_width=True)

    if send_clicked and user_text.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        with st.spinner("DistilBot is thinking…"):
            reply = get_ai_response(user_text, st.session_state.last_params,
                                    st.session_state.last_pred,
                                    st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # Clear button
    if st.button("🗑️ Clear chat", use_container_width=False):
        st.session_state.chat_history = []
        st.rerun()

with info_col:
    st.markdown("""
    <div class="card">
      <div style="font-family:'Space Mono';font-size:.72rem;color:#00D4AA;letter-spacing:.12em;margin-bottom:10px">
        SUGGESTED QUESTIONS
      </div>
      <div style="font-size:.82rem;color:#7A8BA6;line-height:2">
        💬 "Is this purity result good?"<br>
        💬 "How can I improve purity?"<br>
        💬 "What is reflux ratio?"<br>
        💬 "Why is top temperature important?"<br>
        💬 "Explain reboiler intensity"<br>
        💬 "What causes flooding?"
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div style="font-family:'Space Mono';font-size:.72rem;color:#00D4AA;letter-spacing:.12em;margin-bottom:10px">
        AI STATUS
      </div>
    """, unsafe_allow_html=True)
    api_key, *_ = _get_api_cfg()
    if api_key:
        st.success("🟢 Live AI connected (Groq Cloud ✓)")
    else:
        st.info("🔵 Rule-based mode (no API key)")
        st.caption("Add GROQ_API_KEY to .env or use the sidebar to enable live AI.")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 13. DOCUMENTATION TABS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">06 · Documentation & Reference</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📐 Features", "📈 Model", "🔧 Troubleshoot", "📊 Feature Chart"])

with tab1:
    st.markdown("""
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1–6 | P, L, V, D, B, F | Core | Direct user inputs |
| 7 | Temp_Bottom | Reference | T1 + 20 °C (reboiler approx.) |
| 8–13 | Reflux/Reboiler/Condenser/Feed_Norm/D_W/B_W | Derived | Operating ratios |
| 14–18 | Reflux×T_top, Reflux×T_diff, Reboiler×T_bot, Feed×Reflux, Feed×Reboiler | Interaction | Non-linear physics |
| 19–21 | Column_Load, Sep_Duty, Col_Efficiency | Efficiency | Combined performance |
""")

with tab2:
    c1, c2 = st.columns(2)
    c1.metric("Algorithm", "Random Forest")
    c2.metric("Training points", "1,200+")
    c1.metric("R² Score", "≥ 0.98")
    c2.metric("Validation", "5-Fold CV")
    st.markdown("""
**Typical Operating Ranges**

| Parameter | Normal | High Risk | Low Risk |
|-----------|--------|-----------|----------|
| Top Temp T1 | 77–80°C | > 85°C | < 75°C |
| Reflux Ratio | 1.5–2.2 | > 3.0 | < 0.8 |
| Vapor V | 900–1100 | > 1200 | < 800 |
""")

with tab3:
    st.markdown("""
**🔴 Purity < 0.70**
Increase reflux flow L. Check T1 — values above 82°C mean water vapour is entering the product.
Reduce feed F if > 620 kmol/hr.

**🟡 Model files not found**
Place `model.pkl`, `scaler.pkl`, `features_names.pkl` in the same folder as `main.py`.

**🔵 Why T1 + 20°C for bottom temperature?**
Bottom temperature was measured as T14 during training.
Since only T1 (top) is input here, the physics approximation T_bottom ≈ T_top + 20°C
ensures interaction features remain accurate.

**🟣 Chatbot not responding?**
Check your API key in the sidebar or `.env` file. Without a key, rule-based fallback is used.
""")

with tab4:
    image_path = "feature_importance.png"
    if PIL_AVAILABLE and os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, caption="Random Forest Feature Importance", use_container_width=True)
    else:
        st.info("Place `feature_importance.png` in the app folder to display this chart.")
        # Draw a simple placeholder bar chart with st.bar_chart
        importance_data = pd.DataFrame({
            "Feature": ["Reflux×T_top", "Temp_Bottom", "Reflux_Ratio",
                        "Reboiler×T_bot", "Column_Load", "Feed×Reflux",
                        "Sep_Duty", "Col_Efficiency", "Pressure", "T1"],
            "Importance": [0.28, 0.19, 0.14, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02],
        }).set_index("Feature")
        st.bar_chart(importance_data)
        st.caption("📊 Illustrative importance — replace with your actual feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14. FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#2A3450;font-size:.78rem;
            padding:24px;margin-top:32px;font-family:'Space Mono'">
  ⚗️ ETHANOLIQ v2.0 &nbsp;·&nbsp; RANDOM FOREST &nbsp;·&nbsp; 21-FEATURE PHYSICS ENGINE<br>
  <span style="opacity:.6">Built for educational purposes · Not for critical process control</span>
</div>
""", unsafe_allow_html=True)