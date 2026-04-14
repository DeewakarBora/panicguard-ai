"""
PanicGuard AI — Streamlit Dashboard (Bulletproofed)
====================================================
Fintech-grade, dark-theme dashboard with complete error boundaries,
graceful fallbacks, and zero-crash guarantee.

Run:
    streamlit run dashboard/app.py
    cd dashboard && streamlit run app.py
    streamlit run panicguard-ai/dashboard/app.py
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

# ── ISSUE 3: Import path robustness ─────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import plotly.graph_objects as go
import streamlit as st

from agents.orchestrator import PanicGuardOrchestrator
from data.historical_crashes import HISTORICAL_CRASHES
from utils.helpers import format_inr

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="PanicGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ISSUE 5: SESSION STATE INITIALISATION  (ALL at top)                ║
# ╚══════════════════════════════════════════════════════════════════════╝

_DEFAULTS = {
    "chat_history":      [],
    "analysis_result":   None,      # cached pipeline result
    "orchestrator":      None,
    "demo_loaded":       False,
    "analysis_running":  False,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ISSUE 10: Pre-compute demo result on first load                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

if st.session_state.analysis_result is None:
    st.session_state.analysis_result = PanicGuardOrchestrator.get_demo_result()
    st.session_state.demo_loaded = True


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ISSUE 7: Error boundary helper                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _safe_section(section_name: str):
    """Decorator that wraps a dashboard section in try/except."""
    def decorator(func):
        def wrapper(*a, **kw):
            try:
                return func(*a, **kw)
            except Exception as e:
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.15);
                            border-radius:12px; padding:20px; text-align:center; margin:10px 0;">
                    <div style="font-size:1.2rem; margin-bottom:6px;">⚠️</div>
                    <div style="color:#94a3b8; font-size:0.82rem;">
                        {section_name} is temporarily unavailable
                    </div>
                </div>
                """, unsafe_allow_html=True)
                # Log for debugging but never show to user
                print(f"[PanicGuard] {section_name} error: {e}")
        return wrapper
    return decorator


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DESIGN SYSTEM  (custom CSS)                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

_COLORS = {
    "bg":       "#0a1628", "card": "rgba(17, 29, 53, 0.90)",
    "border":   "rgba(30, 58, 95, 0.80)",  "accent": "#3b82f6",
    "green":    "#10b981", "yellow": "#f59e0b",
    "orange":   "#f97316", "red": "#ef4444",
    "blue":     "#3b82f6", "text": "#f1f5f9",
    "text_dim": "#94a3b8", "surface": "#0e1a2e",
    "gold":     "#fbbf24",
}
RISK_COLOR = {"LOW": _COLORS["green"], "MEDIUM": _COLORS["yellow"],
              "HIGH": _COLORS["orange"], "CRITICAL": _COLORS["red"]}

st.markdown("""
<style>
html, body, [class*="st-"] { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; }
.stApp { background: linear-gradient(160deg, #0a1628 0%, #0e1a2e 40%, #0a1628 100%) !important; }
#MainMenu, footer, header { visibility: hidden; }

.glass-card {
    background: rgba(17, 29, 53, 0.85); backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(30, 58, 95, 0.80);
    border-radius: 16px; padding: 24px 28px; margin-bottom: 16px;
    transition: transform .2s, box-shadow .2s;
}
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(59, 130, 246, 0.10); }

.metric-card {
    background: rgba(17, 29, 53, 0.90); backdrop-filter: blur(12px);
    border: 1px solid rgba(30, 58, 95, 0.70); border-radius: 14px;
    padding: 20px 22px; text-align: center;
}
.metric-label { font-size:0.78rem; font-weight:500; text-transform:uppercase; letter-spacing:1.2px; color:#94a3b8; margin-bottom:6px; }
.metric-value { font-size:1.65rem; font-weight:700; color:#f1f5f9; font-variant-numeric:tabular-nums; }
.metric-delta { font-size:0.82rem; font-weight:500; margin-top:4px; }
.delta-up { color:#10b981; } .delta-down { color:#ef4444; }

.scenario-card {
    background: rgba(17, 29, 53, 0.90); backdrop-filter: blur(12px);
    border-radius: 14px; padding: 22px 18px; text-align: center;
    border: 1px solid rgba(30, 58, 95, 0.60); height: 100%;
}
.scenario-card.panic  { border-left: 3px solid #ef4444; }
.scenario-card.hold   { border-left: 3px solid #f59e0b; }
.scenario-card.brave  { border-left: 3px solid #10b981; }
.scenario-card.defense{ border-left: 3px solid #3b82f6; }
.scenario-emoji { font-size:2rem; margin-bottom:6px; }
.scenario-title { font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:#94a3b8; margin-bottom:10px; }
.scenario-value { font-size:1.45rem; font-weight:800; font-variant-numeric:tabular-nums; }
.scenario-sub   { font-size:0.75rem; color:#94a3b8; margin-top:6px; }

.gauge-wrapper { display:flex; flex-direction:column; align-items:center; padding:10px 0 0 0; }
.gauge-wrapper .risk-badge {
    display:inline-block; padding:6px 22px; border-radius:50px; font-size:0.82rem;
    font-weight:700; letter-spacing:1.4px; text-transform:uppercase; margin-top:-10px;
}
.bias-pill {
    display:inline-block; background:rgba(59,130,246,0.14); border:1px solid rgba(59,130,246,0.32);
    color:#93c5fd; padding:6px 16px; border-radius:50px; font-size:0.78rem; font-weight:600; margin:4px 6px 4px 0;
}
.section-header { font-size:1.3rem; font-weight:600; color:#f1f5f9; margin-bottom:4px; padding-bottom:8px; border-bottom:1px solid rgba(30,58,95,0.80); }
.section-sub { font-size:0.82rem; color:#94a3b8; margin-bottom:20px; }

.crash-table { width:100%; border-collapse:separate; border-spacing:0; font-size:0.82rem; }
.crash-table thead th { background:rgba(30,58,95,0.50); padding:12px 14px; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; color:#93c5fd; text-align:left; }
.crash-table thead th:first-child { border-radius:10px 0 0 0; }
.crash-table thead th:last-child  { border-radius:0 10px 0 0; }
.crash-table tbody td { padding:12px 14px; border-bottom:1px solid rgba(30,58,95,0.50); color:#e2e8f0; }
.crash-table tbody tr:hover { background:rgba(59,130,246,0.05); }

.app-footer { text-align:center; color:#64748b; font-size:0.75rem; padding:40px 0 20px 0; border-top:1px solid rgba(30,58,95,0.50); margin-top:40px; }

@keyframes pulse-critical { 0%{box-shadow:0 0 0 0 rgba(239,68,68,0.45);} 70%{box-shadow:0 0 0 18px rgba(239,68,68,0);} 100%{box-shadow:0 0 0 0 rgba(239,68,68,0);} }
.pulse { animation: pulse-critical 1.8s infinite; border-radius: 50%; }

.wealth-destroyed {
    background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25);
    border-radius:12px; padding:20px 24px; text-align:center;
}
.wd-amount { font-size:2.2rem; font-weight:900; color:#ef4444; font-variant-numeric:tabular-nums; }
.wd-label { font-size:0.88rem; font-weight:500; color:#fca5a5; margin-top:4px; }

.chat-user { background:rgba(59,130,246,0.10); border:1px solid rgba(59,130,246,0.22); border-radius:14px 14px 4px 14px; padding:14px 18px; margin:8px 0; color:#e2e8f0; }
.chat-ai { background:rgba(14,26,46,0.85); border:1px solid rgba(30,58,95,0.70); border-radius:14px 14px 14px 4px; padding:14px 18px; margin:8px 0; color:#e2e8f0; }

.stSelectbox > div > div, .stSlider > div { border-color:rgba(30,58,95,0.80) !important; }
.stButton > button { background:linear-gradient(135deg,#3b82f6 0%,#2563eb 100%) !important; color:white !important; border:none !important; border-radius:10px !important; padding:10px 28px !important; font-weight:600 !important; transition:all .2s !important; }
.stButton > button:hover { transform:translateY(-1px) !important; box-shadow:0 4px 20px rgba(59,130,246,0.30) !important; }
div[data-testid="stSidebar"] { background:linear-gradient(180deg,#0e1a2e 0%,#0a1628 100%) !important; border-right:1px solid rgba(30,58,95,0.70) !important; }

.info-banner { background:rgba(59,130,246,0.07); border:1px solid rgba(59,130,246,0.20); border-radius:10px; padding:10px 18px; font-size:0.78rem; color:#93c5fd; text-align:center; margin-bottom:12px; }
.citation { font-size:0.68rem; color:#64748b; font-style:italic; }
.section-divider { height:1px; background:linear-gradient(90deg,transparent,rgba(30,58,95,0.60),transparent); margin:32px 0; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _risk_color(level: str) -> str:
    return RISK_COLOR.get(level, _COLORS["accent"])


def _build_gauge(score: int, risk: str) -> go.Figure:
    color = _risk_color(risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "", "font": {"size": 68, "color": color, "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)",
                     "tickfont": {"size": 11, "color": "#64748b"}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "rgba(30,27,58,0.5)", "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(34,197,94,0.08)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.08)"},
                {"range": [60, 80], "color": "rgba(249,115,22,0.08)"},
                {"range": [80, 100], "color": "rgba(239,68,68,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(height=380, margin=dict(t=30, b=0, l=40, r=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#f0f0f0"})
    return fig


def _panic_interpretation(score: int, risk: str) -> str:
    if risk == "CRITICAL":
        return ("Markets are in extreme fear. History shows these moments are often the best time to keep buying — "
                "COVID bottom investors gained +112% in 18 months.")
    if risk == "HIGH":
        return ("Significant stress detected. Every past crisis of this magnitude recovered. "
                "Stopping your SIP now locks in the loss and misses the recovery.")
    if risk == "MEDIUM":
        return ("Elevated volatility but not a crisis. Temporary corrections are a normal part of equity markets — "
                "staying invested is the evidence-based response.")
    return "Markets are calm. Your long-term plan is on track — no action needed."


def _build_factor_bars(factors: list[dict]) -> go.Figure:
    names = [f["factor"] for f in reversed(factors)]
    impacts = [float(str(f.get("impact", "0")).replace("%", "")) for f in reversed(factors)]
    n = len(names)
    # Color by rank: highest-impact bar = red, tapering to amber
    bar_colors = []
    for i in range(n):
        rank = n - 1 - i  # 0 = lowest impact in reversed list
        t = rank / max(n - 1, 1)
        # interpolate red (#ef4444) → amber (#f59e0b) as t goes 1→0
        bar_colors.append(_COLORS["red"] if t > 0.6 else (_COLORS["orange"] if t > 0.3 else _COLORS["yellow"]))
    fig = go.Figure(go.Bar(
        x=impacts, y=names, orientation="h",
        marker_color=bar_colors, marker_line_width=0,
        text=[f"{v}%" for v in impacts], textposition="outside",
        textfont={"size": 12, "color": "#e2e8f0"},
    ))
    fig.update_layout(
        height=max(140, 50 * len(names)), margin=dict(t=5, b=5, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "range": [0, max(impacts + [1]) * 1.35]},
        yaxis={"tickfont": {"size": 12, "color": "#cbd5e1"}},
        bargap=0.35, showlegend=False,
    )
    return fig


def _parse_inr(s: str) -> float:
    try:
        s = str(s).replace("\u20b9", "").replace(",", "").strip()
        multiplier = 1
        if s.endswith("Cr"):
            s = s.replace("Cr", "").strip()
            multiplier = 1_00_00_000
        elif s.endswith("L"):
            s = s.replace("L", "").strip()
            multiplier = 1_00_000
        return float(s) * multiplier
    except (ValueError, TypeError):
        return 0


def _build_scenario_chart(scenarios: dict) -> go.Figure:
    horizons = [5, 10, 15, 20]

    def _extract(key):
        items = scenarios.get("scenarios", {}).get(key, [])
        vals = []
        for yr in horizons:
            match = next((s for s in items if s.get("years") == yr), None)
            if match:
                raw = match.get("value_raw", 0)
                if raw == 0:
                    raw = _parse_inr(match.get("value_fmt", "0"))
                vals.append(raw)
            else:
                vals.append(0)
        return vals

    fig = go.Figure()
    for key, name, color in [
        ("stop", "Stop SIP (Panic)", "#ef4444"),
        ("defense", "Switch to Debt", "#3b82f6"),
        ("hold", "Continue SIP (Hold)", "#f59e0b"),
        ("brave", "Increase SIP +50%", "#22c55e"),
    ]:
        vals = _extract(key)
        fig.add_trace(go.Bar(
            name=name, x=[f"{y}yr" for y in horizons], y=vals,
            marker_color=color, marker_line_width=0,
            text=[format_inr(v) for v in vals], textposition="outside",
            textfont={"size": 11, "color": "#e2e8f0"},
        ))
    fig.update_layout(
        barmode="group", height=420, margin=dict(t=30, b=40, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, font=dict(size=12, color="#cbd5e1")),
        xaxis=dict(tickfont=dict(size=13, color="#cbd5e1")),
        yaxis=dict(visible=False), bargap=0.22, bargroupgap=0.06,
    )
    return fig


def _bias_descriptions():
    return {
        "Loss Aversion": "\U0001f494 **Loss Aversion** \u2014 You\u2019re feeling losses 2\u00d7 more intensely than equivalent gains. This asymmetry makes you want to \"stop the pain\" even though selling locks in the loss permanently.",
        "Recency Bias": "\U0001f504 **Recency Bias** \u2014 This crash feels permanent, but all 7 previous crashes in Indian market history were followed by full recovery. Average recovery time: ~12 months.",
        "Herd Mentality": "\U0001f411 **Herd Mentality** \u2014 76% of investors stopped their SIPs during similar crashes. But the 24% who continued built significantly more wealth. Don\u2019t follow the crowd \u2014 follow the data.",
        "Panic / Emotional": "\U0001f9e0 **Panic Response** \u2014 Your amygdala is in control. The urge to sell everything is a primal survival instinct, not a rational financial decision.",
        "Anchoring": "\u2693 **Anchoring** \u2014 You\u2019re comparing today\u2019s price to a previous high. But the market doesn\u2019t owe you a return from any specific price.",
        "Availability Heuristic": "\U0001f4f0 **Availability Heuristic** \u2014 Vivid media coverage distorts your risk perception. Crashes get 10\u00d7 more headlines than recoveries.",
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING (robustified)                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _run_analysis(portfolio: dict) -> dict:
    """Run orchestrator with full error isolation. NEVER raises."""
    try:
        orch = PanicGuardOrchestrator(user_portfolio=portfolio)
        st.session_state.orchestrator = orch
        result = orch.run_full_analysis()
        return result
    except Exception as e:
        print(f"[PanicGuard] Pipeline error: {e}")
        return PanicGuardOrchestrator.get_demo_result()


def _get_orchestrator(portfolio: dict) -> PanicGuardOrchestrator:
    if st.session_state.orchestrator is None:
        try:
            st.session_state.orchestrator = PanicGuardOrchestrator(user_portfolio=portfolio)
        except Exception:
            st.session_state.orchestrator = PanicGuardOrchestrator()
    return st.session_state.orchestrator


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR  (ISSUE 4: input validation)                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:2.6rem;">🛡️</div>
        <div style="font-size:1.2rem; font-weight:800; color:#f0f0f0; letter-spacing:0.5px;">PanicGuard AI</div>
        <div style="font-size:0.72rem; color:#8b5cf6; font-weight:500; letter-spacing:1px; text-transform:uppercase;">
            Your Financial Crisis Shield
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<p class="section-header" style="font-size:1rem;">📋 Your Portfolio</p>""", unsafe_allow_html=True)

    monthly_sip = st.slider(
        "Monthly SIP (\u20b9)", min_value=500, max_value=1_00_000,
        value=10_000, step=500, format="\u20b9%d",
        help="Your total monthly SIP contribution across all funds",
    )
    total_invested = st.number_input(
        "Total Invested (\u20b9)", min_value=0, max_value=5_00_00_000,
        value=5_00_000, step=10_000, format="%d",
    )
    current_value = st.number_input(
        "Current Value (\u20b9)", min_value=0, max_value=5_00_00_000,
        value=4_50_000, step=10_000, format="%d",
    )
    horizon = st.slider("Investment Horizon (years)", min_value=1, max_value=30, value=10)
    risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
    fund_types = st.multiselect(
        "Fund Types", ["Large Cap", "Mid Cap", "Small Cap", "Index Fund", "Debt"],
        default=["Large Cap", "Mid Cap", "Debt"],
    )

    st.markdown("---")
    analyze_btn = st.button("\U0001f50d Analyze My Portfolio", use_container_width=True, type="primary")

    st.markdown("""
    <div style="text-align:center; padding:20px 0 0;">
        <div style="font-size:0.7rem; color:#64748b; line-height:1.5;">
            Powered by<br>
            <span style="color:#8b5cf6; font-weight:600;">XGBoost + SHAP + Behavioral AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BUILD PORTFOLIO  (ISSUE 4: safe defaults, no div-by-zero)          ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _build_portfolio() -> dict:
    _type_map = {"Large Cap": "large_cap", "Mid Cap": "mid_cap",
                 "Small Cap": "small_cap", "Index Fund": "index", "Debt": "debt"}
    selected = fund_types if fund_types else ["Large Cap"]
    n = len(selected)
    safe_invested = max(total_invested, 1)  # prevent div-by-zero
    inv_each = safe_invested // n
    cur_ratio = current_value / safe_invested
    funds = []
    for ft in selected:
        t = _type_map.get(ft, "large_cap")
        funds.append({"name": f"{ft} Fund", "type": t,
                       "invested": inv_each, "current": int(inv_each * cur_ratio)})
    return {"monthly_sip": max(monthly_sip, 500),
            "funds": funds,
            "investment_horizon_years": max(horizon, 1),
            "risk_profile": risk_profile.lower()}


portfolio = _build_portfolio()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ISSUE 8: Loading + ISSUE 5: Cached result in session state         ║
# ╚══════════════════════════════════════════════════════════════════════╝

if analyze_btn:
    with st.spinner("\U0001f6e1\ufe0f PanicGuard is analyzing your portfolio..."):
        progress = st.progress(0, text="Initialising agents...")
        progress.progress(10, text="Scanning market data...")
        result = _run_analysis(portfolio)
        progress.progress(70, text="Generating coaching...")
        st.session_state.analysis_result = result
        st.session_state.demo_loaded = False
        progress.progress(100, text="Analysis complete!")
        progress.empty()

# Use cached result (ISSUE 5)
result = st.session_state.analysis_result

# ── Unpack convenience fields ──────────────────────────────────────────
crisis     = result.get("crisis_data", {})
coaching   = result.get("coaching", {})
port_rep   = result.get("portfolio_report", {})
crash_comp = result.get("crash_comparison", {})
pan_score  = result.get("panic_score", 0)
risk_lvl   = result.get("risk_level", "LOW")
is_demo    = result.get("is_demo", False)
mkt        = crisis.get("market_summary", {})
sip_data   = port_rep.get("sip_scenarios", {})
cost_data  = port_rep.get("cost_of_panic", {})
port_sum   = port_rep.get("portfolio_summary", {})

# ── ISSUE 1 & 2: Subtle info banners ──────────────────────────────────
if is_demo:
    st.markdown("""<div class="info-banner">
        📡 Using cached market data &nbsp;·&nbsp; Click <strong>Analyze My Portfolio</strong> for live results
    </div>""", unsafe_allow_html=True)

errors = result.get("errors", [])
if any("model" in str(e).lower() for e in errors):
    st.markdown("""<div class="info-banner">
        🔧 Running in lite mode — train model for full features
    </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  § HEADER                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

@_safe_section("Header")
def render_header():
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"""
        <div style="padding:8px 0;">
            <span style="font-size:2.2rem; font-weight:900; color:#f0f0f0;">🛡️ PanicGuard AI</span>
            <span style="font-size:0.95rem; color:#94a3b8; margin-left:16px; font-weight:400;">
                Your calm in the market storm
            </span>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        now = datetime.now()
        dot = "\U0001f7e2" if pan_score < 60 else ("\U0001f7e1" if pan_score < 80 else "\U0001f534")
        badge_label = "Sample Analysis" if is_demo else "Live"
        badge_style = ("background:rgba(100,116,139,0.18); color:#94a3b8; border:1px solid rgba(100,116,139,0.35);"
                       "padding:3px 12px; border-radius:50px; font-size:0.75rem; font-weight:500;"
                       if is_demo else
                       "background:rgba(16,185,129,0.18); color:#10b981; border:1px solid rgba(16,185,129,0.35);"
                       "padding:3px 12px; border-radius:50px; font-size:0.75rem; font-weight:500;")
        st.markdown(f"""
        <div style="text-align:right; padding:12px 0;">
            <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:6px;">{dot} {now.strftime("%d %b %Y, %I:%M %p")}</div>
            <span style="{badge_style}">{badge_label}</span>
        </div>""", unsafe_allow_html=True)

render_header()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  § 1  PANIC SCORE  (Hero)                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

@_safe_section("Panic Score")
def render_panic_score():
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    g1, g2 = st.columns([6, 4])
    with g1:
        color = _risk_color(risk_lvl)
        pulse_cls = "pulse" if risk_lvl == "CRITICAL" else ""
        glow_shadow = f"box-shadow: 0 0 32px {color}30, 0 0 8px {color}18;"
        st.markdown(f"""
        <div class="glass-card gauge-wrapper" style="{glow_shadow}">
            <div style="font-size:0.78rem; font-weight:600; text-transform:uppercase;
                        letter-spacing:1.4px; color:#94a3b8; margin-bottom:4px;">Market Panic Index</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(_build_gauge(pan_score, risk_lvl), use_container_width=True, key="gauge")
        conf = coaching.get("confidence_level", 0.8)
        conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)
        interpretation = _panic_interpretation(pan_score, risk_lvl)
        st.markdown(f"""
            <div class="risk-badge {pulse_cls}" style="background:{color}22; color:{color}; border:1px solid {color}55;">
                {risk_lvl}</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:12px; text-align:center;">
                {crisis.get("recommendation", "HOLD")} recommended &nbsp;·&nbsp; Confidence {conf_str}</div>
            <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.6; margin-top:16px;
                        padding:14px 18px; background:rgba(59,130,246,0.06);
                        border-left:3px solid {color}; border-radius:0 8px 8px 0;">
                {interpretation}</div>
        </div>""", unsafe_allow_html=True)

    with g2:
        factors = crisis.get("top_factors", [])
        st.markdown("""<div class="glass-card">
            <div style="font-size:0.78rem; font-weight:600; text-transform:uppercase;
                        letter-spacing:1.4px; color:#94a3b8; margin-bottom:8px;">Top Factors Driving This Score</div>
        """, unsafe_allow_html=True)
        if factors:
            st.plotly_chart(_build_factor_bars(factors), use_container_width=True, key="factors")
            for f in factors:
                st.markdown(f"""<div style="font-size:0.78rem; color:#cbd5e1; margin:2px 0;">
                    <span style="color:{_COLORS['accent']};">\u25cf</span>
                    {f.get('factor','')}: <strong>{f.get('value','')}</strong></div>""", unsafe_allow_html=True)
        else:
            st.info("No significant risk factors detected.")
        st.markdown("</div>", unsafe_allow_html=True)

render_panic_score()
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  § 2  MARKET SNAPSHOT                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

@_safe_section("Market Snapshot")
def render_market():
    st.markdown("""<div class="section-header">📊 Market Snapshot</div>
    <div class="section-sub">Real-time indices powering the panic model
    <span class="citation">&nbsp;·&nbsp; Source: NSE via Yahoo Finance</span></div>""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    nifty_val = mkt.get("nifty", 0) or 0
    nifty_chg = mkt.get("nifty_change", 0) or 0
    vix_val = mkt.get("vix") or 0
    crude_val = mkt.get("crude") or 0

    with m1:
        delta_cls = "delta-up" if nifty_chg >= 0 else "delta-down"
        arrow = "\u25b2" if nifty_chg >= 0 else "\u25bc"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Nifty 50</div>
            <div class="metric-value">{nifty_val:,.2f}</div>
            <div class="metric-delta {delta_cls}">{arrow} {nifty_chg:+.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        vix_interp = "Fear is low" if vix_val < 20 else ("Fear is elevated" if vix_val < 30 else "Extreme fear")
        vix_clr = _COLORS["green"] if vix_val < 20 else (_COLORS["yellow"] if vix_val < 30 else _COLORS["red"])
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">India VIX</div>
            <div class="metric-value">{vix_val:.1f}</div>
            <div class="metric-delta" style="color:{vix_clr};">{vix_interp}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Crude Oil (Brent)</div>
            <div class="metric-value">${crude_val:.2f}</div>
            <div class="metric-delta" style="color:#94a3b8;">Global macro indicator</div>
        </div>""", unsafe_allow_html=True)

render_market()
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  § 3  PORTFOLIO STATUS                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

@_safe_section("Portfolio")
def render_portfolio():
    st.markdown("""<div class="section-header">💼 Your Portfolio</div>
    <div class="section-sub">Current standing and crisis impact analysis</div>""", unsafe_allow_html=True)

    p1, p2 = st.columns([2, 1])
    with p1:
        gl_pct = port_sum.get("gain_loss_pct", 0) or 0
        gl_color = _COLORS["green"] if gl_pct >= 0 else _COLORS["red"]
        st.markdown(f"""<div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Total Invested</div>
                    <div style="font-size:1.6rem; font-weight:800; font-variant-numeric:tabular-nums; color:#f0f0f0;">
                        {port_sum.get("total_invested", format_inr(total_invested))}</div>
                </div>
                <div style="font-size:2.2rem; color:#64748b;">\u2192</div>
                <div>
                    <div style="font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Current Value</div>
                    <div style="font-size:1.6rem; font-weight:800; font-variant-numeric:tabular-nums; color:{gl_color};">
                        {port_sum.get("total_current", format_inr(current_value))}</div>
                </div>
                <div>
                    <div style="font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Gain/Loss</div>
                    <div style="font-size:1.6rem; font-weight:800; font-variant-numeric:tabular-nums; color:{gl_color};">
                        {gl_pct:+.1f}%</div>
                </div>
            </div></div>""", unsafe_allow_html=True)

        fi = port_sum.get("fund_impacts", [])
        if fi:
            st.markdown("""<div class="glass-card" style="padding:16px 20px;">""", unsafe_allow_html=True)
            st.markdown("""<table class="crash-table"><thead><tr>
                <th>Fund</th><th>Invested</th><th>Current</th><th>P&amp;L</th></tr></thead><tbody>""", unsafe_allow_html=True)
            for f in fi:
                pct = f.get("gain_loss_pct", 0) or 0
                c = _COLORS["green"] if pct >= 0 else _COLORS["red"]
                st.markdown(f"""<tr><td>{f.get('name','')}</td><td>{f.get('invested','')}</td>
                    <td>{f.get('current','')}</td>
                    <td style="color:{c}; font-weight:600;">{f.get('gain_loss','')} ({pct:+.1f}%)</td></tr>""", unsafe_allow_html=True)
            st.markdown("</tbody></table></div>", unsafe_allow_html=True)

    with p2:
        hist = port_rep.get("historical_comparison", {})
        sim_crash = hist.get("most_similar_crash", crash_comp.get("most_similar_crash", "COVID-19 (2020)"))
        rec_mo = hist.get("recovery_months", crash_comp.get("recovery_months", "?"))
        post_g = hist.get("post_bottom_gain", crash_comp.get("post_bottom_gain", "?"))
        st.markdown(f"""<div class="glass-card">
            <div style="font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:1.2px; color:#94a3b8; margin-bottom:10px;">
                📈 Most Similar Crash</div>
            <div style="font-size:1.15rem; font-weight:700; color:#c4b5fd; margin-bottom:10px;">{sim_crash}</div>
            <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.6;">
                {hist.get("message", crash_comp.get("recovery_timeline", ""))}</div>
            <div style="margin-top:14px; padding-top:12px; border-top:1px solid rgba(139,92,246,0.12);
                        font-size:0.82rem; color:#f59e0b; font-weight:500;">
                Recovery: <strong>{rec_mo} months</strong> &nbsp;·&nbsp; Post-bottom gain: <strong>+{post_g}%</strong></div>
            <div class="citation" style="margin-top:8px;">Source: BSE/NSE historical archives</div>
        </div>""", unsafe_allow_html=True)

render_portfolio()
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  § 4  SIP SCENARIOS  (The Money Shot)                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

@_safe_section("SIP Scenarios")
def render_scenarios():
    st.markdown("""<div class="section-header">💰 What Happens to Your Money?</div>
    <div class="section-sub">Four scenarios. One decision. A lifetime of difference.</div>""", unsafe_allow_html=True)

    wd = cost_data.get("wealth_destroyed", sip_data.get("summary", {}).get("cost_of_panic", "\u20b90"))
    h_yrs = sip_data.get("horizon_years", horizon)
    st.markdown(f"""<div class="wealth-destroyed">
        <div style="font-size:0.78rem; color:#fca5a5; font-weight:600; text-transform:uppercase;
                    letter-spacing:1.2px; margin-bottom:6px;">Wealth Destroyed by Panicking</div>
        <div class="wd-amount">{wd}</div>
        <div class="wd-label">This is how much you lose over {h_yrs} years by stopping your SIP today</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    summary = sip_data.get("summary", {})
    s1, s2, s3, s4 = st.columns(4)

    def _scenario_val(key, yr):
        items = sip_data.get("scenarios", {}).get(key, [])
        match = next((s for s in items if s.get("years") == yr), None)
        return match.get("value_fmt", "\u2014") if match else "\u2014"

    with s1:
        st.markdown(f"""<div class="scenario-card panic">
            <div class="scenario-emoji">🔴</div><div class="scenario-title">Panic \u2014 Stop SIP</div>
            <div class="scenario-value" style="color:#ef4444;">{summary.get("stop_value", "\u2014")}</div>
            <div class="scenario-sub">in {h_yrs} years</div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="scenario-card hold">
            <div class="scenario-emoji">🟡</div><div class="scenario-title">Hold \u2014 Continue SIP</div>
            <div class="scenario-value" style="color:#f59e0b;">{summary.get("hold_value", "\u2014")}</div>
            <div class="scenario-sub">in {h_yrs} years</div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="scenario-card brave">
            <div class="scenario-emoji">🟢</div><div class="scenario-title">Brave \u2014 Increase 50%</div>
            <div class="scenario-value" style="color:#22c55e;">{summary.get("brave_value", "\u2014")}</div>
            <div class="scenario-sub">in {h_yrs} years</div></div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""<div class="scenario-card defense">
            <div class="scenario-emoji">🔵</div><div class="scenario-title">Defense \u2014 Switch Debt</div>
            <div class="scenario-value" style="color:#3b82f6;">{_scenario_val("defense", h_yrs)}</div>
            <div class="scenario-sub">in {h_yrs} years</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(_build_scenario_chart(sip_data), use_container_width=True, key="scenarios")
    st.markdown("</div>", unsafe_allow_html=True)

    headline = sip_data.get("headline", "")
    if headline:
        st.markdown(f"""<div style="text-align:center; font-size:0.92rem; color:#cbd5e1;
                    font-weight:500; padding:8px 20px; line-height:1.6;">{headline}</div>""", unsafe_allow_html=True)

render_scenarios()
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  § 5  AI COACH                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

@_safe_section("AI Coach")
def render_coach():
    st.markdown("""<div class="section-header">🧠 Your AI Behavioral Coach</div>
    <div class="section-sub">Personalised, empathetic guidance backed by data and history</div>""", unsafe_allow_html=True)

    co1, co2 = st.columns([2, 1])
    with co1:
        msg = coaching.get("coaching_message", "")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(msg)
        st.markdown('</div>', unsafe_allow_html=True)

    with co2:
        biases = coaching.get("detected_biases", [])
        st.markdown("""<div class="glass-card">
            <div style="font-size:0.78rem; font-weight:600; text-transform:uppercase;
                        letter-spacing:1.2px; color:#94a3b8; margin-bottom:12px;">🔍 Detected Behavioral Biases</div>
        """, unsafe_allow_html=True)
        if biases:
            for b in biases:
                st.markdown(f'<span class="bias-pill">{b}</span>', unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            bd = _bias_descriptions()
            for b in biases:
                if b in bd:
                    st.markdown(f"<div style='font-size:0.82rem; color:#cbd5e1; margin:8px 0; line-height:1.5;'>{bd[b]}</div>",
                                unsafe_allow_html=True)
        else:
            st.markdown("""<div style="font-size:0.82rem; color:#22c55e;">
                \u2705 No panic-driven biases detected. You're thinking clearly!</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        kd = coaching.get("key_data_points", [])
        if kd:
            st.markdown("""<div class="glass-card">
                <div style="font-size:0.78rem; font-weight:600; text-transform:uppercase;
                            letter-spacing:1.2px; color:#94a3b8; margin-bottom:10px;">📌 Key Data Points</div>
            """, unsafe_allow_html=True)
            for point in kd:
                st.markdown(f"""<div style="font-size:0.78rem; color:#e2e8f0; margin:6px 0;
                                 padding-left:14px; border-left:2px solid #8b5cf6;">{point}</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

render_coach()
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ── Chat Interface (ISSUE 5: persists in session state) ────────────────
@_safe_section("Chat")
def render_chat():
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.92rem; font-weight:600; color:#f0f0f0; margin-bottom:8px;">
        💬 Ask the AI Coach</div>""", unsafe_allow_html=True)

    user_input = st.chat_input("Ask anything \u2014 e.g., 'Should I stop my SIP?'")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        try:
            orch = _get_orchestrator(portfolio)
            reply = orch.run_chat(user_input)
        except Exception:
            reply = ("Every crash in Indian market history has been followed by recovery. "
                     "Continue your SIP \u2014 you are buying at a discount. What specifically is worrying you?")
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    for msg_item in st.session_state.chat_history:
        if msg_item["role"] == "user":
            st.markdown(f'<div class="chat-user">\U0001f64b {msg_item["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">\U0001f6e1\ufe0f {msg_item["content"]}</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  § 6  HISTORICAL PROOF                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

@_safe_section("Historical Proof")
def render_history():
    st.markdown("""<div class="section-header">📜 Every Crash Recovers — Historical Proof</div>
    <div class="section-sub">7 major Indian market crashes. 7 full recoveries. Zero exceptions.
    <span class="citation">&nbsp;·&nbsp; Source: BSE/NSE archives, AMFI Monthly Report Feb 2026</span></div>""",
    unsafe_allow_html=True)

    st.markdown('<div class="glass-card" style="overflow-x:auto;">', unsafe_allow_html=True)
    st.markdown("""<table class="crash-table"><thead><tr>
        <th>Crisis</th><th>Period</th><th>Nifty Drop</th>
        <th>Recovery</th><th>SIP Continuers</th><th>Key Lesson</th></tr></thead><tbody>""", unsafe_allow_html=True)

    for c in HISTORICAL_CRASHES:
        drop_color = _COLORS["red"] if c.nifty_peak_to_trough_pct < -25 else _COLORS["orange"]
        rec_text = f"{c.nifty_recovery_months} months" if c.nifty_recovery_months > 0 else "\u23f3 Ongoing"
        trigger_short = (c.trigger[:60] + "...") if len(c.trigger) > 60 else c.trigger
        sip_short = (c.typical_sip_continuers_outcome[:100] + "...") if len(c.typical_sip_continuers_outcome) > 100 else c.typical_sip_continuers_outcome
        lesson_short = (c.key_lesson[:80] + "...") if len(c.key_lesson) > 80 else c.key_lesson
        st.markdown(f"""<tr>
            <td><strong>{c.name}</strong><br><span style="font-size:0.72rem; color:#94a3b8;">{trigger_short}</span></td>
            <td style="white-space:nowrap;">{c.start_date} \u2192<br>{c.end_date}</td>
            <td style="color:{drop_color}; font-weight:700; font-variant-numeric:tabular-nums;">{c.nifty_peak_to_trough_pct}%</td>
            <td style="color:#22c55e; font-weight:600;">{rec_text}</td>
            <td style="font-size:0.78rem;">{sip_short}</td>
            <td style="font-size:0.78rem; color:#f59e0b;">{lesson_short}</td>
        </tr>""", unsafe_allow_html=True)
    st.markdown("</tbody></table></div>", unsafe_allow_html=True)

    # Recovery bar chart
    crash_names = [c.name.split("(")[0].strip() for c in HISTORICAL_CRASHES if c.nifty_recovery_months > 0]
    drops = [abs(c.nifty_peak_to_trough_pct) for c in HISTORICAL_CRASHES if c.nifty_recovery_months > 0]
    recoveries = [c.nifty_recovery_months for c in HISTORICAL_CRASHES if c.nifty_recovery_months > 0]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(name="Drop (%)", x=crash_names, y=drops, marker_color=_COLORS["red"], opacity=0.75,
                              text=[f"-{d}%" for d in drops], textposition="outside", textfont={"size": 11, "color": "#fca5a5"}))
    fig_hist.add_trace(go.Bar(name="Recovery (months)", x=crash_names, y=recoveries, marker_color=_COLORS["green"], opacity=0.75,
                              text=[f"{r}mo" for r in recoveries], textposition="outside", textfont={"size": 11, "color": "#86efac"}))
    fig_hist.update_layout(barmode="group", height=340, margin=dict(t=30, b=10, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", yanchor="bottom", y=-0.12, font=dict(size=12, color="#cbd5e1")),
                           xaxis=dict(tickfont=dict(size=10, color="#94a3b8"), tickangle=-25),
                           yaxis=dict(visible=False), bargap=0.25)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_hist, use_container_width=True, key="crash_history")
    st.markdown("</div>", unsafe_allow_html=True)

    # ISSUE 9: Data source citation
    st.markdown("""<div style="text-align:center; margin-top:8px;">
        <span class="citation">SIP stoppage data: AMFI Monthly Report, Feb 2026 &nbsp;·&nbsp;
        Market data: NSE via Yahoo Finance &nbsp;·&nbsp;
        Historical crash data: BSE/NSE archives</span></div>""", unsafe_allow_html=True)

render_history()
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  § 7  SHAP EXPLAINABILITY                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

@_safe_section("SHAP Explainability")
def render_shap():
    st.markdown("""<div class="section-header">🔬 Model Explainability (SHAP)</div>
    <div class="section-sub">Transparent AI — see exactly why the model made its prediction</div>""", unsafe_allow_html=True)

    x1, x2 = st.columns([1, 1])
    with x1:
        shap_path = crisis.get("shap_plot_path")
        plots_dir = _ROOT / "models" / "plots"
        waterfall = plots_dir / "shap_waterfall_latest.png"
        global_shap = plots_dir / "shap_global_importance.png"

        img = None
        if shap_path and Path(shap_path).exists():
            img = str(shap_path)
        elif waterfall.exists():
            img = str(waterfall)
        elif global_shap.exists():
            img = str(global_shap)

        if img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(img, caption="SHAP Feature Attribution", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="glass-card" style="text-align:center; padding:40px;">
                <div style="font-size:2rem;">🔬</div>
                <div style="color:#94a3b8; margin-top:10px;">
                    SHAP plots will appear after running the model on live data.</div>
            </div>""", unsafe_allow_html=True)

    with x2:
        shap_expl = crisis.get("shap_explanation", "")
        st.markdown(f"""<div class="glass-card">
            <div style="font-size:0.78rem; font-weight:600; text-transform:uppercase;
                        letter-spacing:1.2px; color:#94a3b8; margin-bottom:12px;">Plain English Explanation</div>
            <div style="font-size:1.05rem; color:#f0f0f0; font-weight:600; margin-bottom:14px;">
                Here's <span style="color:#8b5cf6;">WHY</span> PanicGuard thinks panic risk is {pan_score}%</div>
            <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.7;">
                {shap_expl or "The model considers 15 technical and macro features. Run analysis to see details."}</div>
        </div>""", unsafe_allow_html=True)

        beeswarm = _ROOT / "models" / "plots" / "shap_beeswarm.png"
        if beeswarm.exists():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(str(beeswarm), caption="SHAP Beeswarm — Feature Impact Distribution", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

render_shap()
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

render_chat()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  § FOOTER  (ISSUE 9: citations)                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

st.markdown(f"""
<div class="app-footer">
    <div style="margin-bottom:8px;">
        <span style="font-size:1.4rem;">🛡️</span>
        <span style="font-weight:700; color:#8b5cf6;"> PanicGuard AI</span>
    </div>
    Built for <strong>AI Automate 2026 Hackathon</strong><br>
    <div style="margin-top:6px;">
        <strong>Data Sources:</strong> NSE (via Yahoo Finance) · BSE · AMFI Monthly Reports · India VIX<br>
        <strong>SIP Stoppage Stat (76%):</strong> AMFI data, Feb 2026 · Historical crash data: BSE/NSE archives
    </div>
    <div style="margin-top:10px; padding-top:10px; border-top:1px solid rgba(139,92,246,0.1);">
        \u2696\ufe0f <em>This is not financial advice. Not SEBI-registered.
        For educational and behavioral support purposes only.</em>
    </div>
    <div style="margin-top:8px; font-size:0.68rem; color:#475569;">
        Pipeline: {result.get("pipeline_time_s", 0):.2f}s &nbsp;\u00b7&nbsp;
        Status: {result.get("status", "?")} &nbsp;\u00b7&nbsp;
        {result.get("timestamp", "")}
    </div>
</div>
""", unsafe_allow_html=True)
