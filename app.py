"""
AI Workforce Risk Intelligence — Streamlit App
===============================================
Full analytics platform using both datasets + trained RandomForest model.

Datasets:
  - final_ai_workforce_dataset.csv   (14,995 rows, 28 cols, encoded labels)
  - ai_job_replacement_2020_2026_v2.csv  (15,000 rows, 20 cols, string labels, 2020-2026)

Artifacts:
  - model.pkl    RandomForestClassifier (150 trees, depth 10, 6 features)
  - scaler.pkl   StandardScaler
  - encoders.pkl empty dict (reserved)

Run with:
    .venv\\Scripts\\python.exe -m streamlit run app.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════ #
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════ #
st.set_page_config(
    page_title="AI Workforce Risk Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "**AI Workforce Risk Intelligence** — RandomForest + Streamlit",
    },
)

# ══════════════════════════════════════════════════════════════════════════════ #
#  LABEL MAPS  (Dataset 1 — encoded integers)
# ══════════════════════════════════════════════════════════════════════════════ #
COUNTRY_MAP  = {0:"Australia",1:"Brazil",2:"Canada",3:"China",4:"France",
                5:"Germany",6:"India",7:"Italy",8:"Japan",9:"Netherlands",
                10:"Singapore",11:"Spain",12:"UAE",13:"UK",14:"USA"}
INDUSTRY_MAP = {0:"Education",1:"Finance",2:"Government",3:"Healthcare",
                4:"Manufacturing",5:"Retail",6:"Technology",7:"Transportation"}
JOB_MAP      = {0:"Customer Service",1:"Data Scientist",2:"Driver",3:"Factory Worker",
                4:"Financial Analyst",5:"HR Manager",6:"Marketing Manager",
                7:"Nurse",8:"Software Engineer",9:"Teacher"}
EXP_MAP      = {0:"Entry",1:"Junior",2:"Lead",3:"Mid",4:"Senior"}
EDU_MAP      = {0:"Associate",1:"Bachelor's",2:"High School",3:"Master's",4:"PhD"}
RISK_MAP     = {0:"Low",1:"Medium",2:"High"}

# ── Chart theme constants
PLOT_BG    = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(99,179,237,0.08)"
FONT_COLOR = "#8aafd4"
RISK_COLORS = {"Low":"#68d391","Medium":"#fbd38d","High":"#fc8181"}

# ══════════════════════════════════════════════════════════════════════════════ #
#  CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════ #
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #080c18 0%, #0c1426 40%, #080c18 100%);
    color: #e0e8ff;
}

/* ── Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1228 0%, #080c18 100%);
    border-right: 1px solid rgba(99,179,237,0.12);
}
section[data-testid="stSidebar"] * { color: #c8d8f0 !important; }

/* ── Hero */
.hero-banner {
    background: linear-gradient(135deg, #141c38 0%, #0f2040 50%, #1a2545 100%);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 20px;
    padding: 38px 44px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,179,237,0.06) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(159,122,234,0.05) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem; font-weight: 700; letter-spacing: -0.5px; margin: 0 0 8px 0;
    background: linear-gradient(135deg, #63b3ed 0%, #9f7aea 50%, #63b3ed 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub  { font-size: 1.05rem; color: #7fa8cc; margin: 0 0 16px 0; font-weight: 400; }
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.08); border: 1px solid rgba(99,179,237,0.22);
    border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; color: #7fa8cc;
    margin-right: 6px; margin-top: 4px;
}

/* ── KPI cards */
.metric-card {
    background: linear-gradient(135deg, #111928, #182138);
    border: 1px solid rgba(99,179,237,0.16); border-radius: 14px;
    padding: 20px 22px; text-align: center; position: relative; overflow: hidden;
    transition: border-color .25s, transform .2s;
}
.metric-card:hover { border-color: rgba(99,179,237,0.38); transform: translateY(-2px); }
.metric-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.45), transparent);
}
.metric-value {
    font-size: 1.9rem; font-weight: 700; color: #63b3ed;
    font-family: 'JetBrains Mono', monospace; margin: 0;
}
.metric-label { font-size: 0.73rem; color: #6a90b8; text-transform: uppercase; letter-spacing: 1.3px; margin: 5px 0 0 0; }
.metric-delta { font-size: 0.8rem; margin-top: 6px; font-weight: 500; }
.delta-positive { color: #68d391; }
.delta-negative { color: #fc8181; }
.delta-neutral  { color: #9f7aea; }

/* ── Section headers */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 26px 0 12px 0; padding-bottom: 10px;
    border-bottom: 1px solid rgba(99,179,237,0.13);
}
.section-title { font-size: 0.92rem; font-weight: 600; color: #b8cce0; text-transform: uppercase; letter-spacing: 1.6px; margin: 0; }

/* ── Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
    background: rgba(99,179,237,0.05); border: 1px solid rgba(99,179,237,0.13);
    border-radius: 8px; color: #6a90b8; padding: 8px 18px; font-size: 0.86rem; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.15) !important;
    border-color: rgba(99,179,237,0.38) !important; color: #63b3ed !important;
}

/* ── Risk result */
.risk-result { border-radius: 14px; padding: 26px 30px; text-align: center; border: 1px solid; margin-top: 8px; }
.risk-low    { background: rgba(104,211,145,0.07); border-color: rgba(104,211,145,0.3); }
.risk-medium { background: rgba(251,211,141,0.07); border-color: rgba(251,211,141,0.3); }
.risk-high   { background: rgba(252,129,129,0.07); border-color: rgba(252,129,129,0.3); }
.risk-level  { font-size: 2.1rem; font-weight: 700; margin: 0 0 8px 0; }
.risk-low .risk-level { color: #68d391; } .risk-medium .risk-level { color: #fbd38d; } .risk-high .risk-level { color: #fc8181; }
.risk-description { font-size: 0.88rem; color: #8fa8c8; margin: 0; }

/* ── Prob bars */
.prob-bar-container { margin: 10px 0 5px 0; }
.prob-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: #6a90b8; margin-bottom: 4px; }
.prob-bar-bg { background: rgba(255,255,255,0.06); border-radius: 6px; height: 10px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 6px; }

/* ── Pills */
.insight-pill {
    display: inline-block; background: rgba(99,179,237,0.09);
    border: 1px solid rgba(99,179,237,0.22); border-radius: 20px;
    padding: 5px 13px; font-size: 0.8rem; color: #9abdd8; margin: 3px;
}
.insight-warning { background: rgba(252,129,129,0.09); border-color: rgba(252,129,129,0.22); color: #fc8181; }
.insight-success { background: rgba(104,211,145,0.09); border-color: rgba(104,211,145,0.22); color: #68d391; }

/* ── Info cards */
.info-card {
    background: linear-gradient(135deg, #111928, #182138);
    border: 1px solid rgba(99,179,237,0.13); border-radius: 14px;
    padding: 22px 24px; height: 100%;
}
.info-card h4 { color: #63b3ed; margin: 0 0 12px 0; font-size: 1rem; }
.info-card p  { color: #8fa8c8; font-size: 0.87rem; margin: 0; line-height: 1.75; }

/* ── Trend card */
.trend-card {
    background: rgba(99,179,237,0.04); border: 1px solid rgba(99,179,237,0.12);
    border-radius: 12px; padding: 16px 20px; margin: 6px 0;
}
.trend-title  { font-size: 0.8rem; color: #6a90b8; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 4px 0; }
.trend-value  { font-size: 1.4rem; font-weight: 700; color: #63b3ed; font-family: 'JetBrains Mono', monospace; margin: 0; }
.trend-delta  { font-size: 0.78rem; margin-top: 4px; }

/* ── Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #6b46c1); color: white;
    border: none; border-radius: 10px; padding: 12px 28px; font-weight: 600;
    font-family: 'Space Grotesk', sans-serif; font-size: 0.93rem; width: 100%; transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Footer */
.footer-note {
    text-align: center; color: #3a5070; font-size: 0.76rem;
    margin-top: 44px; padding-top: 16px;
    border-top: 1px solid rgba(99,179,237,0.08);
}

/* ── Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(8,12,24,0.8); }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.25); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════ #
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════ #
def apply_theme(fig, height=360):
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Space Grotesk", color=FONT_COLOR, size=12),
        height=height, margin=dict(l=14, r=14, t=40, b=14),
        legend=dict(bgcolor="rgba(10,16,34,0.7)", bordercolor="rgba(99,179,237,0.15)", borderwidth=1),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(size=11))
    return fig


def sh(icon: str, title: str):
    """Render styled section header."""
    st.markdown(
        f'<div class="section-header"><span>{icon}</span><p class="section-title">{title}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(value: str, label: str, delta: str = "", dtype: str = ""):
    delta_html = f'<p class="metric-delta delta-{dtype}">{delta}</p>' if delta else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<p class="metric-value">{value}</p>'
        f'<p class="metric-label">{label}</p>'
        f'{delta_html}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════ #
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════ #
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="⚙️ Loading ML model…")
def load_model():
    try:
        m = pickle.load(open(os.path.join(BASE, "model.pkl"), "rb"))
        s = pickle.load(open(os.path.join(BASE, "scaler.pkl"), "rb"))
        return m, s
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()


@st.cache_data(show_spinner="📊 Loading datasets…")
def load_datasets():
    # ── Dataset 1: encoded
    p1 = os.path.join(BASE, "final_ai_workforce_dataset.csv")
    try:
        df1 = pd.read_csv(p1)
    except FileNotFoundError:
        st.error(f"❌ Dataset not found: {p1}"); st.stop()

    df1["country_name"]    = df1["country"].map(COUNTRY_MAP)
    df1["industry_name"]   = df1["industry"].map(INDUSTRY_MAP)
    df1["job_role_name"]   = df1["job_role"].map(JOB_MAP)
    df1["exp_level_name"]  = df1["experience_level"].map(EXP_MAP)
    df1["edu_level_name"]  = df1["education_level"].map(EDU_MAP)
    df1["risk_label_name"] = df1["custom_risk_label"].map(RISK_MAP)

    # ── Dataset 2: string labels, 2020-2026 trends
    p2 = os.path.join(BASE, "ai_job_replacement_2020_2026_v2.csv")
    try:
        df2 = pd.read_csv(p2)
    except FileNotFoundError:
        st.error(f"❌ Dataset not found: {p2}"); st.stop()

    return df1, df2


# ── Bootstrap
inject_css()
model, scaler = load_model()
df1, df2 = load_datasets()


# ══════════════════════════════════════════════════════════════════════════════ #
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════ #
with st.sidebar:
    st.markdown(
        '<div style="padding:10px 0 6px">'
        '<div style="font-size:1.6rem;font-weight:700;color:#63b3ed;letter-spacing:-0.5px">🤖 AI Risk</div>'
        '<div style="font-size:0.75rem;color:#3a5070;margin-top:3px;letter-spacing:1px;text-transform:uppercase">Workforce Intelligence</div>'
        '</div>', unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**🔍 Global Filters**")
    st.caption("Applied to Dashboard, Deep Analysis & Data Explorer")

    all_countries  = sorted(df1["country_name"].dropna().unique().tolist())
    all_industries = sorted(df1["industry_name"].dropna().unique().tolist())
    all_exp        = sorted(df1["exp_level_name"].dropna().unique().tolist())

    sel_country  = st.selectbox("🌍 Country",         ["All"] + all_countries)
    sel_industry = st.selectbox("🏭 Industry",        ["All"] + all_industries)
    sel_exp      = st.selectbox("🎓 Experience",      ["All"] + all_exp)
    sel_risk     = st.selectbox("⚠️ Risk Category",   ["All", "Low", "Medium", "High"])

    st.markdown("---")
    # Trend tab filter
    st.markdown("**📅 Trend Analysis Filters**")
    trend_countries = ["All"] + sorted(df2["country"].dropna().unique().tolist())
    trend_industries = ["All"] + sorted(df2["industry"].dropna().unique().tolist())
    sel_trend_country  = st.selectbox("🌍 Trend Country",  trend_countries, key="tc")
    sel_trend_industry = st.selectbox("🏭 Trend Industry", trend_industries, key="ti")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#3a5070;line-height:1.85;">'
        '<b style="color:#4a6a88">📦 Model</b><br>'
        'RandomForest Classifier<br>150 estimators · depth 10<br>6 features · 3-class<br><br>'
        '<b style="color:#4a6a88">📊 Datasets</b><br>'
        'DS1: 14,995 rows · 28 cols<br>DS2: 15,000 rows · 20 cols<br>Years: 2020–2026<br><br>'
        '<b style="color:#4a6a88">🌍 Coverage</b><br>'
        '15 countries · 8 industries<br>10 job roles · 5 edu levels</div>',
        unsafe_allow_html=True,
    )


# ──  Apply DS1 filters
fdf = df1.copy()
if sel_country  != "All": fdf = fdf[fdf["country_name"]    == sel_country]
if sel_industry != "All": fdf = fdf[fdf["industry_name"]   == sel_industry]
if sel_exp      != "All": fdf = fdf[fdf["exp_level_name"]  == sel_exp]
if sel_risk     != "All": fdf = fdf[fdf["risk_label_name"] == sel_risk]

# Apply DS2 trend filters
tdf = df2.copy()
if sel_trend_country  != "All": tdf = tdf[tdf["country"]  == sel_trend_country]
if sel_trend_industry != "All": tdf = tdf[tdf["industry"] == sel_trend_industry]


# ══════════════════════════════════════════════════════════════════════════════ #
#  HERO
# ══════════════════════════════════════════════════════════════════════════════ #
st.markdown(
    '<div class="hero-banner">'
    '<p class="hero-title">AI Workforce Risk Intelligence</p>'
    '<p class="hero-sub">Predict, analyze & track how AI is reshaping jobs across industries, countries and time</p>'
    '<span class="hero-badge">🤖 RandomForest Classifier</span>'
    '<span class="hero-badge">📊 29,995 Records</span>'
    '<span class="hero-badge">🌍 15 Countries</span>'
    '<span class="hero-badge">🏭 8 Industries</span>'
    '<span class="hero-badge">📅 2020–2026</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════ #
#  TABS
# ══════════════════════════════════════════════════════════════════════════════ #
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard",
    "🔬 Deep Analysis",
    "📅 Trend Analysis",
    "🤖 AI Replacement",
    "🎯 Risk Predictor",
    "📋 Data Explorer",
    "ℹ️ About",
])


# ══════════════════════════════════════════════════ TAB 1  —  DASHBOARD ══════ #
with tab1:
    n         = len(fdf)
    avg_auto  = fdf["automation_risk_percent"].mean() if n else 0
    avg_sal_b = fdf["salary_before_usd"].mean()       if n else 0
    avg_sal_a = fdf["salary_after_usd"].mean()        if n else 0
    avg_ai    = fdf["ai_adoption_level"].mean()       if n else 0
    sal_delta = ((avg_sal_a - avg_sal_b) / avg_sal_b * 100) if avg_sal_b else 0
    hi_pct    = (fdf["risk_label_name"] == "High").mean() * 100 if n else 0
    avg_ai_dis = fdf["ai_disruption_intensity"].mean() if n and "ai_disruption_intensity" in fdf.columns else 0

    if n == 0:
        st.warning("⚠️ No records match the current filters. Try broadening your selection.")
        st.stop()

    # ── KPIs (7 cards)
    c = st.columns(7)
    kpis_data = [
        (f"{n:,}",               "Total Records",        "",                                          ""),
        (f"{avg_auto:.1f}%",     "Avg Automation Risk",  "↑ High" if avg_auto>50 else "↓ Moderate",  "negative" if avg_auto>50 else "positive"),
        (f"${avg_sal_b/1000:.0f}K", "Avg Salary Pre-AI", "",                                          ""),
        (f"${avg_sal_a/1000:.0f}K", "Avg Salary Post-AI", f"{'↑' if sal_delta>0 else '↓'} {abs(sal_delta):.1f}%", "positive" if sal_delta>0 else "negative"),
        (f"{avg_ai:.1f}",        "AI Adoption Score",    "",                                          ""),
        (f"{hi_pct:.1f}%",       "High Risk Jobs",       "↑ Watch" if hi_pct>20 else "↓ OK",         "negative" if hi_pct>20 else "positive"),
        (f"{avg_ai_dis:.1f}",    "AI Disruption Index",  "",                                          ""),
    ]
    for col, (v, l, d, dt) in zip(c, kpis_data):
        with col: kpi(v, l, d, dt)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Donut + Industry bar
    ca, cb = st.columns([1, 1.6])
    with ca:
        sh("🎯", "Risk Distribution")
        rc = fdf["risk_label_name"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
        fig = go.Figure(go.Pie(
            labels=rc.index, values=rc.values, hole=0.62,
            marker=dict(colors=["#68d391","#fbd38d","#fc8181"], line=dict(color="#080c18",width=2)),
            textinfo="label+percent", textfont=dict(size=12, color="#e0e8ff"),
            hovertemplate="<b>%{label}</b><br>%{value:,} records<br>%{percent}<extra></extra>",
        ))
        fig.add_annotation(text=f"<b>{n:,}</b>", x=0.5, y=0.5, showarrow=False, font=dict(color="#c0d0e8",size=18))
        fig.update_layout(showlegend=True, margin=dict(l=0,r=0,t=30,b=0))
        apply_theme(fig, 300); st.plotly_chart(fig, use_container_width=True)

    with cb:
        sh("🏭", "Automation Risk by Industry")
        ir = fdf.groupby("industry_name")["automation_risk_percent"].mean().sort_values().reset_index()
        fig = go.Figure(go.Bar(
            y=ir["industry_name"], x=ir["automation_risk_percent"], orientation="h",
            marker=dict(color=ir["automation_risk_percent"],
                        colorscale=[[0,"#68d391"],[0.5,"#fbd38d"],[1,"#fc8181"]],
                        showscale=True, colorbar=dict(thickness=7,len=0.6,tickfont=dict(size=10))),
            text=ir["automation_risk_percent"].round(1).astype(str)+"%",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
        ))
        fig.update_xaxes(range=[0,72], title="Avg Automation Risk %")
        apply_theme(fig, 300); st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Country AI + Salary
    cc, cd = st.columns(2)
    with cc:
        sh("🌍", "AI Adoption by Country")
        cai = fdf.groupby("country_name")["ai_adoption_level"].mean().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(cai, x="country_name", y="ai_adoption_level", color="ai_adoption_level",
                     color_continuous_scale=["#1a365d","#2b6cb0","#63b3ed","#bee3f8"],
                     labels={"country_name":"","ai_adoption_level":"AI Adoption"})
        fig.update_coloraxes(showscale=False); fig.update_xaxes(tickangle=-38)
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    with cd:
        sh("💰", "Salary Pre vs Post AI by Industry")
        sc = fdf.groupby("industry_name").agg(pre=("salary_before_usd","mean"), post=("salary_after_usd","mean")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Pre-AI",  x=sc["industry_name"], y=sc["pre"]/1000,  marker_color="rgba(99,179,237,0.75)",  hovertemplate="<b>%{x}</b><br>$%{y:.0f}K<extra></extra>"))
        fig.add_trace(go.Bar(name="Post-AI", x=sc["industry_name"], y=sc["post"]/1000, marker_color="rgba(159,122,234,0.75)", hovertemplate="<b>%{x}</b><br>$%{y:.0f}K<extra></extra>"))
        fig.update_layout(barmode="group"); fig.update_xaxes(tickangle=-30); fig.update_yaxes(title="Salary ($K)")
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Exp risk line + Scatter
    ce, cf = st.columns(2)
    with ce:
        sh("📈", "Risk Score by Experience Level")
        er = fdf.groupby("exp_level_name")["custom_risk_score"].mean().reindex(["Entry","Junior","Mid","Senior","Lead"]).dropna().reset_index()
        fig = px.line(er, x="exp_level_name", y="custom_risk_score", markers=True,
                      color_discrete_sequence=["#63b3ed"],
                      labels={"exp_level_name":"","custom_risk_score":"Avg Risk Score"})
        fig.update_traces(marker=dict(size=10,color="#9f7aea",line=dict(color="#63b3ed",width=2)), line=dict(width=2.5))
        fig.add_hrect(y0=0.6,y1=1.0, fillcolor="rgba(252,129,129,0.04)", line_width=0,
                      annotation_text="High Risk Zone", annotation_font=dict(size=10,color="#fc8181"), annotation_position="top right")
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    with cf:
        sh("🔥", "Skill Gap vs AI Adoption")
        samp = fdf.sample(min(2500,n), random_state=42)
        fig = px.scatter(samp, x="ai_adoption_level", y="skill_gap_index", color="risk_label_name",
                         color_discrete_map=RISK_COLORS, opacity=0.45,
                         labels={"ai_adoption_level":"AI Adoption","skill_gap_index":"Skill Gap","risk_label_name":"Risk"},
                         hover_data={"industry_name":True,"country_name":True})
        fig.update_traces(marker=dict(size=4))
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Disruption intensity + Education distribution
    cg, ch = st.columns(2)
    with cg:
        sh("⚡", "AI Disruption Intensity by Job Role")
        dis = fdf.groupby("job_role_name")["ai_disruption_intensity"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(dis, x="ai_disruption_intensity", y="job_role_name", orientation="h",
                     color="ai_disruption_intensity",
                     color_continuous_scale=["#1a365d","#63b3ed","#9f7aea","#fc8181"],
                     labels={"ai_disruption_intensity":"Avg Disruption Intensity","job_role_name":""})
        fig.update_coloraxes(showscale=False)
        apply_theme(fig,340); st.plotly_chart(fig, use_container_width=True)

    with ch:
        sh("🎓", "Risk Distribution by Education Level")
        edu_risk = fdf.groupby(["edu_level_name","risk_label_name"]).size().reset_index(name="count")
        fig = px.bar(edu_risk, x="edu_level_name", y="count", color="risk_label_name",
                     color_discrete_map=RISK_COLORS, barmode="stack",
                     category_orders={"edu_level_name":["High School","Associate","Bachelor's","Master's","PhD"],
                                       "risk_label_name":["Low","Medium","High"]},
                     labels={"edu_level_name":"Education","count":"Count","risk_label_name":"Risk"})
        apply_theme(fig,340); st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════ TAB 2  —  DEEP ANALYSIS ═══ #
with tab2:
    feature_names = ["Experience (Yrs)","Salary (Pre-AI)","Remote Score","AI Adoption","Wage Volatility","Skill Demand Growth"]
    imp_df = pd.DataFrame({"Feature":feature_names,"Importance":model.feature_importances_}).sort_values("Importance")

    c1, c2 = st.columns(2)
    with c1:
        sh("📊", "Feature Importance (RandomForest)")
        fig = go.Figure(go.Bar(
            y=imp_df["Feature"], x=imp_df["Importance"], orientation="h",
            marker=dict(color=imp_df["Importance"], colorscale=[[0,"#1a365d"],[0.5,"#2b6cb0"],[1,"#9f7aea"]], showscale=False),
            text=(imp_df["Importance"]*100).round(1).astype(str)+"%", textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        apply_theme(fig,340); st.plotly_chart(fig, use_container_width=True)

    with c2:
        sh("🔗", "Pearson Correlation with Automation Risk")
        ccols = ["automation_risk_percent","skill_gap_index","ai_adoption_level",
                 "years_of_experience","salary_before_usd","wage_volatility_index",
                 "remote_feasibility_score","ai_disruption_intensity","custom_risk_score"]
        exist = [c for c in ccols if c in fdf.columns]
        corr  = fdf[exist].corr()["automation_risk_percent"].drop("automation_risk_percent").sort_values()
        fig = go.Figure(go.Bar(
            y=corr.index, x=corr.values, orientation="h",
            marker=dict(color=corr.values, colorscale=[[0,"#fc8181"],[0.5,"#e0e8ff"],[1,"#68d391"]], cmid=0,
                        showscale=True, colorbar=dict(thickness=7,len=0.7)),
            hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>",
        ))
        apply_theme(fig,340); st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sh("🫧", "Job Role Risk Bubble Chart")
        jb = fdf.groupby("job_role_name").agg(avg_risk=("custom_risk_score","mean"), avg_auto=("automation_risk_percent","mean"), count=("job_id","count")).reset_index()
        fig = px.scatter(jb, x="avg_auto", y="avg_risk", size="count", color="avg_risk", text="job_role_name",
                         color_continuous_scale=["#68d391","#fbd38d","#fc8181"],
                         labels={"avg_auto":"Avg Automation Risk %","avg_risk":"Avg Risk Score"})
        fig.update_traces(textposition="top center", textfont=dict(size=9),
                          hovertemplate="<b>%{text}</b><br>Auto: %{x:.1f}%<br>Risk: %{y:.3f}<extra></extra>")
        fig.update_coloraxes(showscale=False); fig.update_layout(showlegend=False)
        apply_theme(fig,360); st.plotly_chart(fig, use_container_width=True)

    with c4:
        sh("💸", "Salary Change % by Job Role")
        js = fdf.groupby("job_role_name").agg(b=("salary_before_usd","mean"), a=("salary_after_usd","mean")).reset_index()
        js["chg"] = (js["a"]-js["b"])/js["b"]*100
        js = js.sort_values("chg")
        fig = go.Figure(go.Bar(
            y=js["job_role_name"], x=js["chg"], orientation="h",
            marker=dict(color=js["chg"], colorscale=[[0,"#fc8181"],[0.5,"#fbd38d"],[1,"#68d391"]], cmid=0, showscale=False),
            text=js["chg"].round(1).astype(str)+"%", textposition="outside",
            hovertemplate="<b>%{y}</b><br>Δ %{x:.1f}%<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_dash="dot")
        apply_theme(fig,360); st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        sh("📉", "Wage Volatility by Industry")
        wv = fdf.groupby("industry_name")["wage_volatility_index"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(wv, x="industry_name", y="wage_volatility_index", color="wage_volatility_index",
                     color_continuous_scale=["#1a365d","#63b3ed","#fc8181"],
                     labels={"industry_name":"","wage_volatility_index":"Avg Wage Volatility"})
        fig.update_coloraxes(showscale=False); fig.update_xaxes(tickangle=-30)
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    with c6:
        sh("📦", "Skill Gap Distribution by Experience Level")
        exp_order = ["Entry","Junior","Mid","Senior","Lead"]
        colors    = ["#fc8181","#fbd38d","#63b3ed","#68d391","#9f7aea"]
        fig = go.Figure()
        for exp, color in zip(exp_order, colors):
            vals = fdf[fdf["exp_level_name"]==exp]["skill_gap_index"].dropna().values
            if len(vals):
                fig.add_trace(go.Box(y=vals, name=exp, marker_color=color, line_color=color, boxmean=True))
        fig.update_layout(showlegend=False)
        apply_theme(fig); st.plotly_chart(fig, use_container_width=True)

    # Full correlation matrix
    sh("📐", "Full Numeric Correlation Heatmap")
    num_cols = ["automation_risk_percent","skill_gap_index","ai_adoption_level",
                "years_of_experience","salary_before_usd","salary_after_usd",
                "wage_volatility_index","remote_feasibility_score",
                "ai_disruption_intensity","custom_risk_score","skill_demand_growth_percent"]
    num_cols = [c for c in num_cols if c in fdf.columns]
    cm = fdf[num_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
        colorscale=[[0,"#fc8181"],[0.5,"#1a2545"],[1,"#68d391"]], zmid=0,
        text=cm.values.round(2), texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z}<extra></extra>",
        colorbar=dict(title="r", thickness=10, len=0.9),
    ))
    apply_theme(fig, 460); st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════ TAB 3  —  TREND ANALYSIS ═══ #
with tab3:
    n2 = len(tdf)
    if n2 == 0:
        st.warning("⚠️ No trend records match filters."); st.stop()

    # ── Top trend KPIs
    t_kpi_cols = st.columns(5)
    yr_min, yr_max = tdf["year"].min(), tdf["year"].max()
    early = tdf[tdf["year"]==yr_min]["automation_risk_percent"].mean()
    late  = tdf[tdf["year"]==yr_max]["automation_risk_percent"].mean()
    delta_auto = late - early
    avg_resk   = tdf["reskilling_urgency_score"].mean()
    avg_dis    = tdf["ai_disruption_intensity"].mean()
    avg_ai_rep = tdf["ai_replacement_score"].mean()
    avg_stp    = tdf["skill_transition_pressure"].mean()

    trend_kpis = [
        (f"{n2:,}",            "Trend Records",          "", ""),
        (f"{avg_ai_rep:.1f}",  "Avg AI Replacement Score", "", ""),
        (f"{avg_resk:.1f}",    "Avg Reskilling Urgency",   "", ""),
        (f"{avg_dis:.1f}",     "Avg AI Disruption",        "", ""),
        (f"{delta_auto:+.1f}%","Auto Risk Change", "2020→2026", "negative" if delta_auto>0 else "positive"),
    ]
    for col, (v,l,d,dt) in zip(t_kpi_cols, trend_kpis):
        with col: kpi(v,l,d,dt)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Automation risk over time by industry
    ta, tb = st.columns(2)
    with ta:
        sh("📈", "Automation Risk % Over Time by Industry")
        yr_ind = tdf.groupby(["year","industry"])["automation_risk_percent"].mean().reset_index()
        fig = px.line(yr_ind, x="year", y="automation_risk_percent", color="industry",
                      markers=True, color_discrete_sequence=px.colors.qualitative.Set2,
                      labels={"year":"Year","automation_risk_percent":"Avg Automation Risk %","industry":"Industry"})
        fig.update_traces(line_width=2, marker_size=6)
        apply_theme(fig, 360); st.plotly_chart(fig, use_container_width=True)

    with tb:
        sh("🤖", "AI Replacement Score Over Time by Job Role")
        yr_job = tdf.groupby(["year","job_role"])["ai_replacement_score"].mean().reset_index()
        fig = px.line(yr_job, x="year", y="ai_replacement_score", color="job_role",
                      markers=True, color_discrete_sequence=px.colors.qualitative.Pastel,
                      labels={"year":"Year","ai_replacement_score":"Avg AI Replacement Score","job_role":"Job Role"})
        fig.update_traces(line_width=2, marker_size=6)
        apply_theme(fig, 360); st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Salary trend + Reskilling urgency
    tc2, td2 = st.columns(2)
    with tc2:
        sh("💰", "Salary Change % Trend Over Years")
        yr_sal = tdf.groupby(["year","industry"])["salary_change_percent"].mean().reset_index()
        fig = px.area(yr_sal, x="year", y="salary_change_percent", color="industry",
                      color_discrete_sequence=px.colors.qualitative.Set3,
                      labels={"year":"Year","salary_change_percent":"Avg Salary Change %","industry":"Industry"})
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
        apply_theme(fig, 340); st.plotly_chart(fig, use_container_width=True)

    with td2:
        sh("🏥", "Reskilling Urgency Score Over Time")
        yr_resk = tdf.groupby(["year","job_role"])["reskilling_urgency_score"].mean().reset_index()
        fig = px.line(yr_resk, x="year", y="reskilling_urgency_score", color="job_role",
                      markers=True, color_discrete_sequence=px.colors.qualitative.Vivid,
                      labels={"year":"Year","reskilling_urgency_score":"Reskilling Urgency","job_role":"Job Role"})
        fig.update_traces(line_width=2, marker_size=6)
        apply_theme(fig, 340); st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Skill demand growth + AI disruption over time
    te2, tf2 = st.columns(2)
    with te2:
        sh("📚", "Skill Demand Growth % by Country (2020–2026)")
        yr_cnt = tdf.groupby(["year","country"])["skill_demand_growth_percent"].mean().reset_index()
        fig = px.line(yr_cnt, x="year", y="skill_demand_growth_percent", color="country",
                      markers=True, color_discrete_sequence=px.colors.qualitative.Alphabet,
                      labels={"year":"Year","skill_demand_growth_percent":"Skill Demand Growth %","country":"Country"})
        fig.update_traces(line_width=1.5, marker_size=5)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_dash="dot")
        apply_theme(fig, 340); st.plotly_chart(fig, use_container_width=True)

    with tf2:
        sh("⚡", "AI Disruption Intensity Over Years")
        yr_dis = tdf.groupby(["year","industry"])["ai_disruption_intensity"].mean().reset_index()
        fig = px.bar(yr_dis, x="year", y="ai_disruption_intensity", color="industry",
                     barmode="group", color_discrete_sequence=px.colors.qualitative.Bold,
                     labels={"year":"Year","ai_disruption_intensity":"Avg Disruption Intensity","industry":"Industry"})
        apply_theme(fig, 340); st.plotly_chart(fig, use_container_width=True)

    # ── Year-over-year animated bubble
    sh("🌐", "Country AI Adoption vs Automation Risk (Animated 2020–2026)")
    yr_cntry = tdf.groupby(["year","country"]).agg(
        auto=("automation_risk_percent","mean"),
        ai=("ai_adoption_level","mean"),
        sal=("salary_before_usd","mean"),
        n=("job_id","count")
    ).reset_index()
    fig = px.scatter(yr_cntry, x="ai", y="auto", size="n", color="country",
                     animation_frame="year", animation_group="country",
                     size_max=40, opacity=0.85,
                     range_x=[0,100], range_y=[0,100],
                     labels={"ai":"AI Adoption Level","auto":"Automation Risk %","country":"Country","n":"Record Count"},
                     color_discrete_sequence=px.colors.qualitative.Alphabet)
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>AI Adoption: %{x:.1f}<br>Auto Risk: %{y:.1f}%<extra></extra>")
    apply_theme(fig, 460); st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════ TAB 4  —  AI REPLACEMENT ═════ #
with tab4:
    sh("🤖", "AI Replacement Score — Overview")

    r_kpi_cols = st.columns(5)
    rep_kpis = [
        (f"{df2['ai_replacement_score'].mean():.1f}", "Avg AI Replacement Score", "", ""),
        (f"{df2[df2['automation_risk_category']=='High'].shape[0]:,}", "High Risk Records", "", "negative"),
        (f"{df2['reskilling_urgency_score'].mean():.1f}", "Avg Reskilling Urgency", "", ""),
        (f"{df2['skill_transition_pressure'].mean():.1f}", "Avg Skill Transition Pressure", "", ""),
        (f"{df2['ai_disruption_intensity'].mean():.1f}", "Avg AI Disruption Intensity", "", ""),
    ]
    for col, (v,l,d,dt) in zip(r_kpi_cols, rep_kpis):
        with col: kpi(v,l,d,dt)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Violin + Category bar
    ra, rb = st.columns(2)
    with ra:
        sh("🎻", "AI Replacement Score Distribution by Job Role")
        fig = go.Figure()
        job_roles = sorted(df2["job_role"].unique())
        colors_v  = px.colors.qualitative.Pastel
        for i, jr in enumerate(job_roles):
            vals = df2[df2["job_role"]==jr]["ai_replacement_score"].values
            fig.add_trace(go.Violin(
                y=vals, name=jr, box_visible=True, meanline_visible=True,
                line_color=colors_v[i % len(colors_v)], fillcolor=colors_v[i % len(colors_v)].replace("rgb","rgba").replace(")",",.3)"),
                opacity=0.8,
            ))
        fig.update_layout(violinmode="overlay", showlegend=True)
        apply_theme(fig, 400); st.plotly_chart(fig, use_container_width=True)

    with rb:
        sh("📊", "Automation Risk Category Distribution by Industry")
        cat_ind = df2.groupby(["industry","automation_risk_category"]).size().reset_index(name="count")
        fig = px.bar(cat_ind, x="industry", y="count", color="automation_risk_category",
                     color_discrete_map=RISK_COLORS, barmode="stack",
                     category_orders={"automation_risk_category":["Low","Medium","High"]},
                     labels={"industry":"","count":"Count","automation_risk_category":"Risk Category"})
        fig.update_xaxes(tickangle=-30)
        apply_theme(fig, 400); st.plotly_chart(fig, use_container_width=True)

    # Row 2: Reskilling heatmap + Skill transition
    rc2, rd2 = st.columns(2)
    with rc2:
        sh("🗺️", "Reskilling Urgency — Country × Job Role Heatmap")
        resk_piv = df2.pivot_table(values="reskilling_urgency_score", index="country", columns="job_role", aggfunc="mean").round(1)
        fig = go.Figure(go.Heatmap(
            z=resk_piv.values, x=resk_piv.columns.tolist(), y=resk_piv.index.tolist(),
            colorscale=[[0,"#1a3a5c"],[0.5,"#fbd38d"],[1,"#fc8181"]],
            text=resk_piv.values.round(1), texttemplate="%{text}", textfont=dict(size=9),
            hovertemplate="<b>%{y} × %{x}</b><br>Urgency: %{z:.1f}<extra></extra>",
            colorbar=dict(title="Urgency", thickness=9, len=0.85),
        ))
        apply_theme(fig, 420); st.plotly_chart(fig, use_container_width=True)

    with rd2:
        sh("⚙️", "Skill Transition Pressure by Industry")
        stp = df2.groupby("industry")["skill_transition_pressure"].mean().sort_values(ascending=False).reset_index()
        fig = go.Figure(go.Bar(
            x=stp["industry"], y=stp["skill_transition_pressure"],
            marker=dict(color=stp["skill_transition_pressure"],
                        colorscale=[[0,"#1a365d"],[0.5,"#63b3ed"],[1,"#fc8181"]], showscale=False),
            text=stp["skill_transition_pressure"].round(1), textposition="outside",
            hovertemplate="<b>%{x}</b><br>STP: %{y:.1f}<extra></extra>",
        ))
        fig.update_xaxes(tickangle=-30)
        apply_theme(fig, 360); st.plotly_chart(fig, use_container_width=True)

    # Row 3: AI Disruption vs Salary scatter + radar
    re2, rf2 = st.columns(2)
    with re2:
        sh("💥", "AI Disruption Intensity vs Salary Change")
        samp2 = df2.sample(min(3000, len(df2)), random_state=99)
        fig = px.scatter(samp2, x="ai_disruption_intensity", y="salary_change_percent",
                         color="automation_risk_category", color_discrete_map=RISK_COLORS,
                         opacity=0.5, size="wage_volatility_index", size_max=12,
                         labels={"ai_disruption_intensity":"AI Disruption Intensity",
                                 "salary_change_percent":"Salary Change %",
                                 "automation_risk_category":"Risk Category"},
                         hover_data={"job_role":True,"country":True})
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_dash="dot")
        apply_theme(fig, 380); st.plotly_chart(fig, use_container_width=True)

    with rf2:
        sh("🕸️", "Multi-Metric Radar by Automation Risk Category")
        metrics = ["automation_risk_percent","ai_replacement_score","reskilling_urgency_score",
                   "skill_transition_pressure","ai_disruption_intensity","wage_volatility_index"]
        radar_df = df2.groupby("automation_risk_category")[metrics].mean().reset_index()
        # Normalize each metric 0–1
        norm = radar_df.copy()
        for m in metrics:
            rng = norm[m].max() - norm[m].min()
            norm[m] = (norm[m]-norm[m].min())/rng if rng else 0

        fig = go.Figure()
        for _, row in norm.iterrows():
            cat = row["automation_risk_category"]
            vals = [row[m] for m in metrics] + [row[metrics[0]]]
            lbls = [m.replace("_"," ").title() for m in metrics] + [metrics[0].replace("_"," ").title()]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=lbls, fill='toself', name=cat,
                line_color=RISK_COLORS.get(cat,"#63b3ed"),
                fillcolor=RISK_COLORS.get(cat,"#63b3ed").replace("#","rgba(") if False else RISK_COLORS.get(cat,"#63b3ed"),
                opacity=0.55,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID_COLOR),
                       angularaxis=dict(gridcolor=GRID_COLOR)),
            showlegend=True,
        )
        apply_theme(fig, 380); st.plotly_chart(fig, use_container_width=True)

    # ── AI Replacement table
    sh("📋", "Top 10 Highest AI Replacement Score Records")
    top_rep = df2.nlargest(10,"ai_replacement_score")[["job_role","industry","country","year","ai_replacement_score","reskilling_urgency_score","automation_risk_category"]]
    top_rep.columns = ["Job Role","Industry","Country","Year","AI Replacement Score","Reskilling Urgency","Risk Category"]
    st.dataframe(top_rep.style.background_gradient(subset=["AI Replacement Score","Reskilling Urgency"], cmap="RdYlGn_r").format({"AI Replacement Score":"{:.2f}","Reskilling Urgency":"{:.2f}"}),
                 use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════ TAB 5  —  RISK PREDICTOR ════ #
with tab5:
    st.markdown(
        '<div style="background:rgba(99,179,237,0.05);border:1px solid rgba(99,179,237,0.14);'
        'border-radius:12px;padding:16px 20px;margin-bottom:20px;">'
        '<p style="margin:0;color:#7fa8cc;font-size:0.9rem;">🧠 Uses a trained '
        '<b style="color:#63b3ed">Random Forest Classifier</b> (150 trees, max depth 10) — '
        '6 input features → 3-class output: <b>Low / Medium / High</b> displacement risk.</p></div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1.15, 1])
    with col_l:
        st.markdown("#### ⚙️ Input Parameters")
        c1, c2 = st.columns(2)
        with c1:
            yrs_exp  = st.slider("🎓 Years of Experience",   0, 17, 5,  help="Total years in current role/field")
            remote   = st.slider("🏠 Remote Feasibility",   10, 100, 55, help="How easily could this job be done remotely? (0–100)")
            wage_vol = st.slider("📉 Wage Volatility Index", 0, 39, 8,  help="Historical wage instability index")
        with c2:
            salary   = st.number_input("💵 Current Salary (USD)", 15000, 500000, 80000, step=5000)
            ai_adopt = st.slider("🤖 AI Adoption Level",    0, 100, 50, help="How much AI is already used in this sector? (0–100)")
            skill_gr = st.slider("📈 Skill Demand Growth %", -32, 50, 5, help="Year-over-year change in skill demand")

        predict_btn = st.button("🔮  Predict My Risk Level", use_container_width=True)

        # Show scaler means for context
        with st.expander("📊 Dataset Averages (for reference)", expanded=False):
            means = scaler.mean_
            st.markdown(f"""
| Feature | Your Value | Dataset Avg |
|---------|-----------|------------|
| Years Experience | {yrs_exp} | {means[0]:.1f} |
| Salary (USD) | ${salary:,} | ${means[1]:,.0f} |
| Remote Feasibility | {remote} | {means[2]:.1f} |
| AI Adoption | {ai_adopt} | {means[3]:.1f} |
| Wage Volatility | {wage_vol} | {means[4]:.1f} |
| Skill Growth % | {skill_gr} | {means[5]:.1f} |
""")

    with col_r:
        st.markdown("#### 📊 Prediction Result")
        if predict_btn:
            arr   = np.array([[yrs_exp, salary, remote, ai_adopt, wage_vol, skill_gr]])
            scaled = scaler.transform(arr)
            pred   = model.predict(scaled)[0]
            proba  = model.predict_proba(scaled)[0]
            label  = RISK_MAP[pred]

            icons = {"Low":"🟢","Medium":"🟡","High":"🔴"}
            descs = {
                "Low":    "Low displacement risk. Skills remain relevant and AI adoption is manageable.",
                "Medium": "Moderate risk. Upskilling recommended to stay ahead of AI-driven changes.",
                "High":   "High displacement risk. Significant AI exposure — reskilling or role transition advisable.",
            }
            st.markdown(
                f'<div class="risk-result risk-{label.lower()}">'
                f'<p class="risk-level">{icons[label]} {label} Risk</p>'
                f'<p class="risk-description">{descs[label]}</p></div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>**Probability Breakdown**")
            for lbl, prob, color in zip(["Low","Medium","High"], proba, ["#68d391","#fbd38d","#fc8181"]):
                pct = prob*100
                st.markdown(
                    f'<div class="prob-bar-container">'
                    f'<div class="prob-label"><span>{lbl}</span><span>{pct:.1f}%</span></div>'
                    f'<div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div></div>'
                    '</div>', unsafe_allow_html=True,
                )

            st.markdown("<br>**🔍 Key Risk Drivers**")
            pills = []
            if ai_adopt > 70: pills.append(("⚠️ High AI adoption in role","warning"))
            if yrs_exp  < 3:  pills.append(("⚠️ Low experience — higher vulnerability","warning"))
            if skill_gr > 10: pills.append(("✅ Strong skill demand — protective","success"))
            if remote   > 70: pills.append(("⚠️ High remote feasibility — automation target","warning"))
            if wage_vol > 15: pills.append(("⚠️ High wage volatility — unstable market","warning"))
            if salary > 150000: pills.append(("✅ Higher salary often AI-complementary","success"))
            if not pills: pills.append(("ℹ️ Balanced risk profile",""))
            html = "".join(f'<span class="insight-pill insight-{k}">{t}</span>' if k else f'<span class="insight-pill">{t}</span>' for t,k in pills)
            st.markdown(f'<div style="margin-top:8px">{html}</div>', unsafe_allow_html=True)

        else:
            st.markdown(
                '<div style="background:rgba(99,179,237,0.03);border:1px dashed rgba(99,179,237,0.18);'
                'border-radius:12px;padding:50px 20px;text-align:center;color:#3a5070">'
                '<div style="font-size:3rem;margin-bottom:14px">🎯</div>'
                '<p style="margin:0;font-size:0.9rem">Set your parameters and click<br>'
                '<b style="color:#63b3ed">Predict My Risk Level</b></p></div>',
                unsafe_allow_html=True,
            )

    # ── Benchmark
    st.markdown("---")
    sh("📊", "Your Profile vs Dataset Averages")
    b_cols = st.columns(6)
    bench_items = [
        ("AI Adoption Level",   ai_adopt,  "ai_adoption_level",""),
        ("Remote Feasibility",  remote,    "remote_feasibility_score",""),
        ("Years Experience",    yrs_exp,   "years_of_experience"," yrs"),
        ("Wage Volatility",     wage_vol,  "wage_volatility_index",""),
        ("Automation Risk",     fdf["automation_risk_percent"].mean() if len(fdf) else 0, "automation_risk_percent","%"),
        ("Skill Gap Avg",       fdf["skill_gap_index"].mean() if len(fdf) else 0, "skill_gap_index",""),
    ]
    for col, (label, uval, cname, unit) in zip(b_cols, bench_items):
        avg = fdf[cname].mean() if len(fdf) else 0
        clr = "#63b3ed" if uval >= avg else "#fbd38d"
        delta = f"{'↑' if uval>=avg else '↓'} vs avg {avg:.1f}{unit}"
        with col:
            st.markdown(
                f'<div class="metric-card"><p class="metric-label">{label}</p>'
                f'<p class="metric-value" style="font-size:1.4rem">{uval}{unit}</p>'
                f'<p class="metric-delta" style="color:{clr}">{delta}</p></div>',
                unsafe_allow_html=True,
            )

    # ── Feature importance radar for this prediction vs average
    st.markdown("---")
    sh("🕸️", "Risk Factor Radar — Your Profile vs Dataset Average")
    radar_cols = ["years_of_experience","salary_before_usd","remote_feasibility_score",
                  "ai_adoption_level","wage_volatility_index","skill_demand_growth_percent"]
    avgs = [fdf[c].mean() if c in fdf.columns else 0 for c in radar_cols]
    user_vals = [yrs_exp, salary, remote, ai_adopt, wage_vol, skill_gr]
    labels_r  = ["Experience","Salary","Remote","AI Adoption","Wage Volatility","Skill Growth"]

    # Normalize 0–1 together
    combined = [list(map(lambda x: max(x,1e-6), user_vals)), list(map(lambda x: max(x,1e-6), avgs))]
    max_vals = [max(user_vals[i], avgs[i]) for i in range(len(labels_r))]
    norm_user = [user_vals[i]/max_vals[i] if max_vals[i] else 0 for i in range(len(labels_r))]
    norm_avg  = [avgs[i]/max_vals[i] if max_vals[i] else 0 for i in range(len(labels_r))]

    fig = go.Figure()
    for vals, name, color in [(norm_user+[norm_user[0]], "Your Profile","#63b3ed"), (norm_avg+[norm_avg[0]], "Dataset Avg","#9f7aea")]:
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels_r+[labels_r[0]], fill='toself',
            name=name, line_color=color, opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID_COLOR),
                   angularaxis=dict(gridcolor=GRID_COLOR)),
        showlegend=True,
    )
    apply_theme(fig, 400); st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════ TAB 6  —  DATA EXPLORER ═════ #
with tab6:
    # ── Dataset selector
    ds_choice = st.radio("Choose Dataset", ["📊 Dataset 1 — Encoded (14,995 rows)","📅 Dataset 2 — Trend (15,000 rows)"], horizontal=True)
    active_df = fdf if "Dataset 1" in ds_choice else tdf

    col_t, col_s = st.columns([2,1])
    with col_t:
        sh("📋", "Filtered Data Table")
        n_rows = st.selectbox("Rows to display", [25,50,100,250,500], index=0)
        st.dataframe(active_df.head(n_rows), use_container_width=True, height=380)
        st.caption(f"Showing {min(n_rows,len(active_df)):,} of {len(active_df):,} records")
        csv_bytes = active_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Filtered CSV", data=csv_bytes,
                           file_name=f"ai_workforce_filtered_{'ds1' if 'Dataset 1' in ds_choice else 'ds2'}.csv",
                           mime="text/csv")

    with col_s:
        sh("📐", "Summary Statistics")
        num_only = active_df.select_dtypes(include="number")
        stat = num_only.describe().round(2)
        stat.index = ["Count","Mean","Std","Min","25%","50%","75%","Max"]
        st.dataframe(stat, use_container_width=True, height=300)

        if "Dataset 1" in ds_choice:
            st.markdown("**🏆 Top 5 High-Risk Roles**")
            top = fdf.groupby("job_role_name")["custom_risk_score"].mean().sort_values(ascending=False).head(5).reset_index()
            top.columns=["Job Role","Avg Risk Score"]; top["Avg Risk Score"]=top["Avg Risk Score"].round(4)
            st.dataframe(top, use_container_width=True, hide_index=True)
        else:
            st.markdown("**🏆 Top 5 Highest Reskilling Urgency Roles**")
            top2 = df2.groupby("job_role")["reskilling_urgency_score"].mean().sort_values(ascending=False).head(5).reset_index()
            top2.columns=["Job Role","Avg Reskilling Urgency"]; top2["Avg Reskilling Urgency"]=top2["Avg Reskilling Urgency"].round(2)
            st.dataframe(top2, use_container_width=True, hide_index=True)

    # ── Country × Industry heatmap
    if "Dataset 1" in ds_choice:
        sh("🗺️", "Country × Industry — Custom Risk Score Heatmap")
        pivot = fdf.pivot_table(values="custom_risk_score", index="country_name", columns="industry_name", aggfunc="mean").round(3)
    else:
        sh("🗺️", "Country × Industry — Automation Risk % Heatmap")
        pivot = tdf.pivot_table(values="automation_risk_percent", index="country", columns="industry", aggfunc="mean").round(1)

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#1a3a5c"],[0.5,"#63b3ed"],[1,"#fc8181"]],
        text=pivot.values.round(2), texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="<b>%{y} × %{x}</b><br>Value: %{z}<extra></extra>",
        colorbar=dict(title="Value", thickness=10, len=0.85),
    ))
    apply_theme(fig, 450); st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════ TAB 7  —  ABOUT ══════ #
with tab7:
    st.markdown("<br>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown('<div class="info-card"><h4>🤖 ML Model</h4><p>'
                    'A <b>Random Forest Classifier</b> trained on 14,995 synthetic workforce records.<br><br>'
                    '• 150 decision trees<br>• Max depth: 10<br>• Criterion: Gini impurity<br>'
                    '• 6 input features<br>• 3-class output: Low / Medium / High<br>• StandardScaler preprocessing'
                    '</p></div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="info-card"><h4>📊 Datasets</h4><p>'
                    '<b>Dataset 1</b> — Encoded labels, 28 columns<br>'
                    '14,995 records · 15 countries<br>8 industries · 10 job roles<br><br>'
                    '<b>Dataset 2</b> — String labels, 20 columns<br>'
                    '15,000 records · 9 countries<br>8 industries · 10 job roles<br>'
                    'Years: 2020–2026 · Includes AI replacement score, reskilling urgency, disruption intensity'
                    '</p></div>', unsafe_allow_html=True)
    with a3:
        st.markdown('<div class="info-card"><h4>⚙️ 6 Model Features</h4><p>'
                    '1. <b>Years of Experience</b><br>'
                    '2. <b>Salary (Pre-AI, USD)</b><br>'
                    '3. <b>Remote Feasibility Score</b><br>'
                    '4. <b>AI Adoption Level</b><br>'
                    '5. <b>Wage Volatility Index</b><br>'
                    '6. <b>Skill Demand Growth %</b>'
                    '</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="info-card"><h4>🛠️ Tech Stack</h4><p>'
                    '• <b>Streamlit 1.56</b> — web framework<br>'
                    '• <b>Plotly 6.7</b> — interactive charts<br>'
                    '• <b>scikit-learn 1.8</b> — ML model & scaler<br>'
                    '• <b>Pandas 3.0 / NumPy 2.4</b> — data processing<br>'
                    '• <b>Google Fonts</b> — Space Grotesk + JetBrains Mono<br>'
                    '• <b>Custom CSS</b> — glassmorphism dark theme'
                    '</p></div>', unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="info-card"><h4>🚀 How to Run</h4><p>'
                    '1. Activate virtualenv:<br>'
                    '<code>.venv\\Scripts\\activate</code><br><br>'
                    '2. Install dependencies:<br>'
                    '<code>pip install -r requirements.txt</code><br><br>'
                    '3. Launch the app:<br>'
                    '<code>python -m streamlit run app.py</code><br><br>'
                    'Opens at <b>http://localhost:8501</b>'
                    '</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sh("📋", "Model Feature Importances")
    imp_full = pd.DataFrame({
        "Feature": ["Experience (Yrs)","Salary (Pre-AI)","Remote Score","AI Adoption","Wage Volatility","Skill Demand Growth"],
        "Importance": model.feature_importances_,
        "Importance %": (model.feature_importances_*100).round(2),
    }).sort_values("Importance %", ascending=False)

    fi1, fi2 = st.columns([1,2])
    with fi1:
        st.dataframe(imp_full[["Feature","Importance %"]], use_container_width=True, hide_index=True)
    with fi2:
        fig = px.bar(imp_full.sort_values("Importance"), x="Importance %", y="Feature", orientation="h",
                     color="Importance %", color_continuous_scale=["#1a365d","#63b3ed","#9f7aea"],
                     labels={"Importance %":"Importance (%)","Feature":""})
        fig.update_coloraxes(showscale=False)
        apply_theme(fig, 280); st.plotly_chart(fig, use_container_width=True)

    # Model performance simulation from scaler stats
    sh("🎯", "Model Input Space Summary (from trained StandardScaler)")
    mean_vals  = scaler.mean_
    scale_vals = scaler.scale_
    feat_names = ["Experience (Yrs)","Salary (Pre-AI, USD)","Remote Feasibility","AI Adoption Level","Wage Volatility","Skill Demand Growth %"]
    scaler_df  = pd.DataFrame({"Feature":feat_names, "Mean":mean_vals.round(2), "Std Dev":scale_vals.round(2)})
    scaler_df["Min Estimate"] = (mean_vals - 2*scale_vals).round(1)
    scaler_df["Max Estimate"] = (mean_vals + 2*scale_vals).round(1)
    st.dataframe(scaler_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════ #
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════ #
st.markdown(
    '<div class="footer-note">'
    'AI Workforce Risk Intelligence &nbsp;·&nbsp; RandomForest Classifier &nbsp;·&nbsp; '
    '29,995 total records · 15 countries · 8 industries · 10 job roles · 2020–2026<br>'
    'Built with Streamlit + Plotly &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; Python &nbsp;·&nbsp; '
    'Datasets: DS1 (14,995 rows) + DS2 (15,000 rows)'
    '</div>',
    unsafe_allow_html=True,
)
