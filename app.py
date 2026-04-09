import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="AI Workforce Risk Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────── #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 40%, #0a1020 100%); color: #e0e8ff; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1530 0%, #0a1020 100%); border-right: 1px solid rgba(99,179,237,0.15); }
section[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
.hero-banner { background: linear-gradient(135deg, #1a1f3a 0%, #0f2040 50%, #1a2545 100%); border: 1px solid rgba(99,179,237,0.2); border-radius: 16px; padding: 36px 40px; margin-bottom: 28px; position: relative; overflow: hidden; }
.hero-title { font-size: 2.4rem; font-weight: 700; background: linear-gradient(135deg, #63b3ed, #9f7aea, #63b3ed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 8px 0; letter-spacing: -0.5px; }
.hero-sub { font-size: 1rem; color: #7fa8cc; margin: 0; font-weight: 400; }
.metric-card { background: linear-gradient(135deg, #141c30 0%, #1a2545 100%); border: 1px solid rgba(99,179,237,0.18); border-radius: 12px; padding: 20px 24px; text-align: center; position: relative; overflow: hidden; }
.metric-card::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, rgba(99,179,237,0.5), transparent); }
.metric-value { font-size: 2rem; font-weight: 700; color: #63b3ed; font-family: 'JetBrains Mono', monospace; margin: 0; }
.metric-label { font-size: 0.78rem; color: #7fa8cc; text-transform: uppercase; letter-spacing: 1.2px; margin: 4px 0 0 0; }
.metric-delta { font-size: 0.82rem; margin-top: 6px; font-weight: 500; }
.delta-positive { color: #68d391; }
.delta-negative { color: #fc8181; }
.section-header { display: flex; align-items: center; gap: 10px; margin: 28px 0 14px 0; padding-bottom: 10px; border-bottom: 1px solid rgba(99,179,237,0.15); }
.section-title { font-size: 1rem; font-weight: 600; color: #c8d8f0; text-transform: uppercase; letter-spacing: 1.5px; margin: 0; }
.risk-result { border-radius: 14px; padding: 28px 32px; text-align: center; border: 1px solid; margin-top: 8px; }
.risk-low { background: rgba(104,211,145,0.08); border-color: rgba(104,211,145,0.35); }
.risk-medium { background: rgba(251,211,141,0.08); border-color: rgba(251,211,141,0.35); }
.risk-high { background: rgba(252,129,129,0.08); border-color: rgba(252,129,129,0.35); }
.risk-level { font-size: 2.2rem; font-weight: 700; margin: 0 0 8px 0; }
.risk-low .risk-level { color: #68d391; }
.risk-medium .risk-level { color: #fbd38d; }
.risk-high .risk-level { color: #fc8181; }
.risk-description { font-size: 0.9rem; color: #8fa8c8; margin: 0; }
.prob-bar-container { margin: 10px 0 5px 0; }
.prob-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: #7fa8cc; margin-bottom: 4px; }
.prob-bar-bg { background: rgba(255,255,255,0.07); border-radius: 6px; height: 10px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 6px; }
.insight-pill { display: inline-block; background: rgba(99,179,237,0.1); border: 1px solid rgba(99,179,237,0.25); border-radius: 20px; padding: 6px 14px; font-size: 0.82rem; color: #a0c4e8; margin: 4px; }
.insight-warning { background: rgba(252,129,129,0.1); border-color: rgba(252,129,129,0.25); color: #fc8181; }
.insight-success { background: rgba(104,211,145,0.1); border-color: rgba(104,211,145,0.25); color: #68d391; }
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: rgba(99,179,237,0.06); border: 1px solid rgba(99,179,237,0.15); border-radius: 8px; color: #7fa8cc; padding: 8px 20px; font-size: 0.88rem; }
.stTabs [aria-selected="true"] { background: rgba(99,179,237,0.18) !important; border-color: rgba(99,179,237,0.4) !important; color: #63b3ed !important; }
.stButton > button { background: linear-gradient(135deg, #2b6cb0, #6b46c1); color: white; border: none; border-radius: 10px; padding: 12px 28px; font-weight: 600; font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; width: 100%; }
.footer-note { text-align: center; color: #4a6080; font-size: 0.78rem; margin-top: 40px; padding-top: 16px; border-top: 1px solid rgba(99,179,237,0.1); }
</style>
""", unsafe_allow_html=True)

# ─── LABEL MAPS ─────────────────────────────────────────────────────────────── #
COUNTRY_MAP  = {0:"Australia",1:"Brazil",2:"Canada",3:"China",4:"France",5:"Germany",6:"India",7:"Italy",8:"Japan",9:"Netherlands",10:"Singapore",11:"Spain",12:"UAE",13:"UK",14:"USA"}
INDUSTRY_MAP = {0:"Education",1:"Finance",2:"Government",3:"Healthcare",4:"Manufacturing",5:"Retail",6:"Technology",7:"Transportation"}
JOB_MAP      = {0:"Customer Service",1:"Data Scientist",2:"Driver",3:"Factory Worker",4:"Financial Analyst",5:"HR Manager",6:"Marketing Manager",7:"Nurse",8:"Software Engineer",9:"Teacher"}
EXP_MAP      = {0:"Entry",1:"Junior",2:"Lead",3:"Mid",4:"Senior"}
EDU_MAP      = {0:"Associate",1:"Bachelor's",2:"High School",3:"Master's",4:"PhD"}
RISK_MAP     = {0:"Low",1:"Medium",2:"High"}

PLOT_BG    = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(99,179,237,0.08)"
FONT_COLOR = "#3b77c6"

def theme(fig, h=360):
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Space Grotesk", color=FONT_COLOR, size=12),
        height=h, margin=dict(l=16,r=16,t=40,b=16),
        legend=dict(bgcolor="rgba(14,20,40,0.6)", bordercolor="rgba(99,179,237,0.15)", borderwidth=1))
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(size=11))
    return fig

# ─── LOAD DATA ──────────────────────────────────────────────────────────────── #
@st.cache_resource
def load_model():
    m = pickle.load(open("model.pkl","rb"))
    s = pickle.load(open("scaler.pkl","rb"))
    return m, s

@st.cache_data
def load_df():
    df = pd.read_csv("final_ai_workforce_dataset.csv")
    df["country_name"]   = df["country"].map(COUNTRY_MAP)
    df["industry_name"]  = df["industry"].map(INDUSTRY_MAP)
    df["job_role_name"]  = df["job_role"].map(JOB_MAP)
    df["exp_level_name"] = df["experience_level"].map(EXP_MAP)
    df["edu_level_name"] = df["education_level"].map(EDU_MAP)
    df["risk_label_name"]= df["custom_risk_label"].map(RISK_MAP)
    return df

model, scaler = load_model()
df = load_df()

# ─── SIDEBAR ────────────────────────────────────────────────────────────────── #
with st.sidebar:
    st.markdown('<div style="padding:12px 0 20px;"><div style="font-size:1.6rem;font-weight:700;color:#63b3ed;">🤖 AI Risk Workforce Analytics</div></div>', unsafe_allow_html=True)
    st.markdown("**🔍 Global Filters**")
    sel_country  = st.selectbox("Country",  ["All"]+sorted(df["country_name"].dropna().unique().tolist()))
    sel_industry = st.selectbox("Industry", ["All"]+sorted(df["industry_name"].dropna().unique().tolist()))
    sel_exp      = st.selectbox("Experience Level", ["All"]+sorted(df["exp_level_name"].dropna().unique().tolist()))
    sel_risk     = st.selectbox("Risk Category", ["All","Low","Medium","High"])
    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#4a6080;line-height:1.7;"><b style="color:#5a7898;">Model Info</b><br>RandomForest Classifier<br>150 estimators · depth 10<br>6 features · 14,995 records<br>3-class risk prediction</div>', unsafe_allow_html=True)

fdf = df.copy()
if sel_country  != "All": fdf = fdf[fdf["country_name"]   == sel_country]
if sel_industry != "All": fdf = fdf[fdf["industry_name"]  == sel_industry]
if sel_exp      != "All": fdf = fdf[fdf["exp_level_name"] == sel_exp]
if sel_risk     != "All": fdf = fdf[fdf["risk_label_name"]== sel_risk]

# ─── HERO ───────────────────────────────────────────────────────────────────── #
st.markdown('<div class="hero-banner"><p class="hero-title">AI Workforce Risk Intelligence</p><p class="hero-sub">Predict, analyze, and understand how AI is reshaping job markets across industries and countries</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔬 Deep Analysis", "🎯 Risk Predictor", "📋 Data Explorer"])

# ═══════════════════════════════════════════════════════ TAB 1 — DASHBOARD ══ #
with tab1:
    n = len(fdf)
    avg_auto   = fdf["automation_risk_percent"].mean() if n else 0
    avg_sal_b  = fdf["salary_before_usd"].mean() if n else 0
    avg_sal_a  = fdf["salary_after_usd"].mean() if n else 0
    avg_ai     = fdf["ai_adoption_level"].mean() if n else 0
    sal_delta  = ((avg_sal_a - avg_sal_b)/avg_sal_b*100) if avg_sal_b else 0
    hi_pct     = (fdf["risk_label_name"]=="High").mean()*100 if n else 0

    cols = st.columns(6)
    kpis = [
        (f"{n:,}","Total Records","",""),
        (f"{avg_auto:.1f}%","Avg Automation Risk","↑ High" if avg_auto>50 else "↓ Moderate","negative" if avg_auto>50 else "positive"),
        (f"${avg_sal_b/1000:.0f}K","Avg Salary Pre-AI","",""),
        (f"${avg_sal_a/1000:.0f}K","Avg Salary Post-AI",f"{'↑' if sal_delta>0 else '↓'} {abs(sal_delta):.1f}%","positive" if sal_delta>0 else "negative"),
        (f"{avg_ai:.1f}","AI Adoption Score","",""),
        (f"{hi_pct:.1f}%","High Risk Jobs","↑ Watch" if hi_pct>20 else "↓ OK","negative" if hi_pct>20 else "positive"),
    ]
    for col,(val,label,delta,dtype) in zip(cols,kpis):
        with col:
            delta_html = f'<p class="metric-delta delta-{dtype}">{delta}</p>' if delta else ""
            st.markdown(f'<div class="metric-card"><p class="metric-value">{val}</p><p class="metric-label">{label}</p>{delta_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ca, cb = st.columns([1,1.6])
    with ca:
        st.markdown('<div class="section-header"><span>🎯</span><p class="section-title">Risk Distribution</p></div>', unsafe_allow_html=True)
        rc = fdf["risk_label_name"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
        fig = go.Figure(go.Pie(labels=rc.index, values=rc.values, hole=0.6,
            marker=dict(colors=["#68d391","#fbd38d","#fc8181"], line=dict(color="#0a0e1a",width=2)),
            textinfo="label+percent", textfont=dict(size=12,color="#e0e8ff"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"))
        fig.add_annotation(text=f"<b>{n:,}</b>", x=0.5, y=0.5, showarrow=False, font=dict(color="#c8d8f0",size=16))
        fig.update_layout(showlegend=True, margin=dict(l=0,r=0,t=30,b=0))
        theme(fig, 300); st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown('<div class="section-header"><span>🏭</span><p class="section-title">Automation Risk by Industry</p></div>', unsafe_allow_html=True)
        ir = fdf.groupby("industry_name")["automation_risk_percent"].mean().sort_values().reset_index()
        fig = go.Figure(go.Bar(y=ir["industry_name"], x=ir["automation_risk_percent"], orientation='h',
            marker=dict(color=ir["automation_risk_percent"], colorscale=[[0,"#68d391"],[0.5,"#fbd38d"],[1,"#fc8181"]], showscale=True, colorbar=dict(thickness=8,len=0.6,tickfont=dict(size=10))),
            text=ir["automation_risk_percent"].round(1).astype(str)+"%", textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>"))
        fig.update_xaxes(range=[0,65], title="Avg Automation Risk %")
        theme(fig, 300); st.plotly_chart(fig, use_container_width=True)

    cc, cd = st.columns(2)
    with cc:
        st.markdown('<div class="section-header"><span>🌍</span><p class="section-title">AI Adoption by Country</p></div>', unsafe_allow_html=True)
        cai = fdf.groupby("country_name")["ai_adoption_level"].mean().sort_values(ascending=False).head(12).reset_index()
        fig = px.bar(cai, x="country_name", y="ai_adoption_level", color="ai_adoption_level",
            color_continuous_scale=["#1a365d","#2b6cb0","#63b3ed","#bee3f8"],
            labels={"country_name":"","ai_adoption_level":"AI Adoption"})
        fig.update_coloraxes(showscale=False); fig.update_xaxes(tickangle=-35)
        theme(fig); st.plotly_chart(fig, use_container_width=True)

    with cd:
        st.markdown('<div class="section-header"><span>💰</span><p class="section-title">Salary Pre vs Post AI by Industry</p></div>', unsafe_allow_html=True)
        sc = fdf.groupby("industry_name").agg(pre=("salary_before_usd","mean"), post=("salary_after_usd","mean")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Pre-AI", x=sc["industry_name"], y=sc["pre"]/1000, marker_color="rgba(99,179,237,0.7)", hovertemplate="<b>%{x}</b><br>$%{y:.0f}K<extra></extra>"))
        fig.add_trace(go.Bar(name="Post-AI", x=sc["industry_name"], y=sc["post"]/1000, marker_color="rgba(159,122,234,0.7)", hovertemplate="<b>%{x}</b><br>$%{y:.0f}K<extra></extra>"))
        fig.update_layout(barmode="group"); fig.update_xaxes(tickangle=-30); fig.update_yaxes(title="Salary ($K)")
        theme(fig); st.plotly_chart(fig, use_container_width=True)

    ce, cf = st.columns(2)
    with ce:
        st.markdown('<div class="section-header"><span>📈</span><p class="section-title">Risk Score by Experience Level</p></div>', unsafe_allow_html=True)
        er = fdf.groupby("exp_level_name")["custom_risk_score"].mean().reindex(["Entry","Junior","Mid","Senior","Lead"]).dropna().reset_index()
        fig = px.line(er, x="exp_level_name", y="custom_risk_score", markers=True, color_discrete_sequence=["#63b3ed"],
            labels={"exp_level_name":"","custom_risk_score":"Avg Risk Score"})
        fig.update_traces(marker=dict(size=10,color="#9f7aea",line=dict(color="#63b3ed",width=2)), line=dict(width=2.5))
        fig.add_hrect(y0=0.6,y1=1.0,fillcolor="rgba(252,129,129,0.05)",line_width=0,annotation_text="High Risk Zone",annotation_font=dict(size=10,color="#fc8181"),annotation_position="top right")
        theme(fig); st.plotly_chart(fig, use_container_width=True)

    with cf:
        st.markdown('<div class="section-header"><span>🔥</span><p class="section-title">Skill Gap vs AI Adoption</p></div>', unsafe_allow_html=True)
        samp = fdf.sample(min(2000,n), random_state=42) if n>0 else fdf
        fig = px.scatter(samp, x="ai_adoption_level", y="skill_gap_index", color="risk_label_name",
            color_discrete_map={"Low":"#68d391","Medium":"#fbd38d","High":"#fc8181"}, opacity=0.5,
            labels={"ai_adoption_level":"AI Adoption","skill_gap_index":"Skill Gap","risk_label_name":"Risk"},
            hover_data={"industry_name":True,"country_name":True})
        fig.update_traces(marker=dict(size=4))
        theme(fig); st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════ TAB 2 — DEEP ANALYSIS ══ #
with tab2:
    features_names = ['Experience (Yrs)','Salary (Pre-AI)','Remote Score','AI Adoption','Wage Volatility','Skill Demand Growth']
    imp_df = pd.DataFrame({"Feature":features_names,"Importance":model.feature_importances_}).sort_values("Importance")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header"><span>📊</span><p class="section-title">Feature Importance</p></div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(y=imp_df["Feature"], x=imp_df["Importance"], orientation='h',
            marker=dict(color=imp_df["Importance"], colorscale=[[0,"#1a365d"],[0.5,"#2b6cb0"],[1,"#9f7aea"]], showscale=False),
            text=(imp_df["Importance"]*100).round(1).astype(str)+"%", textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>"))
        theme(fig,340); st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header"><span>🔗</span><p class="section-title">Correlation with Automation Risk</p></div>', unsafe_allow_html=True)
        ccols = ["automation_risk_percent","skill_gap_index","ai_adoption_level","years_of_experience","salary_before_usd","wage_volatility_index","remote_feasibility_score"]
        corr = fdf[ccols].corr()["automation_risk_percent"].drop("automation_risk_percent").sort_values()
        fig = go.Figure(go.Bar(y=corr.index, x=corr.values, orientation='h',
            marker=dict(color=corr.values, colorscale=[[0,"#fc8181"],[0.5,"#e0e8ff"],[1,"#68d391"]], cmid=0, showscale=True, colorbar=dict(thickness=8,len=0.7)),
            hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>"))
        theme(fig,340); st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-header"><span>👤</span><p class="section-title">Job Role Risk Bubble Chart</p></div>', unsafe_allow_html=True)
        jb = fdf.groupby("job_role_name").agg(avg_risk=("custom_risk_score","mean"), avg_auto=("automation_risk_percent","mean"), count=("job_id","count")).reset_index()
        fig = px.scatter(jb, x="avg_auto", y="avg_risk", size="count", color="avg_risk", text="job_role_name",
            color_continuous_scale=["#68d391","#fbd38d","#fc8181"],
            labels={"avg_auto":"Avg Automation Risk %","avg_risk":"Avg Risk Score"})
        fig.update_traces(textposition="top center", textfont=dict(size=9), hovertemplate="<b>%{text}</b><br>Auto: %{x:.1f}%<br>Risk: %{y:.3f}<extra></extra>")
        fig.update_coloraxes(showscale=False); fig.update_layout(showlegend=False)
        theme(fig,360); st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header"><span>💸</span><p class="section-title">Salary Change % by Job Role</p></div>', unsafe_allow_html=True)
        js = fdf.groupby("job_role_name").agg(b=("salary_before_usd","mean"), a=("salary_after_usd","mean")).reset_index()
        js["chg"] = (js["a"]-js["b"])/js["b"]*100
        js = js.sort_values("chg")
        fig = go.Figure(go.Bar(y=js["job_role_name"], x=js["chg"], orientation='h',
            marker=dict(color=js["chg"], colorscale=[[0,"#fc8181"],[0.5,"#fbd38d"],[1,"#68d391"]], cmid=0, showscale=False),
            text=js["chg"].round(1).astype(str)+"%", textposition="outside",
            hovertemplate="<b>%{y}</b><br>Δ %{x:.1f}%<extra></extra>"))
        fig.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_dash="dot")
        theme(fig,360); st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="section-header"><span>📉</span><p class="section-title">Wage Volatility by Industry</p></div>', unsafe_allow_html=True)
        wv = fdf.groupby("industry_name")["wage_volatility_index"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(wv, x="industry_name", y="wage_volatility_index", color="wage_volatility_index",
            color_continuous_scale=["#1a365d","#63b3ed","#fc8181"],
            labels={"industry_name":"","wage_volatility_index":"Avg Wage Volatility"})
        fig.update_coloraxes(showscale=False); fig.update_xaxes(tickangle=-30)
        theme(fig); st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.markdown('<div class="section-header"><span>📦</span><p class="section-title">Skill Gap Distribution by Experience</p></div>', unsafe_allow_html=True)
        fig = go.Figure()
        exp_order = ["Entry","Junior","Mid","Senior","Lead"]
        colors    = ["#fc8181","#fbd38d","#63b3ed","#68d391","#9f7aea"]
        for exp, color in zip(exp_order, colors):
            vals = fdf[fdf["exp_level_name"]==exp]["skill_gap_index"].dropna().values
            if len(vals): fig.add_trace(go.Box(y=vals, name=exp, marker_color=color, line_color=color, boxmean=True))
        fig.update_layout(showlegend=False)
        theme(fig); st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════ TAB 3 — RISK PREDICTOR ══ #
with tab3:
    st.markdown('<div style="background:rgba(99,179,237,0.06);border:1px solid rgba(99,179,237,0.15);border-radius:12px;padding:16px 20px;margin-bottom:20px;"><p style="margin:0;color:#7fa8cc;font-size:0.9rem;">🧠 Uses a trained <b style="color:#63b3ed">Random Forest Classifier</b> (150 trees, max depth 10) — 6 input features, 3-class output: Low / Medium / High risk.</p></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2,1])
    with col_l:
        st.markdown("#### ⚙️ Input Parameters")
        c1, c2 = st.columns(2)
        with c1:
            yrs_exp     = st.slider("🎓 Years of Experience", 0, 17, 5)
            remote      = st.slider("🏠 Remote Feasibility", 10, 100, 55)
            wage_vol    = st.slider("📉 Wage Volatility Index", 0, 39, 8)
        with c2:
            salary      = st.number_input("💵 Current Salary (USD)", 15000, 500000, 80000, step=5000)
            ai_adopt    = st.slider("🤖 AI Adoption Level", 0, 100, 50)
            skill_gr    = st.slider("📈 Skill Demand Growth %", -32, 50, 5)
        predict_btn = st.button("🔮 Predict Risk Level")

    with col_r:
        st.markdown("#### 📊 Prediction Result")
        if predict_btn:
            arr = np.array([[yrs_exp, salary, remote, ai_adopt, wage_vol, skill_gr]])
            pred  = model.predict(scaler.transform(arr))[0]
            proba = model.predict_proba(scaler.transform(arr))[0]
            label = RISK_MAP[pred]
            icons = {"Low":"🟢","Medium":"🟡","High":"🔴"}
            descs = {
                "Low":"Low displacement risk. Skills remain competitive and AI adoption is manageable.",
                "Medium":"Moderate risk detected. Upskilling recommended to stay ahead of AI-driven changes.",
                "High":"High displacement risk. Significant AI exposure — reskilling or role transition advisable."
            }
            st.markdown(f'<div class="risk-result risk-{label.lower()}"><p class="risk-level">{icons[label]} {label} Risk</p><p class="risk-description">{descs[label]}</p></div>', unsafe_allow_html=True)
            st.markdown("<br>**Probability Breakdown**")
            bar_colors = ["#68d391","#fbd38d","#fc8181"]
            for lbl, prob, color in zip(["Low","Medium","High"], proba, bar_colors):
                pct = prob*100
                st.markdown(f'<div class="prob-bar-container"><div class="prob-label"><span>{lbl}</span><span>{pct:.1f}%</span></div><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div></div></div>', unsafe_allow_html=True)
            st.markdown("<br>**🔍 Key Drivers**")
            pills = []
            if ai_adopt > 70:    pills.append(("⚠️ High AI adoption in role","warning"))
            if yrs_exp < 3:      pills.append(("⚠️ Low experience — higher vulnerability","warning"))
            if skill_gr > 10:    pills.append(("✅ Strong skill demand — protective","success"))
            if remote > 70:      pills.append(("⚠️ High remote feasibility — automation target","warning"))
            if wage_vol > 15:    pills.append(("⚠️ High wage volatility — unstable market","warning"))
            if salary > 150000:  pills.append(("✅ Higher salary roles often AI-complementary","success"))
            if not pills:        pills.append(("ℹ️ Balanced risk profile",""))
            html = "".join(f'<span class="insight-pill insight-{k}">{t}</span>' if k else f'<span class="insight-pill">{t}</span>' for t,k in pills)
            st.markdown(f'<div style="margin-top:8px;">{html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:rgba(99,179,237,0.04);border:1px dashed rgba(99,179,237,0.2);border-radius:12px;padding:40px 20px;text-align:center;color:#4a6080;"><div style="font-size:2.5rem;margin-bottom:12px;">🎯</div><p style="margin:0;font-size:0.9rem;">Set your parameters and click<br><b style="color:#63b3ed">Predict Risk Level</b> to see results</p></div>', unsafe_allow_html=True)

    # Benchmark
    st.markdown("---")
    st.markdown('<div class="section-header"><span>📊</span><p class="section-title">Benchmark vs Dataset Average</p></div>', unsafe_allow_html=True)
    bench = st.columns(3)
    for col, (label, uval, cname, unit) in zip(bench,[
        ("AI Adoption Level", ai_adopt, "ai_adoption_level", ""),
        ("Remote Feasibility", remote, "remote_feasibility_score", ""),
        ("Years of Experience", yrs_exp, "years_of_experience", " yrs"),
    ]):
        avg = fdf[cname].mean()
        clr = "#63b3ed" if uval > avg else "#fbd38d"
        delta = f"{'↑' if uval>avg else '↓'} vs avg {avg:.1f}{unit}"
        with col:
            st.markdown(f'<div class="metric-card"><p class="metric-label">{label}</p><p class="metric-value" style="font-size:1.5rem;">{uval}{unit}</p><p class="metric-delta" style="color:{clr};">{delta}</p></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════ TAB 4 — DATA EXPLORER ══ #
with tab4:
    col_t, col_s = st.columns([2,1])
    with col_t:
        st.markdown('<div class="section-header"><span>📋</span><p class="section-title">Filtered Dataset</p></div>', unsafe_allow_html=True)
        disp = fdf[["job_role_name","industry_name","country_name","exp_level_name","edu_level_name",
                     "years_of_experience","salary_before_usd","salary_after_usd","salary_change_percent",
                     "automation_risk_percent","ai_adoption_level","skill_gap_index","remote_feasibility_score",
                     "wage_volatility_index","custom_risk_score","risk_label_name"]].copy()
        disp.columns = ["Job Role","Industry","Country","Experience","Education","Exp Yrs",
                         "Salary Before","Salary After","Salary Δ%","Auto Risk%","AI Adoption",
                         "Skill Gap","Remote Score","Wage Volatility","Risk Score","Risk Level"]
        n_rows = st.selectbox("Rows to display",[25,50,100,250], index=0)
        st.dataframe(
            disp.head(n_rows).style
                .background_gradient(subset=["Risk Score","Auto Risk%","AI Adoption"], cmap="RdYlGn_r")
                .format({"Salary Before":"${:,.0f}","Salary After":"${:,.0f}","Salary Δ%":"{:.1f}%","Auto Risk%":"{:.1f}%","Risk Score":"{:.3f}"}),
            use_container_width=True, height=380)
        st.caption(f"Showing {min(n_rows,len(fdf)):,} of {len(fdf):,} records")

    with col_s:
        st.markdown('<div class="section-header"><span>📐</span><p class="section-title">Summary Statistics</p></div>', unsafe_allow_html=True)
        stat = fdf[["automation_risk_percent","skill_gap_index","ai_adoption_level",
                     "remote_feasibility_score","wage_volatility_index","custom_risk_score"]].describe().round(2)
        stat.index = ["Count","Mean","Std","Min","25%","50%","75%","Max"]
        stat.columns = ["Auto Risk","Skill Gap","AI Adopt","Remote","Wage Vol","Risk Score"]
        st.dataframe(stat, use_container_width=True, height=300)
        st.markdown("**Top 5 High-Risk Roles**")
        top = fdf.groupby("job_role_name")["custom_risk_score"].mean().sort_values(ascending=False).head(5).reset_index()
        top.columns = ["Job Role","Avg Risk Score"]
        top["Avg Risk Score"] = top["Avg Risk Score"].round(4)
        st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header"><span>🗺️</span><p class="section-title">Country × Industry Risk Heatmap</p></div>', unsafe_allow_html=True)
    pivot = fdf.pivot_table(values="custom_risk_score", index="country_name", columns="industry_name", aggfunc="mean").round(3)
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,"#1a3a5c"],[0.5,"#63b3ed"],[1,"#fc8181"]],
        text=pivot.values.round(2), texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="<b>%{y} × %{x}</b><br>Risk: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Risk Score", thickness=10, len=0.8)))
    theme(fig, 440); st.plotly_chart(fig, use_container_width=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────── #
st.markdown('<div class="footer-note">AI Workforce Risk Intelligence &nbsp;·&nbsp; RandomForest Classifier &nbsp;·&nbsp; 14,995 records · 15 countries · 8 industries · 10 job roles<br>Built with Streamlit + Plotly &nbsp;·&nbsp; sklearn</div>', unsafe_allow_html=True)