# app.py
import base64
from pathlib import Path

import streamlit as st

from components.data import N_CATARACT, N_NORMAL, load_results
from tabs.overview import render_overview
from tabs.screen_yourself import render_screen_yourself
from tabs.under_the_lens import render_under_the_lens
from tabs.what_is_cataract import render_what_is_cataract

# =====================================================================
# PAGE CONFIG  (set ONCE at the top level)
# =====================================================================
st.set_page_config(page_title="IPVC Cataract Detection", page_icon="👁️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "assets" / "img"


def img_to_base64(filename: str) -> str:
    with open(IMG_DIR / filename, "rb") as f:
        return base64.b64encode(f.read()).decode()


# =====================================================================
# GLOBAL STYLES
# =====================================================================
st.markdown(
    """
    <style>
        :root { color-scheme: light !important; }
        html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif !important; }
        .stApp, [data-testid="stAppViewContainer"] { background: #F3F8F9 !important; }
        [data-testid="stHeader"] { background: transparent !important; }
        .block-container { max-width: 1280px; padding-top: 1.4rem; padding-bottom: 2rem; }

        /* ---------- sidebar ---------- */
        section[data-testid="stSidebar"] { background: linear-gradient(185deg, #0B2E3F 0%, #0D4552 58%, #0F7F7A 135%) !important; }
        section[data-testid="stSidebar"] * { color: #E8F3F4; }
        section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px !important; flex-direction: column; align-items: stretch; }
        section[data-testid="stSidebar"] div[role="radiogroup"] label { width: 100%; padding: 10px 14px; border-radius: 12px;
            cursor: pointer; transition: all .18s ease; font-weight: 600; }
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none !important; }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,0.09); transform: translateX(3px); }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(90deg, #14B8A6, #0F9D8A); box-shadow: 0 6px 16px rgba(20,184,166,0.35); }
        .brand { text-align: center; padding: 6px 0 4px 0; }
        .brand-title { font-size: 22px; font-weight: 800; color: #fff !important; line-height: 1.05; margin-top: 6px; }
        .brand-sub { font-size: 13px; color: #A8CBD0 !important; }
        .brand-line { height: 3px; width: 56px; margin: 10px auto 4px auto; border-radius: 3px; background: #2DD4BF; }
        .side-card { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12); border-radius: 14px;
            padding: 12px 14px; margin-top: 12px; font-size: 13px; }
        .side-card h5 { margin: 0 0 8px 0; font-size: 14px; color: #fff !important; }
        .side-chip { background: rgba(45,212,191,0.18); border: 1px solid rgba(45,212,191,0.45); border-radius: 9px;
            padding: 6px 10px; font-weight: 700; font-size: 12.5px; margin-bottom: 8px; }
        .side-row { display: flex; justify-content: space-between; padding: 3px 0; }
        .side-row b { color: #5EEAD4 !important; }

        /* ---------- hero ---------- */
        .hero { display: flex; gap: 16px; align-items: stretch; margin-bottom: 14px; }
        .hero-bar { width: 6px; border-radius: 6px; background: linear-gradient(180deg, #0F9D8A, #2DD4BF); flex-shrink: 0; }
        .hero-title { font-size: 31px; font-weight: 800; color: #0B2E3F; line-height: 1.12; }
        .hero-title em { font-style: normal; color: #0F9D8A; }
        .hero-text { font-size: 15px; color: #4B6470; margin-top: 4px; max-width: 880px; }
        .page-tag { font-size: 11.5px; letter-spacing: 1.6px; font-weight: 800; color: #0F9D8A; margin-bottom: 2px; }

        /* ---------- KPI tiles ---------- */
        .kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin: 6px 0 14px 0; }
        .kpi { border-radius: 16px; padding: 14px 16px; box-shadow: 0 4px 14px rgba(11,46,63,0.06); transition: transform .18s ease; }
        .kpi:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(11,46,63,0.12); }
        .kpi-top { display: flex; align-items: center; gap: 8px; }
        .kpi-icon { width: 30px; height: 30px; border-radius: 9px; display: inline-flex; align-items: center;
            justify-content: center; font-size: 15px; }
        .kpi-label { font-weight: 700; font-size: 13.5px; color: #1f3a47; }
        .kpi-value { font-size: 30px; font-weight: 800; margin-top: 6px; line-height: 1.05; }
        .kpi-cap { font-size: 11.5px; color: #6b8089; margin-top: 2px; }
        @media (max-width: 1000px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }

        /* ---------- cards ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border: 1px solid #E2EEF0 !important;
            border-radius: 18px !important; box-shadow: 0 6px 18px rgba(11,46,63,0.06); }
        div[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #E2EEF0; border-radius: 18px;
            box-shadow: 0 6px 18px rgba(11,46,63,0.06); padding: 6px 8px; }
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stPlotlyChart"] {
            background: transparent; border: none; box-shadow: none; padding: 0; }
        div[data-testid="stMetric"] { background: #fff; border: 1px solid #E2EEF0; border-radius: 14px; padding: 10px 14px; }
        .panel-title { font-weight: 800; font-size: 16px; color: #0B2E3F; margin-bottom: 2px; }
        .panel-sub { font-weight: 500; font-size: 12.5px; color: #6b8089; margin-left: 8px; }
        .two-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; align-items: stretch; }
        .equal-card { background: #fff; border-radius: 18px; padding: 22px; border: 1px solid #E2EEF0;
            box-shadow: 0 6px 18px rgba(11,46,63,0.06); }
        @media (max-width: 900px) { .two-grid { grid-template-columns: 1fr; } }
        .video-frame { position: relative; width: 100%; padding-top: 56.25%; border-radius: 14px; overflow: hidden; }
        .video-frame iframe { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0; }
        .info-card { background: #fff; border-radius: 18px; padding: 8px 22px 14px 22px; border: 1px solid #E2EEF0;
            box-shadow: 0 6px 18px rgba(11,46,63,0.06); height: 100%; }
        .info-card h4 { color: #0B2E3F; }
        .info-card ul { padding-left: 18px; margin: 0; }
        .info-card li { margin-bottom: 8px; font-size: 14px; }
        .info-card li span { color: #64808a; font-size: 13px; }
        .step-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .step { border-radius: 14px; padding: 12px 14px; font-size: 13.5px; }
        .step b { display: block; font-size: 14.5px; margin-bottom: 2px; }
        .step-no { display:inline-block; width:22px; height:22px; border-radius:99px; background:#0B2E3F; color:#fff !important;
            text-align:center; line-height:22px; font-size:12px; margin-right:6px; }

        /* ---------- story ---------- */
        .chapter { display: flex; gap: 14px; align-items: center; margin: 22px 0 8px 0; }
        .chapter-no { width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, #0F9D8A, #2DD4BF);
            color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 800; flex-shrink: 0;
            box-shadow: 0 6px 14px rgba(15,157,138,0.35); }
        .chapter-title { font-size: 20px; font-weight: 800; color: #0B2E3F; line-height: 1.1; }
        .chapter-q { font-size: 14px; color: #4B6470; }
        .takeaway { border-radius: 14px; padding: 14px 18px; margin: 10px 0 6px 0; font-size: 14.5px; color: #1f3a47; }
        .takeaway.finding { background: #E3F5F2; border-left: 6px solid #0F9D8A; }
        .takeaway.next { background: #FFF1DE; border-left: 6px solid #F2A541; }
        .case-list { color: #334155; font-size: 14px; padding-left: 20px; }
        .result-card { background: #fff; border-radius: 16px; padding: 16px 20px; border: 1px solid #E2EEF0; }
        .result-pill { display: inline-block; padding: 10px 26px; border-radius: 999px; color: #fff; font-size: 24px;
            font-weight: 800; box-shadow: 0 8px 18px rgba(0,0,0,0.15); }
        .disclaimer { background: #FFF6E5; border: 1px solid #F6DDA9; border-left: 6px solid #F2A541; border-radius: 12px;
            padding: 10px 16px; font-size: 13.5px; color: #5a4a1e; margin-bottom: 12px; }

        /* ---------- buttons ---------- */
        .stButton > button, button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
            background: linear-gradient(135deg, #0F9D8A, #14B8A6) !important; color: #fff !important; border: none !important;
            border-radius: 12px !important; padding: 10px 26px !important; font-weight: 700 !important;
            box-shadow: 0 6px 14px rgba(15,157,138,0.30); transition: transform .15s ease, box-shadow .15s ease; }
        .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(15,157,138,0.40); }
        .stButton > button p { color: #fff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# NAVIGATION (sidebar, with jump flags set by the "next" buttons)
# =====================================================================
NAV = {"overview": "🏠  Overview", "what": "📖  What is Cataract", "lens": "🔬  Under the Lens",
       "screen": "🩺  Screen a Photo"}

if "nav_tabs" not in st.session_state:
    st.session_state["nav_tabs"] = NAV["overview"]

for flag, target in [("go_to_what_is", "what"), ("go_to_under_the_lens", "lens"), ("go_to_screen", "screen")]:
    if st.session_state.get(flag):
        st.session_state["nav_tabs"] = NAV[target]
        st.session_state[flag] = False

res = load_results()
best = res.sort_values(["mcc", "f1"], ascending=False).iloc[0]
author_b64 = img_to_base64("author.png")

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <svg width="74" height="74" viewBox="0 0 74 74" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8 22V8h14M52 8h14v14M66 52v14H52M22 66H8V52" stroke="#2DD4BF" stroke-width="3.5" stroke-linecap="round"/>
            <path d="M10 37C18 25 28 19 37 19s19 6 27 18c-8 12-18 18-27 18S18 49 10 37Z" stroke="#fff" stroke-width="3" fill="#0B2E3F"/>
            <circle cx="37" cy="37" r="11" fill="#F0644F"/><circle cx="37" cy="37" r="5" fill="#0B2E3F"/>
            <circle cx="41" cy="33" r="2.2" fill="#fff"/>
          </svg>
          <div class="brand-title">IPVC<br>Cataract Detection</div>
          <div class="brand-sub">under class imbalance</div>
          <div class="brand-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.radio("Navigation", list(NAV.values()), key="nav_tabs", label_visibility="collapsed")
    st.markdown(
        f"""
        <div class="side-card"><h5>🏆 Best model</h5>
          <div class="side-chip">{best['model']} + {best['strategy']}</div>
          <div class="side-row"><span>Accuracy</span><b>{best['accuracy']:.1%}</b></div>
          <div class="side-row"><span>Recall</span><b>{best['recall']:.1%}</b></div>
          <div class="side-row"><span>F1</span><b>{best['f1']:.3f}</b></div>
          <div class="side-row"><span>MCC</span><b>{best['mcc']:.3f}</b></div>
        </div>
        <div class="side-card"><h5>🗂️ Dataset</h5>
          {N_NORMAL + N_CATARACT} fundus images<br>
          Class imbalance: {N_NORMAL} normal / {N_CATARACT} cataract
        </div>
        <div class="side-card" style="display:flex; gap:10px; align-items:center;">
          <img src="data:image/png;base64,{author_b64}" style="width:46px;height:46px;border-radius:99px;object-fit:cover;border:2px solid #2DD4BF;">
          <div><b style="color:#fff;">BAYU FATWA NEGARA</b><br><span style="font-size:11.5px;">XBDS3024N · Image Processing and Computer Vision</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab = st.session_state["nav_tabs"]

if tab == NAV["overview"]:
    render_overview()
elif tab == NAV["what"]:
    render_what_is_cataract()
elif tab == NAV["lens"]:
    render_under_the_lens()
else:
    render_screen_yourself()
