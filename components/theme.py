"""One stylesheet for the whole dashboard: bento cards, one type scale, one spacing unit (16 px)."""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');
:root { color-scheme: light !important;
  --ink:#0B2E3F; --muted:#5B7480; --line:#DCE8EA; --bg:#EDF3F4; --card:#FFFFFF;
  --teal:#0F9D8A; --teal-soft:#E3F5F2; --coral:#F0644F; --amber:#F2A541;
  --r:22px; --gap:16px; --shadow:0 1px 2px rgba(11,46,63,.04), 0 8px 24px rgba(11,46,63,.06);
  --head:'Plus Jakarta Sans', Inter, system-ui, sans-serif; --body:Inter, system-ui, -apple-system, 'Segoe UI', sans-serif; }

html, body, [class*="css"], .stApp, button, input, textarea, select { font-family: var(--body) !important; }
h1,h2,h3,h4,h5,.hero-title,.panel-title,.chapter-title,.kpi-value,.brand-title,.bento-title { font-family: var(--head) !important; letter-spacing:-.01em; }
.stApp, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--ink); }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 1240px; padding: 1.6rem 1.6rem 3rem; }
h3 { font-size: 1.15rem !important; font-weight: 800 !important; margin: 1.4rem 0 .6rem !important; padding: 0 !important; color: var(--ink); }
h4 { font-size: 1rem !important; font-weight: 700 !important; color: var(--ink); }
p, li { line-height: 1.6; }
[data-testid="stCaptionContainer"] { color: var(--muted); font-size: .82rem; }

/* one rhythm: every stack and every row of cards uses the same gap */
div[data-testid="stVerticalBlock"] { gap: var(--gap); }
div[data-testid="stHorizontalBlock"] { gap: var(--gap); align-items: stretch; }

/* ---------- bento card ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--card); border: 1px solid var(--line) !important;
  border-radius: var(--r) !important; box-shadow: var(--shadow); padding: 4px 6px; }
div[data-testid="stPlotlyChart"] { background: transparent; border: none; box-shadow: none; padding: 0; }
div[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--line) !important; border-radius: 18px !important;
  box-shadow: var(--shadow); overflow: hidden; }
div[data-testid="stExpander"] summary { font-family: var(--head); font-weight: 700; }
div[data-testid="stAlert"] { border-radius: 16px; }
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
img { border-radius: 14px; }
.panel-title { font-family: var(--head); font-weight: 800; font-size: 1rem; color: var(--ink); line-height: 1.25; }
.panel-sub { display:block; font-family: var(--body); font-weight: 500; font-size: .8rem; color: var(--muted); margin-top: 2px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(185deg, #0B2E3F 0%, #0D4552 58%, #0F7F7A 135%) !important; }
section[data-testid="stSidebar"] * { color: #E8F3F4; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 6px !important; flex-direction: column; align-items: stretch; }
section[data-testid="stSidebar"] div[role="radiogroup"] label { width: 100%; padding: 11px 14px; border-radius: 14px; cursor: pointer;
  transition: background .18s ease; font-family: var(--head); font-weight: 700; font-size: .92rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,.09); }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background: linear-gradient(90deg,#14B8A6,#0F9D8A); box-shadow: 0 6px 16px rgba(20,184,166,.35); }
.brand { text-align: center; padding: 6px 0 4px; }
.brand-title { font-size: 1.35rem; font-weight: 800; color: #fff !important; line-height: 1.1; margin-top: 8px; }
.brand-sub { font-size: .8rem; color: #A8CBD0 !important; }
.brand-line { height: 3px; width: 56px; margin: 12px auto 6px; border-radius: 3px; background: #2DD4BF; }
.side-card { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); border-radius: 16px; padding: 14px; margin-top: 12px; font-size: .82rem; }
.side-card h5 { margin: 0 0 8px; font-size: .88rem; color: #fff !important; }
.side-chip { background: rgba(45,212,191,.18); border: 1px solid rgba(45,212,191,.45); border-radius: 10px; padding: 6px 10px; font-weight: 700; font-size: .78rem; margin-bottom: 8px; }
.side-row { display: flex; justify-content: space-between; padding: 3px 0; }
.side-row b { color: #5EEAD4 !important; }

/* ---------- page header (hero card + animation card, same fixed height) ---------- */
.st-key-hero { background: linear-gradient(135deg, #0B2E3F 0%, #0D4F5C 70%, #0F7F7A 130%) !important; border: none !important; }
.st-key-hero * { color: #fff; }
.hero { display: flex; flex-direction: column; justify-content: center; height: 100%; padding: 8px 14px; }
.page-tag { font-family: var(--head); font-size: .72rem; letter-spacing: .14em; font-weight: 800; color: #5EEAD4 !important; margin-bottom: 6px; }
.hero-title { font-size: 2rem; font-weight: 800; line-height: 1.12; color: #fff; }
.hero-title em { font-style: normal; color: #5EEAD4; }
.hero-text { font-size: .95rem; color: #BFD9DD !important; margin-top: 8px; max-width: 760px; }

/* ---------- KPI bento tiles ---------- */
.kpi-row { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: var(--gap); }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: var(--r); padding: 16px 18px 14px; box-shadow: var(--shadow);
  display: flex; flex-direction: column; gap: 6px; position: relative; overflow: hidden; }
.kpi::before { content:''; position:absolute; left:0; top:0; right:0; height:4px; background: var(--tone); }
.kpi-top { display: flex; align-items: center; gap: 8px; }
.kpi-icon { width: 28px; height: 28px; border-radius: 9px; display: inline-flex; align-items: center; justify-content: center; font-size: .9rem; background: var(--tint); }
.kpi-label { font-size: .72rem; font-weight: 700; color: var(--muted); letter-spacing: .06em; text-transform: uppercase; }
.kpi-value { font-size: 1.9rem; font-weight: 800; line-height: 1.05; color: var(--tone); }
.kpi-cap { font-size: .76rem; color: var(--muted); }
@media (max-width: 1000px) { .kpi-row { grid-template-columns: repeat(2, minmax(0,1fr)); } }

/* ---------- generic bento grids built in HTML ---------- */
.bento { display: grid; gap: var(--gap); grid-template-columns: repeat(12, minmax(0,1fr)); }
.bento > .cell { background: var(--card); border: 1px solid var(--line); border-radius: var(--r); box-shadow: var(--shadow); padding: 20px 22px; }
.cell.s5 { grid-column: span 5; } .cell.s6 { grid-column: span 6; } .cell.s7 { grid-column: span 7; } .cell.s12 { grid-column: span 12; }
@media (max-width: 900px) { .bento > .cell { grid-column: span 12 !important; } }
.bento-title { font-size: 1rem; font-weight: 800; color: var(--ink); margin: 0 0 12px; }
.bento-sub { font-size: .8rem; color: var(--muted); margin: -8px 0 12px; }
.mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.mini { border-radius: 16px; padding: 14px 16px; background: var(--tint); }
.mini b { display: flex; align-items: center; gap: 8px; font-family: var(--head); font-size: .92rem; color: var(--ink); margin-bottom: 4px; }
.mini span.q { font-size: .82rem; color: #3E5967; line-height: 1.45; }
.step-no { display:inline-flex; align-items:center; justify-content:center; width: 24px; height: 24px; border-radius: 99px; background: var(--ink); color:#fff !important; font-size: .72rem; font-weight: 700; flex-shrink: 0; }
.flow { display: flex; flex-direction: column; gap: 10px; }
.flow-row { display: grid; grid-template-columns: 34px 1fr; gap: 12px; align-items: center; padding: 10px 12px; border-radius: 14px; background: #F4F9F9; }
.flow-no { background: var(--teal-soft); color: var(--teal); font-family: var(--head); font-weight: 800; font-size: .76rem; border-radius: 10px; text-align: center; padding: 7px 0; }
.flow-row b { display: block; font-family: var(--head); font-size: .9rem; color: var(--ink); }
.flow-row small { color: var(--muted); font-size: .78rem; }

.edu-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: var(--gap); }
.edu-card { background: var(--card); border: 1px solid var(--line); border-radius: var(--r); box-shadow: var(--shadow); padding: 20px 22px; }
.edu-card h4 { margin: 0 0 8px; font-size: 1rem !important; }
.edu-card p { margin: 0; font-size: .9rem; color: #3E5967; }
.edu-card p b { color: var(--ink); }
@media (max-width: 800px) { .edu-grid { grid-template-columns: 1fr; } }
.clarity-values { display:flex; flex-wrap:wrap; gap:8px; margin: 6px 0; }
.clarity-values span { background: var(--teal-soft); border-radius: 10px; padding: 6px 10px; font-size: .78rem; color: var(--ink); }
.clarity-values b { color: #0B665B; margin-right: 5px; }

/* ---------- story ---------- */
.chapter { display: flex; gap: 14px; align-items: center; margin: 8px 0 0; }
.chapter-no { width: 38px; height: 38px; border-radius: 12px; background: linear-gradient(135deg,#0F9D8A,#2DD4BF); color:#fff; display:flex; align-items:center; justify-content:center; font-family: var(--head); font-weight: 800; flex-shrink: 0; }
.chapter-title { font-size: 1.1rem; font-weight: 800; color: var(--ink); line-height: 1.15; }
.chapter-q { font-size: .85rem; color: var(--muted); }
.takeaway { border-radius: 16px; padding: 14px 18px; font-size: .9rem; color: #1f3a47; }
.takeaway.finding { background: var(--teal-soft); border-left: 5px solid var(--teal); }
.takeaway.next { background: #FFF1DE; border-left: 5px solid var(--amber); }
.result-pill { display: inline-block; padding: 10px 24px; border-radius: 999px; color: #fff; font-family: var(--head); font-size: 1.25rem; font-weight: 800; box-shadow: 0 8px 18px rgba(0,0,0,.15); }
.disclaimer { background: #FFF6E5; border: 1px solid #F6DDA9; border-left: 5px solid var(--amber); border-radius: 16px; padding: 12px 18px; font-size: .86rem; color: #5a4a1e; }

.st-key-gauge3d svg.main-svg { filter: drop-shadow(0 10px 8px rgba(11,46,63,.35)) drop-shadow(0 2px 1px rgba(11,46,63,.4)); }
.st-key-gauge3d { background: radial-gradient(ellipse at 50% 30%, #FFFFFF 0%, #E9F1F3 100%); border-radius: 18px; }

/* ---------- buttons ---------- */
.stButton > button, button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
  background: linear-gradient(135deg,#0F9D8A,#14B8A6) !important; color:#fff !important; border:none !important; border-radius: 14px !important;
  padding: 10px 24px !important; font-family: var(--head) !important; font-weight: 700 !important; box-shadow: 0 6px 14px rgba(15,157,138,.28);
  transition: transform .15s ease, box-shadow .15s ease; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(15,157,138,.38); }
.stButton > button p { color:#fff !important; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)
