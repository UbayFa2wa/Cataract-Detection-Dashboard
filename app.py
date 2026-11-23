"""Four-page IPVC dashboard: original identity, authoritative final results."""
import base64
from pathlib import Path
import streamlit as st
from components.theme import inject
from components.data import N_NORMAL, N_CATARACT, load_results
from tabs.overview import render_overview
from tabs.what_is_cataract import render_what_is_cataract
from tabs.under_the_lens import render_under_the_lens
from tabs.screen_yourself import render_screen_yourself

st.set_page_config(page_title="IPVC Cataract Detection", page_icon="👁️", layout="wide")
ROOT = Path(__file__).resolve().parent

def img_to_base64(filename):
    return base64.b64encode((ROOT / "assets" / "img" / filename).read_bytes()).decode("ascii")

inject()

NAV = {
    "overview": "🏠  Overview",
    "what": "📖  What is Cataract",
    "lens": "🔬  Under the Lens",
    "screen": "🩺  Screen a Photo",
}

if "nav_tabs" not in st.session_state:
    st.session_state["nav_tabs"] = NAV["overview"]

for flag, target in [("go_to_what_is", "what"), ("go_to_under_the_lens", "lens"),
                     ("go_to_screen", "screen")]:
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
          <svg width="74" height="74" viewBox="0 0 74 74" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <path d="M8 22V8h14M52 8h14v14M66 52v14H52M22 66H8V52" stroke="#2DD4BF" stroke-width="3.5" stroke-linecap="round"/>
            <path d="M10 37C18 25 28 19 37 19s19 6 27 18c-8 12-18 18-27 18S18 49 10 37Z" stroke="#fff" stroke-width="3" fill="#0B2E3F"/>
            <circle cx="37" cy="37" r="11" fill="#F0644F"/><circle cx="37" cy="37" r="5" fill="#0B2E3F"/>
            <circle cx="41" cy="33" r="2.2" fill="#fff"/>
          </svg>
          <div class="brand-title">IPVC<br>Cataract Detection</div>
          <div class="brand-sub">under class imbalance</div>
          <div class="brand-line"></div>
        </div>
        """, unsafe_allow_html=True,
    )
    st.radio("Navigation", list(NAV.values()), key="nav_tabs", label_visibility="collapsed")
    st.markdown(
        f"""
        <div class="side-card"><h5>🏆 Highest MCC</h5>
          <div class="side-chip">{best['model']} · {best['strategy']}</div>
          <div class="side-row"><span>Accuracy</span><b>{best['accuracy']:.2%}</b></div>
          <div class="side-row"><span>Sensitivity</span><b>{best['recall']:.0%}</b></div>
          <div class="side-row"><span>Specificity</span><b>{best['specificity']:.0%}</b></div>
          <div class="side-row"><span>MCC</span><b>{best['mcc']:.3f}</b></div>
        </div>
        <div class="side-card"><h5>🗂️ Dataset</h5>
          <b>{N_NORMAL + N_CATARACT} fundus images</b><br>
          {N_NORMAL} Normal · {N_CATARACT} Cataract<br>
          <span style="font-size:11px;color:#A8CBD0">21 experiments · fixed 80-image test set</span>
        </div>
        <div class="side-card" style="display:flex;gap:10px;align-items:center;">
          <img src="data:image/png;base64,{author_b64}" alt="Project author" style="width:46px;height:46px;border-radius:99px;object-fit:cover;border:2px solid #2DD4BF;"/>
          <div><b style="color:#fff">BAYU FATWA NEGARA</b><br><span style="font-size:11px">XBDS3024N · IPVC</span></div>
        </div>
        """, unsafe_allow_html=True,
    )
    st.caption("University research prototype · not a clinical diagnosis")

page = st.session_state["nav_tabs"]
if page == NAV["overview"]:
    render_overview()
elif page == NAV["what"]:
    render_what_is_cataract()
elif page == NAV["lens"]:
    render_under_the_lens()
else:
    render_screen_yourself()
