"""Page 2 - DESCRIPTIVE analytics: what cataract is, who it affects, and what the data looks like."""
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_lottie import st_lottie

from components.data import N_CATARACT, N_NORMAL, TEST_CATARACT, TEST_NORMAL
from components.fundus_utils import clarity_maps, preprocess, read_image
from components.home_lottie import lottie_doctor
from components.ui import CATARACT, NORMAL, GREY, goto, hero, show, style, takeaway

IMG_DIR = Path(__file__).resolve().parents[1] / "assets" / "img"

RISKS = [
    ("Age", "Lens proteins change and clump over the decades."),
    ("Diabetes", "High blood sugar speeds up clouding of the lens."),
    ("Smoking", "Linked to earlier and denser cataract."),
    ("Sunlight (UV)", "Long exposure without eye protection adds up."),
    ("Steroid use", "Long-term steroid medicine can cloud the lens."),
    ("Eye injury or inflammation", "Past trauma or uveitis raises the risk."),
]
SYMPTOMS = [
    ("Blurred or cloudy vision", "Like looking through a frosted window."),
    ("Glare and halos", "Headlights and lamps look streaky at night."),
    ("Faded colours", "Colours look yellow or washed out."),
    ("Poor night vision", "Dim light becomes harder to cope with."),
    ("Frequent new glasses", "Prescription changes often but vision stays soft."),
    ("Double vision in one eye", "A doubled image that glasses do not fix."),
]


def _cards(title, items, colour):
    rows = "".join(
        f"<li><b>{name}</b><br><span>{text}</span></li>" for name, text in items
    )
    st.markdown(
        f"""<div class="info-card" style="border-top:5px solid {colour};">
              <h4>{title}</h4><ul>{rows}</ul></div>""",
        unsafe_allow_html=True,
    )


@st.cache_data
def _example(name):
    return read_image((IMG_DIR / name).read_bytes())


def _blur_demo():
    """Blur a normal fundus photograph and watch the clarity measurements fall."""
    base = _example("normal_1.jpg")
    left, right = st.columns([1, 2])
    with left:
        strength = st.slider("Simulated lens clouding", 0, 10, 0, help="Gaussian blur strength (sigma, in pixels).")
        enhanced = preprocess(base)
        if strength:
            size = int(strength * 4) | 1
            enhanced = cv2.GaussianBlur(enhanced, (size, size), strength)
        edges, corners, clarity = clarity_maps(enhanced)
        st.metric("Edge density", f"{clarity['edge_density']:.4f}")
        st.metric("Laplacian variance", f"{clarity['laplacian_variance']:.0f}")
    with right:
        a, b = st.columns(2)
        a.image(enhanced, caption="Photograph after simulated clouding", use_container_width=True)
        b.image(edges, caption="Edges the computer can still find", use_container_width=True, clamp=True)
    st.caption("A teaching simulation on one normal photograph. Real cataract also changes colour and contrast.")


def _dataset_sunburst():
    train_n = N_NORMAL - TEST_NORMAL
    train_c = N_CATARACT - TEST_CATARACT
    fig = go.Figure(go.Sunburst(
        ids=["all", "N", "C", "N-train", "N-test", "C-train", "C-test"],
        labels=["400 images", "Normal", "Cataract", "Training", "Test", "Training", "Test"],
        parents=["", "all", "all", "N", "N", "C", "C"],
        values=[0, 0, 0, train_n, TEST_NORMAL, train_c, TEST_CATARACT],
        branchvalues="remainder",
        marker=dict(colors=["white", NORMAL, CATARACT, "#7FCBC0", "#B5E3DC", "#F7A89C", "#FAC9C1"]),
        hovertemplate="<b>%{label}</b><br>%{value} images<extra></extra>",
        insidetextorientation="radial",
    ))
    return style(fig, height=420, legend=False, title="How the 400 images are split: class, then training or test")


def render_what_is_cataract():
    # ---------------- header ----------------
    hero_l, hero_r = st.columns([4, 1])
    with hero_l:
        hero("STEP 1 · DESCRIPTIVE ANALYTICS", "What is <em>cataract</em>, and what does the data look like?",
             "A <b>cataract</b> is a clouding of the eye's natural lens. Light scatters instead of focusing cleanly "
             "on the retina, so vision turns blurry, hazy or dull. The Global Burden of Disease study estimated "
             "<b>15.2 million</b> cases of blindness from cataract among adults aged 50 and older in 2020 "
             "(GBD 2019 Blindness and Vision Impairment Collaborators, 2021). It is treatable by surgery, and "
             "early checks help people act in time.")
    with hero_r:
        st_lottie(lottie_doctor, height=170, key="doctor_lottie")

    # ---------------- risk and symptoms ----------------
    c1, c2 = st.columns(2)
    with c1:
        _cards("Who is more likely to develop it?", RISKS, CATARACT)
    with c2:
        _cards("What might a person notice?", SYMPTOMS, NORMAL)
    st.caption("General information for awareness, not medical advice.")

    st.markdown("---")

    # ---------------- why a photo of the retina helps ----------------
    st.markdown("### Why would a photo of the back of the eye show a cataract?")
    st.markdown(
        "A **fundus photograph** shows the retina. The clouded lens sits in front of the retina, so "
        "the picture comes out **softer**: fewer sharp edges and less fine vessel detail. The study turns "
        "that loss of sharpness into numbers. Drag the slider to blur a normal photograph and watch the "
        "numbers fall."
    )
    _blur_demo()

    ex1, ex2 = st.columns(2)
    ex1.image(str(IMG_DIR / "normal_2.jpg"), caption="Normal fundus photograph", use_container_width=True)
    ex2.image(str(IMG_DIR / "cataract_2.jpg"), caption="Cataract fundus photograph", use_container_width=True)

    st.markdown("---")

    # ---------------- the data ----------------
    st.markdown("### The data behind the study")
    left, right = st.columns([3, 2])
    with left:
        show(_dataset_sunburst())
    with right:
        st.markdown(
            f"""<div class="info-card" style="border-top:5px solid {NORMAL};">
              <h4>400 fundus photographs</h4>
              <ul>
                <li><b>300 normal and 100 cataract</b><br><span>From a public retinal image dataset.</span></li>
                <li><b>320 for training, 80 for testing</b><br><span>A stratified split keeps the same 3 to 1 mix in both parts.</span></li>
                <li><b>Only 20 cataract images in the test set</b><br><span>Every result on the next pages is scored on these 80 unseen images.</span></li>
              </ul></div>""",
            unsafe_allow_html=True,
        )
        st.caption("Click a ring in the chart to zoom in.")

    takeaway(
        "Three of every four images are normal. A model that answers \"normal\" every time would score "
        "**75% accuracy** and find **no** cataract at all. This class imbalance is the central problem of the "
        "project, and the next page shows how it fools some models."
    )

    st.write("")
    mid = st.columns([3, 3, 3])[1]
    with mid:
        if st.button("NEXT: LOOK UNDER THE LENS ➜", key="to_diagnostic"):
            goto("go_to_under_the_lens")
