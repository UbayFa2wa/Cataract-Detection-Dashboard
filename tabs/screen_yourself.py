"""Original three-panel Screen a Photo dashboard, connected to final validated model assets.

This module intentionally restores the old published visual flow, but it never
uses the earlier test_scores.csv or silently substitutes a different classifier.
"""
from pathlib import Path
import hashlib

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.data import load_clarity, model_result
from components.fundus_utils import (
    cnn_available, svm_available, preprocess, read_image, validate_fundus,
    score_cnn, score_svm, clarity_maps,
)
from components.ui import (
    CATARACT, NORMAL, CLASS_COLOUR, page_header, panel_title, kpi_tiles,
    chapter, show, style,
)
from components.gauge import gauge_svg
from components.home_lottie import lottie_doctor

PANEL_H = 560

IMG_DIR = Path(__file__).resolve().parents[1] / "assets" / "img"
SAMPLES = {
    "Normal 1": "normal_1.jpg", "Normal 2": "normal_2.jpg",
    "Cataract 1": "cataract_1.jpg", "Cataract 2": "cataract_2.jpg",
}
CHOICES = {
    "EfficientNetB0 + class weights (higher sensitivity)": ("EfficientNetB0", "class weights"),
    "EfficientNetB0 baseline (highest MCC)": ("EfficientNetB0", "baseline"),
    "SVM + SMOTE (classical comparison)": ("SVM", "SMOTE"),
}
DISCLAIMER = (
    "Research demonstration for a university course. This is **not a medical device** "
    "and cannot diagnose or exclude cataract."
)


@st.cache_data(show_spinner=False, max_entries=24)
def _analyse(img_bytes, model, strategy):
    """Preprocessing and predictions are exactly those in final components/fundus_utils.py."""
    if len(img_bytes) > 10 * 1024 * 1024:
        return {"ok": False, "msg": "Please upload a photograph smaller than 10 MB."}
    rgb = read_image(img_bytes)
    if rgb is None:
        return {"ok": False, "msg": "The file could not be decoded as a photograph."}
    ok, message = validate_fundus(rgb)
    if not ok:
        return {"ok": False, "original": rgb, "msg": message}
    enhanced = preprocess(rgb)
    edges, corners, clarity = clarity_maps(enhanced)
    if model == "SVM":
        score = score_svm(enhanced)
    else:
        score = score_cnn(enhanced, strategy=strategy)
    return {
        "ok": True, "original": rgb, "enhanced": enhanced,
        "edges": edges, "corners": corners, "clarity": clarity,
        "score": float(score),
    }


def _bands(lo, hi, light, deep, n=6):
    """Split [lo, hi] into n bands that darken from `light` to `deep`, which gives the dial a bevelled, solid look."""
    def mix(i):
        f = i / (n - 1)
        return "#%02X%02X%02X" % tuple(round(int(light[k:k + 2], 16) * (1 - f) + int(deep[k:k + 2], 16) * f) for k in (1, 3, 5))
    edges = [lo + (hi - lo) * i / n for i in range(n + 1)]
    return [{"range": [edges[i], edges[i + 1]], "color": mix(i)} for i in range(n)]


def _gauge(score, cutoff, is_cnn):
    """Bold, bevelled score dial; score != calibrated disease probability."""
    if is_cnn:
        lo, hi, mid = 0.0, 1.0, cutoff
        title, fmt = "Uncalibrated model score", ".3f"
    else:
        hi = max(2.0, min(10.0, abs(score) * 1.25)); lo, mid = -hi, 0.0
        title, fmt = "SVM decision value", ".2f"
    steps = _bands(lo, mid, "#7FE3D3", "#07806F") + _bands(mid, hi, "#FF9A86", "#D2321B")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(valueformat=fmt, font=dict(size=46, color="#0B2E3F", family="Plus Jakarta Sans, Inter, sans-serif")),
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, family="Plus Jakarta Sans, Inter, sans-serif")),
        gauge=dict(
            axis=dict(range=[lo, hi], tickwidth=2, tickcolor="#0B2E3F", tickfont=dict(size=12, color="#0B2E3F")),
            bar=dict(color="#0B2E3F", thickness=0.34, line=dict(color="#FFFFFF", width=2)),
            bgcolor="#FFFFFF", borderwidth=5, bordercolor="#0B2E3F", steps=steps,
            threshold=dict(line=dict(color="#111111", width=6), thickness=0.95, value=mid),
        ),
    ))
    return style(fig, height=250, legend=False)


def _processing_figure(out):
    corner_img = cv2.dilate(out["corners"].astype(np.uint8) * 255,
                            np.ones((3, 3), np.uint8))
    panels = [cv2.resize(out["original"], (224, 224)), out["enhanced"],
              np.dstack([out["edges"]] * 3), np.dstack([corner_img] * 3)]
    captions = ["Original", "Crop + CLAHE", "Canny edges", "Harris corners"]
    fig = make_subplots(rows=1, cols=4, subplot_titles=captions, horizontal_spacing=0.01)
    for i, img in enumerate(panels, start=1):
        fig.add_trace(go.Image(z=img, hoverinfo="skip"), row=1, col=i)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=280, legend=False)


def _clarity_figure(out):
    """Descriptive overlay, not evidence for independent cataract diagnosis."""
    df = load_clarity()
    items = [
        ("Edge density", "edge_density"),
        ("Corner density", "corner_density"),
        ("Laplacian variance", "laplacian_variance"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[name for name, _ in items])
    for i, (_, key) in enumerate(items, start=1):
        for cls in ("Normal", "Cataract"):
            fig.add_trace(go.Box(
                y=df.loc[df["class"] == cls, key], name=cls, boxpoints=False,
                marker_color=CLASS_COLOUR[cls], showlegend=(i == 1),
                hoverinfo="skip"), row=1, col=i)
        fig.add_hline(y=out["clarity"][key], line=dict(color="#0B2E3F", dash="dash", width=2),
                      row=1, col=i)
    return style(fig, height=330, legend_below=True)


def _store_sample(name, filename):
    # No upload override unless a *new* upload is selected.
    st.session_state["screen_selected_img"] = {
        "name": name, "bytes": (IMG_DIR / filename).read_bytes(), "source": "sample",
    }


def render_screen_yourself():
    page_header(
        "STEP 3 · PREDICTIVE + STEP 4 · PRESCRIPTIVE",
        "Screen a <em>photograph</em>, then decide what to do",
        "Choose a model and fundus photograph. Explore its score, see the result and compare "
        "the model's test-set metrics.",
        lottie_doctor, "doctor_screen",
    )
    st.markdown(f"<div class='disclaimer'>⚠️ {DISCLAIMER}</div>", unsafe_allow_html=True)

    models = {
        title: pair for title, pair in CHOICES.items()
        if (svm_available() if pair[0] == "SVM" else cnn_available(pair[1]))
    }
    if not models:
        st.error("No saved model is available. Check assets/models/ and TensorFlow installation.")
        return

    # Restore the original horizontal control bar.
    with st.container(border=True):
        b1, b2, b3 = st.columns([2.2, 3, 1.3])
        selected = b1.selectbox("Select model", list(models), key="dashboard_selected_model")
        model, strategy = models[selected]
        is_cnn = model == "EfficientNetB0"
        cutoff = b2.slider(
            "Decision threshold (cataract when score is at least this)",
            min_value=0.05, max_value=0.95, value=0.50, step=0.01,
            disabled=not is_cnn, key="dashboard_cutoff",
            help="Original reported EfficientNetB0 metrics use 0.50. Changing this slider "
                 "affects only the current demonstration prediction and does NOT update test metrics. "
                 "The SVM uses the zero decision boundary.",
        )
        show_explanation = b3.toggle("Show explanation", value=True, key="dashboard_show_explanation")
    if is_cnn and cutoff != 0.50:
        st.caption("Exploratory cutoff %.2f for this photograph only. KPI cards below always report "
                   "the original 0.50 test-set cutoff." % cutoff)

    # Restore 3-panel dashboard from the originally published screenshot.
    col_a, col_b, col_c = st.columns([1.2, 1, 1.1])
    with col_a:
        with st.container(border=True, height=PANEL_H):
            panel_title("Choose a fundus image")
            samples_tab, upload_tab = st.tabs(["Samples", "Upload"])
            with samples_tab:
                grid = st.columns(2)
                for i, (name, filename) in enumerate(SAMPLES.items()):
                    with grid[i % 2]:
                        st.image(str(IMG_DIR / filename), use_container_width=True)
                        st.button(name, key=f"dashboard_pick_{name}", use_container_width=True,
                                  on_click=_store_sample, args=(name, filename))
                st.caption("Samples come from the source dataset and are demonstrations, "
                           "not an independent accuracy test.")
            with upload_tab:
                uploaded = st.file_uploader(
                    "Upload JPG/PNG (max 10 MB)", type=["jpg", "jpeg", "png"],
                    key="dashboard_upload",
                )
                if uploaded is not None:
                    content = uploaded.getvalue()
                    fingerprint = hashlib.sha256(content).hexdigest()
                    if fingerprint != st.session_state.get("dashboard_uploaded_fingerprint"):
                        st.session_state["dashboard_uploaded_fingerprint"] = fingerprint
                        st.session_state["screen_selected_img"] = {
                            "name": uploaded.name, "bytes": content, "source": "upload",
                        }
                with st.expander("Which images are accepted?"):
                    st.write("Use an individual colour retinal fundus photograph (JPG/PNG) "
                             "with a dark background, similar to the samples. "
                             "The image-format check is approximate and does not verify medical content.")

    chosen = st.session_state.get("screen_selected_img")
    result = None
    if chosen is not None:
        try:
            with st.spinner("Analysing fundus image..."):
                result = _analyse(chosen["bytes"], model, strategy)
        except Exception as exc:
            result = {"ok": False,
                      "msg": f"The selected model could not run ({type(exc).__name__}). "
                             "No other model was substituted."}

    with col_b:
        with st.container(border=True, height=PANEL_H):
            panel_title("Fundus image", chosen["name"] if chosen else None)
            if chosen is None:
                st.info("No image yet. Pick a sample or upload one.")
            elif result is not None and result.get("original") is not None:
                st.image(cv2.resize(result["original"], (224, 224)), use_container_width=True)
                if result["ok"]:
                    st.caption("Fundus image accepted for demonstration.")
            else:
                st.info("Image preview unavailable.")

    signal = None
    with col_c:
        with st.container(border=True, height=PANEL_H):
            panel_title("Prediction result")
            if chosen is None:
                st.info("The result appears here.")
            elif not result["ok"]:
                st.error(f"Image not accepted or model unavailable. {result['msg']}")
            else:
                threshold = cutoff if is_cnn else 0.0
                signal = result["score"] >= threshold
                colour = CATARACT if signal else NORMAL
                word = "Cataract signal" if signal else "No cataract signal"
                st.markdown(
                    f"<div style='text-align:center;margin:7px 0 10px 0'>"
                    f"<span class='result-pill' style='background:{colour};'>"
                    f"{'👁️' if signal else '✅'} {word}</span></div>",
                    unsafe_allow_html=True,
                )
                st.caption("Research prediction only, not a medical diagnosis or clearance.")
                st.markdown(gauge_svg(result["score"], threshold, is_cnn), unsafe_allow_html=True)
                st.caption(f"{('Uncalibrated sigmoid score' if is_cnn else 'SVM decision value')}: "
                           f"{result['score']:.4f}")

    # The cards always come from authoritative 21-result CSV at documented decision rules.
    row = model_result(model, strategy)
    chapter("★", f"Test-set performance of {model}", f"{strategy} · fixed 80-image hold-out set")
    kpi_tiles([
        ("Accuracy", f"{row.accuracy:.2%}", "🎯", "teal", "80 held-out images"),
        ("Recall", f"{row.recall:.0%}", "🔎", "mint", f"{int(row.FN)} of 20 cataracts missed"),
        ("Specificity", f"{row.specificity:.0%}", "🛡️", "sky", f"{int(row.FP)} false alarms of 60"),
        ("F1-score", f"{row.f1:.3f}", "⚖️", "violet", "Cataract F1"),
        ("MCC", f"{row.mcc:.3f}", "📈", "peach", "Balanced-class metric"),
    ])
    st.caption("Metrics above refer to the documented fixed-threshold 80-image test set. "
               "They do not represent the probability that the uploaded image is correctly labelled.")

    if result is not None and result["ok"] and show_explanation:
        chapter(1, "What the computer sees", "Same preprocessing and clarity measures as the notebook.")
        show(_processing_figure(result))
        with st.expander("How does this photo compare with the dataset?"):
            show(_clarity_figure(result))
            st.caption("Clarity descriptors are not clinical cataract biomarkers. "
                       "Camera settings, focus and lighting also affect these measurements.")

    chapter(2, "What to do next", "The result should be interpreted cautiously.")
    if signal is None:
        st.info("Choose a sample or upload an accepted fundus image to see the short guidance.")
    elif signal:
        st.warning("A cataract signal was detected. Arrange an eye examination with an "
                   "optometrist or ophthalmologist; this model cannot confirm cataract.")
    else:
        st.info("No cataract signal was detected by this model. This does not rule out "
                "cataract or other eye conditions. Seek professional advice if symptoms occur.")
    st.caption("No extra cutoff or model was selected using the historical test-scores file. "
               "Only the selected, explicitly named saved model is used.")
