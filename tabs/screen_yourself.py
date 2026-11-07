"""Page 4 - PREDICTIVE and PRESCRIPTIVE analytics.

Predictive: upload or pick a fundus photograph, validate it, show what the computer measures and score it.
Prescriptive: turn the score into actions - advice for the result, how far to trust a positive, and a
cost-based policy lab that recommends a model and a cutoff.
"""
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.data import CLARITY_SUMMARY, load_clarity, load_results, load_test_scores, metrics_at
from components.fundus_utils import (clarity_maps, cnn_available, preprocess, read_image, score_cnn,
                                     score_svm, svm_available, validate_fundus)
from components.glossary import metrics_glossary
from components.ui import (CATARACT, CLASS_COLOUR, FAMILY_COLOUR, NAVY, NORMAL, chapter, hero, kpi_tiles,
                           panel_title, show, style, takeaway)

IMG_DIR = Path(__file__).resolve().parents[1] / "assets" / "img"
SAMPLES = {"Normal 1": "normal_1.jpg", "Normal 2": "normal_2.jpg",
           "Cataract 1": "cataract_1.jpg", "Cataract 2": "cataract_2.jpg"}
CNN_LABEL = "EfficientNetB0 + class weights (recommended)"
SVM_LABEL = "SVM + SMOTE (lighter classical model)"

RULES = """**Accepted:** a colour fundus photograph of one eye. It shows a round, red to orange retina on a dark
background, like the samples below. JPG or PNG, at least 200 px on the short side.

**Not accepted:** faces, ordinary eye close-ups, screenshots, black-and-white images and scans of printed pictures.
These return "Image not accepted".

The models were trained on 400 images from one public dataset. Uploaded images are processed in memory and are
not stored."""

HOSPITAL = """**Hospital Shah Alam (HSAS), Department of Ophthalmology**
- Address: Persiaran Kayangan, Seksyen 7, 40000 Shah Alam, Selangor
- Services: secondary and tertiary eye care (including medical retina), 24-hour emergency, optometry
- Telephone: +60 3-5526 3000
- Website: https://jknselangor.moh.gov.my/hsas/

Confirm opening hours and referral requirements by phone before visiting. Any eye-care professional can examine
the lens; HSAS is one example."""

ADVICE_CATARACT = """#### Suggested next steps
1. **Book a full eye examination** with an optometrist or ophthalmologist. Only an examination of the lens can
   confirm a cataract.
2. **Note the symptoms** you have noticed: blurred or cloudy vision, glare, faded colours, trouble seeing at night.
3. **Do not wait** if there is sudden vision loss, eye pain or redness. Seek urgent care.
4. **Ask about risk factors** with your doctor, especially diabetes, long-term steroid use and smoking."""

ADVICE_NORMAL = """#### Suggested next steps
1. **Keep routine eye check-ups**, especially with age, diabetes or other risk factors.
2. **Watch for symptoms**: blurred vision, glare, faded colours or night-vision trouble. See an eye-care
   professional if they appear.
3. **Protect your eyes**: sunglasses with UV protection, no smoking, good blood-sugar control.
4. **Remember the limits**: "no signal" is not a clearance. In testing, the models still missed some cataract
   images (see the policy lab below)."""

DISCLAIMER = ("Research demonstration for a university course. It is **not a medical device** and does not give a "
              "diagnosis. Only an eye-care professional can diagnose a cataract.")


# ----------------------------------------------------------------------------
# Predictive helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _analyse(img_bytes, model_key):
    """Validate, preprocess and score one image. Cached per image and model."""
    rgb = read_image(img_bytes)
    if rgb is None:
        return {"ok": False, "msg": "The file could not be read as an image."}
    ok, why = validate_fundus(rgb)
    if not ok:
        return {"ok": False, "msg": why}
    enhanced = preprocess(rgb)
    edges, corners, clarity = clarity_maps(enhanced)
    note, used = "", model_key
    try:
        score = score_cnn(enhanced) if model_key == "cnn" else score_svm(enhanced)
    except Exception as exc:  # TensorFlow or Keras problem on the server: fall back to the SVM
        if model_key == "cnn" and svm_available():
            score, used = score_svm(enhanced), "svm"
            note = f"EfficientNetB0 could not be loaded on this server ({type(exc).__name__}). The SVM was used instead."
        else:
            return {"ok": False, "msg": f"The model could not run ({type(exc).__name__})."}
    return {"ok": True, "original": cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA),
            "enhanced": enhanced, "edges": edges, "corners": corners, "clarity": clarity,
            "score": score, "model": used, "note": note}


def _processing_figure(out):
    corners = cv2.dilate(out["corners"].astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    panels = [out["original"], out["enhanced"], np.dstack([out["edges"]] * 3), np.dstack([corners] * 3)]
    titles = ["1. Original", "2. Crop, resize, CLAHE", f"3. Edges ({out['clarity']['edge_density']:.3f})",
              f"4. Corners ({out['clarity']['corner_density']:.4f})"]
    fig = make_subplots(rows=1, cols=4, subplot_titles=titles, horizontal_spacing=0.01)
    for i, panel in enumerate(panels, start=1):
        fig.add_trace(go.Image(z=panel, hoverinfo="skip"), row=1, col=i)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=300, legend=False, title="What the computer sees")


def _clarity_position(out):
    """Where does this photograph sit among the two classes? Links back to Chapter 1 of the diagnostic page."""
    df = load_clarity()
    measures = list(CLARITY_SUMMARY)
    keys = ["edge_density", "corner_density", "laplacian_variance"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=measures)
    closer = 0
    for i, (measure, key) in enumerate(zip(measures, keys), start=1):
        value = out["clarity"][key]
        if df is not None:
            for cls in ["Normal", "Cataract"]:
                fig.add_trace(go.Box(y=df.loc[df["class"] == cls, measure], name=cls, boxpoints=False,
                                     marker_color=CLASS_COLOUR[cls], showlegend=False,
                                     hoverinfo="skip"), row=1, col=i)
            centre = {c: df.loc[df["class"] == c, measure].median() for c in ["Normal", "Cataract"]}
        else:
            for cls in ["Normal", "Cataract"]:
                mean, sd = CLARITY_SUMMARY[measure][cls]
                fig.add_trace(go.Bar(x=[cls], y=[mean], error_y=dict(type="data", array=[sd]),
                                     marker_color=CLASS_COLOUR[cls], opacity=0.6, showlegend=False,
                                     hoverinfo="skip"), row=1, col=i)
            centre = {c: CLARITY_SUMMARY[measure][c][0] for c in ["Normal", "Cataract"]}
        fig.add_trace(go.Scatter(x=["Normal", "Cataract"], y=[value, value], mode="lines+markers",
                                 line=dict(color="#111", dash="dash", width=2), marker=dict(size=8),
                                 name="This photograph", showlegend=(i == 1),
                                 hovertemplate=f"This photograph: {value:.4f}<extra></extra>"), row=1, col=i)
        closer += abs(value - centre["Cataract"]) < abs(value - centre["Normal"])
    show(style(fig, height=430, legend_below=True, title="Where this photograph sits among the two classes"))
    return int(closer)


def _gauge(score, cutoff, model):
    if model == "cnn":
        axis, steps = [0, 1], [{"range": [0, cutoff], "color": "#DDF3EF"}, {"range": [cutoff, 1], "color": "#FDE0DB"}]
        title = "Model score (cataract side is right of the line)"
        fmt = ".3f"
    else:
        axis, steps = [-3, 3], [{"range": [-3, 0], "color": "#DDF3EF"}, {"range": [0, 3], "color": "#FDE0DB"}]
        title = "SVM decision value (cataract side is above 0)"
        cutoff, fmt = 0.0, ".2f"
        score = float(np.clip(score, -3, 3))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(valueformat=fmt),
        title=dict(text=title, font=dict(size=14)),
        gauge=dict(axis=dict(range=axis), bar=dict(color="#333"), steps=steps,
                   threshold=dict(line=dict(color=CATARACT, width=4), thickness=0.85, value=cutoff)),
    ))
    return style(fig, height=250, legend=False)


# ----------------------------------------------------------------------------
# Prescriptive helpers
# ----------------------------------------------------------------------------
def _set_cutoff(value):
    st.session_state["cutoff"] = float(value)


def _policy_lab(res):
    ratio = st.slider("A missed cataract is as costly as this many false alarms", 1, 10, 3, key="cost_ratio",
                      help="1 means both mistakes cost the same. 5 means a missed cataract is five times worse "
                           "than sending a healthy person for an extra eye examination.")
    floor = st.slider("Minimum specificity the clinic can tolerate", 0, 95, 50, 5, key="spec_floor",
                      format="%d%%", help="Specificity is the share of healthy eyes correctly cleared.")
    d = res.copy()
    d["cost"] = ratio * d["FN"] + d["FP"]
    ok = d[d["specificity"] >= floor / 100].sort_values(["cost", "mcc"], ascending=[True, False])
    if ok.empty:
        st.warning("No configuration meets that specificity. Lower the minimum.")
        return
    top = ok.iloc[0]
    st.success(
        f"**Recommended configuration: {top['label']}.** With a miss weighted {ratio}× a false alarm, it has the "
        f"lowest expected cost on the test set: {int(top['FN'])} of 20 cataracts missed and {int(top['FP'])} "
        f"false alarms among 60 healthy images (recall {top['recall']:.0%}, specificity {top['specificity']:.0%})."
    )
    shown = ok.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=shown["cost"], y=shown["label"], orientation="h",
        marker_color=[FAMILY_COLOUR[f] for f in shown["family"]],
        text=[f"{int(c)}" for c in shown["cost"]], textposition="outside",
        customdata=np.stack([shown["FN"], shown["FP"], shown["recall"], shown["specificity"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Cost %{x}<br>Missed %{customdata[0]}, false alarms %{customdata[1]}"
                      "<br>Recall %{customdata[2]:.0%}, specificity %{customdata[3]:.0%}<extra></extra>",
    ))
    fig.update_xaxes(title=f"Expected cost = {ratio} × missed + false alarms (lower is better)")
    show(style(fig, height=max(320, 34 * len(shown) + 110), legend=False,
               title="The ten lowest-cost configurations", right=40))

    ts = load_test_scores()
    if ts is not None and "EfficientNetB0 + class weights" in set(ts["label"]):
        d2 = ts[ts["label"] == "EfficientNetB0 + class weights"]
        best = None
        for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
            m = metrics_at(d2["y_true"], d2["score"], t)
            cost = ratio * m["FN"] + m["FP"]
            key = (cost, -m["mcc"])
            if m["specificity"] >= floor / 100 and (best is None or key < best[0]):
                best = (key, t, m)
        if best:
            _, t, m = best
            c1, c2 = st.columns([3, 1])
            c1.markdown(
                f"For EfficientNetB0 with class weights, the cutoff with the lowest cost is **{t:.2f}**: "
                f"{m['FN']} missed, {m['FP']} false alarms (recall {m['recall']:.0%}, specificity "
                f"{m['specificity']:.0%})."
            )
            c2.button("Use this cutoff", key="apply_cutoff", on_click=_set_cutoff, args=(t,))
    st.caption("Counts come from 80 test images, only 20 with cataract. Treat the recommendation as a "
               "direction, not a guarantee.")


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
def render_screen_yourself():
    res = load_results()

    hero("STEP 3 · PREDICTIVE  +  STEP 4 · PRESCRIPTIVE", "Screen a <em>photograph</em>, then decide what to do",
         "Pick a model and a cutoff, choose a fundus photograph and read the result. The lower half turns the "
         "result into advice, a trust estimate and a screening policy.")
    st.markdown(f"<div class='disclaimer'>⚠️ {DISCLAIMER}</div>", unsafe_allow_html=True)

    models = {}
    if cnn_available():
        models[CNN_LABEL] = "cnn"
    if svm_available():
        models[SVM_LABEL] = "svm"
    if not models:
        st.error("No model file was found in assets/models/.")
        return

    # ---------------- control bar ----------------
    with st.container(border=True):
        b1, b2, b3 = st.columns([2.2, 3, 1.3])
        model_label = b1.selectbox("Select model", list(models), key="screen_model")
        is_cnn = models[model_label] == "cnn"
        st.session_state.setdefault("cutoff", 0.5)
        b2.slider("Decision threshold (cataract when score is at least this)", 0.05, 0.95, step=0.01,
                  key="cutoff", disabled=not is_cnn,
                  help="Scores come from the sigmoid output of EfficientNetB0 and are not calibrated. 0.5 is the "
                       "study default. The policy lab can suggest a value. The SVM uses its own boundary.")
        show_expl = b3.toggle("Show explanation", value=True, key="show_expl")
    cutoff = st.session_state["cutoff"] if is_cnn else 0.0

    # ---------------- choose / preview / result ----------------
    colA, colB, colC = st.columns([1.15, 1, 1.15])
    with colA:
        with st.container(border=True):
            panel_title("Choose a fundus image")
            t_sample, t_upload = st.tabs(["Samples", "Upload"])
            with t_sample:
                grid = st.columns(2)
                for n, (name, file) in enumerate(SAMPLES.items()):
                    with grid[n % 2]:
                        st.image(str(IMG_DIR / file), use_container_width=True)
                        if st.button(name, key=f"pick_{name}", use_container_width=True):
                            st.session_state["screen_img"] = {"name": name, "bytes": (IMG_DIR / file).read_bytes()}
                st.caption("Samples come from the public dataset. Some may have been used in training, so a "
                           "correct result here is a demonstration, not evidence of accuracy.")
            with t_upload:
                up = st.file_uploader("JPG or PNG, max 10 MB", type=["jpg", "jpeg", "png"], key="screen_upload")
                if up is not None:
                    st.session_state["screen_img"] = {"name": up.name, "bytes": up.getvalue()}
                with st.expander("Which images are accepted?"):
                    st.markdown(RULES)

    chosen = st.session_state.get("screen_img")
    out = None
    if chosen is not None:
        with st.spinner("Analysing. The first run loads the model and can take a minute."):
            out = _analyse(chosen["bytes"], models[model_label])

    with colB:
        with st.container(border=True):
            panel_title("Fundus image", chosen["name"] if chosen else None)
            if chosen is None:
                st.info("No image yet. Pick a sample or upload one.")
            elif out["ok"]:
                st.image(out["original"], use_container_width=True)
                st.caption("Accepted as a fundus photograph.")
            else:
                st.image(chosen["bytes"], use_container_width=True)

    signal = None
    with colC:
        with st.container(border=True):
            panel_title("Prediction result")
            if chosen is None:
                st.info("The result appears here.")
            elif not out["ok"]:
                st.error(f"**Image not accepted.** {out['msg']}")
            else:
                if out["note"]:
                    st.warning(out["note"])
                used_cnn = out["model"] == "cnn"
                signal = out["score"] >= (cutoff if used_cnn else 0.0)
                colour = CATARACT if signal else NORMAL
                word = "Cataract signal" if signal else "No cataract signal"
                icon = "👁️" if signal else "✅"
                st.markdown(
                    f"<div style='text-align:center;margin:4px 0 8px 0;'><span class='result-pill' "
                    f"style='background:{colour};'>{icon} {word}</span></div>"
                    f"<div style='font-size:13px;color:#4B6470;text-align:center;'>This research prototype "
                    f"flags the image as <b style='color:{colour};'>{'cataract' if signal else 'normal'}</b>. "
                    f"Use for academic demonstration only.</div>",
                    unsafe_allow_html=True,
                )
                show(_gauge(out["score"], cutoff, out["model"]))

    # ---------------- model KPIs ----------------
    key = (("EfficientNetB0", "class weights") if models[model_label] == "cnn" else ("SVM", "SMOTE"))
    row = res[(res["model"] == key[0]) & (res["strategy"] == key[1])].iloc[0]
    st.markdown(f"##### Test-set performance of {row['label']}")
    kpi_tiles([
        ("Accuracy", f"{row['accuracy']:.1%}", "🎯", "teal", "80 unseen images"),
        ("Recall", f"{row['recall']:.1%}", "🔎", "mint", f"{int(row['FN'])} of 20 cataracts missed"),
        ("Specificity", f"{row['specificity']:.1%}", "🛡️", "sky", f"{int(row['FP'])} false alarms of 60"),
        ("F1-score", f"{row['f1']:.3f}", "⚖️", "violet", "cataract class"),
        ("MCC", f"{row['mcc']:.3f}", "📈", "peach", "balanced single score"),
    ])

    with st.expander("What do these numbers mean?"):
        metrics_glossary()

    # ---------------- explanation ----------------
    if out is not None and out["ok"] and show_expl:
        chapter(1, "What the computer measures", "The same three clarity measurements as the diagnostic page.")
        show(_processing_figure(out))
        closer = _clarity_position(out)
        st.caption(f"On {closer} of 3 measurements this photograph sits closer to the cataract group. "
                   "Dashed line: this photograph.")

    # ---------------- prescriptive ----------------
    chapter(2, "What to do next", "Advice that follows the result.")
    if signal is None:
        st.info("Choose an accepted fundus image to see advice for its result.")
    else:
        a, b = st.columns(2)
        with a:
            with st.container(border=True):
                st.markdown(ADVICE_CATARACT if signal else ADVICE_NORMAL)
        with b:
            with st.container(border=True):
                st.markdown(HOSPITAL)

    chapter(3, "Policy lab", "Which model and cutoff suit a clinic that fears missing cataracts?")
    _policy_lab(res)
    takeaway(
        "A screening programme that cannot afford missed cataracts accepts more false alarms and favours "
        "high recall. A programme with scarce examination slots raises the specificity floor. The data "
        "informs the choice; the clinic makes it.",
        kind="next",
    )
    st.caption(DISCLAIMER)
