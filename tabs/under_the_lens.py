"""Page 3 - DIAGNOSTIC analytics: why the classes differ, why accuracy misleads, who misses what, how sure.

The page is a case file. Each chapter asks one question, answers it with an interactive chart and hands one
finding to the next chapter. The last chapter hands over to the prescriptive page.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu
from streamlit_lottie import st_lottie

from components.charts import confusion_fig
from components.data import (CLARITY_SUMMARY, TEST_CATARACT, load_clarity, load_cv, load_results,
                             load_test_scores, metrics_at, wilson_interval)
from components.glossary import clarity_glossary, metrics_glossary, strategies_glossary
from components.home_lottie import lottie_data_analytics
from components.ui import (CATARACT, CLASSICAL, CLASS_COLOUR, FAMILY_COLOUR, GREY, NORMAL, QUAL, VIOLET, chapter,
                           goto, hero, show, style, takeaway)

METRICS = {"MCC": "mcc", "F1 (cataract)": "f1", "Recall": "recall", "Specificity": "specificity",
           "Precision": "precision", "Accuracy": "accuracy", "ROC-AUC": "roc_auc"}


# ----------------------------------------------------------------------------
# Chapter 1: the evidence in the image
# ----------------------------------------------------------------------------
def _clarity_chapter():
    chapter(1, "The evidence", "Do cataract photographs really look different to a computer?")
    with st.expander("What do these three measurements mean?", expanded=True):
        clarity_glossary()

    df = load_clarity()
    measures = list(CLARITY_SUMMARY)
    fig = make_subplots(rows=1, cols=3, subplot_titles=measures)

    if df is not None:
        for i, measure in enumerate(measures, start=1):
            for cls in ["Normal", "Cataract"]:
                vals = df.loc[df["class"] == cls, measure]
                fig.add_trace(go.Violin(
                    y=vals, name=cls, legendgroup=cls, showlegend=(i == 1),
                    box_visible=True, meanline_visible=True, points="all", jitter=0.35, pointpos=0,
                    marker=dict(size=3, opacity=0.45), line_color=CLASS_COLOUR[cls],
                    fillcolor=CLASS_COLOUR[cls], opacity=0.55,
                    hovertemplate=f"{cls}<br>{measure}: %{{y:.4f}}<extra></extra>",
                ), row=1, col=i)
        show(style(fig, height=470, legend_below=True, title="Clarity measurements of all 400 photographs, by class"))

        rows = []
        for measure in measures:
            a = df.loc[df["class"] == "Normal", measure]
            b = df.loc[df["class"] == "Cataract", measure]
            p = mannwhitneyu(a, b).pvalue
            rows.append({"Measurement": measure,
                         "Median, normal": round(a.median(), 4),
                         "Median, cataract": round(b.median(), 4),
                         "Mann-Whitney p": f"{p:.1e}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("A p-value below 0.05 means the difference between the classes is unlikely to be chance.")
    else:
        for i, measure in enumerate(measures, start=1):
            for cls in ["Normal", "Cataract"]:
                mean, sd = CLARITY_SUMMARY[measure][cls]
                fig.add_trace(go.Bar(
                    x=[cls], y=[mean], error_y=dict(type="data", array=[sd]),
                    marker_color=CLASS_COLOUR[cls], name=cls, legendgroup=cls, showlegend=(i == 1),
                    hovertemplate=f"{cls}<br>mean {mean:.4f}, SD {sd:.4f}<extra></extra>",
                ), row=1, col=i)
        show(style(fig, height=440, legend_below=True, title="Clarity measurements by class (average and spread)"))

    takeaway(
        "On average, cataract photographs have fewer sharp edges, fewer corner points and a lower Laplacian "
        "variance, which is the signature of a softer image. The classes still overlap, so no single measurement "
        "decides the case. The models combine many measurements."
    )


# ----------------------------------------------------------------------------
# Chapter 2: the accuracy trap
# ----------------------------------------------------------------------------
def _paradox_chapter(res):
    chapter(2, "The trap", "If accuracy is high, is the model finding cataracts?")
    by = st.radio("Colour the points by", ["Model family", "Strategy", "Model"], horizontal=True,
                  key="paradox_by")
    key = {"Model family": "family", "Strategy": "strategy", "Model": "model"}[by]
    fig = go.Figure()
    groups = list(res.groupby(key))
    for n, (name, g) in enumerate(groups):
        colour = FAMILY_COLOUR.get(name, QUAL[n % len(QUAL)])
        # Experiments that share the same score sit on the same spot, so marker sizes are staggered
        # (largest first, smallest on top) to keep every group visible.
        size = max(9, 24 - 4 * n)
        fig.add_trace(go.Scatter(
            x=g["accuracy"], y=g["recall"], mode="markers", name=str(name),
            marker=dict(size=size, color=colour, opacity=0.85, line=dict(width=1.5, color="white")),
            customdata=np.stack([g["label"], g["mcc"], g["FN"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>Accuracy %{x:.1%}<br>Recall %{y:.0%}"
                          "<br>MCC %{customdata[1]:.3f}<br>Missed cataracts %{customdata[2]} of 20<extra></extra>",
        ))
    fig.add_vline(x=0.75, line_dash="dash", line_color=GREY)
    fig.add_annotation(x=0.75, y=1.06, text="Answer \"normal\" every time: 75% accuracy",
                       showarrow=False, xanchor="left", font=dict(size=12, color="#555"))
    fig.update_xaxes(title="Accuracy on the 80-image test set", tickformat=".0%", range=[0.6, 1.0])
    fig.update_yaxes(title="Recall: share of cataract images found", tickformat=".0%", range=[-0.03, 1.12])
    show(style(fig, height=470, title="Accuracy versus recall for all 21 experiments"))
    st.caption("Some experiments share exactly the same scores, so their markers overlap. Marker sizes are "
               "staggered so every group stays visible.")

    worst = res.sort_values(["recall", "accuracy"]).iloc[0]
    takeaway(
        f"**{worst['label']}** reaches {worst['accuracy']:.0%} accuracy yet finds {worst['recall']:.0%} of the "
        "cataract images. It matches the all-normal answer. Accuracy hides this failure, so the study judges "
        "models by recall, specificity, F1 and the Matthews correlation coefficient (MCC) instead."
    )
    with st.expander("What do accuracy, recall, specificity, F1 and MCC mean?"):
        metrics_glossary()


# ----------------------------------------------------------------------------
# Chapter 3: the misses
# ----------------------------------------------------------------------------
def _errors_chapter(res):
    chapter(3, "The misses", "Which models miss cataracts, and which raise false alarms?")
    st.markdown("A **missed cataract** is a cataract image the model called normal. A **false alarm** is a normal "
                "image the model called cataract.")
    strategies = list(res["strategy"].unique())
    strategy = st.selectbox("Imbalance strategy", strategies, index=strategies.index("class weights")
                            if "class weights" in strategies else 0, key="err_strategy")
    d = res[res["strategy"] == strategy].sort_values("FN", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["model"], y=d["FN"], name="Missed cataracts (out of 20)", marker_color=CATARACT,
                         text=d["FN"], textposition="outside", cliponaxis=False,
                         hovertemplate="%{x}<br>Missed %{y} of 20<extra></extra>"))
    fig.add_trace(go.Bar(x=d["model"], y=d["FP"], name="False alarms (out of 60 normal)", marker_color=VIOLET,
                         text=d["FP"], textposition="outside", cliponaxis=False,
                         hovertemplate="%{x}<br>False alarms %{y} of 60<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Number of images", range=[0, max(d["FN"].max(), d["FP"].max()) + 6])
    show(style(fig, height=420, title=f"Mistakes per model with the {strategy} strategy"))

    top, bottom = d.iloc[0], d.sort_values("FN").iloc[0]
    st.markdown(
        f"Under **{strategy}**, **{top['model']}** missed the most cataract images ({int(top['FN'])} of 20) "
        f"and **{bottom['model']}** the fewest ({int(bottom['FN'])} of 20)."
    )
    corr = res[["FN", "FP"]].corr().iloc[0, 1]
    pull = ("In these 21 experiments, fewer missed cataracts usually came with more false alarms. "
            if corr < -0.2 else "")
    takeaway(f"{pull}The two mistakes pull against each other, so a model cannot be judged on one of them alone. "
             "A screening programme decides how many false alarms it accepts to avoid a missed cataract.")


# ----------------------------------------------------------------------------
# Chapter 4: does the fix work?
# ----------------------------------------------------------------------------
def _strategy_chapter(res):
    chapter(4, "The fix", "Which imbalance strategy helps which model?")
    with st.expander("What do baseline, class weights, ROS, SMOTE and augmentation mean?", expanded=True):
        strategies_glossary()
    metric_name = st.radio("Metric", ["Recall", "Specificity", "MCC", "F1 (cataract)"], horizontal=True,
                           key="heat_metric")
    metric = METRICS[metric_name]
    pivot = res.pivot_table(index="model", columns="strategy", values=metric)
    order = [c for c in ["baseline", "class weights", "ROS", "SMOTE", "augmentation"] if c in pivot.columns]
    pivot = pivot[order]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=order, y=pivot.index, colorscale=[[0, "#EAF6F4"], [0.5, "#7FCBC0"], [1, NORMAL]],
        zmin=0, zmax=1,
        text=[["" if np.isnan(v) else f"{v:.2f}" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=14),
        hovertemplate="%{y} + %{x}<br>" + metric_name + " %{z:.3f}<extra></extra>",
        colorbar=dict(title=metric_name), xgap=3, ygap=3,
    ))
    fig.update_yaxes(autorange="reversed", automargin=True)
    show(style(fig, height=380, legend=False, title=f"{metric_name} for every model and strategy"))
    st.caption("Empty cells were not run. SMOTE and ROS work on numeric features, so the neural networks use "
               "class weights or augmentation instead.")

    sv = res[(res["model"] == "SVM")].set_index("strategy")
    if {"baseline", "class weights"} <= set(sv.index):
        gain = sv.loc["class weights", "recall"] - sv.loc["baseline", "recall"]
        takeaway(
            f"For the SVM, class weights lift recall from {sv.loc['baseline', 'recall']:.0%} to "
            f"{sv.loc['class weights', 'recall']:.0%} ({gain * 100:+.0f} points), while specificity moves "
            f"from {sv.loc['baseline', 'specificity']:.0%} to {sv.loc['class weights', 'specificity']:.0%}. "
            "Re-weighting buys recall and pays for it with false alarms. The Small CNN with augmentation "
            "shows the opposite risk: a fix applied badly can leave a model that predicts one class."
        )


# ----------------------------------------------------------------------------
# Chapter 5: how sure are we?
# ----------------------------------------------------------------------------
def _doubt_chapter(res, cv):
    chapter(5, "The doubt", "The test set holds only 20 cataract images. How sure is each result?")
    r = res.copy()
    bounds = [wilson_interval(int(k), TEST_CATARACT) for k in r["TP"]]
    r["lo"] = [b[0] for b in bounds]
    r["hi"] = [b[1] for b in bounds]
    r = r.sort_values("recall")
    fig = go.Figure(go.Scatter(
        x=r["recall"], y=r["label"], mode="markers",
        marker=dict(size=10, color=[FAMILY_COLOUR[f] for f in r["family"]]),
        error_x=dict(type="data", symmetric=False, array=r["hi"] - r["recall"],
                     arrayminus=r["recall"] - r["lo"], color=GREY, thickness=1.5),
        customdata=np.stack([r["lo"], r["hi"], r["TP"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Recall %{x:.0%} (%{customdata[2]} of 20)<br>"
                      "95% interval %{customdata[0]:.0%} to %{customdata[1]:.0%}<extra></extra>",
    ))
    fig.update_xaxes(title="Recall, with the range it could plausibly take (95% interval)", tickformat=".0%",
                     range=[0, 1.02])
    fig.update_yaxes(tickfont=dict(size=11))
    show(style(fig, height=680, legend=False, title="How certain is each recall? (orange: neural, blue: classical)"))

    st.markdown("#### Does the ranking hold on other splits?")
    st.markdown("The classical models were also scored with 5-fold cross-validation repeated 3 times, which tests "
                "each model on 15 different splits of the 400 images.")
    c = cv.sort_values("recall_mean")
    fig2 = go.Figure(go.Bar(
        x=c["recall_mean"], y=c["label"], orientation="h", marker_color=CLASSICAL,
        error_x=dict(type="data", array=c["recall_sd"], color="#333", thickness=1.5),
        text=[f"{v:.0%}" for v in c["recall_mean"]], textposition="inside", insidetextanchor="start",
        textfont=dict(color="white", size=13),
        customdata=np.stack([c["f1_mean"], c["mcc_mean"], c["mcc_sd"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Recall %{x:.1%}<br>F1 %{customdata[0]:.3f}<br>"
                      "MCC %{customdata[1]:.3f} ± %{customdata[2]:.3f}<extra></extra>",
    ))
    fig2.update_xaxes(title="Average recall (line: spread across the 15 splits)", tickformat=".0%", range=[0, 1])
    show(style(fig2, height=420, legend=False, title="Cross-validated recall of the classical models"))

    takeaway(
        "With 20 positives, one extra image moves recall by five points, and the intervals overlap widely. "
        "The ranking is a guide, not proof that one model beats the next. Repeated cross-validation agrees "
        "that SVM with re-weighting or oversampling recalls more cataracts than SVM alone, and that random "
        "forest recalls fewer."
    )


# ----------------------------------------------------------------------------
# Chapter 6: the cutoff is a decision (shown only when the exported scores are present)
# ----------------------------------------------------------------------------
def _cutoff_chapter(ts):
    chapter(6, "The cutoff", "A model gives a score. Where should the line between normal and cataract sit?")
    labels = sorted(ts["label"].unique())
    default = labels.index("EfficientNetB0 + class weights") if "EfficientNetB0 + class weights" in labels else 0
    c1, c2 = st.columns([1, 2])
    sel = c1.selectbox("Neural model", labels, index=default, key="cut_model")
    t = c2.slider("Call an image cataract when its score is at least", 0.0, 1.0, 0.5, 0.01, key="cut_t",
                  help="Scores come from the sigmoid output of the network (uncalibrated).")
    d = ts[ts["label"] == sel]
    m, m0 = metrics_at(d["y_true"], d["score"], t), metrics_at(d["y_true"], d["score"], 0.5)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Recall", f"{m['recall']:.0%}", f"{(m['recall'] - m0['recall']) * 100:+.0f} pts vs 0.5")
    k2.metric("Specificity", f"{m['specificity']:.0%}", f"{(m['specificity'] - m0['specificity']) * 100:+.0f} pts vs 0.5")
    k3.metric("MCC", f"{m['mcc']:.3f}", f"{m['mcc'] - m0['mcc']:+.3f} vs 0.5")
    k4.metric("Missed cataracts", f"{m['FN']} of {m['TP'] + m['FN']}")
    k5.metric("False alarms", f"{m['FP']} of {m['TN'] + m['FP']}")

    left, right = st.columns([3, 2])
    grid = np.linspace(0, 1, 101)
    curve = [metrics_at(d["y_true"], d["score"], g) for g in grid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=[c["recall"] for c in curve], name="Recall", mode="lines",
                             line=dict(color=CATARACT, width=3),
                             hovertemplate="Cutoff %{x:.2f}<br>Recall %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=[c["specificity"] for c in curve], name="Specificity", mode="lines",
                             line=dict(color=NORMAL, width=3),
                             hovertemplate="Cutoff %{x:.2f}<br>Specificity %{y:.0%}<extra></extra>"))
    fig.add_vline(x=t, line_dash="dash", line_color="#333")
    fig.add_trace(go.Scatter(x=[t, t], y=[m["recall"], m["specificity"]], mode="markers", showlegend=False,
                             marker=dict(size=12, color=[CATARACT, NORMAL], line=dict(width=2, color="white")),
                             hoverinfo="skip"))
    fig.update_xaxes(title="Cutoff on the model score", range=[0, 1])
    fig.update_yaxes(title="Share of test images", tickformat=".0%", range=[0, 1.03])
    with left:
        show(style(fig, height=400, title="Recall and specificity as the cutoff moves"))
    with right:
        show(confusion_fig(m["TP"], m["FN"], m["TN"], m["FP"], f"Results at cutoff {t:.2f}", height=400))

    takeaway(
        "Moving the cutoff trades false alarms for missed cataracts. No setting removes the trade-off. "
        "The right cutoff depends on what a missed cataract costs compared with an unnecessary eye "
        "examination. The next page lets the screening result meet that decision.",
        kind="next",
    )


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
def render_under_the_lens():
    res, cv, ts = load_results(), load_cv(), load_test_scores()

    questions = ["Do the classes look different?", "Is accuracy a fair judge?", "Who misses cataracts?",
                 "Which fix helps?", "How sure are the results?"]
    if ts is not None:
        questions.append("Where should the cutoff sit?")
    head_l, head_r = st.columns([4, 1])
    with head_l:
        hero("STEP 2 · DIAGNOSTIC ANALYTICS", "Under the lens: why some models see cataract and others do not",
             f"A case file for data enthusiasts. {len(questions)} questions lead from the image to a decision.")
        st.markdown("<ol class='case-list'>" + "".join(f"<li>{q}</li>" for q in questions) + "</ol>",
                    unsafe_allow_html=True)
    with head_r:
        st_lottie(lottie_data_analytics, height=170, key="analytics_lottie")

    _clarity_chapter()
    st.markdown("---")
    _paradox_chapter(res)
    st.markdown("---")
    _errors_chapter(res)
    st.markdown("---")
    _strategy_chapter(res)
    st.markdown("---")
    _doubt_chapter(res, cv)
    if ts is not None:
        st.markdown("---")
        _cutoff_chapter(ts)

    st.write("")
    mid = st.columns([3, 2, 3])[1]
    with mid:
        if st.button("NEXT: SCREEN A PHOTO ➜", key="to_screen"):
            goto("go_to_screen")
