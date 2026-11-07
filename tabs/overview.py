"""Page 1 - Overview: the headline numbers, the dataset and the whole experiment at a glance."""
import streamlit as st
from streamlit_lottie import st_lottie

from components.charts import confusion_fig, donut_fig
from components.data import load_results
from components.glossary import metrics_glossary
from components.home_lottie import lottie_thinking
from components.ui import TINT, goto, hero, kpi_tiles, panel_title, show

VIDEO_ID = "BXEfJNdXlYM"


def render_overview():
    res = load_results()
    best = res.sort_values(["mcc", "f1"], ascending=False).iloc[0]

    left, right = st.columns([5, 1])
    with left:
        hero("OVERVIEW", "IPVC <em>Cataract Detection</em> under Class Imbalance",
             "A Streamlit prototype for fundus image classification. It compares classical machine learning and "
             "neural networks when only one image in four shows cataract, then turns the best model into a "
             "screening tool.")
    with right:
        st_lottie(lottie_thinking, height=120, key="home_lottie")

    kpi_tiles([
        ("Accuracy", f"{best['accuracy']:.1%}", "🎯", "teal", "share of all images right"),
        ("Recall", f"{best['recall']:.1%}", "🔎", "mint", f"{int(best['TP'])} of 20 cataracts found"),
        ("Specificity", f"{best['specificity']:.1%}", "🛡️", "sky", f"{int(best['TN'])} of 60 normals cleared"),
        ("F1-score", f"{best['f1']:.3f}", "⚖️", "violet", "recall and precision balanced"),
        ("MCC", f"{best['mcc']:.3f}", "📈", "peach", "honest score, -1 to +1"),
    ])
    st.caption(f"Best configuration by MCC: **{best['label']}**, scored on 80 unseen test images.")
    with st.expander("What do these numbers mean?"):
        metrics_glossary()

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            show(donut_fig("Class distribution: 3 normal for every 1 cataract"))
    with c2:
        with st.container(border=True):
            show(confusion_fig(int(best["TP"]), int(best["FN"]), int(best["TN"]), int(best["FP"]),
                               f"Best model on the 80 test images: {int(best['FN'])} missed, {int(best['FP'])} false alarms"))

    with st.container(border=True):
        panel_title("Experiment summary", "all 21 experiments, same 80 test images, best MCC first")
        fam = st.radio("Family", ["All", "Classical", "Neural"], horizontal=True, key="ov_family",
                       label_visibility="collapsed")
        d = res if fam == "All" else res[res["family"] == fam]
        d = d.sort_values("mcc", ascending=False)
        table = d[["model", "strategy", "family"]].copy()
        table.columns = ["Model", "Strategy", "Family"]
        for col, name in [("accuracy", "Accuracy"), ("recall", "Recall"), ("specificity", "Specificity")]:
            table[name] = d[col] * 100
        table["F1"], table["MCC"] = d["f1"], d["mcc"]
        cc = st.column_config
        st.dataframe(
            table, hide_index=True, use_container_width=True, height=420,
            column_config={
                "Accuracy": cc.ProgressColumn("Accuracy", min_value=0, max_value=100, format="%.1f%%"),
                "Recall": cc.ProgressColumn("Recall", min_value=0, max_value=100, format="%.1f%%"),
                "Specificity": cc.ProgressColumn("Specificity", min_value=0, max_value=100, format="%.1f%%"),
                "F1": cc.NumberColumn("F1", format="%.3f"),
                "MCC": cc.NumberColumn("MCC", format="%.3f"),
            },
        )

    st.markdown("&nbsp;")
    steps = [("Descriptive", "What cataract is and what the dataset looks like.", "teal"),
             ("Diagnostic", "Why classes differ, why accuracy misleads, who misses what.", "violet"),
             ("Predictive", "Upload a fundus photograph and read the model's score.", "peach"),
             ("Prescriptive", "Advice and a screening policy built on the result.", "mint")]
    cards = "".join(
        f"<div class='step' style='background:{TINT[t][0]};'><b><span class='step-no'>{i}</span>{name}</b>{text}</div>"
        for i, (name, text, t) in enumerate(steps, start=1))
    st.markdown(
        f"""
        <div class="two-grid">
          <div class="equal-card"><div class="panel-title">Watch first</div><div style="height:10px"></div>
            <div class="video-frame"><iframe src="https://www.youtube.com/embed/{VIDEO_ID}" title="Cataract video"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowfullscreen></iframe></div></div>
          <div class="equal-card"><div class="panel-title">The four-step story</div><div style="height:10px"></div>
            <div class="step-grid">{cards}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    mid = st.columns([3, 2, 3])[1]
    with mid:
        if st.button("START EXPLORING ➜", key="go_explore"):
            goto("go_to_what_is")
