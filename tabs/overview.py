"""Overview: header, KPI row, two chart cards, then one bento block for analytics and workflow."""
import streamlit as st
from components.charts import class_chart, confusion_chart
from components.data import load_results, model_result
from components.home_lottie import lottie_thinking
from components.ui import TINT, card, goto, kpi_tiles, page_header, show


def _best():
    return load_results().sort_values(["mcc", "f1"], ascending=False).iloc[0]


def render_overview():
    page_header("OVERVIEW", "IPVC <em>Cataract Detection</em>",
                "Class imbalance in 400 fundus photographs · 6 models · 21 experiments",
                lottie_thinking, "thinking_hero")

    best = _best()
    tp, fn, tn, fp = int(best.TP), int(best.FN), int(best.TN), int(best.FP)
    kpi_tiles([
        ("Accuracy", f"{best.accuracy:.1%}", "🎯", "teal", f"{tp + tn} / 80 test images correct"),
        ("Sensitivity", f"{best.recall:.0%}", "🔎", "mint", f"{tp} / 20 cataracts detected"),
        ("Specificity", f"{best.specificity:.1%}", "🛡️", "sky", f"{tn} / 60 normal images cleared"),
        ("F1-score", f"{best.f1:.3f}", "⚖️", "violet", "Cataract F1"),
        ("MCC", f"{best.mcc:.3f}", "📈", "peach", "Highest of 21 experiments"),
    ])
    st.caption(f"{best.model} · {best.strategy} · highest MCC on one 80-image hold-out test set.")

    left, right = st.columns(2)
    with left:
        with card("Dataset", "3:1 class imbalance · 300 normal, 100 cataract"):
            show(class_chart())
    with right:
        with card("Best model", f"{best.model} · {best.strategy} · confusion matrix"):
            show(confusion_chart(best))

    analytics = [
        ("Descriptive", "What is in the dataset?", "teal"),
        ("Diagnostic", "What changed with imbalance handling?", "violet"),
        ("Predictive", "What does the saved model predict?", "peach"),
        ("Prescriptive", "What is the appropriate next step?", "mint"),
    ]
    minis = "".join(
        f"<div class='mini' style='--tint:{TINT[t][0]}'><b><span class='step-no'>{i}</span>{name}</b>"
        f"<span class='q'>{question}</span></div>" for i, (name, question, t) in enumerate(analytics, 1))
    flow = [("01", "Collect", "400 images · 300 normal, 100 cataract"), ("02", "Split", "320 train · 80 test"),
            ("03", "Process", "Crop · 224×224 · CLAHE"), ("04", "Experiment", "6 models · 21 conditions"),
            ("05", "Evaluate", "Recall · specificity · F1 · MCC")]
    rows = "".join(f"<div class='flow-row'><div class='flow-no'>{n}</div><div><b>{t}</b><small>{d}</small></div></div>"
                   for n, t, d in flow)
    st.markdown(
        f"<div class='bento'>"
        f"<div class='cell s7'><div class='bento-title'>Four analytics perspectives</div><div class='mini-grid'>{minis}</div>"
        f"<div class='bento-sub' style='margin:14px 0 0'>The 21 experiments cover 12 classical and 9 neural configurations; "
        f"image processing and test split were held constant.</div></div>"
        f"<div class='cell s5'><div class='bento-title'>Research workflow</div><div class='flow'>{rows}</div></div>"
        f"</div>", unsafe_allow_html=True)

    base = model_result("EfficientNetB0", "baseline")
    wt = model_result("EfficientNetB0", "class weights")
    with st.expander("Baseline vs class weights · screening trade-off"):
        st.dataframe([
            {"Configuration": "Baseline · highest MCC", "Sensitivity": f"{base.recall:.0%}", "Specificity": f"{base.specificity:.1%}",
             "FN": int(base.FN), "FP": int(base.FP), "MCC": round(base.mcc, 3)},
            {"Configuration": "Class weights · highest recall", "Sensitivity": f"{wt.recall:.0%}", "Specificity": f"{wt.specificity:.1%}",
             "FN": int(wt.FN), "FP": int(wt.FP), "MCC": round(wt.mcc, 3)},
        ], hide_index=True, use_container_width=True)
        st.caption(f"Class weights detect {int(base.FN - wt.FN)} additional cataracts but produce "
                   f"{int(wt.FP - base.FP)} additional false alarms on this test set.")

    with st.expander("View all 21 experiment results"):
        df = load_results().sort_values(["mcc", "f1"], ascending=False)
        st.dataframe(df[["family", "model", "strategy", "accuracy", "recall", "specificity", "f1", "mcc", "FN", "FP"]]
                     .rename(columns={"family": "Family", "model": "Model", "strategy": "Strategy", "accuracy": "Accuracy",
                                      "recall": "Sensitivity", "specificity": "Specificity", "f1": "F1", "mcc": "MCC"}),
                     hide_index=True, use_container_width=True)

    st.caption("Prototype for academic comparison only. Scores are not clinically validated.")
    if st.button("EXPLORE THE DATA ➜", key="go_explore"):
        goto("go_to_what_is")
