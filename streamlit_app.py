import cv2
import numpy as np
import streamlit as st

import logic as L

st.set_page_config(page_title="Cataract Awareness and Screening Dashboard", layout="wide")

ABOUT = """## Cataract: what it is and why it matters
A **cataract** is a clouding of the eye's natural lens. Light scatters instead of focusing cleanly on the retina, so vision gets blurry, hazy or dull. Cataract was the leading cause of blindness among adults aged 50 and older in 2020, with an estimated 15.2 million cases worldwide (GBD 2019 Blindness and Vision Impairment Collaborators, 2021). It is treatable, and early checks help."""

RISKS = """**Common risk factors**
- Increasing age
- Diabetes
- Smoking
- Long-term sun (UV) exposure
- Long-term steroid use
- Past eye injury or inflammation, family history"""

SYMPTOMS = """**Common symptoms**
- Blurred or cloudy vision
- Glare or halos around lights
- Colours looking faded or yellowed
- Trouble seeing at night
- Frequent changes in glasses prescription
- Double vision in one eye"""

HOW = """### How can a photo of the back of the eye help?
A **fundus photograph** shows the retina at the back of the eye. A clouded lens sits in front of the retina, so the photograph looks **less sharp**: fewer clear edges, softer vessel detail. The study measures this loss of clarity with image processing and trains models to tell normal and cataract photographs apart.

### About this dashboard
It follows four steps: **1 Descriptive** (this page), **2 Diagnostic** (the evidence, for data enthusiasts), **3 Predictive** (try the model on a fundus photograph) and **4 Prescriptive** (what to do next). It is a student research demonstration and **not a diagnostic tool**."""

RULES = """**Accepted images.** A colour fundus photograph of one eye: a round, red-orange retina on a dark background (like the samples below), JPG or PNG, at least 200 px on the short side.
**Not accepted.** Faces, ordinary eye close-ups, screenshots, black-and-white images, scans of printed pictures. These return "Image not accepted".
The model was trained on 400 images from one public dataset. Uploaded images are processed in memory and not saved."""

SAMPLES = {"Normal 1": "assets/examples/normal_1.jpg", "Normal 2": "assets/examples/normal_2.jpg",
           "Cataract 1": "assets/examples/cataract_1.jpg", "Cataract 2": "assets/examples/cataract_2.jpg"}

st.title("Cataract Awareness and Screening Dashboard")
st.caption("XBDS3024N Image Processing and Computer Vision · Research demonstration, not a diagnosis")

t1, t2, t3, t4 = st.tabs(["1. Descriptive", "2. Diagnostic (for data enthusiasts)", "3. Predictive: try it", "4. Prescriptive: what next?"])

# ---------------- 1 DESCRIPTIVE ----------------
with t1:
    st.markdown(ABOUT)
    c1, c2 = st.columns(2)
    c1.markdown(RISKS); c2.markdown(SYMPTOMS)
    st.caption("General information, not medical advice.")
    st.markdown(HOW)
    st.markdown("### The data behind the study")
    a, b = st.columns([1, 2])
    a.pyplot(L.class_distribution_fig())
    b.image("assets/sample_images.png", caption="Normal and cataract fundus photographs after preprocessing")

# ---------------- 2 DIAGNOSTIC ----------------
with t2:
    st.markdown("**Why do the classes separate, and why do some models fail?** Each chart answers one question.")
    a, b = st.columns(2)
    a.pyplot(L.clarity_fig())
    b.markdown("Cataract images show lower edge density, corner density and Laplacian variance (Mann–Whitney p = 3.6e-39, 7.7e-5, 4.1e-14). Edge density differs most, consistent with lens opacity blurring retinal detail. These are clarity indicators, not clinical biomarkers.")
    st.markdown("**Accuracy hides missed cataracts.** Several models reach 85 to 89% accuracy with recall of 45 to 60%. The Small CNN with augmentation scores 75% accuracy and 0% recall because every sigmoid score stayed below 0.5 (range 0.456 to 0.480).")
    a, b = st.columns(2)
    a.pyplot(L.accuracy_paradox_fig())
    strat2 = b.selectbox("Compare errors under strategy", ["baseline", "class weights", "ROS", "SMOTE", "augmentation"], index=1)
    b.pyplot(L.errors_fig(strat2))
    b.markdown(L.diagnosis_text(strat2))
    st.markdown("**Class weighting trades false negatives for false alarms.** It raised recall for five of six models (EfficientNetB0 from 70% to 95%) and lowered MCC for four.")
    st.image("assets/recspec.png", caption="Recall versus specificity, all 21 experiments", width=620)
    st.markdown("### All 21 experiments (test set, 80 images)")
    f1, f2 = st.columns(2)
    fam = f1.selectbox("Family", ["All", "Classical", "Neural"])
    strat = f2.selectbox("Strategy", ["All", "baseline", "class weights", "ROS", "SMOTE", "augmentation"])
    st.dataframe(L.results_table(fam, strat), use_container_width=True, hide_index=True)
    st.image("assets/mcc21.png", caption="MCC ranking of all 21 experiments", width=700)

# ---------------- 3 PREDICTIVE ----------------
with t3:
    st.markdown("### Check a fundus photograph")
    st.markdown("EfficientNetB0 with class weights (best in the study: MCC 0.726, recall 95%) looks for a cataract signal.")
    st.markdown(RULES)
    up = st.file_uploader("Upload a fundus photograph", type=["jpg", "jpeg", "png"])
    st.markdown("**Or try a sample** (taken from the dataset, so the model has seen these)")
    cols = st.columns(4)
    for col, (name, path) in zip(cols, SAMPLES.items()):
        col.image(path, caption=name, use_container_width=True)
        if col.button(f"Use {name}", key=name):
            st.session_state["sample"] = path
            st.session_state.pop("result", None)

    img = None
    if up is not None:
        arr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            st.error("The file could not be read as an image.")
    elif st.session_state.get("sample"):
        img = cv2.cvtColor(cv2.imread(st.session_state["sample"]), cv2.COLOR_BGR2RGB)
        st.info(f"Using sample: {st.session_state['sample'].split('/')[-1]}")

    if img is not None:
        left, right = st.columns(2)
        left.image(img, caption="Input image", use_container_width=True)
        if st.button("Check image", type="primary"):
            with st.spinner("Analysing"):
                st.session_state["result"] = L.predict(img)
        res = st.session_state.get("result")
        if res:
            text, enhanced, fig, advice = res
            right.markdown(text)
            if enhanced is not None:
                right.image(enhanced, caption="Preprocessed input (224 × 224, CLAHE)")
            if fig is not None:
                st.pyplot(fig)
            if advice != L.ADVICE_NONE:
                st.success("Result ready. Open the **4. Prescriptive** tab for next steps, shown below as well.")
                st.markdown(advice)
    st.caption("Research demonstration. The score is not a calibrated probability and the result is not a diagnosis.")

# ---------------- 4 PRESCRIPTIVE ----------------
with t4:
    res = st.session_state.get("result")
    st.markdown(res[3] if res else L.ADVICE_NONE)
