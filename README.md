# IPVC Cataract Dashboard — restored original dashboard design

**UI:** Original published navy/teal theme and sidebar, original author photograph and fundus samples, original Lottie illustrations, and the original YouTube educational video on the *What is Cataract* page.

**Detection page:** Original control-bar + three-panel Samples/Upload → Image → Result dashboard, plus five original-style KPI cards. Three saved models can be explicitly selected. A cutoff slider is available for *demonstrating a single prediction*; **all reported test metrics always use the original notebook decision rules** (0.50 for EfficientNetB0 and zero for SVM). Do not present a user-selected slider threshold as validated performance. No model fallback and no test-score policy lab.

**Research:** 21 documented experiments; EfficientNetB0 baseline highest observed MCC 0.722, 90.0% accuracy, 70% sensitivity; weighted EfficientNetB0 highest observed sensitivity 90% (MCC 0.687), on the one 80-image hold-out set. Research prototype only.

**Run:** `py verify_assets.py` then `py -m streamlit run app.py`. Never upload secrets or `.env` files.

**Deployment:** Copy the contents of this ZIP into the existing GitHub clone; do not replace `.git`. Remove stale `assets/data/test_scores.csv` if it remains in the old clone. Run the verification and inference checks before commit/push.

---

# IPVC Cataract Dashboard — original visuals + final results

This version **restores the original GitHub dashboard visual theme** (navy/teal sidebar, coral category accents, author image, fundus photos and three Lottie animations), while **preserving the newer concise four-page content**, model selection and final 21-experiment results.

**Pages:** Overview (including four analytics perspectives and five-step methodology overview), What is Cataract (optional image-clarity demonstration), Under the Lens (results and expandable methods), Screen a Photo (fixed-threshold research prediction).

- **Primary comparison winner:** EfficientNetB0 baseline: 90.0% accuracy, 70% cataract sensitivity, 96.7% normal specificity, MCC 0.722.
- **Higher-sensitivity alternative:** EfficientNetB0 class weights: 90% cataract sensitivity, MCC 0.687.
- **Classical reference:** SVM + SMOTE.

The selected model is always explicit; app inference does not silently fall back to another model. EfficientNetB0 threshold is 0.5 and SVM uses a zero decision-function boundary. Scores are not clinical probabilities.

## Local validation

Run these from the repository root:

```powershell
py -m pip install -r requirements.txt
py verify_assets.py
py -m streamlit run app.py
```

If copying these files into an existing GitHub clone, delete the **old tracked** `assets/data/test_scores.csv` and any superseded files after backup, because simply overwriting files does not remove old files. Do not copy `.git` from ZIP archives. Review `git status` and verify no credentials before pushing.

**Evidence:** Model/data SHA-256 and static validation are available via `verify_assets.py`. Re-testing both TensorFlow models on the full 80-image test set, and running Streamlit in the actual cloud runtime, remain necessary; restoring the theme does not alter model preprocessing or saved weights.

No API key is required for local saved-model prediction. Do not commit `.env` or `.streamlit/secrets.toml`. The nonessential insecure old devcontainer config was removed from this package.
