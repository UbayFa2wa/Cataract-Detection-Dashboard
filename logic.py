"""Data, charts and prediction helpers (no Gradio import, so they can be tested alone)."""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = pd.read_csv(os.path.join(HERE, "data", "results.csv"))
CV = pd.read_csv(os.path.join(HERE, "data", "repeated_cv.csv"))
RES["label"] = RES["model"] + " + " + RES["strategy"]
COL = {"Classical": "#4C72B0", "Neural": "#DD8452"}
MODEL_PATH = os.path.join(HERE, "cataract_efficientnetb0_class_weight.keras")
SVM_PATH = os.path.join(HERE, "cataract_svm_smote.joblib")
IMG_SIZE = 224


# ---------- Descriptive ----------
def class_distribution_fig():
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.bar(["Normal", "Cataract"], [300, 100], color=["#4C72B0", "#C44E52"])
    for i, v in enumerate([300, 100]):
        ax.text(i, v + 6, f"{v} ({v / 4:.0f}%)", ha="center")
    ax.set_ylabel("Images"); ax.set_ylim(0, 340)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); return fig


def accuracy_paradox_fig():
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for fam, g in RES.groupby("family"):
        ax.scatter(g["accuracy"], g["recall"], c=COL[fam], label=fam, s=45, edgecolor="white")
    ax.axvline(0.75, ls="--", c="#888"); ax.text(0.752, 0.02, "all-normal baseline\n(75% accuracy)", fontsize=8, color="#555")
    ax.set_xlabel("Accuracy"); ax.set_ylabel("Cataract recall"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.25)
    fig.tight_layout(); return fig


# ---------- Diagnostic ----------
def clarity_fig():
    names = ["Canny edge\ndensity", "Harris corner\ndensity", "Laplacian variance\n(÷1000)"]
    normal = [0.0970, 0.0115, 0.44387]; n_sd = [0.0274, 0.0084, 0.13446]
    cat = [0.0419, 0.0084, 0.32575]; c_sd = [0.0232, 0.0079, 0.12188]
    fig, ax = plt.subplots(figsize=(6, 3.8)); x = np.arange(3)
    ax.bar(x - .18, normal, .36, yerr=n_sd, capsize=3, color="#4C72B0", label="Normal")
    ax.bar(x + .18, cat, .36, yerr=c_sd, capsize=3, color="#C44E52", label="Cataract")
    ax.set_xticks(x, names); ax.legend(frameon=False); ax.set_ylabel("Mean ± SD")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); return fig


def errors_fig(strategy):
    d = RES[RES["strategy"] == strategy].sort_values("model")
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); x = np.arange(len(d))
    ax.bar(x - .18, d["FN"], .36, color="#C44E52", label="False negatives (missed cataract, of 20)")
    ax.bar(x + .18, d["FP"], .36, color="#8172B3", label="False positives (false alarms, of 60)")
    ax.set_xticks(x, d["model"], rotation=20, ha="right")
    ax.set_ylabel("Count"); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); return fig


def diagnosis_text(strategy):
    d = RES[RES["strategy"] == strategy]
    if d.empty:
        return ""
    worst = d.sort_values("FN", ascending=False).iloc[0]
    best = d.sort_values("FN").iloc[0]
    return (f"Under **{strategy}**, the most missed cataract images came from **{worst['model']}** "
            f"({int(worst['FN'])} of 20), and the fewest from **{best['model']}** ({int(best['FN'])} of 20).")


# ---------- Modelling / ranking ----------
def ranking_fig():
    d = RES.sort_values("mcc")
    fig, ax = plt.subplots(figsize=(6.6, 6))
    ax.barh(d["label"], d["mcc"], color=[COL[f] for f in d["family"]])
    for i, v in enumerate(d["mcc"]):
        ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_xlabel("MCC (80-image test set)"); ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); return fig


def results_table(family="All", strategy="All"):
    d = RES.copy()
    if family != "All": d = d[d["family"] == family]
    if strategy != "All": d = d[d["strategy"] == strategy]
    d = d.sort_values(["mcc", "f1"], ascending=False)
    out = d[["model", "strategy", "accuracy", "recall", "specificity", "precision", "f1", "mcc", "roc_auc", "FN", "FP"]].copy()
    for c in ["accuracy", "recall", "specificity", "precision"]:
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    return out


# ---------- Prescriptive ----------
def recommend(cost_ratio, min_spec):
    """Rank experiments by expected cost = ratio * FN + FP, subject to a minimum specificity."""
    d = RES.copy()
    d["cost"] = cost_ratio * d["FN"] + d["FP"]
    ok = d[d["specificity"] >= min_spec / 100].sort_values(["cost", "mcc"], ascending=[True, False])
    if ok.empty:
        return "No experiment meets that specificity floor. Lower it.", pd.DataFrame(), None
    top = ok.iloc[0]
    msg = (f"**Recommended: {top['label']}.** Missing a cataract is weighted {cost_ratio:g}× a false alarm. "
           f"On the test set this model missed {int(top['FN'])} of 20 cataract images and raised {int(top['FP'])} false alarms "
           f"among 60 normal images (recall {top['recall']:.0%}, specificity {top['specificity']:.0%}, MCC {top['mcc']:.3f}).")
    tab = ok.head(5)[["label", "FN", "FP", "recall", "specificity", "mcc", "cost"]].copy()
    tab["recall"] = (tab["recall"] * 100).round(1).astype(str) + "%"
    tab["specificity"] = (tab["specificity"] * 100).round(1).astype(str) + "%"
    tab.columns = ["Configuration", "Missed", "False alarms", "Recall", "Specificity", "MCC", "Cost"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    t5 = ok.head(8).iloc[::-1]
    ax.barh(t5["label"], t5["cost"], color=[COL[f] for f in t5["family"]])
    ax.set_xlabel("Expected cost (false-alarm units)"); ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return msg, tab, fig


# ---------- Predictive ----------
def preprocess(rgb):
    """Crop black border, resize to 224x224 (INTER_AREA), CLAHE on LAB L channel. Input/output: RGB uint8."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > 10)
    if len(ys):
        rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def validate_fundus(rgb):
    """Return (ok, message). Accepts colour retinal photographs with a dark background, like the dataset."""
    h, w = rgb.shape[:2]
    if min(h, w) < 200:
        return False, "The image is too small. Use a photograph at least 200 pixels on the short side."
    if not 0.6 <= w / h <= 2.0:
        return False, "The image proportions are unusual for a fundus photograph."
    small = cv2.resize(rgb, (256, max(32, int(256 * h / w))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    mask = g > 10
    frac = float(mask.mean())
    k = max(4, int(0.06 * min(g.shape)))
    corners = np.mean([g[:k, :k].mean(), g[:k, -k:].mean(), g[-k:, :k].mean(), g[-k:, -k:].mean()])
    r = small[..., 0][mask].mean() if mask.any() else 0
    gr = small[..., 1][mask].mean() if mask.any() else 1
    if corners >= 25 or not 0.25 <= frac <= 0.8:
        return False, "This does not look like a fundus photograph. A fundus image shows a round retina on a dark background."
    if r / max(gr, 1) < 1.1:
        return False, "The colours do not match a colour fundus photograph (the retina should look red to orange)."
    return True, ""


def processing_fig(rgb, enhanced):
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    resp = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    corner = (resp > 0.01 * resp.max())
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, im, t, cm in zip(axes, [rgb, enhanced, edges, corner.astype(np.uint8) * 255],
                             ["1. Original", "2. Crop, resize, CLAHE", f"3. Edges ({(edges > 0).mean():.3f})", f"4. Corners ({corner.mean():.4f})"],
                             [None, None, "gray", "gray"]):
        ax.imshow(im, cmap=cm); ax.set_title(t, fontsize=9); ax.axis("off")
    fig.tight_layout(); return fig


# ---------- Classical features (identical to the notebook) ----------
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern


def colour_histogram(image, bins=16):
    feats = []
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256]).flatten()
        feats.extend(hist / (hist.sum() + 1e-8))
    return feats


def glcm_features(gray, levels=32):
    quantised = (gray // (256 // levels)).astype(np.uint8)
    glcm = graycomatrix(quantised, distances=[1, 3], angles=[0, np.pi / 2],
                        levels=levels, symmetric=True, normed=True)
    props = ["contrast", "correlation", "energy", "homogeneity"]
    return np.concatenate([graycoprops(glcm, p).ravel() for p in props])


def lbp_histogram(gray, P=8, R=1):
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)
    return hist


def hog_features(gray):
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    return hog(small, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2))


def extract_feature_vector(image, edge_map, corner_dens):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    green = image[..., 1]
    clarity = [edge_map.mean() / 255.0, corner_dens, cv2.Laplacian(gray, cv2.CV_64F).var()]
    return np.concatenate([colour_histogram(image), glcm_features(green),
                           lbp_histogram(green), hog_features(green), clarity])


def features_from_enhanced(enhanced):
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    resp = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    cd = (resp > 0.01 * resp.max()).mean()
    return np.asarray(extract_feature_vector(enhanced, edges, cd), dtype=np.float32).reshape(1, -1)


HOSPITAL = """**Hospital Shah Alam (HSAS)**, Department of Ophthalmology
- Address: Persiaran Kayangan, Seksyen 7, 40000 Shah Alam, Selangor
- Services: secondary and tertiary eye care (including medical retina), 24-hour emergency, optometry
- Telephone: +60 3-5526 3000
- Website: https://jknselangor.moh.gov.my/hsas/

Confirm opening hours and referral requirements by phone before visiting."""

ADVICE_CATARACT = """## Cataract signal detected
The model found image features that resemble the cataract class. This is a screening signal, not a diagnosis.

**What to do next**
1. Book a full eye examination with an eye-care professional. Only an examination of the lens can confirm cataract.
2. Bring this result if you wish, along with a list of symptoms: blurred or cloudy vision, glare, faded colours, trouble seeing at night.
3. Seek urgent care for sudden vision loss, eye pain or redness.

**Where to go (example)**

""" + HOSPITAL

ADVICE_NORMAL = """## No cataract signal found
The model did not find features that resemble the cataract class. This is not a clearance. In the study, the models still missed some cataract images on the test set (the SVM + SMOTE model missed 3 of 20), and they were tested on one dataset only.

**What to do next**
1. Keep routine eye check-ups, especially with age, diabetes or other risk factors.
2. See an eye-care professional if blurred vision, glare, faded colours or night-vision trouble appear.
3. Seek urgent care for sudden vision loss, eye pain or redness.

**Where to go if symptoms appear (example)**

""" + HOSPITAL

ADVICE_NONE = "Upload a fundus photograph in the Predictive tab. Advice for the result appears here."

_model = None
_svm = None
MODEL_NAME = "SVM + SMOTE"


def _score_svm(enhanced):
    global _svm
    import joblib
    if _svm is None:
        _svm = joblib.load(SVM_PATH)
    z = _svm["pca"].transform(_svm["scaler"].transform(features_from_enhanced(enhanced)))
    score = float(_svm["classifier"].decision_function(z)[0])
    return score, score >= 0.0, f"SVM decision value **{score:.2f}** (above 0 means cataract signal; not a probability)."


def _score_cnn(enhanced):
    global _model
    import tensorflow as tf
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    score = float(_model.predict(enhanced[None].astype("float32"), verbose=0)[0, 0])
    return score, score >= 0.5, f"Model score **{score:.3f}** (threshold 0.5, not a calibrated probability)."


def predict(rgb):
    """Returns result markdown, preprocessed image, processing figure, advice markdown."""
    if rgb is None:
        return "Upload a fundus photograph first.", None, None, ADVICE_NONE
    rgb = np.asarray(rgb).astype("uint8")
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, -1)
    rgb = rgb[..., :3]
    ok, why = validate_fundus(rgb)
    if not ok:
        return f"### Image not accepted\n{why}\n\nSee the accepted-image rules above.", None, None, ADVICE_NONE
    enhanced = preprocess(rgb)
    fig = processing_fig(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)), enhanced)
    scorer, name = None, ""
    if os.path.isfile(MODEL_PATH):
        try:
            import tensorflow  # noqa: F401
            scorer, name = _score_cnn, "EfficientNetB0 + class weights"
        except ImportError:
            pass
    if scorer is None and os.path.isfile(SVM_PATH):
        scorer, name = _score_svm, "SVM + SMOTE"
    if scorer is None:
        return "No model file found. Add `cataract_svm_smote.joblib` to the app folder.", enhanced, fig, ADVICE_NONE
    _, cat, detail = scorer(enhanced)
    res = f"### Result: {'Cataract signal' if cat else 'No cataract signal'}\n{detail}\n\nModel used: {name}."
    return res, enhanced, fig, (ADVICE_CATARACT if cat else ADVICE_NORMAL)
