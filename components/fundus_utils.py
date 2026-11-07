"""Image checks, preprocessing, feature extraction and model scoring.

This module does not import Streamlit, so it can be tested on its own.
The preprocessing and the 401 classical features copy the notebook exactly
(Sections 4, 6 and 15B), otherwise the saved models would receive different inputs.
"""
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "assets" / "models"
CNN_PATH = MODEL_DIR / "cataract_efficientnetb0_class_weight.keras"
SVM_PATH = MODEL_DIR / "cataract_svm_smote.joblib"
IMG_SIZE = 224


# ----------------------------------------------------------------------------
# 1. Is this a fundus photograph?
# ----------------------------------------------------------------------------
def validate_fundus(rgb):
    """Return (ok, message). Accepts colour retinal photographs on a dark background.

    Rule set checked on all dataset images (all accepted) and on ordinary photographs (rejected):
    minimum size, aspect ratio, dark corners, share of bright pixels, red-to-green ratio.
    """
    h, w = rgb.shape[:2]
    if min(h, w) < 200:
        return False, "The image is too small. Use a photograph at least 200 pixels on the short side."
    if not 0.6 <= w / h <= 2.0:
        return False, "The image proportions are unusual for a fundus photograph."
    small = cv2.resize(rgb, (256, max(32, int(256 * h / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    mask = gray > 10
    frac = float(mask.mean())
    k = max(4, int(0.06 * min(gray.shape)))
    corners = np.mean([gray[:k, :k].mean(), gray[:k, -k:].mean(),
                       gray[-k:, :k].mean(), gray[-k:, -k:].mean()])
    red = small[..., 0][mask].mean() if mask.any() else 0
    green = small[..., 1][mask].mean() if mask.any() else 1
    if corners >= 25 or not 0.25 <= frac <= 0.8:
        return False, ("This does not look like a fundus photograph. "
                       "A fundus image shows a round retina on a dark background.")
    if red / max(green, 1) < 1.1:
        return False, "The colours do not match a colour fundus photograph (the retina should look red to orange)."
    return True, ""


# ----------------------------------------------------------------------------
# 2. Preprocessing (crop black border, resize, CLAHE on the L channel)
# ----------------------------------------------------------------------------
def preprocess(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > 10)
    if len(ys):
        rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def clarity_maps(enhanced):
    """Edge map, corner mask and the three clarity measurements used in the study."""
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    response = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    corners = response > 0.01 * response.max()
    clarity = {
        "edge_density": float(edges.mean() / 255.0),
        "corner_density": float(corners.mean()),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }
    return edges, corners, clarity


# ----------------------------------------------------------------------------
# 3. Classical features (401 values, identical to the notebook)
# ----------------------------------------------------------------------------
def _colour_histogram(image, bins=16):
    feats = []
    for channel in range(3):
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256]).flatten()
        feats.extend(hist / (hist.sum() + 1e-8))
    return feats


def _glcm_features(gray, levels=32):
    quantised = (gray // (256 // levels)).astype(np.uint8)
    glcm = graycomatrix(quantised, distances=[1, 3], angles=[0, np.pi / 2],
                        levels=levels, symmetric=True, normed=True)
    props = ["contrast", "correlation", "energy", "homogeneity"]
    return np.concatenate([graycoprops(glcm, p).ravel() for p in props])


def _lbp_histogram(gray, P=8, R=1):
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)
    return hist


def _hog_features(gray):
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    return hog(small, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2))


def feature_vector(enhanced, edges, corner_density, lap_var):
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    green = enhanced[..., 1]
    clarity = [edges.mean() / 255.0, corner_density, lap_var]
    vec = np.concatenate([_colour_histogram(enhanced), _glcm_features(green),
                          _lbp_histogram(green), _hog_features(green), clarity])
    return np.asarray(vec, dtype=np.float32).reshape(1, -1)


# ----------------------------------------------------------------------------
# 4. Models (loaded once per server process)
# ----------------------------------------------------------------------------
def cnn_available():
    return CNN_PATH.is_file() and find_spec("tensorflow") is not None


def svm_available():
    return SVM_PATH.is_file()


@lru_cache(maxsize=1)
def _load_cnn():
    import tensorflow as tf
    return tf.keras.models.load_model(CNN_PATH)


@lru_cache(maxsize=1)
def _load_svm():
    import joblib
    return joblib.load(SVM_PATH)


def score_cnn(enhanced):
    """Sigmoid score of EfficientNetB0 (uncalibrated). Cataract when the score is at least the threshold."""
    model = _load_cnn()
    return float(model.predict(enhanced[None].astype("float32"), verbose=0)[0, 0])


def score_svm(enhanced):
    """SVM decision value. Above 0 means the cataract side of the boundary."""
    edges, corners, clarity = clarity_maps(enhanced)
    vec = feature_vector(enhanced, edges, clarity["corner_density"], clarity["laplacian_variance"])
    bundle = _load_svm()
    z = bundle["pca"].transform(bundle["scaler"].transform(vec))
    return float(bundle["classifier"].decision_function(z)[0])


def read_image(file_or_bytes):
    """Decode an uploaded file (bytes or file-like) to an RGB uint8 array, or None."""
    data = file_or_bytes.read() if hasattr(file_or_bytes, "read") else file_or_bytes
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if arr is None else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
