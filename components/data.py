"""Cached loaders and small statistics helpers shared by all pages."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "assets" / "data"

# Class sizes of the study (3:1 imbalance) and the 80-image test set.
N_NORMAL, N_CATARACT = 300, 100
TEST_NORMAL, TEST_CATARACT = 60, 20

# Mean and SD of the three clarity measurements per class (full 400-image run in the notebook).
# Used only when assets/data/clarity_features.csv has not been exported yet.
CLARITY_SUMMARY = {
    "Edge density": {"Normal": (0.0970, 0.0274), "Cataract": (0.0419, 0.0232)},
    "Corner density": {"Normal": (0.0115, 0.0084), "Cataract": (0.0084, 0.0079)},
    "Laplacian variance": {"Normal": (443.87, 134.46), "Cataract": (325.75, 121.88)},
}
CLARITY_COLUMNS = {"edge_density": "Edge density",
                   "corner_density": "Corner density",
                   "laplacian_variance": "Laplacian variance"}


@st.cache_data
def load_results():
    df = pd.read_csv(DATA_DIR / "results.csv")
    df["label"] = df["model"] + " + " + df["strategy"]
    return df


@st.cache_data
def load_cv():
    df = pd.read_csv(DATA_DIR / "repeated_cv.csv")
    df["label"] = df["model"] + " + " + df["strategy"]
    return df


@st.cache_data
def load_clarity():
    """Per-image clarity values, or None if the notebook export has not been added yet."""
    path = DATA_DIR / "clarity_features.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path).rename(columns=CLARITY_COLUMNS)
    return df


@st.cache_data
def load_test_scores():
    """Test-set labels and neural scores (one row per image per configuration), or None."""
    path = DATA_DIR / "test_scores.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df["label"] = df["model"] + " + " + df["strategy"]
    return df


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def wilson_interval(k, n, z=1.96):
    """Wilson 95% interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def metrics_at(y_true, score, threshold):
    """Confusion counts and metrics when images with score >= threshold are called cataract."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "recall": recall, "specificity": spec,
            "precision": prec, "f1": f1, "mcc": mcc, "accuracy": (tp + tn) / len(y)}


def predictive_values(recall, specificity, prevalence):
    """Positive and negative predictive value for an assumed share of people with cataract."""
    tp = recall * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    tn = specificity * (1 - prevalence)
    fn = (1 - recall) * prevalence
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return ppv, npv
