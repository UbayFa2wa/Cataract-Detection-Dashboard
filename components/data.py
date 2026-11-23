"""Final 21-experiment data from the submitted CATARACT notebook."""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'assets' / 'data'
N_NORMAL, N_CATARACT = 300, 100
TEST_NORMAL, TEST_CATARACT = 60, 20


@st.cache_data

def load_results():
    df = pd.read_csv(DATA / 'results.csv')
    required = {'family','model','strategy','accuracy','recall','specificity',
                'f1','mcc','TN','FP','FN','TP'}
    missing = required - set(df.columns)
    if missing or len(df) != 21:
        raise ValueError(f'Expected 21 final experiments; missing fields: {sorted(missing)}')
    if not ((df.TN + df.FP == TEST_NORMAL) & (df.TP + df.FN == TEST_CATARACT)).all():
        raise ValueError('Results are not from the documented 60/20 test split.')
    df['label'] = df['model'] + ' + ' + df['strategy']
    return df


@st.cache_data

def load_cv():
    df = pd.read_csv(DATA / 'repeated_cv.csv')
    if len(df) != 12:
        raise ValueError('Expected 12 classical cross-validation configurations.')
    df['label'] = df['model'] + ' + ' + df['strategy']
    return df


@st.cache_data

def load_clarity():
    df = pd.read_csv(DATA / 'clarity_features.csv')
    if len(df) != 400 or df['class'].value_counts().to_dict() != {'Normal':300, 'Cataract':100}:
        raise ValueError('Clarity data must match the 300/100 full dataset.')
    return df


def model_result(model, strategy):
    d = load_results()
    found = d.loc[(d.model == model) & (d.strategy == strategy)]
    if len(found) != 1:
        raise ValueError(f'Missing performance row for {model} / {strategy}')
    return found.iloc[0]
