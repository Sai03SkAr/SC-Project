"""
UGRansome (Nkongolo et al., 2021) loader. The CSV comes from Kaggle (login required):
https://www.kaggle.com/datasets/nkongolo/ugransome-dataset - place it in data/ugransome/.

UGRansome has no benign class: every row is ransomware-related network traffic, labelled
A (anomaly - unknown, zero-day threat without a signature), S (signature - well-known threat)
or SS (synthetic signature - a mix of both). The binary task used here, matching the
dataset's stated purpose, is zero-day detection: A (1) vs S (0). SS rows are dropped as
ambiguous.

Each original attribute stays ONE feature (categorical text gets a category code - sorted
labels, no use of the target - not one-hot), so the feature space stays small enough to enumerate exhaustively.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import RANDOM_STATE, Dataset


def _norm(c: str) -> str:
    return re.sub(r"[^a-z]", "", c.lower())


def build_ugransome(folder: Path) -> Dataset:
    csvs = sorted(Path(folder).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No UGRansome CSV in {folder} - download it from Kaggle first.")
    df = pd.read_csv(csvs[0])
    cols = {_norm(c): c for c in df.columns}
    label = cols.get("prediction")
    if label is None:
        raise ValueError(f"No Prediction column in {csvs[0].name}: {list(df.columns)}")

    lab = df[label].astype(str).str.strip().str.upper()
    df = df[lab.isin(["A", "S"])].reset_index(drop=True)
    y = (lab[lab.isin(["A", "S"])].reset_index(drop=True) == "A").astype(np.int8).to_numpy()

    feats, names = [], []
    for c in df.columns:
        if c == label:
            continue
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() > 0.95:
            feats.append(num.fillna(-1).to_numpy(float))
        else:                                   # category code; frequency encoding could merge
            codes = pd.Categorical(s.astype(str).str.strip()).codes   # equally common categories
            feats.append(codes.astype(float))
        names.append(str(c).strip())
    X = np.column_stack(feats).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=RANDOM_STATE)
    return Dataset(name="ugransome", title="UGRansome", X_train=X_tr, y_train=y_tr,
                   X_test=X_te, y_test=y_te, feature_names=names,
                   feature_tier={n: "network" for n in names},
                   notes={"split": "stratified 80/20, random_state=42", "file": csvs[0].name,
                          "task": "zero-day (A) vs known-signature (S); SS dropped",
                          "rows_used": int(len(df))})
