"""Shared evaluation: train once on the full training split, score once on the sealed test set.

Scores are reported twice - on the whole test set, and on only the test samples whose exact
feature vector never appears in training ("unseen"). MLRan's predefined split has a third of
its test vectors duplicated in training, which rewards memorising over detecting.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

from datasets import RANDOM_STATE


def unseen_mask(ds):
    train_rows = {r.tobytes() for r in np.ascontiguousarray(ds.X_train)}
    return np.array([r.tobytes() not in train_rows for r in np.ascontiguousarray(ds.X_test)])


def rates(y, pred, proba=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"accuracy": (tp + tn) / len(y), "recall": tp / (tp + fn) if tp + fn else np.nan,
           "precision": tp / (tp + fp) if tp + fp else np.nan,
           "fpr": fp / (fp + tn) if fp + tn else np.nan, "n": int(len(y))}
    if proba is not None and len(np.unique(y)) == 2:
        out["auc"] = roc_auc_score(y, proba)
    return {k: float(v) for k, v in out.items()}


def evaluate_subset(ds, mask, clf=None, unseen=None):
    mask = np.asarray(mask, dtype=bool)
    clf = clf or RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    t0 = time.time()
    clf.fit(ds.X_train[:, mask], ds.y_train)
    train_s = time.time() - t0
    proba = clf.predict_proba(ds.X_test[:, mask])[:, 1] if hasattr(clf, "predict_proba") else None
    pred = clf.predict(ds.X_test[:, mask])
    unseen = unseen_mask(ds) if unseen is None else unseen
    res = {"n_features": int(mask.sum()), "train_s": train_s,
           "test_all": rates(ds.y_test, pred, proba),
           "test_unseen": rates(ds.y_test[unseen], pred[unseen],
                                None if proba is None else proba[unseen])}
    return res
