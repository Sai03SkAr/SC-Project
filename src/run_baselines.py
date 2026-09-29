"""
Phase 03 - baseline classifiers on ALL 111 features.

This establishes the control point every reduced-feature result is compared
against: what does the problem look like with no feature selection at all?

Reports the same objectives as the search (detection cost, FNR, FPR) plus the
standard classification metrics, for three classifiers:

    Random Forest  - primary; used inside the SPEA2/NSGA-II search itself
    SVM            - baseline; needs scaling, sensitive to the -1 sentinels
    KNN            - baseline; distance-based, most damaged by irrelevant
                     features, so reduction benefits should show up most here

Data discipline: fit on the 80% training split only, evaluate ONCE on the
sealed 20% test split - the same split make_splits() produces for the search,
so results are directly comparable.

Run:  python src/run_baselines.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))

from cost_model import max_detection_cost_ms  # noqa: E402
from problem import RANDOM_STATE, load_dataset, make_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"


def evaluate(clf, X_train, y_train, X_test, y_test, scale: bool = False):
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_s = time.time() - t0

    t0 = time.time()
    pred = clf.predict(X_test)
    total_s = time.time() - t0
    latency_ms = (total_s / len(X_test)) * 1000

    proba = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        proba = clf.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn)

    return {
        "accuracy": float(acc),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if proba is not None else None,
        "fnr": float(fnr),
        "fpr": float(fpr),
        "train_time_s": train_s,
        "prediction_latency_ms_per_sample": latency_ms,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)

    X, y, names = load_dataset()
    splits = make_splits(X, y, search_subsample=None)  # full training data
    X_train, y_train = splits["X_train_full"], splits["y_train_full"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    print(f"train: {X_train.shape}   test (sealed): {X_test.shape}")
    print(f"all-111-features detection cost: {max_detection_cost_ms():.2f} ms\n")

    classifiers = {
        "random_forest": (
            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
            False,
        ),
        "svm": (SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE), True),
        "knn": (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True),
    }

    results = {}
    for name, (clf, scale) in classifiers.items():
        print(f"training {name}...", flush=True)
        t0 = time.time()
        results[name] = evaluate(clf, X_train, y_train, X_test, y_test, scale=scale)
        results[name]["scaled"] = scale
        print(f"  accuracy={results[name]['accuracy']:.4f}  "
              f"recall={results[name]['recall']:.4f}  "
              f"FNR={results[name]['fnr']:.4f}  "
              f"FPR={results[name]['fpr']:.4f}  "
              f"({time.time() - t0:.1f}s)")

    results["_meta"] = {
        "n_features": len(names),
        "detection_cost_ms": max_detection_cost_ms(),
        "random_state": RANDOM_STATE,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
    }

    out = TABLES / "baselines_all_111_features.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out.relative_to(ROOT)}")

    print("\n=== SUMMARY (control point for all later comparisons) ===")
    print(f"{'classifier':<16}{'accuracy':>10}{'recall':>10}{'FNR':>9}{'FPR':>9}{'latency(ms)':>13}")
    for name, r in results.items():
        if name == "_meta":
            continue
        print(f"{name:<16}{r['accuracy']:>10.4f}{r['recall']:>10.4f}"
              f"{r['fnr']:>9.4f}{r['fpr']:>9.4f}{r['prediction_latency_ms_per_sample']:>13.4f}")
    print(f"{'(all features)':<16}{'':>10}{'':>10}{'':>9}{'':>9}"
          f"{max_detection_cost_ms():>13.2f}  <- feature-extraction cost, separate from model latency")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
