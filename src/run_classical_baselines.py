"""
Phase 03 (continued) - classical reduction baselines: PCA and a filter method.

These are what SPEA2/NSGA-II have to beat. Neither optimises for detection
cost or the FNR/FPR split the way the multi-objective search does, so they
are evaluated only on accuracy-family metrics at a chosen component/feature
count, for comparison with a similarly-sized point on the Pareto front.

PCA is included specifically to make the interpretability argument concrete:
its output is blended components, not named features, so it cannot answer
"which checks are worth computing" - only SPEA2/NSGA-II can.

Run:  python src/run_classical_baselines.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from problem import RANDOM_STATE, load_dataset, make_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

# Match the filter's k to a representative point on the SPEA2 front once
# available; 25 is the documented "target" subset size from docs/08.
DEFAULT_K = 25


def evaluate_subset(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
    }


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    X, y, names = load_dataset()
    splits = make_splits(X, y, search_subsample=None)
    X_train, y_train = splits["X_train_full"], splits["y_train_full"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    results = {}

    # ---------------------------------------------------------------- PCA
    print("PCA (95% variance)...")
    t0 = time.time()
    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    Xtr_pca = pca.fit_transform(Xtr_s)
    Xte_pca = pca.transform(Xte_s)
    n_components = pca.n_components_

    pca_result = evaluate_subset(Xtr_pca, y_train, Xte_pca, y_test)
    pca_result["n_components"] = int(n_components)
    pca_result["elapsed_s"] = time.time() - t0
    results["pca"] = pca_result
    print(f"  {n_components} components -> accuracy={pca_result['accuracy']:.4f} "
          f"recall={pca_result['recall']:.4f}  ({pca_result['elapsed_s']:.1f}s)")
    print("  NOTE: components are blended combinations of all 111 features -")
    print("  this result cannot be turned into a list of 'checks worth computing'.")

    # ------------------------------------------------------------- filter
    print(f"\nMutual information filter (top {DEFAULT_K})...")
    t0 = time.time()
    mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    top_k_idx = np.argsort(mi_scores)[::-1][:DEFAULT_K]
    top_k_names = [names[i] for i in top_k_idx]

    filt_result = evaluate_subset(
        X_train[:, top_k_idx], y_train, X_test[:, top_k_idx], y_test
    )
    filt_result["k"] = DEFAULT_K
    filt_result["selected_features"] = top_k_names
    filt_result["elapsed_s"] = time.time() - t0
    results["mutual_info_filter"] = filt_result
    print(f"  top {DEFAULT_K} features -> accuracy={filt_result['accuracy']:.4f} "
          f"recall={filt_result['recall']:.4f}  ({filt_result['elapsed_s']:.1f}s)")
    print(f"  top 10 by MI score: {top_k_names[:10]}")

    out = TABLES / "classical_baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
