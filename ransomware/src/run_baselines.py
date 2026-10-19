"""
Baselines: all features with RF / SVM / KNN, all-static-features RF, PCA, and a
mutual-information filter. Every method is trained on the full training split and scored
once on the sealed test set (all rows, and unseen rows only).

Run:  python src/run_baselines.py --dataset rd2024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
from cost import cost_model_for  # noqa: E402
from datasets import RANDOM_STATE, load  # noqa: E402
from evaluate import evaluate_subset, unseen_mask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--filter-k", type=int, default=25)
    a = ap.parse_args()
    ds = load(a.dataset)
    cost = cost_model_for(ds)
    unseen = unseen_mask(ds)
    all_mask = np.ones(ds.n_features, bool)
    out = {}

    def record(name, mask, clf=None, note=""):
        r = evaluate_subset(ds, mask, clf=clf, unseen=unseen)
        r["cost_ms"] = cost(mask); r["note"] = note
        out[name] = r
        t, u = r["test_all"], r["test_unseen"]
        print(f"{name:<26}{r['n_features']:>5} feat  cost {r['cost_ms']:>10.2f}  "
              f"recall {t['recall']:.2%} FPR {t['fpr']:.2%} | unseen recall {u['recall']:.2%} "
              f"FPR {u['fpr']:.2%} (n={u['n']})", flush=True)

    record("rf_all_features", all_mask)
    record("knn_all_features", all_mask,
           make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)))
    record("svm_all_features", all_mask,
           make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)))

    if hasattr(cost, "tier_masks") and "static" in cost.tier_masks:
        record("rf_all_static_features", cost.tier_masks["static"],
               note="pre-execution ceiling: every static feature, no sandbox")

    # PCA (blended components need every input feature)
    pca_clf = make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=RANDOM_STATE),
                            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))
    record("pca_95", all_mask, pca_clf, note="blended components; needs all inputs")
    out["pca_95"]["n_components"] = int(pca_clf.named_steps["pca"].n_components_)

    mi = mutual_info_classif(ds.X_train, ds.y_train, random_state=RANDOM_STATE)
    k = min(a.filter_k, ds.n_features // 2)   # small spaces: top half, else "top k" = everything
    top = np.zeros(ds.n_features, bool); top[np.argsort(mi)[::-1][:k]] = True
    record(f"mi_filter_top{k}", top, note="filter; ignores feature interactions")
    out[f"mi_filter_top{k}"]["selected"] = [ds.feature_names[i] for i in np.flatnonzero(top)]

    d = ROOT / "results" / ds.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "baselines.json").write_text(json.dumps(out, indent=2))
    print(f"saved results/{ds.name}/baselines.json  (unseen test rows: {unseen.sum()}/{len(unseen)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
