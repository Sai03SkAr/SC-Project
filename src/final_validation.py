"""
Phase 06 final validation: evaluate each method's chosen feature subset ONCE on
the sealed test set, with the full-strength classifier, so every row of the
comparison table is measured the same way.

Run:  python src/final_validation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from cost_model import build_feature_tiers, detection_cost_ms  # noqa: E402
from problem import RANDOM_STATE, load_dataset, make_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"


def evaluate(cols, names, X_tr, y_tr, X_te, y_te, tiers):
    idx = [names.index(c) for c in cols]
    mask = np.zeros(len(names), dtype=bool)
    mask[idx] = True
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    t0 = time.time()
    clf.fit(X_tr[:, idx], y_tr)
    train_s = time.time() - t0
    pred = clf.predict(X_te[:, idx])
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    return {
        "n_features": len(idx),
        "cost_ms": float(detection_cost_ms(mask, names, tiers)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)),
        "fpr": float(fp / (fp + tn)),
        "train_time_s": train_s,
    }


def main() -> int:
    X, y, names = load_dataset()
    s = make_splits(X, y, search_subsample=None)
    X_tr, y_tr, X_te, y_te = s["X_train_full"], s["y_train_full"], s["X_test"], s["y_test"]
    tiers = build_feature_tiers(names)

    subsets = {"all_features": names}
    for algo in ("spea2", "nsga2"):
        d = json.loads((TABLES / f"{algo}_pareto_analysis.json").read_text())
        for prof in ("email_gateway", "browser_extension"):
            subsets[f"{algo}_{prof}"] = d["deployment_profiles"][prof]["selected_features"]
    for r in json.loads((TABLES / "pso_weighted_baseline.json").read_text()):
        subsets[f"pso_{r['weights']['name']}"] = r["selected_features"]
    filt = json.loads((TABLES / "classical_baselines.json").read_text())["mutual_info_filter"]
    subsets["mi_filter_top25"] = filt["selected_features"]

    out = {}
    for name, cols in subsets.items():
        out[name] = evaluate(cols, names, X_tr, y_tr, X_te, y_te, tiers)
        r = out[name]
        print(f"{name:<30}{r['n_features']:>4} feat  {r['cost_ms']:>9.2f} ms  "
              f"acc {r['accuracy']:.2%}  recall {r['recall']:.2%}  FPR {r['fpr']:.2%}  "
              f"train {r['train_time_s']:.1f}s")

    (TABLES / "final_validation_test_set.json").write_text(json.dumps(out, indent=2))
    print("\nsaved: results/tables/final_validation_test_set.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
