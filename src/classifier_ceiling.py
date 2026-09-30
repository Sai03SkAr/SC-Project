"""
Does a stronger classifier move the false-alarm ceiling?

Random Forest reaches 2% FPR only at ~90% recall. This tests other classifiers on the same
splits. For each, the cut-off is chosen on the validation split (never the test set) for two
targets - FPR <= 2%, and recall >= 95% - and then measured once on the sealed test set.

Run:  python src/classifier_ceiling.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from problem import RANDOM_STATE, load_dataset, make_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"


def rates(y, p, thr):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return tp / (tp + fn), fp / (fp + tn)


def pick(y, p, cond):
    grid = np.linspace(0.05, 0.99, 941)
    ok = [t for t in grid if cond(*rates(y, p, t))]
    return ok


def main() -> int:
    X, y, names = load_dataset()
    s = make_splits(X, y, search_subsample=None)
    spea = json.loads((TABLES / "spea2_pareto_analysis.json").read_text())
    subsets = {
        "all_features": list(range(len(names))),
        "spea2_37": [names.index(c) for c in
                     spea["deployment_profiles"]["email_gateway"]["selected_features"]],
    }
    models = {
        "random_forest_200": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "random_forest_500": lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE),
        "extra_trees_500": lambda: ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=RANDOM_STATE),
    }

    out = {}
    print(f"{'subset':<14}{'model':<24}{'AUC':>7}  {'@FPR<=2%: recall / FPR':>24}  {'@recall>=95%: recall / FPR':>28}  both?")
    for sname, idx in subsets.items():
        for mname, make in models.items():
            clf = make()
            t0 = time.time()
            clf.fit(s["X_fit"][:, idx], s["y_fit"])
            pv = clf.predict_proba(s["X_val"][:, idx])[:, 1]
            pt = clf.predict_proba(s["X_test"][:, idx])[:, 1]
            auc = roc_auc_score(s["y_test"], pt)

            t_fpr = min(pick(s["y_val"], pv, lambda r, f: f <= 0.02) or [0.99])
            t_rec = max(pick(s["y_val"], pv, lambda r, f: r >= 0.95) or [0.05])
            r1, f1 = rates(s["y_test"], pt, t_fpr)
            r2, f2 = rates(s["y_test"], pt, t_rec)
            both = [t for t in pick(s["y_val"], pv, lambda r, f: r >= 0.95 and f <= 0.02)]
            both_test = rates(s["y_test"], pt, both[0]) if both else None

            out[f"{sname}/{mname}"] = {
                "test_auc": float(auc), "fit_s": time.time() - t0,
                "fpr2_threshold": float(t_fpr), "fpr2_test_recall": float(r1), "fpr2_test_fpr": float(f1),
                "rec95_threshold": float(t_rec), "rec95_test_recall": float(r2), "rec95_test_fpr": float(f2),
                "both_targets_on_validation": bool(both),
                "both_targets_test": None if both_test is None else [float(both_test[0]), float(both_test[1])],
            }
            print(f"{sname:<14}{mname:<24}{auc:>7.4f}  {r1:>11.2%} / {f1:>6.2%}     {r2:>12.2%} / {f2:>6.2%}     "
                  f"{'YES' if both else 'no'}")

    (TABLES / "classifier_ceiling.json").write_text(json.dumps(out, indent=2))
    print("\nsaved: results/tables/classifier_ceiling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
