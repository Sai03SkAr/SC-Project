"""
Decision-threshold tuning for the false-alarm (FPR) target.

The classifier flags a site as phishing when P(phishing) >= 0.5 by default. Raising the
threshold lowers the false-alarm rate at some cost to recall. The threshold is chosen on
the internal VALIDATION split (part of the training data) and then applied ONCE to the
sealed test set - the test set never influences the choice.

Run:  python src/threshold_tuning.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from problem import RANDOM_STATE, load_dataset, make_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"


def rates(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tp / (tp + fn), fp / (fp + tn)


def threshold_for_fpr(y_val, p_val, target_fpr):
    """Lowest threshold whose validation FPR is at or below the target."""
    for thr in np.linspace(0.30, 0.99, 691):
        _, fpr = rates(y_val, p_val, thr)
        if fpr <= target_fpr:
            return float(thr)
    return 0.99


def main() -> int:
    X, y, names = load_dataset()
    s = make_splits(X, y, search_subsample=None)
    X_fit, y_fit, X_val, y_val = s["X_fit"], s["y_fit"], s["X_val"], s["y_val"]
    X_te, y_te = s["X_test"], s["y_test"]

    spea = json.loads((TABLES / "spea2_pareto_analysis.json").read_text())
    nsga = json.loads((TABLES / "nsga2_pareto_analysis.json").read_text())
    subsets = {
        "all_features": names,
        "spea2_gateway": spea["deployment_profiles"]["email_gateway"]["selected_features"],
        "nsga2_gateway": nsga["deployment_profiles"]["email_gateway"]["selected_features"],
    }

    out = {}
    for name, cols in subsets.items():
        idx = [names.index(c) for c in cols]
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
        clf.fit(X_fit[:, idx], y_fit)
        p_val = clf.predict_proba(X_val[:, idx])[:, 1]
        p_te = clf.predict_proba(X_te[:, idx])[:, 1]

        rows = []
        r, f = rates(y_te, p_te, 0.5)
        rows.append({"target_fpr": None, "threshold": 0.5, "test_recall": r, "test_fpr": f})
        for target in (0.04, 0.03, 0.02, 0.01):
            thr = threshold_for_fpr(y_val, p_val, target)
            r, f = rates(y_te, p_te, thr)
            rows.append({"target_fpr": target, "threshold": thr, "test_recall": r, "test_fpr": f})
        out[name] = {"n_features": len(idx), "test_auc": float(roc_auc_score(y_te, p_te)),
                     "operating_points": rows}

        print(f"\n{name}  ({len(idx)} features, test ROC-AUC {out[name]['test_auc']:.4f})")
        print(f"  {'FPR target':<12}{'threshold':>10}{'test recall':>13}{'test FPR':>10}")
        for row in rows:
            t = "default" if row["target_fpr"] is None else f"<= {row['target_fpr']:.0%}"
            print(f"  {t:<12}{row['threshold']:>10.3f}{row['test_recall']:>13.2%}{row['test_fpr']:>10.2%}")

    (TABLES / "threshold_tuning.json").write_text(json.dumps(out, indent=2))
    print("\nsaved: results/tables/threshold_tuning.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
