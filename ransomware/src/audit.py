"""
Phase 01 - data audit for one dataset.

Checks shape and balance, constant features, duplicate rows within and ACROSS the
train/test split (cross-split duplicates silently inflate test scores), the static/dynamic
tier split, and single-feature "shortcuts": any one feature that alone separates the
classes almost perfectly usually signals a collection artifact rather than real behaviour.

Run:  python src/audit.py --dataset rd2024
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from datasets import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    ds = load(a.dataset)
    out_dir = ROOT / "results" / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.vstack([ds.X_train, ds.X_test]); y = np.concatenate([ds.y_train, ds.y_test])
    rep = {"dataset": ds.title, **ds.notes}
    rep["n_train"], rep["n_test"], rep["n_features"] = len(ds.y_train), len(ds.y_test), ds.n_features
    rep["ransomware"], rep["benign"] = int(y.sum()), int((y == 0).sum())
    rep["tiers"] = dict(Counter(ds.feature_tier.values()))

    const = [n for i, n in enumerate(ds.feature_names) if np.unique(X[:, i]).size == 1]
    rep["constant_features"] = len(const)

    def rows(A):
        return {r.tobytes() for r in np.ascontiguousarray(A)}
    tr_rows, te_rows = rows(ds.X_train), rows(ds.X_test)
    rep["duplicate_rows_total"] = int(len(X) - len(rows(X)))
    rep["test_rows_also_in_train"] = int(sum(r.tobytes() in tr_rows for r in np.ascontiguousarray(ds.X_test)))

    # label conflicts: identical feature vectors with different labels
    seen: dict[bytes, set] = {}
    for r, lab in zip(np.ascontiguousarray(X), y):
        seen.setdefault(r.tobytes(), set()).add(int(lab))
    rep["conflicting_duplicate_vectors"] = int(sum(len(s) > 1 for s in seen.values()))

    aucs = []
    for i, n in enumerate(ds.feature_names):
        col = ds.X_train[:, i]
        if np.unique(col).size < 2:
            continue
        auc = roc_auc_score(ds.y_train, col)
        aucs.append((max(auc, 1 - auc), n, ds.feature_tier[n]))
    aucs.sort(reverse=True)
    rep["top_single_feature_auc"] = [{"feature": n, "tier": t, "auc": round(v, 4)} for v, n, t in aucs[:10]]
    rep["shortcut_features_auc_over_0.95"] = [n for v, n, _ in aucs if v > 0.95]

    (out_dir / "audit.json").write_text(json.dumps(rep, indent=2))
    print(f"=== {ds.title} ===")
    for k, v in rep.items():
        if k != "top_single_feature_auc":
            print(f"{k:34} {v}")
    print("top single-feature AUCs (train):")
    for r in rep["top_single_feature_auc"]:
        print(f"   {r['auc']:.4f}  [{r['tier']:7}] {r['feature'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
