"""
Pool every seed's Pareto front, measure stability, extract deployment points, and
evaluate those points ONCE on the sealed test set.

Deployment points (chosen on validation data, from the pooled front):
    pre_execution  - lowest balanced error among solutions using static features only (no sandbox)
    full_analysis  - lowest balanced error at any cost (sandbox allowed)
For count-cost datasets (no static/dynamic split):
    best           - lowest balanced error at any feature count
    compact        - fewest features within 1 percentage point of the best balanced error

Balanced error = (FNR + FPR) / 2. "Best recall" is NOT used: a detector that flags almost
everything has near-100% recall, is Pareto-optimal (nothing beats its FNR), and is useless.
Balanced error needs no hand-picked target and penalises both kinds of mistake equally.

Run:  python src/analyze.py --dataset rd2024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from cost import cost_model_for  # noqa: E402
from datasets import load  # noqa: E402
from evaluate import evaluate_subset, unseen_mask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_runs(ds_name, algo):
    d = ROOT / "results" / ds_name / "fronts"
    runs = []
    for js in sorted(d.glob(f"{algo}_pop100_gen50_seed*.json"),
                     key=lambda p: int(p.stem.split("seed")[1])):
        npz = np.load(js.with_suffix(".npz"), allow_pickle=True)
        runs.append({"meta": json.loads(js.read_text()), "F": npz["F"],
                     "X": npz["X"].astype(bool), "hv_history": npz["hv_history"]})
    return runs


def non_dominated(F, X):
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            if dom.any():
                keep[i] = False
    # drop exact duplicate masks
    _, uniq = np.unique(X[keep], axis=0, return_index=True)
    idx = np.flatnonzero(keep)[np.sort(uniq)]
    return F[idx], X[idx]


def balanced_error(F):
    return (F[:, 1] + F[:, 2]) / 2


def pick(F, idx):
    """Lowest balanced error, ties broken by lower cost."""
    idx = np.asarray(idx)
    if idx.size == 0:
        return None
    order = np.lexsort((F[idx, 0], balanced_error(F[idx])))
    return int(idx[order[0]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    ds = load(a.dataset)
    cost = cost_model_for(ds)
    unseen = unseen_mask(ds)
    report = {"dataset": ds.name, "algorithms": {}}

    hv_by_algo = {}
    for algo in ("spea2", "nsga2"):
        runs = load_runs(ds.name, algo)
        if not runs:
            continue
        hv = np.array([r["meta"]["hypervolume"] for r in runs])
        hv_by_algo[algo] = hv
        F, X = non_dominated(np.vstack([r["F"] for r in runs]), np.vstack([r["X"] for r in runs]))
        nfeat = X.sum(axis=1)
        static_only = np.array([cost.is_static_only(m) for m in X])

        profiles = {}
        if cost.kind == "tier":
            profiles["pre_execution"] = pick(F, np.flatnonzero(static_only))
            profiles["full_analysis"] = pick(F, np.arange(len(F)))
        else:
            best = pick(F, np.arange(len(F)))
            profiles["best"] = best
            be = balanced_error(F)
            ok = np.flatnonzero(be <= be[best] + 0.01)
            profiles["compact"] = int(ok[np.lexsort((be[ok], nfeat[ok]))[0]])

        prof_out = {}
        for pname, i in profiles.items():
            if i is None:
                prof_out[pname] = None
                continue
            ev = evaluate_subset(ds, X[i], unseen=unseen)
            ev.update({"cost_ms": float(F[i, 0]), "val_recall": float(1 - F[i, 1]),
                       "val_fpr": float(F[i, 2]),
                       "selected": [ds.feature_names[j] for j in np.flatnonzero(X[i])]})
            prof_out[pname] = ev

        freq = X.mean(axis=0)
        top = np.argsort(freq)[::-1][:15]
        tier_share = {}
        if cost.kind == "tier":
            for t, m in cost.tier_masks.items():
                tier_share[t] = {"available": int(m.sum()),
                                 "mean_selected": float(X[:, m].sum(axis=1).mean())}
        report["algorithms"][algo] = {
            "seeds": [r["meta"]["seed"] for r in runs],
            "hv": hv.tolist(), "hv_median": float(np.median(hv)), "hv_mean": float(hv.mean()),
            "hv_std": float(hv.std()), "hv_worst": float(hv.min()),
            "pooled_front": int(len(F)), "static_only_solutions": int(static_only.sum()),
            "features_median": float(np.median(nfeat)), "features_min": int(nfeat.min()),
            "tier_share": tier_share,
            "top_features": [{"feature": ds.feature_names[j], "tier": ds.feature_tier[ds.feature_names[j]],
                              "share": float(freq[j])} for j in top],
            "profiles": prof_out,
        }
        print(f"\n{ds.title} - {algo.upper()}: HV median {np.median(hv):.4f}, mean {hv.mean():.4f} "
              f"± {hv.std():.4f}, worst {hv.min():.4f} | pooled front {len(F)}, "
              f"static-only {static_only.sum()}, features {nfeat.min()}-{nfeat.max()}")
        for pname, ev in prof_out.items():
            if ev is None:
                print(f"   {pname:<14} none on front"); continue
            t, u = ev["test_all"], ev["test_unseen"]
            print(f"   {pname:<14} {ev['n_features']:>4} feat  cost {ev['cost_ms']:>10.2f}  test recall "
                  f"{t['recall']:.2%} FPR {t['fpr']:.2%} | unseen recall {u['recall']:.2%} FPR {u['fpr']:.2%}")

    if len(hv_by_algo) == 2:
        u, p = mannwhitneyu(hv_by_algo["spea2"], hv_by_algo["nsga2"], alternative="two-sided")
        report["mann_whitney"] = {"U": float(u), "p": float(p)}
        print(f"\nMann-Whitney SPEA2 vs NSGA-II hypervolume: U={u:.1f}, p={p:.3f}")

    out = ROOT / "results" / ds.name / "analysis.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
