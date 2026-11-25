"""
Live terminal demo of the ransomware study (a few minutes, not the full multi-hour run).

  1. load a dataset and show its feature tiers
  2. baseline: Random Forest on all features, scored on the sealed test set
  3. a SHORT live SPEA2 search (population 40, 15 generations) with per-generation progress
  4. pick the best no-sandbox detector from that front and score it on the test set
  5. print the saved results of the full experiment (10 seeds x pop 100 x 50 generations)

Run from the SC Project folder:
    .venv/bin/python ransomware/demo.py                    # RD2024 (default)
    .venv/bin/python ransomware/demo.py --dataset mlran
    .venv/bin/python ransomware/demo.py --dataset ugransome
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from analyze import non_dominated, pick  # noqa: E402
from datasets import load  # noqa: E402
from evaluate import evaluate_subset, unseen_mask  # noqa: E402
from problem import RansomwareFeatureSelection, SparseBinarySampling  # noqa: E402


def title(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}", flush=True)


def show(label, ev, cost):
    t = ev["test_all"]
    print(f"  {label:<34} {ev['n_features']:>4} features  cost {cost:<10}"
          f"recall {t['recall']:.2%}   false alarms {t['fpr']:.2%}", flush=True)


def cost_str(problem, mask):
    if problem.cost.kind == "tier":
        return "50 ms" if problem.cost.is_static_only(mask) else "120 s"
    return f"{int(mask.sum())} feat"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rd2024", choices=["mlran", "rd2024", "ugransome"])
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gen", type=int, default=15)
    a = ap.parse_args()

    title(f"1. DATASET: {a.dataset}")
    ds = load(a.dataset)
    n_tr, n_te = len(ds.y_train), len(ds.y_test)
    print(f"  {ds.title}: {n_tr:,} training rows, {n_te:,} sealed test rows, {ds.n_features} features")
    print(f"  positive class share: {ds.y_train.mean():.1%}")
    tiers = {}
    for t in ds.feature_tier.values():
        tiers[t] = tiers.get(t, 0) + 1
    print(f"  feature tiers: {tiers}")

    problem = RansomwareFeatureSelection(ds)
    unseen = unseen_mask(ds)
    all_mask = np.ones(ds.n_features, bool)

    title("2. BASELINE: Random Forest on ALL features (sealed test set)")
    show("all features", evaluate_subset(ds, all_mask, unseen=unseen), cost_str(problem, all_mask))

    title(f"3. LIVE SPEA2 SEARCH (population {a.pop}, {a.gen} generations)")
    print("  objectives: f1 detection cost, f2 missed ransomware (FNR), f3 false alarms (FPR)")
    print("  each evaluation trains a Random Forest on the candidate's features\n", flush=True)
    algo = SPEA2(pop_size=a.pop, sampling=SparseBinarySampling(), crossover=TwoPointCrossover(),
                 mutation=BitflipMutation(prob=1.0 / problem.n_var), eliminate_duplicates=True)
    t0 = time.time()
    res = minimize(problem, algo, ("n_gen", a.gen), seed=1, verbose=True)
    F, X = non_dominated(np.atleast_2d(res.F), np.atleast_2d(res.X).astype(bool))
    print(f"\n  done in {time.time() - t0:.0f} s, {problem.n_evaluations} classifier trainings, "
          f"{len(F)} Pareto-optimal detectors found")

    print("\n  a sample of the Pareto front (validation data):")
    print(f"  {'features':>8}  {'cost':<9}{'recall':>8}  {'false alarms':>12}")
    for i in np.argsort(F[:, 0] + F[:, 1])[:8]:
        print(f"  {int(X[i].sum()):>8}  {cost_str(problem, X[i]):<9}{1 - F[i, 1]:>8.2%}  {F[i, 2]:>12.2%}")

    title("4. CHOSEN DETECTORS from the live front, scored on the sealed test set")
    if problem.cost.kind == "tier":
        static = [i for i in range(len(F)) if problem.cost.is_static_only(X[i])]
        choices = {"pre-execution (no sandbox)": pick(F, static), "full analysis": pick(F, range(len(F)))}
    else:
        choices = {"best detector": pick(F, range(len(F)))}
    for label, i in choices.items():
        if i is None:
            print(f"  {label}: none found in this short run")
            continue
        show(label, evaluate_subset(ds, X[i], unseen=unseen), cost_str(problem, X[i]))
        names = [ds.feature_names[j] for j in np.flatnonzero(X[i])]
        print(f"    features: {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")

    title("5. SAVED RESULTS OF THE FULL EXPERIMENT (10 seeds, pop 100, 50 generations)")
    an = json.loads((ROOT / "results" / ds.name / "analysis.json").read_text())
    for al, r in an["algorithms"].items():
        print(f"  {al.upper():<8} median hypervolume {r['hv_median']:.4f}  "
              f"(mean {r['hv_mean']:.4f} ± {r['hv_std']:.4f})")
        for p, ev in r["profiles"].items():
            if ev:
                show(f"  {p}", ev, cost_str(problem, np.isin(ds.feature_names, ev["selected"])))
    mw = an.get("mann_whitney")
    if mw:
        print(f"\n  SPEA2 vs NSGA-II (Mann-Whitney on hypervolume): p = {mw['p']:.3f}")
    print("\n  full write-up: ransomware/ransomware-report.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
