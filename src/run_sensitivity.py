"""
Phase 08 - cost sensitivity analysis (E9 in docs/06-evaluation-protocol.md).

Tier latencies in cost_model.py are estimates, not measurements, so the
selected feature subsets must be shown robust to them. Re-runs SPEA2 under
three scenarios and compares which tiers survive selection in each.

Uses a smaller budget (3 seeds, not 5) since this is a robustness check, not
the headline result - see docs/05-implementation-plan.md Phase 08.

Run:  python src/run_sensitivity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

from cost_model import (  # noqa: E402
    SENSITIVITY_SCENARIOS,
    max_detection_cost_ms,
    tier_composition,
)
from problem import PhishingFeatureSelection, load_dataset, make_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTS = ROOT / "results" / "fronts" / "sensitivity"
TABLES = ROOT / "results" / "tables"

HV_REF_POINT = np.array([1.1, 1.1, 1.1])
SEEDS = [1, 2, 3]
POP, GEN = 100, 50


def run_scenario(scenario_name, tier_latency, X, y, names, seed):
    splits = make_splits(X, y, search_subsample=15_000)
    problem = PhishingFeatureSelection(
        splits, names, n_estimators=30, tier_latency=tier_latency
    )
    algorithm = SPEA2(
        pop_size=POP, sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(), mutation=BitflipMutation(prob=1 / problem.n_var),
        eliminate_duplicates=True,
    )
    res = minimize(problem, algorithm, ("n_gen", GEN), seed=seed, verbose=False)
    F = np.atleast_2d(res.F)
    X_front = np.atleast_2d(res.X).astype(bool)

    hv = HV(ref_point=HV_REF_POINT)
    hypervolume = float(hv(problem.normalise(F)))

    # tier composition of the best-recall solution
    best = int(np.argmin(F[:, 1]))
    composition = tier_composition(X_front[best], names, problem.feature_tiers)
    n_features_best = int(X_front[best].sum())

    return {
        "scenario": scenario_name, "seed": seed,
        "hypervolume": hypervolume, "front_size": len(F),
        "max_cost_ms": problem.max_cost,
        "best_recall_features": n_features_best,
        "best_recall_tier_composition": composition,
        "best_recall_recall": float(1 - F[best, 1]),
        "best_recall_fpr": float(F[best, 2]),
        "best_recall_cost_ms": float(F[best, 0]),
        "selected_features": [names[i] for i in range(len(names)) if X_front[best][i]],
    }


def main() -> int:
    FRONTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    X, y, names = load_dataset()

    all_results = []
    for scenario_name, tier_latency in SENSITIVITY_SCENARIOS.items():
        print(f"\n=== {scenario_name.upper()} "
              f"(all-features cost = {max_detection_cost_ms(tier_latency):.1f} ms) ===")
        for seed in SEEDS:
            t0 = time.time()
            r = run_scenario(scenario_name, tier_latency, X, y, names, seed)
            r["elapsed_s"] = time.time() - t0
            all_results.append(r)
            print(f"  seed {seed}: HV={r['hypervolume']:.4f}  "
                  f"best-recall features={r['best_recall_features']}  "
                  f"recall={r['best_recall_recall']:.2%}  "
                  f"({r['elapsed_s']:.0f}s)")

    out = TABLES / "sensitivity_analysis.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nsaved: {out.relative_to(ROOT)}")

    # summary: is the selected feature SET stable across scenarios?
    print(f"\n{'=' * 70}\nSTABILITY CHECK\n{'=' * 70}")
    for scenario_name in SENSITIVITY_SCENARIOS:
        sets = [set(r["selected_features"]) for r in all_results
                if r["scenario"] == scenario_name]
        if len(sets) > 1:
            common = set.intersection(*sets)
            union = set.union(*sets)
            jaccard = len(common) / len(union) if union else 1.0
            print(f"{scenario_name:<14} within-scenario Jaccard similarity: {jaccard:.2f}")

    opt_sets = [set(r["selected_features"]) for r in all_results if r["scenario"] == "optimistic"]
    pess_sets = [set(r["selected_features"]) for r in all_results if r["scenario"] == "pessimistic"]
    if opt_sets and pess_sets:
        opt_union = set.union(*opt_sets)
        pess_union = set.union(*pess_sets)
        cross_jaccard = len(opt_union & pess_union) / len(opt_union | pess_union)
        print(f"\noptimistic vs pessimistic cross-scenario Jaccard: {cross_jaccard:.2f}")
        print("(1.0 = identical feature sets chosen regardless of cost assumptions;")
        print(" low values mean the trade-off genuinely depends on deployment conditions)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
