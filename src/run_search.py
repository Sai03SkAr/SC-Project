"""
Run a multi-objective feature-selection search and save the Pareto front.

Usage:
    python src/run_search.py --algorithm spea2 --pop 100 --gen 50 --seed 1
    python src/run_search.py --algorithm nsga2 --pop 100 --gen 50 --seed 1

Every algorithm gets the SAME problem object, operators, population size,
generation count and data splits, so the comparison isolates the algorithm
itself rather than its configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

from cost_model import tier_composition  # noqa: E402
from problem import (  # noqa: E402
    RANDOM_STATE,
    PhishingFeatureSelection,
    load_dataset,
    make_splits,
)

ROOT = Path(__file__).resolve().parent.parent
FRONTS = ROOT / "results" / "fronts"

# Fixed across every algorithm and seed so hypervolumes stay comparable.
HV_REF_POINT = np.array([1.1, 1.1, 1.1])

ALGORITHMS = {"spea2": SPEA2, "nsga2": NSGA2}


def build_algorithm(name: str, pop_size: int, n_var: int):
    cls = ALGORITHMS[name]
    return cls(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(prob=1.0 / n_var),
        eliminate_duplicates=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="spea2")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--subsample", type=int, default=15_000)
    ap.add_argument("--n-estimators", type=int, default=30)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    FRONTS.mkdir(parents=True, exist_ok=True)

    X, y, names = load_dataset()
    splits = make_splits(X, y, search_subsample=args.subsample)
    problem = PhishingFeatureSelection(
        splits, names, n_estimators=args.n_estimators
    )

    algorithm = build_algorithm(args.algorithm, args.pop, problem.n_var)

    print(f"algorithm   : {args.algorithm.upper()}")
    print(f"population  : {args.pop}   generations: {args.gen}   seed: {args.seed}")
    print(f"budget      : {args.pop * args.gen:,} evaluations")
    print(f"search data : fit={splits['X_fit'].shape}  val={splits['X_val'].shape}")
    print("running...", flush=True)

    t0 = time.time()
    res = minimize(
        problem, algorithm, ("n_gen", args.gen),
        seed=args.seed, save_history=True, verbose=False,
    )
    elapsed = time.time() - t0

    F = np.atleast_2d(res.F)
    X_front = np.atleast_2d(res.X).astype(bool)

    hv_indicator = HV(ref_point=HV_REF_POINT)
    hypervolume = float(hv_indicator(problem.normalise(F)))

    # Hypervolume per generation - the convergence figure.
    hv_history = []
    for entry in res.history:
        gen_F = np.atleast_2d(entry.opt.get("F"))
        hv_history.append(float(hv_indicator(problem.normalise(gen_F))))

    # --------------------------------------------------------------- report
    n_features = X_front.sum(axis=1)
    best_fnr = int(np.argmin(F[:, 1]))
    cheapest = int(np.argmin(F[:, 0]))

    print(f"\ndone in {elapsed / 60:.1f} min")
    print(f"front size        : {len(F)}")
    print(f"hypervolume       : {hypervolume:.4f}")
    print(f"evaluations       : {problem.n_evaluations:,} "
          f"(cache hits {problem.n_cache_hits:,}, "
          f"hit rate {problem.cache_hit_rate:.1%})")
    print(f"features selected : {n_features.min()}-{n_features.max()} "
          f"(median {int(np.median(n_features))})")
    print(f"cost range        : {F[:, 0].min():.2f} - {F[:, 0].max():.2f} ms")
    print(f"FNR range         : {F[:, 1].min():.4f} - {F[:, 1].max():.4f}")
    print(f"FPR range         : {F[:, 2].min():.4f} - {F[:, 2].max():.4f}")

    print(f"\nbest recall solution : {n_features[best_fnr]} features, "
          f"cost {F[best_fnr, 0]:.2f} ms, "
          f"recall {1 - F[best_fnr, 1]:.2%}, FPR {F[best_fnr, 2]:.2%}")
    print(f"cheapest solution    : {n_features[cheapest]} features, "
          f"cost {F[cheapest, 0]:.2f} ms, "
          f"recall {1 - F[cheapest, 1]:.2%}, FPR {F[cheapest, 2]:.2%}")

    # Network-free solutions are the headline result: no lookups at all.
    network_free = [
        i for i in range(len(F))
        if set(tier_composition(X_front[i], names, problem.feature_tiers)) <= {"T0"}
    ]
    if network_free:
        best = min(network_free, key=lambda i: F[i, 1])
        print(f"\nNETWORK-FREE solutions: {len(network_free)} of {len(F)}")
        print(f"  best: {n_features[best]} features, cost {F[best, 0]:.2f} ms, "
              f"recall {1 - F[best, 1]:.2%}, FPR {F[best, 2]:.2%}")
    else:
        print("\nNo network-free solution on this front.")

    # ----------------------------------------------------------------- save
    tag = f"_{args.tag}" if args.tag else ""
    stem = f"{args.algorithm}_pop{args.pop}_gen{args.gen}_seed{args.seed}{tag}"

    np.savez_compressed(
        FRONTS / f"{stem}.npz",
        F=F, X=X_front, hv_history=np.array(hv_history),
        feature_names=np.array(names),
    )
    meta = {
        "algorithm": args.algorithm, "pop_size": args.pop, "n_gen": args.gen,
        "seed": args.seed, "subsample": args.subsample,
        "n_estimators": args.n_estimators, "random_state": RANDOM_STATE,
        "hypervolume": hypervolume, "hv_ref_point": HV_REF_POINT.tolist(),
        "front_size": len(F), "elapsed_sec": elapsed,
        "n_evaluations": problem.n_evaluations,
        "n_cache_hits": problem.n_cache_hits,
        "cache_hit_rate": problem.cache_hit_rate,
        "n_network_free": len(network_free),
    }
    (FRONTS / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved: results/fronts/{stem}.npz + .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
