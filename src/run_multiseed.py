"""
Orchestrator: run an algorithm across multiple seeds sequentially, in-process.

Runs sequentially rather than in parallel processes deliberately - each fit
inside the search already uses n_jobs=-1 (all cores), so parallel OS
processes would fight over the same cores rather than adding throughput.

Usage:
    python src/run_multiseed.py --algorithm spea2 --seeds 1 2 3 4 5
    python src/run_multiseed.py --algorithm nsga2 --seeds 1 2 3 4 5
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
from problem import PhishingFeatureSelection, load_dataset, make_splits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTS = ROOT / "results" / "fronts"
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


def run_one(algorithm_name, pop, gen, seed, X, y, names, subsample, n_estimators):
    splits = make_splits(X, y, search_subsample=subsample)
    problem = PhishingFeatureSelection(splits, names, n_estimators=n_estimators)
    algorithm = build_algorithm(algorithm_name, pop, problem.n_var)

    t0 = time.time()
    res = minimize(problem, algorithm, ("n_gen", gen), seed=seed,
                    save_history=True, verbose=False)
    elapsed = time.time() - t0

    F = np.atleast_2d(res.F)
    X_front = np.atleast_2d(res.X).astype(bool)

    hv_indicator = HV(ref_point=HV_REF_POINT)
    hypervolume = float(hv_indicator(problem.normalise(F)))

    hv_history = [
        float(hv_indicator(problem.normalise(np.atleast_2d(e.opt.get("F")))))
        for e in res.history
    ]

    n_features = X_front.sum(axis=1)
    network_free = [
        i for i in range(len(F))
        if set(tier_composition(X_front[i], names, problem.feature_tiers)) <= {"T0"}
    ]

    stem = f"{algorithm_name}_pop{pop}_gen{gen}_seed{seed}"
    np.savez_compressed(
        FRONTS / f"{stem}.npz",
        F=F, X=X_front, hv_history=np.array(hv_history),
        feature_names=np.array(names),
    )
    meta = {
        "algorithm": algorithm_name, "pop_size": pop, "n_gen": gen, "seed": seed,
        "subsample": subsample, "n_estimators": n_estimators,
        "hypervolume": hypervolume, "hv_ref_point": HV_REF_POINT.tolist(),
        "front_size": len(F), "elapsed_sec": elapsed,
        "n_evaluations": problem.n_evaluations,
        "n_cache_hits": problem.n_cache_hits,
        "cache_hit_rate": problem.cache_hit_rate,
        "n_network_free": len(network_free),
        "features_min": int(n_features.min()), "features_max": int(n_features.max()),
        "features_median": float(np.median(n_features)),
        "fnr_min": float(F[:, 1].min()), "fnr_max": float(F[:, 1].max()),
        "fpr_min": float(F[:, 2].min()), "fpr_max": float(F[:, 2].max()),
        "cost_min": float(F[:, 0].min()), "cost_max": float(F[:, 0].max()),
    }
    (FRONTS / f"{stem}.json").write_text(json.dumps(meta, indent=2))

    print(f"  seed {seed}: {elapsed/60:.1f}min  HV={hypervolume:.4f}  "
          f"front={len(F)}  features={n_features.min()}-{n_features.max()}  "
          f"cache_hit={problem.cache_hit_rate:.1%}  network_free={len(network_free)}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=sorted(ALGORITHMS), required=True)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--subsample", type=int, default=15_000)
    ap.add_argument("--n-estimators", type=int, default=30)
    args = ap.parse_args()

    FRONTS.mkdir(parents=True, exist_ok=True)

    X, y, names = load_dataset()
    print(f"{args.algorithm.upper()}  pop={args.pop} gen={args.gen}  "
          f"seeds={args.seeds}  budget/run={args.pop*args.gen:,} evals")
    print(f"data: {X.shape}\n")

    t_all = time.time()
    all_meta = []
    for seed in args.seeds:
        meta = run_one(args.algorithm, args.pop, args.gen, seed,
                        X, y, names, args.subsample, args.n_estimators)
        all_meta.append(meta)

    total = time.time() - t_all
    hvs = [m["hypervolume"] for m in all_meta]
    print(f"\n{args.algorithm.upper()} complete: {total/60:.1f} min total")
    print(f"hypervolume: mean={np.mean(hvs):.4f}  std={np.std(hvs):.4f}  "
          f"min={min(hvs):.4f}  max={max(hvs):.4f}")

    summary_path = FRONTS / f"{args.algorithm}_summary.json"
    summary_path.write_text(json.dumps({
        "algorithm": args.algorithm, "seeds": args.seeds,
        "pop": args.pop, "gen": args.gen,
        "hypervolume_mean": float(np.mean(hvs)),
        "hypervolume_std": float(np.std(hvs)),
        "total_elapsed_sec": total,
        "runs": all_meta,
    }, indent=2))
    print(f"summary saved: {summary_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
