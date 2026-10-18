"""
Run SPEA2 or NSGA-II on one dataset across several seeds, saving every Pareto front.

Identical settings to the phishing project's final runs: population 100, 50 generations,
sparse initial sampling (see problem.SparseBinarySampling), two-point crossover, bit-flip mutation (1/n), duplicate elimination.

Run:  python src/run_multiseed.py --dataset rd2024 --algorithm spea2 --seeds 1 2 3 4 5 6 7 8 9 10
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
from datasets import load  # noqa: E402
from problem import RansomwareFeatureSelection, SparseBinarySampling  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HV_REF = np.array([1.1, 1.1, 1.1])
ALGOS = {"spea2": SPEA2, "nsga2": NSGA2}


def run_one(ds, algo, pop, gen, seed, init="sparse"):
    problem = RansomwareFeatureSelection(ds)
    sampler = SparseBinarySampling() if init == "sparse" else BinaryRandomSampling()
    alg = ALGOS[algo](pop_size=pop, sampling=sampler,
                      crossover=TwoPointCrossover(),
                      mutation=BitflipMutation(prob=1.0 / problem.n_var),
                      eliminate_duplicates=True)
    t0 = time.time()
    res = minimize(problem, alg, ("n_gen", gen), seed=seed, save_history=True, verbose=False)
    elapsed = time.time() - t0
    F = np.atleast_2d(res.F); X = np.atleast_2d(res.X).astype(bool)
    hv = HV(ref_point=HV_REF)
    hypervolume = float(hv(problem.normalise(F)))
    hv_hist = [float(hv(problem.normalise(np.atleast_2d(e.opt.get("F"))))) for e in res.history]
    n_feat = X.sum(axis=1)
    static_only = [i for i in range(len(F)) if problem.cost.is_static_only(X[i])]

    out = ROOT / "results" / ds.name / ("fronts" if init == "sparse" else "ablation_dense_init")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{algo}_pop{pop}_gen{gen}_seed{seed}"
    np.savez_compressed(out / f"{stem}.npz", F=F, X=X, hv_history=np.array(hv_hist),
                        feature_names=np.array(ds.feature_names))
    meta = {"dataset": ds.name, "algorithm": algo, "init": init, "pop_size": pop, "n_gen": gen, "seed": seed,
            "hypervolume": hypervolume, "front_size": len(F), "elapsed_sec": elapsed,
            "n_evaluations": problem.n_evaluations, "cache_hit_rate": problem.cache_hit_rate,
            "n_static_only": len(static_only),
            "features_min": int(n_feat.min()), "features_max": int(n_feat.max()),
            "features_median": float(np.median(n_feat)),
            "fnr_min": float(F[:, 1].min()), "fpr_min": float(F[:, 2].min()),
            "cost_min": float(F[:, 0].min())}
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"  {ds.name} {algo} seed {seed}: {elapsed/60:.1f}min HV={hypervolume:.4f} "
          f"front={len(F)} features={n_feat.min()}-{n_feat.max()} static_only={len(static_only)}",
          flush=True)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--algorithm", choices=sorted(ALGOS), required=True)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--init", choices=["sparse", "dense"], default="sparse")
    a = ap.parse_args()
    ds = load(a.dataset)
    print(f"{ds.title}: {a.algorithm.upper()} pop={a.pop} gen={a.gen} seeds={a.seeds} "
          f"features={ds.n_features} init={a.init}", flush=True)
    metas = [run_one(ds, a.algorithm, a.pop, a.gen, s, a.init) for s in a.seeds]
    hv = np.array([m["hypervolume"] for m in metas])
    print(f"  {a.algorithm.upper()} HV median={np.median(hv):.4f} mean={hv.mean():.4f} "
          f"std={hv.std():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
