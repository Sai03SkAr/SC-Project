"""
Weighted-sum binary PSO baseline (single-objective). Each weight setting yields one point;
this demonstrates the scalarisation limitation, as in the phishing project.

The swarm starts from the same sparse initial sampling as SPEA2 and NSGA-II (pyswarms'
default is ~50% density, which on MLRan never reaches a sandbox-free subset), so the
comparison differs only in the search method.

Run:  python src/run_pso.py --dataset rd2024
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyswarms as ps

sys.path.insert(0, str(Path(__file__).parent))
from datasets import load  # noqa: E402
from evaluate import evaluate_subset, unseen_mask  # noqa: E402
from problem import RansomwareFeatureSelection, SparseBinarySampling  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = [
    {"name": "balanced", "w_cost": 1 / 3, "w_fnr": 1 / 3, "w_fpr": 1 / 3},
    {"name": "cost_dominant", "w_cost": 0.6, "w_fnr": 0.3, "w_fpr": 0.1},
    {"name": "recall_dominant", "w_cost": 0.1, "w_fnr": 0.7, "w_fpr": 0.2},
    {"name": "fpr_dominant", "w_cost": 0.1, "w_fnr": 0.2, "w_fpr": 0.7},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--particles", type=int, default=30)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    ds = load(a.dataset)
    prob = RansomwareFeatureSelection(ds)
    unseen = unseen_mask(ds)
    cache = {}

    def objectives(mask):                     # same fitness, memoised across weightings
        k = mask.tobytes()
        if k not in cache:
            cache[k] = prob.objectives(mask)
        return cache[k]

    out = []
    for w in WEIGHTS:
        def fitness(swarm):
            vals = []
            for pos in swarm:
                c, fnr, fpr = objectives(pos > 0.5)
                vals.append(w["w_cost"] * c / prob.max_cost + w["w_fnr"] * fnr + w["w_fpr"] * fpr)
            return np.array(vals)
        np.random.seed(1)
        init = SparseBinarySampling()._do(prob, a.particles, random_state=np.random.default_rng(1))
        opt = ps.discrete.BinaryPSO(n_particles=a.particles, dimensions=ds.n_features,
                                    options={"c1": 1.5, "c2": 1.5, "w": 0.9, "k": 15, "p": 2},
                                    init_pos=init.astype(int))
        t0 = time.time()
        best, pos = opt.optimize(fitness, iters=a.iters, verbose=False)
        mask = pos > 0.5
        c, fnr, fpr = objectives(mask)
        test = evaluate_subset(ds, mask, unseen=unseen)
        rec = {"weights": w, "n_features": int(mask.sum()), "cost_ms": c,
               "val_recall": 1 - fnr, "val_fpr": fpr, "static_only": prob.cost.is_static_only(mask),
               "elapsed_s": time.time() - t0, **test,
               "selected": [ds.feature_names[i] for i in np.flatnonzero(mask)]}
        out.append(rec)
        print(f"{w['name']:<16} {rec['n_features']:>4} feat cost {c:>10.2f}  test recall "
              f"{test['test_all']['recall']:.2%} FPR {test['test_all']['fpr']:.2%}  "
              f"({rec['elapsed_s']/60:.1f} min)", flush=True)
    d = ROOT / "results" / ds.name
    (d / "pso_weighted.json").write_text(json.dumps(out, indent=2))
    print(f"saved results/{ds.name}/pso_weighted.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
