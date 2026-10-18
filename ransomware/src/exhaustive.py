"""
Exhaustive search - only feasible for small feature spaces (UGRansome).

Evaluates every non-empty feature subset with the same fitness function the algorithms use,
giving the TRUE Pareto front. Then measures how close SPEA2 and NSGA-II came to it:
    recovered  - share of true Pareto-optimal subsets each algorithm found (pooled over seeds)
    IGD        - inverted generational distance from the true front (normalised objectives;
                 0 = the algorithm's front covers the true front exactly)

Run:  python src/exhaustive.py --dataset ugransome
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.indicators.igd import IGD

sys.path.insert(0, str(Path(__file__).parent))
from analyze import load_runs, non_dominated  # noqa: E402
from datasets import load  # noqa: E402
from problem import RansomwareFeatureSelection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ugransome")
    a = ap.parse_args()
    ds = load(a.dataset)
    if ds.n_features > 16:
        print(f"{ds.n_features} features - 2^{ds.n_features} subsets is too many to enumerate.")
        return 1
    prob = RansomwareFeatureSelection(ds)
    n = ds.n_features
    t0 = time.time()
    masks, F = [], []
    for bits in itertools.product([0, 1], repeat=n):
        m = np.array(bits, bool)
        if not m.any():
            continue
        masks.append(m); F.append(prob.objectives(m))
    F = np.array(F); masks = np.array(masks)
    TF, TX = non_dominated(F, masks)
    print(f"evaluated {len(F)} subsets in {(time.time()-t0)/60:.1f} min; true front = {len(TF)}")

    out = {"n_subsets": int(len(F)), "true_front_size": int(len(TF)),
           "true_front": [{"features": [ds.feature_names[j] for j in np.flatnonzero(x)],
                           "n": int(x.sum()), "fnr": float(f[1]), "fpr": float(f[2])}
                          for f, x in sorted(zip(TF, TX), key=lambda t: t[0][0])]}
    igd = IGD(prob.normalise(TF))
    true_keys = {x.tobytes() for x in TX}
    for algo in ("spea2", "nsga2"):
        runs = load_runs(ds.name, algo)
        if not runs:
            continue
        per_seed = [float(igd(prob.normalise(r["F"]))) for r in runs]
        PF, PX = non_dominated(np.vstack([r["F"] for r in runs]), np.vstack([r["X"] for r in runs]))
        rec = sum(x.tobytes() in true_keys for x in PX) / len(true_keys)
        out[algo] = {"igd_per_seed": per_seed, "igd_median": float(np.median(per_seed)),
                     "pooled_recovered_share": float(rec)}
        print(f"{algo.upper()}: median IGD {np.median(per_seed):.4f}, pooled front recovers "
              f"{rec:.0%} of the true Pareto set")
    (ROOT / "results" / ds.name / "exhaustive.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
