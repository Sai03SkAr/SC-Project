"""
E2 (docs/06-evaluation-protocol.md) - the single-objective, weighted-sum PSO baseline.

This is not legacy work: it is the empirical argument for using SPEA2/NSGA-II at all.
A weighted sum collapses (cost, FNR, FPR) into one score using arbitrary weights, and
can only ever return ONE point on the trade-off curve per weight setting - see
docs/03-methodology.md section 2 for why this is a genuine mathematical limitation, not
just a stylistic preference.

Run at several weight settings and compare the resulting points against the SPEA2/
NSGA-II Pareto front: the weighted-sum points should land ON or near the front (they are
valid solutions) but the front covers far more of the trade-off space than any number of
weighted-sum runs practically could.

Run:  python src/run_pso_weighted.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyswarms as ps

sys.path.insert(0, str(Path(__file__).parent))

from cost_model import detection_cost_ms, build_feature_tiers, max_detection_cost_ms  # noqa: E402
from problem import RANDOM_STATE, load_dataset, make_splits  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTS = ROOT / "results" / "fronts"
TABLES = ROOT / "results" / "tables"

# Several weight settings, deliberately including "obviously wrong" ones (e.g. almost
# all weight on one objective) to show the full range of what scalarisation produces.
WEIGHT_SETTINGS = [
    {"name": "balanced",        "w_cost": 1 / 3, "w_fnr": 1 / 3, "w_fpr": 1 / 3},
    {"name": "cost_dominant",   "w_cost": 0.6,   "w_fnr": 0.3,   "w_fpr": 0.1},
    {"name": "recall_dominant", "w_cost": 0.1,   "w_fnr": 0.7,   "w_fpr": 0.2},
    {"name": "fpr_dominant",    "w_cost": 0.1,   "w_fnr": 0.2,   "w_fpr": 0.7},
]

N_PARTICLES = 30
N_ITERS = 50
SEED = 1


def make_fitness_fn(X_fit, y_fit, X_val, y_val, names, tiers, max_cost, weights):
    w_cost, w_fnr, w_fpr = weights["w_cost"], weights["w_fnr"], weights["w_fpr"]

    def fitness(swarm_positions):
        # pyswarms calls this with a (n_particles, n_dims) array of {0,1}-ish values
        costs = np.empty(swarm_positions.shape[0])
        for i, pos in enumerate(swarm_positions):
            mask = pos > 0.5
            if not mask.any():
                costs[i] = 1.0  # worst possible scalar score
                continue
            cost_ms = detection_cost_ms(mask, names, tiers)
            clf = RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=RANDOM_STATE)
            clf.fit(X_fit[:, mask], y_fit)
            pred = clf.predict(X_val[:, mask])
            tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
            fnr = fn / (fn + tp) if (fn + tp) else 1.0
            fpr = fp / (fp + tn) if (fp + tn) else 1.0
            costs[i] = w_cost * (cost_ms / max_cost) + w_fnr * fnr + w_fpr * fpr
        return costs

    return fitness


def main() -> int:
    FRONTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    X, y, names = load_dataset()
    splits = make_splits(X, y, search_subsample=15_000)
    tiers = build_feature_tiers(names)
    max_cost = max_detection_cost_ms()

    results = []
    for weights in WEIGHT_SETTINGS:
        print(f"\n=== weights: {weights['name']} "
              f"(cost={weights['w_cost']}, fnr={weights['w_fnr']}, fpr={weights['w_fpr']}) ===")

        fitness_fn = make_fitness_fn(
            splits["X_fit"], splits["y_fit"], splits["X_val"], splits["y_val"],
            names, tiers, max_cost, weights,
        )

        options = {"c1": 1.5, "c2": 1.5, "w": 0.9, "k": 15, "p": 2}
        optimizer = ps.discrete.BinaryPSO(
            n_particles=N_PARTICLES, dimensions=len(names), options=options,
        )

        t0 = time.time()
        best_cost, best_pos = optimizer.optimize(fitness_fn, iters=N_ITERS, verbose=False)
        elapsed = time.time() - t0

        mask = best_pos > 0.5
        n_features = int(mask.sum())
        f1 = detection_cost_ms(mask, names, tiers)

        clf = RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=RANDOM_STATE)
        clf.fit(splits["X_fit"][:, mask], splits["y_fit"])
        pred = clf.predict(splits["X_val"][:, mask])
        tn, fp, fn, tp = confusion_matrix(splits["y_val"], pred, labels=[0, 1]).ravel()
        fnr = fn / (fn + tp) if (fn + tp) else 1.0
        fpr = fp / (fp + tn) if (fp + tn) else 1.0

        print(f"  {n_features} features, cost={f1:.2f}ms, "
              f"recall={1-fnr:.2%}, FPR={fpr:.2%}  scalar={best_cost:.4f}  ({elapsed:.0f}s)")

        results.append({
            "weights": weights, "n_features": n_features,
            "cost_ms": float(f1), "fnr": float(fnr), "fpr": float(fpr),
            "recall": float(1 - fnr), "scalar_fitness": float(best_cost),
            "elapsed_s": elapsed,
            "selected_features": [names[i] for i in range(len(names)) if mask[i]],
        })

    out = TABLES / "pso_weighted_baseline.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out.relative_to(ROOT)}")

    print(f"\n{'=' * 70}\nTHE LIMITATION, DEMONSTRATED\n{'=' * 70}")
    print(f"{len(WEIGHT_SETTINGS)} weight settings produced {len(results)} isolated points:")
    for r in results:
        print(f"  {r['weights']['name']:<16} -> {r['n_features']:3d} features, "
              f"cost={r['cost_ms']:8.2f}ms, recall={r['recall']:.2%}, FPR={r['fpr']:.2%}")
    print(f"\nCompare: SPEA2 pooled front has 201 non-dominated solutions spanning")
    print(f"the full cost range 0.01-1320ms with recall 85-96% - a continuum this")
    print(f"baseline could only ever sample at {len(WEIGHT_SETTINGS)} arbitrary points,")
    print(f"each requiring a fresh optimisation run to obtain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
