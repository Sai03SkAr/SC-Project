"""
Phase 06/07 - Pareto front analysis.

Pulls together every saved front for one algorithm across seeds and produces
the tables specified in docs/06-evaluation-protocol.md:

    Table 2  algorithm stability across seeds (HV, features, runtime)
    Table 3  surviving features by group / tier
    Table 4  feature selection frequency across all runs
    Deployment profiles: browser extension / email gateway / offline audit

Run:  python src/analyze_fronts.py --algorithm spea2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from cost_model import build_feature_tiers, TIER_DESCRIPTION  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTS = ROOT / "results" / "fronts"
TABLES = ROOT / "results" / "tables"

# Deployment budgets from docs/08-objective-benchmarks.md
PROFILES = {
    "browser_extension": {"max_cost_ms": 1.0, "recall_target": 0.92, "fpr_target": 0.05},
    "email_gateway":      {"max_cost_ms": 500.0, "recall_target": 0.95, "fpr_target": 0.02},
    "offline_audit":      {"max_cost_ms": float("inf"), "recall_target": 0.96, "fpr_target": 0.01},
}


def load_runs(algorithm: str, pop: int, gen: int, seeds: list[int]):
    runs = []
    for seed in seeds:
        stem = f"{algorithm}_pop{pop}_gen{gen}_seed{seed}"
        npz_path = FRONTS / f"{stem}.npz"
        json_path = FRONTS / f"{stem}.json"
        if not npz_path.exists():
            print(f"  (missing: {stem})")
            continue
        data = np.load(npz_path, allow_pickle=True)
        meta = json.loads(json_path.read_text())
        runs.append({
            "seed": seed, "F": data["F"], "X": data["X"].astype(bool),
            "hv_history": data["hv_history"],
            "feature_names": list(data["feature_names"]),
            "meta": meta,
        })
    return runs


def table2_stability(runs, algorithm):
    hvs = [r["meta"]["hypervolume"] for r in runs]
    feats_median = [r["meta"]["features_median"] for r in runs]
    times = [r["meta"]["elapsed_sec"] for r in runs]
    cache_rates = [r["meta"]["cache_hit_rate"] for r in runs]

    print(f"\n{'=' * 70}\nTABLE 2 - {algorithm.upper()} stability across {len(runs)} seeds\n{'=' * 70}")
    print(f"hypervolume     : {np.mean(hvs):.4f} +/- {np.std(hvs):.4f}  "
          f"[{min(hvs):.4f}, {max(hvs):.4f}]")
    print(f"median features : {np.mean(feats_median):.1f} +/- {np.std(feats_median):.1f}")
    print(f"runtime (s)     : {np.mean(times):.1f} +/- {np.std(times):.1f}")
    print(f"cache hit rate  : {np.mean(cache_rates):.1%} +/- {np.std(cache_rates):.1%}")

    return {
        "algorithm": algorithm, "n_seeds": len(runs),
        "hypervolume_mean": float(np.mean(hvs)), "hypervolume_std": float(np.std(hvs)),
        "hypervolume_min": float(min(hvs)), "hypervolume_max": float(max(hvs)),
        "features_median_mean": float(np.mean(feats_median)),
        "runtime_mean_s": float(np.mean(times)),
        "cache_hit_rate_mean": float(np.mean(cache_rates)),
    }


def pooled_pareto_set(runs):
    """Combine every seed's front and re-filter to non-dominated only.

    This is the front used for the feature-frequency and tier-composition
    analyses - it should be at least as good as any single seed's front.
    """
    all_F, all_X = [], []
    for r in runs:
        all_F.append(r["F"])
        all_X.append(r["X"])
    F = np.vstack(all_F)
    X = np.vstack(all_X)

    # Non-dominated filter (simple O(n^2), fine at this scale).
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            # j dominates i?
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False
                break
    return F[keep], X[keep]


def table3_tier_composition(F, X, feature_names, feature_tiers):
    counts_by_tier_group = Counter()
    group_totals = Counter()
    for name in feature_names:
        tier = feature_tiers[name]
        group_totals[tier] += 1

    for row in X:
        for bit, name in zip(row, feature_names):
            if bit:
                counts_by_tier_group[feature_tiers[name]] += 1

    n_solutions = len(X)
    print(f"\n{'=' * 70}\nTABLE 3 - Tier composition across {n_solutions} Pareto-optimal solutions\n{'=' * 70}")
    print(f"{'tier':<6}{'description':<26}{'available':>10}{'avg selected':>14}{'% of tier kept':>16}")
    rows = []
    for tier in sorted(group_totals):
        avail = group_totals[tier]
        avg_selected = counts_by_tier_group[tier] / n_solutions if n_solutions else 0
        pct = avg_selected / avail if avail else 0
        print(f"{tier:<6}{TIER_DESCRIPTION[tier]:<26}{avail:>10}{avg_selected:>14.2f}{pct:>15.1%}")
        rows.append({"tier": tier, "description": TIER_DESCRIPTION[tier],
                      "available": avail, "avg_selected": avg_selected, "pct_kept": pct})

    n_network_free = sum(
        1 for row in X
        if not any(feature_tiers[name] != "T0" for bit, name in zip(row, feature_names) if bit)
    )
    print(f"\nnetwork-free solutions: {n_network_free} of {n_solutions} "
          f"({n_network_free / n_solutions:.1%})" if n_solutions else "no solutions")
    return rows, n_network_free


def table4_feature_frequency(X, feature_names, feature_tiers, top_n=25):
    n_solutions = len(X)
    freq = X.sum(axis=0)
    order = np.argsort(freq)[::-1]

    print(f"\n{'=' * 70}\nTABLE 4 - Feature selection frequency (top {top_n} of {len(feature_names)})\n{'=' * 70}")
    print(f"{'feature':<28}{'tier':<6}{'in # solutions':>15}{'% of front':>12}")
    rows = []
    for idx in order[:top_n]:
        name = feature_names[idx]
        pct = freq[idx] / n_solutions if n_solutions else 0
        print(f"{name:<28}{feature_tiers[name]:<6}{int(freq[idx]):>15}{pct:>11.1%}")
        rows.append({"feature": name, "tier": feature_tiers[name],
                      "count": int(freq[idx]), "pct": pct})
    return rows


def deployment_profiles(F, X, feature_names, feature_tiers):
    print(f"\n{'=' * 70}\nDEPLOYMENT PROFILES\n{'=' * 70}")
    results = {}
    for profile, cfg in PROFILES.items():
        eligible = np.where(F[:, 0] <= cfg["max_cost_ms"])[0]
        if len(eligible) == 0:
            print(f"\n{profile}: NO SOLUTION within {cfg['max_cost_ms']} ms budget")
            results[profile] = None
            continue
        # best recall (lowest FNR) within budget
        best = eligible[np.argmin(F[eligible, 1])]
        n_feat = int(X[best].sum())
        recall = 1 - F[best, 1]
        fpr = F[best, 2]
        cost = F[best, 0]
        meets_target = recall >= cfg["recall_target"] and fpr <= cfg["fpr_target"]

        tiers_used = sorted({feature_tiers[name] for bit, name in
                              zip(X[best], feature_names) if bit})

        print(f"\n{profile}  (budget <= {cfg['max_cost_ms']} ms)")
        print(f"  features: {n_feat}   cost: {cost:.2f} ms")
        print(f"  recall: {recall:.2%}  (target >= {cfg['recall_target']:.0%})")
        print(f"  FPR:    {fpr:.2%}  (target <= {cfg['fpr_target']:.0%})")
        print(f"  meets both targets: {'YES' if meets_target else 'no'}")
        print(f"  tiers used: {tiers_used}")

        results[profile] = {
            "n_features": n_feat, "cost_ms": float(cost),
            "recall": float(recall), "fpr": float(fpr),
            "meets_targets": bool(meets_target), "tiers_used": tiers_used,
            "selected_features": [feature_names[i] for i in range(len(feature_names)) if X[best][i]],
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", required=True)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.algorithm, args.pop, args.gen, args.seeds)
    if not runs:
        print("No runs found - run src/run_multiseed.py first.")
        return 1

    feature_names = runs[0]["feature_names"]
    feature_tiers = build_feature_tiers(feature_names)

    stability = table2_stability(runs, args.algorithm)
    F, X = pooled_pareto_set(runs)
    print(f"\npooled non-dominated set across all seeds: {len(F)} solutions "
          f"(from {sum(len(r['F']) for r in runs)} total)")

    tier_rows, n_network_free = table3_tier_composition(F, X, feature_names, feature_tiers)
    freq_rows = table4_feature_frequency(X, feature_names, feature_tiers)
    profiles = deployment_profiles(F, X, feature_names, feature_tiers)

    out = {
        "algorithm": args.algorithm,
        "stability": stability,
        "pooled_front_size": len(F),
        "tier_composition": tier_rows,
        "n_network_free_solutions": n_network_free,
        "feature_frequency_top25": freq_rows,
        "deployment_profiles": profiles,
    }
    out_path = TABLES / f"{args.algorithm}_pareto_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
