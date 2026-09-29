"""
Phase 06 - required figures, per docs/06-evaluation-protocol.md section 5.

Produces (as many as have data available):
    F1  hypervolume vs generation, mean +/- std band, per algorithm
    F2  Pareto front 2D projections (cost x FNR, cost x FPR, FNR x FPR)
    F6  feature-selection frequency, coloured by tier
    F7  cost vs recall trade-off with deployment budgets marked

Run:  python src/make_figures.py --algorithms spea2 nsga2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from cost_model import build_feature_tiers  # noqa: E402
from analyze_fronts import load_runs, pooled_pareto_set, PROFILES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTS = ROOT / "results" / "fronts"
FIGURES = ROOT / "results" / "figures"

COLORS = {"spea2": "#1F3A93", "nsga2": "#C0392B", "mopso": "#27824C", "pso_weighted": "#8A5A12"}


def fig1_hypervolume_convergence(all_runs: dict[str, list]):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, runs in all_runs.items():
        if not runs:
            continue
        histories = [r["hv_history"] for r in runs]
        min_len = min(len(h) for h in histories)
        histories = np.array([h[:min_len] for h in histories])
        mean = histories.mean(axis=0)
        std = histories.std(axis=0)
        gens = np.arange(1, min_len + 1)
        c = COLORS.get(algo, "#333333")
        ax.plot(gens, mean, label=algo.upper(), color=c, linewidth=2)
        ax.fill_between(gens, mean - std, mean + std, color=c, alpha=0.15)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Hypervolume convergence (mean ± std across seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "f1_hypervolume_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")


def fig2_pareto_projections(pooled: dict[str, tuple]):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    pairs = [(0, 1, "Detection cost (ms)", "FNR (1 - recall)"),
             (0, 2, "Detection cost (ms)", "FPR"),
             (1, 2, "FNR (1 - recall)", "FPR")]

    for ax, (i, j, xl, yl) in zip(axes, pairs):
        for algo, (F, X) in pooled.items():
            c = COLORS.get(algo, "#333333")
            ax.scatter(F[:, i], F[:, j], s=18, alpha=0.7, label=algo.upper(), color=c)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_xscale("log")

    axes[0].legend()
    fig.suptitle("Pareto front projections (pooled across seeds)")
    fig.tight_layout()
    out = FIGURES / "f2_pareto_projections.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")


def fig6_feature_frequency(pooled: dict[str, tuple], feature_names, feature_tiers, top_n=30):
    tier_color = {
        "T0": "#5A6070",
        **{f"T{i}": "#C0392B" for i in range(1, 10)},
    }

    for algo, (F, X) in pooled.items():
        freq = X.sum(axis=0) / len(X)
        order = np.argsort(freq)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(8, 7))
        names = [feature_names[i] for i in order]
        vals = [freq[i] for i in order]
        colors = [tier_color[feature_tiers[n]] for n in names]

        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Fraction of Pareto-optimal solutions containing feature")
        ax.set_title(f"{algo.upper()} - feature selection frequency (top {top_n})")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3, axis="x")

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#5A6070", label="T0 (URL text, cheap)"),
            Patch(color="#C0392B", label="T1-T9 (network lookup, expensive)"),
        ], loc="lower right", fontsize=8)

        fig.tight_layout()
        out = FIGURES / f"f6_feature_frequency_{algo}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out.relative_to(ROOT)}")


def fig7_cost_recall_tradeoff(pooled: dict[str, tuple]):
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, (F, X) in pooled.items():
        c = COLORS.get(algo, "#333333")
        recall = 1 - F[:, 1]
        ax.scatter(F[:, 0], recall, s=20, alpha=0.7, label=algo.upper(), color=c)

    for name, cfg in PROFILES.items():
        if cfg["max_cost_ms"] < float("inf"):
            ax.axvline(cfg["max_cost_ms"], linestyle="--", color="#888888", linewidth=1)
            ax.text(cfg["max_cost_ms"], ax.get_ylim()[0], f" {name}", rotation=90,
                    fontsize=7, va="bottom", color="#666666")

    ax.set_xscale("log")
    ax.set_xlabel("Detection cost (ms, log scale)")
    ax.set_ylabel("Recall")
    ax.set_title("Cost vs recall trade-off, with deployment budgets marked")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "f7_cost_recall_tradeoff.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithms", nargs="+", default=["spea2", "nsga2"])
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)

    all_runs, pooled = {}, {}
    feature_names, feature_tiers = None, None

    for algo in args.algorithms:
        runs = load_runs(algo, args.pop, args.gen, args.seeds)
        all_runs[algo] = runs
        if runs:
            F, X = pooled_pareto_set(runs)
            pooled[algo] = (F, X)
            if feature_names is None:
                feature_names = runs[0]["feature_names"]
                feature_tiers = build_feature_tiers(feature_names)
            print(f"{algo}: {len(runs)} seeds loaded, pooled front = {len(F)}")
        else:
            print(f"{algo}: no runs found, skipping")

    if not pooled:
        print("Nothing to plot - run src/run_multiseed.py first.")
        return 1

    fig1_hypervolume_convergence(all_runs)
    fig2_pareto_projections(pooled)
    fig6_feature_frequency(pooled, feature_names, feature_tiers)
    fig7_cost_recall_tradeoff(pooled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
