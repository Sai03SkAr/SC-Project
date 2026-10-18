"""
Figures for the ransomware study.

Per dataset:  f1 hypervolume convergence (SPEA2 vs NSGA-II, mean ± std over seeds)
              f2 trade-off: recall vs FPR split by tier (or features vs recall when cost = count)
              f3 feature-selection frequency (SPEA2 pooled front), coloured by tier
Combined:     hypervolume per seed, all datasets, both algorithms

Run:  python src/figures.py --datasets mlran rd2024 ugransome
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
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from analyze import load_runs, non_dominated  # noqa: E402
from cost import cost_model_for  # noqa: E402
from datasets import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
COL = {"spea2": "#1F3A93", "nsga2": "#C0392B"}
TIER_COL = {"static": "#2E7D5B", "dynamic": "#B5552B", "network": "#5A6070"}


def per_dataset(name):
    ds = load(name)
    runs = {a: load_runs(name, a) for a in ("spea2", "nsga2")}
    runs = {a: r for a, r in runs.items() if r}
    if not runs:
        print(f"{name}: no runs"); return None

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for a, rs in runs.items():
        h = np.array([r["hv_history"][:min(len(x["hv_history"]) for x in rs)] for r in rs])
        g = np.arange(1, h.shape[1] + 1)
        ax.plot(g, h.mean(0), color=COL[a], lw=2, label=a.upper())
        ax.fill_between(g, h.mean(0) - h.std(0), h.mean(0) + h.std(0), color=COL[a], alpha=0.15)
    ax.set(xlabel="Generation", ylabel="Hypervolume", title=f"{ds.title}: hypervolume convergence")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIG / f"{name}_f1_convergence.png", dpi=150); plt.close(fig)

    cost = cost_model_for(ds)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    if cost.kind == "tier":
        # Only two cost levels exist, so plot the trade-off that remains: recall vs false alarms,
        # split by whether the detector needs the sandbox.
        F, X = non_dominated(np.vstack([r["F"] for r in runs["spea2"]]),
                             np.vstack([r["X"] for r in runs["spea2"]]))
        st = np.array([cost.is_static_only(m) for m in X])
        for m, lab, c, mk in ((st, "static only (no sandbox, 50 ms)", TIER_COL["static"], "o"),
                              (~st, "uses sandbox (~2 min)", TIER_COL["dynamic"], "^")):
            ax.scatter(100 * F[m, 2], 100 * (1 - F[m, 1]), s=18, alpha=0.7, color=c, marker=mk, label=lab)
        an = ROOT / "results" / name / "analysis.json"
        if an.exists():
            prof = json.loads(an.read_text())["algorithms"]["spea2"]["profiles"]
            for pname, ev in prof.items():
                if ev:
                    ax.scatter(100 * ev["val_fpr"], 100 * ev["val_recall"], s=140, facecolors="none",
                               edgecolors="black", lw=1.5)
                    ax.annotate(pname.replace("_", "-"), (100 * ev["val_fpr"], 100 * ev["val_recall"]),
                                textcoords="offset points", xytext=(8, -12), fontsize=8)
        vis = (1 - F[:, 1]) >= 0.85                      # hide the useless low-recall tail
        ax.set(xlim=(0, min(10.0, 100 * F[vis, 2].max() * 1.08)),   # >10% FPR is the flag-everything tail
                ylim=(100 * (1 - F[vis, 1].max()) - 1, 100.3),
               xlabel="False-positive rate % (validation)",
               ylabel="Recall % (validation)",
               title=f"{ds.title}: SPEA2 Pareto front, recall vs false alarms")
    else:
        for a, rs in runs.items():
            F, X = non_dominated(np.vstack([r["F"] for r in rs]), np.vstack([r["X"] for r in rs]))
            ax.scatter(F[:, 0], 1 - F[:, 1], s=16, alpha=0.65, color=COL[a], label=a.upper())
        ax.set(xlabel="Features used", ylabel="Recall (validation)",
               title=f"{ds.title}: features vs recall (Pareto fronts)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(FIG / f"{name}_f2_tradeoff.png", dpi=150); plt.close(fig)

    if "spea2" in runs:
        F, X = non_dominated(np.vstack([r["F"] for r in runs["spea2"]]),
                             np.vstack([r["X"] for r in runs["spea2"]]))
        freq = X.mean(0); top = np.argsort(freq)[::-1][:20]
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        labels = [ds.feature_names[i][:55] for i in top]
        ax.barh(range(len(top)), freq[top],
                color=[TIER_COL.get(ds.feature_tier[ds.feature_names[i]], "#5A6070") for i in top])
        ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=7); ax.invert_yaxis()
        ax.set(xlim=(0, 1), xlabel="Share of SPEA2 Pareto-optimal solutions using the feature",
               title=f"{ds.title}: most-selected features")
        tiers = sorted({ds.feature_tier[ds.feature_names[i]] for i in top})
        ax.legend(handles=[Patch(color=TIER_COL.get(t, "#5A6070"), label=t) for t in tiers],
                  loc="lower right", fontsize=8)
        ax.grid(alpha=0.3, axis="x"); fig.tight_layout()
        fig.savefig(FIG / f"{name}_f3_features.png", dpi=150); plt.close(fig)
    return {a: [r["meta"]["hypervolume"] for r in rs] for a, rs in runs.items()}, ds.title


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mlran", "rd2024", "ugransome"])
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    combined = {}
    for n in a.datasets:
        try:
            r = per_dataset(n)
        except FileNotFoundError:
            print(f"{n}: data not available"); continue
        if r:
            combined[n] = r
            print(f"{n}: figures saved")
    if combined:
        fig, ax = plt.subplots(figsize=(7, 4))
        pos = 0; ticks = []
        for n, (hv, title) in combined.items():
            for k, algo in enumerate(("spea2", "nsga2")):
                if algo in hv:
                    ax.boxplot(hv[algo], positions=[pos + k * 0.8], widths=0.6, patch_artist=True,
                               boxprops=dict(facecolor=COL[algo], alpha=0.35), medianprops=dict(color=COL[algo]))
            ticks.append((pos + 0.4, title)); pos += 2.5
        ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks])
        ax.set_ylabel("Hypervolume (10 seeds)"); ax.set_title("SPEA2 (blue) vs NSGA-II (red) across datasets")
        ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
        fig.savefig(FIG / "combined_hv_boxplot.png", dpi=150); plt.close(fig)
        print("combined figure saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
