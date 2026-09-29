# 06 — Evaluation Protocol

How multi-objective results are measured, compared and presented.

---

## 1. The fundamental shift

**Multi-objective algorithms return sets, not points.** You cannot rank them by accuracy.

| | Single-objective | Multi-objective |
|---|---|---|
| Output | One solution | A Pareto front |
| Comparison | Higher accuracy wins | **Hypervolume**, spread, convergence |
| Reporting | One row in a table | A front, plus selected representative solutions |

Applying single-objective thinking to multi-objective results is the most common way to
misreport this kind of work.

---

## 2. Classification metrics

With **positive class = phishing (`1`)**:

| Metric | Formula | Role |
|--------|---------|------|
| **FNR** | `FN / (FN + TP)` | **Objective f₂** — phishing let through |
| **FPR** | `FP / (FP + TN)` | **Objective f₃** — legitimate sites blocked |
| Recall | `1 − FNR` | Reported form of f₂ |
| Precision | `TP / (TP + FP)` | Derived; reported for interpretability |
| F1 | harmonic mean | Summary only — never an objective here |
| ROC-AUC | area under TPR/FPR | Threshold-independent quality |
| MCC | correlation coefficient | Balanced single number |

**Why FNR and FPR are the objectives rather than recall and precision:** both are minimised, both
are bounded [0,1], and both derive from the same confusion matrix. Uniform minimisation keeps the
Pareto mathematics clean. Convert to recall and precision when *reporting* — they read better.

### The asymmetry, stated explicitly

| Error | Consequence |
|-------|-------------|
| False positive | Legitimate site blocked — annoying |
| **False negative** | **Phishing site passed as safe — user may surrender credentials** |

The project's stated priority is minimising false negatives. The Pareto front makes this
tractable: rather than baking the preference into a weight, **generate the whole trade-off and
apply the preference at selection time.** That is a strictly better methodology, and worth
saying so in the report.

---

## 3. Multi-objective quality indicators

### 3.1 Hypervolume — the primary metric

The volume of objective space dominated by the front, bounded by a reference point.
**Higher is better.** It captures convergence *and* spread in one number.

**Normalise before computing**, since the objectives have wildly different scales
(cost 0.01–1320 ms; FNR and FPR both 0–1):

```
f1_norm = detection_cost / 1320.0        # → [0, 1]
f2_norm = FNR                            # already [0, 1]
f3_norm = FPR                            # already [0, 1]

reference point = (1.1, 1.1, 1.1)
```

The reference point must be **fixed across all algorithms and all seeds** — otherwise
hypervolumes are not comparable. Document it in the experimental setup. Placing it slightly
beyond the nadir (1.1 rather than 1.0) ensures boundary solutions contribute volume.

```python
from pymoo.indicators.hv import HV
hv = HV(ref_point=np.array([1.1, 1.1, 1.1]))
score = hv(normalised_front)
```

### 3.2 Supporting indicators

| Indicator | Measures | Note |
|-----------|----------|------|
| **IGD** | Distance from a reference front | No true front exists — build a reference by merging all algorithms' fronts across all seeds |
| **Spacing** | Evenness of distribution | Lower is better |
| **Front size** | Count of non-dominated solutions | More options is generally better, but not at the cost of hypervolume |
| **Cache hit rate** | Proportion of repeated masks | Evidence for the duplicate-heavy structure motivating SPEA2 |

---

## 4. Experimental protocol

### Splits — fixed for the entire project

```
80 / 20 train-test, stratified, random_state = 42
```

Within the training set, a further internal holdout supplies the search-time validation used by
the fitness function. **The test set is touched once**, at the very end, for the final selected
solutions only.

### Stochasticity

```
5 independent seeds per algorithm
Report mean ± std for: hypervolume, front size, runtime
Merge fronts across seeds for the combined analysis
Report feature selection frequency across all runs
```

### Fair comparison

SPEA2, NSGA-II and MOPSO must share: the same problem object, operators, population size,
generation count, seeds, and data splits. Compare on **equal fitness-evaluation budget**;
report wall-clock separately as a result rather than as the basis of comparison.

---

## 5. Experiment matrix

| # | Experiment | Purpose | Priority |
|---|-----------|---------|----------|
| E1 | Baselines, all 111 features, 3 classifiers | Control point | **Required** |
| E2 | Weighted-sum PSO, ≥2 weight settings | Shows the single-point limitation | **Required** |
| E3 | **SPEA2 × 5 seeds** | **Core result** | **Required** |
| E4 | NSGA-II × 5 seeds, matched budget | Mechanism comparison | **Required** |
| E5 | MOPSO × 5 seeds | Swarm vs evolutionary | **Required** |
| E6 | Hypervolume across all algorithms | Formal comparison | **Required** |
| E7 | Deployment-profile extraction | Headline presentation | **Required** |
| E8 | Feature frequency across the front | Robustness of selection | **Required** |
| E9 | Cost sensitivity (÷3, ×3) | Validates the cost model | **Required** |
| E10 | Knee point + TOPSIS | Formal selection methods | Recommended |
| E11 | `dataset_full.csv` imbalance study | Robustness | Optional |
| E12 | UCI cross-dataset validation | Generalisation | Optional |

E1–E9 constitute a complete project.

---

## 6. Result table templates

### Table 1 — Algorithm comparison (the central table)

| Algorithm | Hypervolume (mean ± std) | Front size | Runtime (s) | Cache hits | Best FNR | Min cost (ms) |
|-----------|--------------------------|------------|-------------|-----------|----------|---------------|
| **SPEA2** | | | | | | |
| NSGA-II | | | | | | |
| MOPSO | | | | | | |
| PSO (weighted) | n/a — single point | 1 | | | | |
| All features | n/a | 1 | | | | 1320 |

### Table 2 — Deployment profiles (the headline table)

| Profile | Budget | Features | Tiers used | Cost (ms) | Recall | Precision | FNR | FPR |
|---------|--------|----------|-----------|-----------|--------|-----------|-----|-----|
| Browser extension | ≤ 1 ms | | T0 only | | | | | |
| Email gateway | ≤ 500 ms | | | | | | | |
| Offline audit | unlimited | | | | | | | |
| *All features* | *—* | *111* | *all* | *1320* | | | | |

### Table 3 — Feature frequency across the Pareto front

| Feature | % of front solutions | Tier | Group |
|---------|---------------------|------|-------|

Sort descending. Features near 100% are structurally essential.

### Table 4 — Tier composition of the front

| Tier | Solutions using it | % of front |
|------|-------------------|-----------|
| T0 (URL parsing) | | |
| T1–T4 (DNS) | | |
| T5 (WHOIS) | | |
| … | | |

**If a large fraction of the front is T0-only, that is the project's headline result.**

### Table 5 — Cost sensitivity

| Scenario | Features selected | Tier composition | Hypervolume | Subset stable? |
|----------|-------------------|------------------|-------------|----------------|
| Optimistic (÷3) | | | | |
| Baseline | | | | |
| Pessimistic (×3) | | | | |

---

## 7. Required figures

| # | Figure | Notes |
|---|--------|-------|
| **F1** | **Hypervolume vs generation**, all algorithms, mean ± std band | **The key convergence figure.** Replaces the single-objective fitness curve |
| **F2** | **Pareto front — 2D projections** (cost×FNR, cost×FPR, FNR×FPR) | Three panels. Far more readable than a 3D scatter |
| F3 | Parallel-coordinates plot of the front | pymoo has PCP built in; shows all 3 objectives at once |
| F4 | SPEA2 vs NSGA-II fronts overlaid | The mechanism comparison, made visual |
| F5 | Weighted-sum PSO points overlaid on the SPEA2 front | **Shows scalarisation returning isolated points while SPEA2 recovers the curve** |
| F6 | Feature frequency bar chart, coloured by tier | |
| F7 | Cost vs recall trade-off with deployment budgets marked | Vertical lines at 1 ms and 500 ms |
| F8 | Confusion matrices — all features vs each deployment profile | Same colour scale |
| F9 | `time_response` distribution | Cost-model calibration evidence |
| F10 | `−1` sentinel rate per external feature | Data-audit evidence |

**F5 is the most persuasive figure in the report.** It shows, in one image, exactly why
multi-objective optimisation was necessary.

**Conventions:** label axes with units, one consistent colour per algorithm across every figure,
save at ≥150 dpi to `results/figures/`.

---

## 8. What success looks like

| Outcome | Interpretation |
|---------|----------------|
| Front spans a wide cost range with usable recall throughout | **Target** — the trade-off is real and mapped |
| T0-only solutions with recall within a few points of baseline | **Best case** — no network calls needed; browser-deployable |
| SPEA2 hypervolume > NSGA-II | Confirms the pre-registered hypothesis |
| SPEA2 ≈ NSGA-II | Honest null result — report it; the mechanism argument was reasonable but unsupported |
| SPEA2 < NSGA-II | **Also fine.** Report it and analyse why. Negative results are results |
| Weighted PSO points sit *on* the SPEA2 front but sparsely | Expected — scalarisation finds valid points, just not the full curve |
| Any FNR ≈ 0 with FPR ≈ 0 | **Suspect leakage.** Investigate before reporting |

The middle rows matter. A pre-registered hypothesis that fails, reported honestly with analysis,
is worth more than a tuned result. Do not adjust parameters until SPEA2 wins.

---

## 9. Reproducibility checklist

- [ ] `RANDOM_STATE = 42` used in every split, classifier and optimiser
- [ ] All 5 seeds recorded explicitly
- [ ] Hypervolume reference point documented and identical everywhere
- [ ] Normalisation constants documented (cost ÷ 1320)
- [ ] All operator and population parameters in the experimental-setup section
- [ ] Every Pareto front saved to `results/fronts/` as `.npz`
- [ ] Result tables saved as CSV, not only pasted into the report
- [ ] Library versions recorded (`pip freeze` into an appendix)
- [ ] Cost-model tier table reproduced in full in the report
