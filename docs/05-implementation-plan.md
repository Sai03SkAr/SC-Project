# 05 — Implementation Plan

The full build sequence for SPEA2-based multi-objective feature selection.

**Nothing is implemented yet. This is the specification.**

---

## 1. Environment

| | |
|---|---|
| **Platform** | Google Colab (recommended — nothing to install) or local Jupyter |
| **Python** | 3.10+ |

### Dependencies

```
pandas           data handling
numpy            numerics
scikit-learn     classifiers, metrics, splits
pymoo            SPEA2, NSGA-II, MOPSO, hypervolume, visualisation
pyswarms         single-objective PSO baseline only
matplotlib       plots
seaborn          heatmaps
```

`pymoo` replaces `pyswarms` as the primary optimisation library. `pyswarms` is retained solely
for the weighted-sum baseline.

### The core API

```python
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
```

SPEA2 is a drop-in algorithm in pymoo using the same modular operators as every other GA, so
swapping SPEA2 ↔ NSGA-II changes exactly one line.

Docs: [SPEA2](https://pymoo.org/algorithms/moo/spea2.html) · [NSGA-II](https://pymoo.org/algorithms/moo/nsga2.html)

---

## 2. Project structure

```
SC Project/
├── README.md
├── docs/                          ← these planning documents
├── data/
│   ├── dataset_small.csv          ← primary: 58,645 × 112
│   ├── dataset_full.csv           ← secondary: 88,647 × 112
│   └── uci_phishing.csv           ← optional cross-dataset
├── src/
│   ├── cost_model.py              ← tier map + detection_cost()
│   ├── problem.py                 ← pymoo problem definition
│   └── evaluation.py              ← metrics, HV, front analysis
├── notebooks/
│   ├── 01_eda_and_audit.ipynb
│   ├── 02_cost_model.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_pso_weighted_baseline.ipynb
│   ├── 05_spea2.ipynb             ← the core
│   ├── 06_nsga2_mopso.ipynb
│   └── 07_pareto_analysis.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   └── fronts/                    ← save every Pareto front as .npz
└── report/
```

Save every Pareto front to disk as it is produced. Re-running a three-hour optimisation because
a notebook kernel restarted is the most avoidable failure in this project.

---

## 3. Phase overview

| Phase | Task | Owner | Estimate |
|-------|------|-------|----------|
| 00 | Setup | Both | 1 h |
| 01 | Data audit — including the `−1` sentinel investigation | A | 4–5 h |
| 02 | Build and calibrate the cost model | B | 3–4 h |
| 03 | Baseline classifiers, all 111 features | A | 3–4 h |
| 04 | Single-objective PSO baseline | B | 3 h |
| **05** | **SPEA2 — the core** | **Both** | **8–10 h** |
| 06 | NSGA-II and MOPSO comparison | B | 4–5 h |
| 07 | Pareto analysis and solution selection | Both | 5–6 h |
| 08 | Cost sensitivity analysis | A | 3 h |
| 09 | Report and presentation | Both | 8–10 h |

**Total: 42–51 hours.** Larger than the original single-objective plan, because multi-objective
adds the cost model, front analysis, and an extra algorithm.

---

## Phase 00 — Setup

- Shared repo or Drive folder; both people able to run notebooks
- `pip install pymoo pyswarms scikit-learn pandas matplotlib seaborn`
- Download `dataset_small.csv` (primary) **and** `dataset_full.csv`
- Fix `RANDOM_STATE = 42` in a shared constants file, used everywhere

> Your current copy is `dataset_full.csv` (88,647 × 112). Get `dataset_small.csv` too — its
> balanced classes make FNR and FPR directly comparable without reweighting.

---

## Phase 01 — Data audit

Standard EDA, plus one investigation specific to this dataset.

### Standard checks

- Shape: 58,645 × 112
- Target `phishing` ∈ {0, 1}; confirm balance (27,998 / 30,647)
- Missing values (source claims none — verify)
- Constant / zero-variance columns — likely among rare-character counts such as
  `qty_space_*`, `qty_asterisk_*`. Free wins for any reduction method; count them
- Within-group correlation — expect heavy redundancy across the five URL sections

### The `−1` sentinel investigation — do not skip

The external-lookup columns contain `−1` where the lookup **failed**, not where a real value
was measured. A domain cannot be −1 days old.

```
FOR each of the 15 external features:
    count and report the proportion of −1 values

Then test:  is  P(phishing | lookup failed)  ≠  P(phishing | lookup succeeded) ?
```

**Why this matters:**

| Model | Effect of `−1` |
|-------|----------------|
| Random Forest | Handles it — splits at `−1` and may correctly treat "lookup failed" as its own signal |
| SVM / KNN | **Misled** — reads `−1` as "slightly less than zero days old", placing failed lookups adjacent to brand-new domains |

**Decision required, and document it:** either (a) leave as-is and rely on tree models, or
(b) add explicit binary "lookup failed" indicator columns. Option (b) is cleaner but adds
features, which interacts with the cost objective — if chosen, the indicator inherits the same
tier as its parent feature, since you only learn the lookup failed by attempting it.

**Deliverable:** `01_eda_and_audit.ipynb`, 4–5 figures, a table of `−1` rates per external
feature, and a written decision on handling.

---

## Phase 02 — Cost model

Implement [`02-cost-model.md`](02-cost-model.md).

```
1. Build FEATURE_TIER — map all 111 features to tiers T0–T9
2. Implement detection_cost(mask) — sum over DISTINCT tiers, not features
3. Unit-check by hand:
      all features          → ~1,320 ms
      T0 only               → 0.01 ms
      one WHOIS feature     → 0.01 + 200 ms
4. Calibrate against data: distribution of `time_response`
      (median / mean / p95) → justify the DNS tier latency
5. Define the three sensitivity scenarios (÷3, baseline, ×3)
```

**Deliverable:** `src/cost_model.py`, a `time_response` distribution figure, and a written
justification of the tier latencies.

---

## Phase 03 — Baselines, all 111 features

```
SPLIT  train/test  80/20, stratified, random_state=42       ← test set now sealed

FOR clf IN [RandomForest, SVM, KNN]:
    IF clf needs scaling:  fit scaler on TRAIN ONLY
    TRAIN on train
    RECORD accuracy, precision, recall, F1, ROC-AUC, MCC
    RECORD FNR, FPR                     ← the actual objectives
    RECORD training time, inference latency
    RECORD detection_cost = ~1,320 ms   ← all tiers required

SELECT Random Forest for the optimisation phases
```

This is the control point in objective space: `(1320 ms, FNR₀, FPR₀)`. Everything later is
measured against it.

---

## Phase 04 — Single-objective PSO baseline

Not legacy work — it is the empirical argument for going multi-objective.

```
fitness(mask) = 0.6·normalised_cost + 0.3·FNR + 0.1·FPR      # weights are arbitrary — that's the point

RUN BinaryPSO, 30 particles × 50 iterations, 5 seeds
RECORD the single solution found, all objectives, runtime
```

**Then run it again with different weights** (e.g. 0.2/0.6/0.2) and show it lands somewhere
else entirely.

**The finding:** each weighting yields one point; the weights have no principled justification;
and the union of several weighted runs still fails to cover the front. That motivates everything
in Phase 05.

---

## Phase 05 — SPEA2 (the core)

### 5.1 Problem definition

```
CLASS FeatureSelectionProblem(ElementwiseProblem):
    n_var  = 111
    n_obj  = 3
    vtype  = binary

    FUNCTION _evaluate(x):
        IF sum(x) == 0:
            RETURN [MAX_COST, 1.0, 1.0]        # worst on every objective

        cols = feature_names WHERE x == 1

        # ---- f1: no training required ----
        f1 = detection_cost(x)

        # ---- f2, f3: ONE model fit yields both ----
        model.fit(X_search[cols], y_search)
        pred = model.predict(X_val[cols])
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()

        f2 = fn / (fn + tp)        # FNR  = 1 − recall
        f3 = fp / (fp + tn)        # FPR

        RETURN [f1, f2, f3]
```

Two properties worth noting in the report:

- **f₁ needs no model.** It is arithmetic over the mask.
- **f₂ and f₃ come from a single confusion matrix.** So three objectives cost exactly the same
  per evaluation as one did. Multi-objective is free here.

### 5.2 Fitness caching — implement this before running anything

This dataset produces enormous numbers of duplicate masks. Cache them:

```
cache = {}

FUNCTION evaluate_cached(x):
    key = x.tobytes()
    IF key IN cache:  RETURN cache[key]
    result = evaluate(x)
    cache[key] = result
    RETURN result
```

Expect a meaningful hit rate given the redundancy in this feature set. Report the hit rate — it
is direct evidence for the duplicate-heavy structure that motivated choosing SPEA2.

### 5.3 Algorithm configuration

```python
algorithm = SPEA2(
    pop_size             = 100,
    sampling             = BinaryRandomSampling(),
    crossover            = TwoPointCrossover(),
    mutation             = BitflipMutation(prob=1/111),
    eliminate_duplicates = True,
)

res = minimize(problem, algorithm, ('n_gen', 50), seed=SEED, save_history=True)
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| `pop_size` | 100 | Standard for MOEAs; enough to populate a 3D front |
| `n_gen` | 50 | 5,000 evaluations per run |
| Mutation prob | 1/111 ≈ 0.009 | Standard heuristic: one bit flip per chromosome on average |
| `eliminate_duplicates` | **True** | Directly addresses the documented diversity-collapse problem |
| `save_history` | True | Required for hypervolume-vs-generation plots |

### 5.4 Runtime budget — plan this before launching

```
per run    :  100 pop × 50 gen              =  5,000 evaluations
per algo   :  5,000 × 5 seeds               = 25,000 evaluations
all 3 MOAs :  25,000 × 3                    = 75,000 evaluations
```

Making each evaluation cheap, **during search only**:

| Setting | Value | Effect |
|---------|-------|--------|
| Subsample | ~15,000 rows | Largest single saving |
| Validation | **Single stratified holdout, not k-fold** | 3× cheaper than 3-fold |
| Classifier | `RandomForest(n_estimators=30, n_jobs=-1)` | Fast, still stable |
| Caching | See 5.2 | Removes repeat work |
| Final validation | Full data, full model, proper CV | Applied only to the final front |

**Before launching the full experiment, time one evaluation and multiply by 5,000.** If a single
run exceeds ~45 minutes, cut `pop_size` to 60 *now* rather than after losing a night.

**Measured, not just estimated.** Actual timing on this machine (`src/problem.py`, tested
directly): **0.071 s per evaluation** — roughly 4× faster than the planned target. A full
5-seed SPEA2 run (100 pop × 50 gen) completed in **~6.3 minutes per seed**, ~32 minutes total,
well under the 45-minute-per-run threshold above. No need to fall back to `pop_size=60`.

The cache hit rate observed in practice is low (~1%), because pymoo's own
`eliminate_duplicates=True` already prevents exact duplicate individuals within a generation
before they reach the fitness function — the cache mainly catches duplicates *across*
generations, which are rarer than expected. It is still worth keeping: it is essentially free
and occasionally saves a real evaluation.

### 5.5 Deliverables

- Pareto front per seed, saved to `results/fronts/`
- Hypervolume-vs-generation curve **(the key convergence figure)**
- Combined front across seeds
- Cache hit rate
- Wall-clock time per run

---

## Phase 06 — NSGA-II and MOPSO

**Identical everything** — same problem object, same operators, same budget, same seeds. Only
the algorithm changes:

```python
algorithm = NSGA2(pop_size=100, sampling=..., crossover=..., mutation=...,
                  eliminate_duplicates=True)
```

Fairness requirement: equal **fitness-evaluation budget**, not equal wall-clock. Report
wall-clock separately as a finding.

**Pre-registered hypothesis:** SPEA2 outperforms NSGA-II on hypervolume here, because its
consistent treatment of duplicate individuals suits this dataset's redundant structure. State
this *before* seeing results — a hypothesis confirmed is worth far more than an observation
rationalised afterwards.

---

## Phase 07 — Pareto analysis and solution selection

The step that turns a plot into conclusions.

### 7.1 Front characterisation

```
Non-dominated solutions found
Spread across each objective (min, max, range)
Feature-count distribution across the front
Tier composition — how many solutions are T0-only?
```

### 7.2 Feature frequency across the front

```
FOR each feature:
    proportion of Pareto-optimal solutions containing it
```

Features present in nearly every solution are robustly essential. Features appearing rarely are
situational. This is a stronger result than any single subset.

### 7.3 Deployment profiles — the headline presentation

Extract the best solution under three realistic budgets:

| Profile | Cost budget | Selection rule |
|---------|-------------|----------------|
| **Browser extension** | ≤ 1 ms | Lowest FNR among T0-only solutions |
| **Email gateway** | ≤ 500 ms | Lowest FNR within budget |
| **Offline audit** | unlimited | Lowest FNR overall |

For each: feature count, tier composition, FNR, FPR, recall, precision, F1.

This converts an abstract front into three concrete engineering recommendations — and
demonstrates directly what multi-objective optimisation delivered, since a single-objective run
could have produced only one of the three.

### 7.4 Formal selection methods

Report alongside the profiles, for methodological completeness:

- **Knee point** — the point of maximum curvature, where a small gain in one objective forces a
  large loss in another. The default choice absent stated preference
- **TOPSIS** — rank by relative closeness to the utopia point versus the nadir point

---

## Phase 08 — Cost sensitivity analysis

Because tier latencies are estimates, results must be shown robust to them.

```
FOR scenario IN [optimistic ÷3, baseline, pessimistic ×3]:
    RE-RUN SPEA2 (3 seeds is sufficient here)
    RECORD front, tier composition, hypervolume

COMPARE: do the selected subsets stay stable?
```

Stable subsets → robust conclusion. Shifting subsets → the trade-off genuinely depends on
deployment conditions, which is itself a legitimate and interesting finding.

---

## Phase 09 — Report

```
1. Introduction            phishing; the real-time constraint; why speed is a hard requirement
2. Literature review       docs/04 — verify every citation first
3. Dataset                 docs/01 — including the −1 sentinel finding
4. Cost model              docs/02 — the methodological contribution
5. Methodology             docs/03 — Pareto concepts, SPEA2 mechanics, why not weighted sum
6. Experimental setup      splits, seeds, parameters, runtime strategy
7. Results                 fronts, hypervolume, deployment profiles
8. Discussion              what survived and why; SPEA2 vs NSGA-II mechanism; limitations
9. Conclusion              and future work
```

---

## 4. Two-person split

| Person A | Person B |
|----------|----------|
| 01 — data audit and `−1` investigation | 02 — cost model |
| 03 — baseline classifiers | 04 — PSO weighted baseline |
| 08 — sensitivity analysis | 06 — NSGA-II and MOPSO |
| **05 — SPEA2 (together)** | **05 — SPEA2 (together)** |
| **07, 09 (together)** | **07, 09 (together)** |

Phases 01/03 and 02/04 run in parallel. Phase 05 is joint — it is what the project is graded on,
and both people must be able to explain it.

---

## 5. Definition of done

- [ ] `−1` sentinel rates quantified per external feature, handling decision documented
- [ ] Cost model implemented, calibrated against `time_response`, hand-verified
- [ ] Baselines recorded for all 111 features across three classifiers
- [ ] Weighted-sum PSO run at ≥2 weight settings, showing single-point limitation
- [ ] SPEA2 run across ≥5 seeds, fronts saved to disk
- [ ] NSGA-II and MOPSO run under identical budget and seeds
- [ ] Hypervolume computed for all algorithms with a documented reference point
- [ ] Hypervolume-vs-generation convergence figure produced
- [ ] Feature-frequency analysis across the Pareto front
- [ ] Three deployment profiles extracted and tabulated
- [ ] Knee point and TOPSIS solutions reported
- [ ] Cost sensitivity analysis across all three scenarios
- [ ] Every final number derived from a single evaluation on the untouched test set
- [ ] Limitations stated explicitly
- [ ] Every citation verified against its primary source
