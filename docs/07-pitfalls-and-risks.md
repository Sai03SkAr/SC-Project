# 07 — Pitfalls and Risks

> **As built (30 Sep 2026).** This document is the pre-implementation plan and is kept as written.
> Deviations from it: **MOPSO was not run** — the weighted-sum PSO baseline
> (`src/run_pso_weighted.py`) covers the swarm/scalarisation comparison instead. Everything else
> described here was implemented; results are in `results/tables/README.md` and the report.

Everything that can invalidate this project, ordered by severity.

---

## 1. Data leakage — the project killer

**Severity: critical. This one invalidates everything.**

### The failure

Running feature selection on the whole dataset and *then* splitting into train and test. The
selection process has already seen the test data, so the reported accuracy is optimistically
biased and meaningless.

It is especially easy to commit here because feature selection feels like "preprocessing" rather
than "training" — but it is a decision learned from data, and must therefore be confined to the
training set.

### The correct ordering

```
1. Split into train / test        ← test set is now sealed
2. Run SPEA2 using an internal validation split INSIDE the training set only
3. Fit final models on the training set using each selected subset
4. Evaluate ONCE on the untouched test set
```

With multi-objective search this needs care: **both f₂ and f₃ are computed from the internal
validation split, never the test set.** The test set is used only for the handful of final
solutions extracted from the front — the deployment profiles, knee point and TOPSIS pick.

### Related leaks to avoid

| Leak | Fix |
|------|-----|
| Fitting a scaler on the full dataset | Fit on train only, then transform both |
| Fitting PCA on the full dataset | Same — train only |
| Computing mutual-information scores on the full dataset | Same — train only |
| Repeatedly evaluating on the test set and picking the best | Evaluate once, at the end |
| Leaving an index column in the features | Drop it — see section 2 |

### How it will be detected

An examiner asks: *"When did you split the data?"* There is no good answer if the split came
second. Symptomatically, accuracy above ~99.5% on this problem should immediately trigger a
leakage check rather than celebration.

---

## 2. The index column trap

**Severity: high. Applies to the UCI Kaggle CSV.**

Kaggle re-uploads of the UCI dataset commonly add a row-number `index` column. If the rows are
sorted by class — phishing first, legitimate after — then the row number becomes an almost
perfect predictor.

The model then achieves near-100% accuracy by learning *"low index means phishing"*, which
transfers to nothing.

**Fix:** inspect columns on load and drop any index column before training.

The primary Vrbančič dataset has no index column, which sidesteps this entirely.

---

## 3. Reporting a single stochastic run

**Severity: high.**

SPEA2, NSGA-II and MOPSO are all randomized. Two runs with different seeds produce different
fronts. Reporting one run reports luck.

**Fix:** run 5 seeds per algorithm, report **hypervolume** as mean ± standard deviation, merge
fronts across seeds for the combined analysis, and report how often each feature appears across
all Pareto-optimal solutions.

The feature-frequency analysis is worth doing for its own sake — a feature present in nearly
every non-dominated solution is a far stronger claim than one appearing in a single lucky run.

---

## 4. Judging by accuracy alone

**Severity: medium-high.**

Missing a phishing site is materially worse than falsely blocking a legitimate one. This is why
FNR and FPR are separate objectives rather than being folded into a single accuracy figure.

**Fix:** report recall, precision and the confusion matrix for every selected solution, and
discuss the asymmetric cost explicitly. Note that the multi-objective formulation handles this
correctly by construction — the preference is applied when *selecting* from the front, not baked
into a weight beforehand.

---

## 5. Empty and degenerate solutions

**Severity: medium — causes crashes, not wrong results.**

The optimiser will eventually propose an all-zeros mask, selecting no features. Training on an
empty feature matrix raises an exception, typically killing a long-running search partway
through.

**Fix:** guard in `_evaluate` — return the worst value on *every* objective:

```
IF sum(x) == 0:  RETURN [MAX_COST, 1.0, 1.0]
```

Returning a good value on any objective (an empty subset does have zero cost) would make the
degenerate solution non-dominated and pollute the entire front. All three must be worst-case.

Add this before the first run, not after losing one.

---

## 6. Merging datasets — considered and rejected

**Severity: high if attempted.**

Combining the Vrbančič and UCI datasets into one larger table fails for three independent
reasons:

| Problem | Detail |
|---------|--------|
| **Incompatible columns** | 111 features vs 30. Only a small intersection exists, discarding most of what makes the primary dataset valuable |
| **Incompatible encodings** | Vrbančič stores raw counts (`length_url = 87`); UCI stores pre-bucketed ternary values (`URL_Length = −1`). Same concept, different representation — stacking them puts two meanings in one column |
| **Dataset shift** | Collected roughly eight years apart from different sources. The model can learn *which dataset a row came from* instead of whether it is phishing, producing inflated accuracy that collapses on new data |

### The sound alternative — cross-dataset validation

Run the **identical pipeline on each dataset separately**, then compare results side by side.
No merging.

This provides everything merging appeared to offer, done correctly:

- Two independent result sets
- A real research question: *does the method generalize, or did it fit one dataset?*
- A genuinely interesting finding either way — if the same **types** of features survive on both
  datasets despite different feature names, that is a meaningful result; if entirely different
  types survive, that is arguably more interesting still

This is standard practice in published work, and there is active research asking exactly whether
phishing-detection features transfer across datasets.

### If a transfer experiment is attempted

Only one direction is possible: **Vrbančič → UCI**. The UCI paper documents the thresholds used
to bucket raw values, so Vrbančič's raw measurements can be converted into UCI-style ternary
encoding. The reverse cannot be done, since UCI already discarded the raw values.

Expect roughly a week of fiddly mapping work. Treat it as a bonus, never as core scope.

---

## 7. Runtime blow-up

**Severity: medium — costs time, not validity. The most likely practical failure.**

Each evaluation trains a model. The full matrix is large:

```
100 pop × 50 gen          =  5,000 evaluations per run
× 5 seeds                 = 25,000 per algorithm
× 3 algorithms            = 75,000 evaluations total
```

**Mitigations** (detailed in [`05-implementation-plan.md`](05-implementation-plan.md)):
subsample to ~15,000 rows during search, use a single stratified holdout rather than k-fold,
`n_estimators=30` while searching, `n_jobs=-1`, **and cache fitness by mask bytes** — this
dataset produces many duplicate masks, so the cache earns its keep.

**Time one evaluation and multiply by 5,000 before launching anything.** If a single run exceeds
~45 minutes, cut `pop_size` to 60 now rather than after losing a night. Discovering this in the
final week is the realistic way this project fails.

---

## 8. Premature convergence and diversity collapse

**Severity: medium — degrades results quietly.**

The population converges on a narrow region of the front and stops exploring. Hypervolume
plateaus early, and the front ends up short and clustered rather than spread.

**Signs:** hypervolume flat after few generations; few non-dominated solutions; the front
covering only a narrow cost range.

This is precisely the failure mode the literature attributes to duplicate solutions in
feature selection — and the reason SPEA2 was chosen, since it treats identical individuals
consistently.

**Fixes:** confirm `eliminate_duplicates=True` is set, raise mutation probability slightly,
increase population size, and check the cache hit rate — an extremely high rate signals the
population has collapsed onto a few repeated masks.

**Plot hypervolume against generation.** That figure makes this visible, and it is the
multi-objective replacement for the single-objective fitness curve.

---

## 9. Unfair algorithm comparison

**Severity: medium — produces a misleading conclusion.**

Giving one algorithm more evaluations, a different classifier, different operators or different
splits measures configuration differences rather than algorithmic ones.

**Fix:** SPEA2, NSGA-II and MOPSO must share the same problem object, operators, population
size, generation count, seeds and splits. Compare on **equal fitness-evaluation budget**; report
wall-clock separately as a result rather than as the comparison basis.

The whole point of the SPEA2-vs-NSGA-II comparison is that only the fitness-assignment and
diversity mechanisms differ. Any other difference destroys that.

---

## 10. Overfitting the feature selection itself

**Severity: medium — subtle, and worth knowing about.**

Even with a correct train/test split, running many SPEA2 configurations and choosing whichever
produced the best test accuracy leaks information through the researcher rather than the code.

**Fix:** make configuration decisions using cross-validation *within* the training set. The test
set answers one question, once: how does the final chosen model perform.

---

## 11. Reproducibility gaps

**Severity: low — but embarrassing at the wrong moment.**

Unfixed seeds, undocumented parameters, or figures existing only inside a notebook cell that has
since been overwritten.

**Fix:** fix `RANDOM_STATE = 42` globally, record library versions, save every table as CSV and
every figure as a file, and document all optimizer parameters in the experimental-setup section.

---

## 11b. Multi-objective specific pitfalls

Added when the project moved from single-objective PSO to SPEA2.

### Comparing algorithms by accuracy

**Severity: high — invalidates the comparison.**

Multi-objective algorithms return *sets*. "SPEA2 got 96.2% accuracy" is not a meaningful
statement — which solution on the front? Use **hypervolume** as the comparison metric, with
front size and spacing as supporting indicators.

### An inconsistent hypervolume reference point

**Severity: high — silently produces wrong conclusions.**

Hypervolume is measured relative to a reference point. Change it between algorithms or seeds and
the numbers become incomparable, while still looking perfectly reasonable.

**Fix:** normalise objectives to [0,1], fix the reference at `(1.1, 1.1, 1.1)`, use it
everywhere, and document it in the experimental setup.

### Unnormalised objectives

**Severity: high.**

Detection cost ranges 0.01–1320; FNR and FPR range 0–1. Computing hypervolume on raw values lets
cost dominate entirely — the other two objectives contribute almost nothing to the volume.
Always normalise first.

### Omitting the FPR objective

**Severity: critical — produces a degenerate result.**

With only cost and FNR as objectives, the optimiser discovers it can classify *everything* as
phishing: FNR = 0, cost minimal. A perfect score on a useless model.

**f₃ (FPR) is not optional.** It is what makes the formulation well-posed.

### Costing features individually instead of by lookup type

**Severity: medium — makes objective f₁ wrong.**

Features sharing a DNS query do not each cost a full lookup. Summing per-feature costs
overstates multi-feature subsets and distorts the entire front.

**Fix:** sum over *distinct tiers required*, per [`02-cost-model.md`](02-cost-model.md).

### Tuning until the preferred algorithm wins

**Severity: high — a research-integrity problem.**

The hypothesis that SPEA2 beats NSGA-II here is pre-registered and mechanistically motivated. If
the data disagrees, **report that.** Adjusting parameters until the favoured algorithm wins,
then presenting it as a finding, is fabrication.

A disconfirmed hypothesis, analysed honestly, is a legitimate and often more interesting result.

### Presenting the front without interpreting it

**Severity: medium — wastes the work.**

A 3D scatter plot of 60 non-dominated solutions tells a reader almost nothing. Extract the
**deployment profiles** (browser / gateway / offline), report the knee point, and analyse the
tier composition. The front is the evidence; the profiles are the conclusion.

---

## 12. Limitations to state proactively

Volunteering these reads as rigour. Being asked about them unprepared does not.

| Limitation | Framing |
|------------|---------|
| **No optimality guarantee** | SPEA2 approximates the Pareto front, not a provably optimal one — inherent to metaheuristics |
| **Dataset-specific results** | The selected features may not transfer; this motivates cross-dataset validation |
| **Fixed collection date** | Phishing tactics evolve; a deployed model would need periodic retraining |
| **URL-only detection has a ceiling** | Page content is never examined, so phishing hosted on a compromised legitimate domain is invisible to this approach |
| **Single classifier family for search** | Feature subsets were selected using Random Forest fitness; a different classifier might prefer different subsets |
| **Latency measured on a laptop** | Indicative, not a production benchmark |

That last-but-one point is a genuine methodological limitation worth naming explicitly: the
selected subset is optimal *for a Random Forest*, and that conditioning should be acknowledged.

---

## 13. Pre-submission checklist

- [ ] Split happens **before** any feature selection, scaling or PCA
- [ ] Test set evaluated exactly once per method
- [ ] Index column dropped if present
- [ ] Metaheuristics run across ≥5 seeds, mean ± std reported
- [ ] Recall and confusion matrix reported, not accuracy alone
- [ ] Empty-mask guard returns worst value on ALL three objectives
- [ ] SPEA2, NSGA-II and MOPSO given matched evaluation budgets
- [ ] Datasets **not** merged
- [ ] Convergence curve plotted and inspected for premature convergence
- [ ] All seeds, parameters and library versions documented
- [ ] Limitations section written
- [ ] Every literature citation verified against its primary source
