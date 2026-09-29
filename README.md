# Phishing Detection via Multi-Objective Feature Selection with SPEA2

**Course:** Soft Computing
**Status:** Planning / research complete — nothing implemented yet
**Last updated:** 7 September 2026

---

## What this project is

Build a model that classifies a website URL as **phishing** or **legitimate**, then use
**SPEA2** — a multi-objective evolutionary algorithm — to find feature subsets that are
simultaneously **fast to compute**, **rarely miss a phishing site**, and **rarely block a
legitimate one**.

The output is not one model. It is a **Pareto front** of best trade-offs, from which concrete
deployment configurations are extracted.

---

## The three objectives

All minimised. These map directly onto the project's stated goals.

| | Objective | Definition | Goal it serves |
|---|---|---|---|
| **f₁** | Detection cost (ms) | Sum of latencies of the distinct network lookups a subset requires | *Minimise time to detect a phishing site* |
| **f₂** | False negative rate | `FN / (FN + TP)` = `1 − recall` | *Never classify a scam site as safe* |
| **f₃** | False positive rate | `FP / (FP + TN)` | Keeps precision honest; prevents a degenerate "flag everything" solution |

---

## Quick facts

| | |
|---|---|
| **Dataset** | Vrbančič Phishing Dataset — `dataset_small.csv` |
| **Size** | 58,645 rows × 112 columns (111 features + target) |
| **Target** | `phishing` — `0` = legitimate, `1` = phishing |
| **Primary algorithm** | **SPEA2** (Strength Pareto Evolutionary Algorithm 2) |
| **Compared against** | NSGA-II, MOPSO, single-objective PSO (weighted sum) |
| **Framework** | `pymoo` |
| **Classifier** | Random Forest (SVM and KNN as baselines) |
| **Search space** | 2¹¹¹ ≈ 2.6 × 10³³ subsets |
| **Cost range** | ~0.01 ms (URL-only) to ~1,320 ms (all features) |
| **Comparison metric** | Hypervolume |
| **Hardware** | None — laptop or free Google Colab |
| **Estimated effort** | 42–51 hours across two people |

---

## Document index

Read in order.

| Document | Contents |
|----------|----------|
| [`docs/01-dataset.md`](docs/01-dataset.md) | All 111 features named and grouped, both variants, encoding, **the `−1` sentinel issue**, download links |
| [`docs/02-cost-model.md`](docs/02-cost-model.md) | **The methodological contribution** — lookup tiers, latency assignment, calibration, sensitivity analysis, deployment profiles |
| [`docs/03-methodology.md`](docs/03-methodology.md) | Problem formulation, Pareto concepts, **SPEA2 mechanics in full**, why not a weighted sum, comparison algorithms |
| [`docs/04-literature-review.md`](docs/04-literature-review.md) | Benchmarks, verified attributions, required citations, the gap this project addresses |
| [`docs/05-implementation-plan.md`](docs/05-implementation-plan.md) | **The build plan** — 10 phases, pymoo configuration, pseudocode, runtime budget, two-person split |
| [`docs/06-evaluation-protocol.md`](docs/06-evaluation-protocol.md) | Hypervolume, experiment matrix, result-table templates, the 10 required figures |
| [`docs/07-pitfalls-and-risks.md`](docs/07-pitfalls-and-risks.md) | Everything that can invalidate the project, including multi-objective-specific traps |
| [`docs/08-objective-benchmarks.md`](docs/08-objective-benchmarks.md) | **Published numbers for each of the three objectives** — target latency, recall and FPR values, and the precision-vs-recall priority tension with commercial practice |

**Also in this folder:** `phishing-summary.html` / `.pdf` and `phishing-detection-plan.html` —
earlier summaries written for the single-objective PSO version. **These are now out of date**
and need regenerating.

---

## Decision log

Why each choice was made — the part that gets forgotten first.

### Chose phishing detection over other topics
Rejected breast cancer diagnosis, mobile price classification, human activity recognition.
Phishing has a genuine real-time speed constraint that makes feature reduction *necessary*
rather than merely tidy, plus a substantial published feature-selection literature and no
hardware component.

### Chose the Vrbančič dataset (111 features) over UCI (30)
At 30 features, simpler methods handle the search adequately, which undercuts the justification
for a metaheuristic — a serious weakness in a soft computing project. At 111 features the search
space is 2¹¹¹ ≈ 2.6 × 10³³ and evolutionary search becomes clearly necessary. UCI is retained as
an optional cross-dataset comparison.

### Rejected merging datasets
Incompatible column sets (111 vs 30), incompatible encodings (raw counts vs pre-bucketed
`−1/0/1`), and roughly eight years apart in collection. Merging risks the model learning *which
dataset a row came from*. **Cross-dataset validation** — running the identical pipeline on each
separately — is the sound alternative.

### Moved from single-objective PSO to multi-objective
The objectives evolved to include detection time, recall and precision — three genuinely
conflicting goals. A weighted-sum fitness would impose arbitrary weights, return a single point,
and is **provably unable to reach concave regions of the Pareto front**. That last point is a
mathematical limitation, not a tuning problem.

### Chose SPEA2 over NSGA-II as primary
Three reasons:

1. **Duplicate handling matches the data.** SPEA2 assigns identical fitness to identical
   individuals; NSGA-II can assign different fitness. This dataset counts the same 17 characters
   across five URL sections, so objective-space duplicates are constant, and the literature
   identifies duplicates as a driver of diversity collapse in NSGA-II.
2. **SPEA2's usual drawback is irrelevant here.** It is computationally heavier, but in a
   wrapper setting ~99.9% of runtime is classifier training. The selection overhead is invisible.
3. **Recent evidence favours it** — IJCAI 2025, *"SPEA2 Beats NSGA-II"*, plus empirical
   comparisons showing SPEA2 ahead on roughly 68% of tested space.

NSGA-II is still run under identical operators and budget, making the comparison a controlled
experiment on exactly the fitness-assignment and diversity mechanisms.

### Rejected NSGA-III, MOEA/D and research variants
NSGA-III solves a many-objective (4+) problem we don't have. MOEA/D decomposes using weight
vectors — philosophically close to the scalarisation being rejected, and awkward to defend.
AF-NSGA-II and Compact NSGA-II address high-dimensional feature selection (thousands of
features); at 111 that problem does not arise, and neither has library support.

### Cost modelled by lookup tier, not feature count
Counting features would rank a 40-feature URL-only subset (~0.01 ms) as worse than a 5-feature
subset requiring WHOIS (~250 ms) — the opposite of the truth. Features inherit the cost of the
network lookup they depend on, and **share** it, so cost sums over *distinct lookup types*.

### Chose `dataset_small.csv` over `dataset_full.csv`
Balanced classes make FNR and FPR directly comparable without reweighting. `dataset_full.csv`
is reserved for an optional imbalance study.

---

## Open items

- [x] Download `dataset_small.csv` and `dataset_full.csv` — both in `data/`
- [x] Quantify `−1` sentinel rates per external feature — see `results/tables/01_data_audit.txt`
      and `docs/01-dataset.md` §6. **Finding:** WHOIS features missing in 32–35% of rows; DNS
      lookup failures show an unexpected *negative* correlation with phishing (lift ≈ 0.5),
      likely a collection artifact — documented as a limitation
- [x] Time one fitness evaluation before committing to the full experiment matrix — **0.071 s**,
      ~4× faster than planned; full 5-seed SPEA2 run took ~32 min total
- [x] Set up local environment (`.venv/`) — pandas, scikit-learn, pymoo 0.6.2, matplotlib all
      working despite Python 3.14 / Homebrew's PEP 668 restrictions
- [x] Implement cost model, problem definition, SPEA2/NSGA-II runner, baselines, Pareto analysis,
      figure generation — all in `src/`
- [x] Added a per-feature tiebreak cost (`PER_FEATURE_EPSILON_MS`) after a pilot run showed the
      tier-only cost model gave no incentive to drop redundant same-tier features — see
      `docs/02-cost-model.md` §6.1
- [ ] Full 5-seed SPEA2 + NSGA-II + baseline run — **in progress**
- [ ] Classical baselines (PCA, mutual-information filter)
- [ ] Cost sensitivity analysis (optimistic/baseline/pessimistic scenarios)
- [ ] Verify every literature citation against its primary source before the report
- [ ] Regenerate the HTML/PDF summaries, which still describe the single-objective plan

---

## Glossary

| Term | Meaning |
|------|---------|
| **Pareto dominance** | `a` dominates `b` if `a` is no worse on every objective and strictly better on at least one |
| **Pareto front** | The set of non-dominated solutions, in objective space — the trade-off surface |
| **Hypervolume** | Volume of objective space dominated by a front, relative to a fixed reference point. Higher is better. The primary comparison metric |
| **Strength `S(i)`** | Number of solutions that individual `i` dominates (SPEA2) |
| **Raw fitness `R(i)`** | Sum of strengths of all solutions dominating `i`. Zero means non-dominated |
| **Density `D(i)`** | `1 / (σᵏ + 2)` — k-th nearest-neighbour distance term breaking ties among non-dominated solutions |
| **Archive truncation** | SPEA2's mechanism for keeping the archive at fixed size while preserving spread and boundary solutions |
| **Crowding distance** | NSGA-II's local diversity measure — the cuboid formed by the two nearest neighbours per objective |
| **Knee point** | The front's point of maximum curvature — best trade-off absent a stated preference |
| **Wrapper method** | Feature selection that trains a model per candidate subset. SPEA2 here is a wrapper |
| **Scalarisation** | Collapsing multiple objectives into one via weights. What this project deliberately avoids |
| **Lookup tier** | A class of network operation (DNS, WHOIS, TLS…) whose cost is shared by all features depending on it |

---

## What is deliberately NOT here

No code. No notebooks, no scripts, no environment. These are research and planning artefacts
only — implementation is a separate, later step.
