# Phishing Detection via Multi-Objective Feature Selection with SPEA2

**Course:** Soft Computing
**Status:** Complete — all experiments run, report written
**Last updated:** 30 September 2026
**Report:** [`project-report.pdf`](project-report.pdf) (6 pages)

---

## What this project is

Build a model that classifies a website URL as **phishing** or **legitimate**, then use
**SPEA2** — a multi-objective evolutionary algorithm — to find feature subsets that are
simultaneously **fast to compute**, **rarely miss a phishing site**, and **rarely block a
legitimate one**.

The output is not one model. It is a **Pareto front** of best trade-offs, from which concrete
deployment configurations are extracted.

---

## Headline result

Measured once on the sealed 20% test set (11,729 websites), Random Forest classifier:

| | Features | Detection cost | Recall | FPR |
|---|---|---|---|---|
| All features (baseline) | 111 | 1,320 ms | 96.39% | 5.04% |
| **SPEA2 — email-gateway point** | **37** | **490 ms** | **96.52%** | **4.89%** |
| SPEA2 — no network lookups | 48 | 0.01 ms | 89.75% | 8.66% |

SPEA2 matched the full-feature detector with **67% fewer features and 63% lower detection
cost**. Detecting with no network lookups at all is possible but costs about 6.6 points of recall.

Other findings: NSGA-II reached a higher and more consistent hypervolume than SPEA2 (1.186 ± 0.008
vs 1.155 ± 0.067), contradicting our pre-registered hypothesis, though the deployable solutions
differ by under 0.3 points. The best achievable recall was stable (95.6–95.8%) when all network
latencies were scaled ÷3 or ×3. A 2% false-alarm rate is reachable only at ~90% recall — the
full 111-feature model hits the same limit — so a 57.6% decision cut-off (95.14% recall, 3.95%
false alarms) is recommended. Full
results: [`results/tables/README.md`](results/tables/README.md).

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
| **Compared against** | NSGA-II, single-objective PSO (weighted sum), PCA, mutual-information filter |
| **Framework** | `pymoo` |
| **Classifier** | Random Forest (SVM and KNN as baselines) |
| **Search space** | 2¹¹¹ ≈ 2.6 × 10³³ subsets |
| **Cost range** | ~0.01 ms (URL-only) to ~1,320 ms (all features) |
| **Comparison metric** | Hypervolume |
| **Hardware** | None — laptop or free Google Colab |


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

**Also in this folder:** [`archive/`](archive/) holds early summaries written for the
superseded single-objective PSO plan. They are kept for the record only.

---

## How to reproduce

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# datasets are not committed - download them into data/
curl -L -o data/dataset_small.csv https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_small.csv
curl -L -o data/dataset_full.csv  https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv

./.venv/bin/python src/audit_data.py                 # Phase 01 data audit
./.venv/bin/python src/run_baselines.py              # all-111-feature baselines
./.venv/bin/python src/run_classical_baselines.py    # PCA + mutual-information filter
./.venv/bin/python src/run_multiseed.py --algorithm spea2   # ~30 min
./.venv/bin/python src/run_multiseed.py --algorithm nsga2   # ~30 min
./.venv/bin/python src/run_pso_weighted.py           # weighted-sum PSO baseline
./.venv/bin/python src/analyze_fronts.py --algorithm spea2
./.venv/bin/python src/analyze_fronts.py --algorithm nsga2
./.venv/bin/python src/final_validation.py           # every subset, once, on the test set
./.venv/bin/python src/threshold_tuning.py           # false-alarm vs recall cut-offs
./.venv/bin/python src/make_figures.py               # figures into results/figures/
./.venv/bin/python src/run_sensitivity.py            # ~55 min
```

On a Mac, prefix long runs with `caffeinate -i` so the machine does not sleep mid-run.

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
- [x] Full 5-seed SPEA2 and NSGA-II runs, all-feature baselines
- [x] Classical baselines (PCA, mutual-information filter)
- [x] Weighted-sum PSO baseline (4 weight settings)
- [x] Final validation of every chosen subset on the sealed test set
- [x] Cost sensitivity analysis (÷3 / ×1 / ×3 latencies)
- [x] Report citations checked. **Correction made:** the "4–6% FNR" and "0.1% FPR" benchmarks come
      from PILFER (Fette, Sadeh & Tomasic, WWW 2007), an *email* phishing classifier; the report now
      labels them as email-phishing results
- [x] Outdated early summaries moved to `archive/`
- [x] Decision cut-off tuning: showed the 2% FPR target is only reachable at ~90% recall, for
      every method including all 111 features
- [ ] **Not done, by decision:** MOPSO was planned as a second swarm comparison but not run — the
      weighted-sum PSO already demonstrates the swarm/scalarisation comparison
- [ ] **Not done, optional:** cross-dataset validation on the UCI dataset, which would test whether
      the failed-DNS-lookup signal is a collection artifact

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

## What is not in the repository

The datasets (`data/*.csv`), the virtual environment (`.venv/`) and the raw Pareto-front arrays
(`results/fronts/*.npz`) are excluded by `.gitignore` to keep the repository small. See
*How to reproduce* above to regenerate them.
