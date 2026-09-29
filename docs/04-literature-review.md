# 04 — Literature Review

Published work to benchmark against, and the raw material for the report's literature chapter.

---

> ## ⚠ Verification notice — read before citing
>
> The figures below were gathered through literature search and are recorded here as **leads**,
> not as verified quotations. Several come from abstracts and secondary summaries rather than
> full texts.
>
> **Before any number in this file goes into a submitted report, open the primary source and
> confirm it.** Check in particular: which dataset the figure refers to, how many features were
> retained, and whether the reported accuracy is cross-validated or test-set.
>
> Citing a misattributed number is worse than citing nothing.

---

## 1. Why this chapter matters

Two concrete benefits:

1. **A comparison target.** Landing at ~97% accuracy with roughly half the features puts this
   project squarely in line with published results. Being able to write *"our result is
   consistent with the published literature"* is substantial credibility for very little work.
2. **A ready-made literature chapter.** The benchmark table below is most of that section.

---

## 2. Benchmark table

Reported results for feature selection applied to phishing detection.

| Approach | Accuracy | Features kept | Dataset | Note |
|----------|----------|---------------|---------|------|
| Random Forest + PSO | 97.84% | 14 | — | Also reported reduced computation time |
| Random Forest + PSO | 97.1% | — | UCI | Same method: 98.7% on a Mendeley set, 90.3% on Zieni |
| ANN + PSO feature selection | 97.81% | — | — | 90.39% on a second, harder dataset |
| CatBoost classifier | 97.64% | — | — | F1 0.9762, AUC-ROC 0.996, ~6s train+test |

### ⚠ Correction — these are NOT phishing results

An earlier draft of this file listed "GA 99.60% with 34 features" and "PSO 99.58% with 32
features" without a dataset attribution. Those figures come from a 2025 **intrusion detection**
study on the **X-IIoTID** dataset, not phishing:

> *Multi-Objective Feature Selection for Intrusion Detection Systems: A Comparative Analysis of
> Bio-Inspired Optimization Algorithms.* Sensors 25(19), 6099, 2025.

Its actual findings, correctly attributed — comparing MOGWO, MOGA, MOPSO and MOACO on a
**bi-objective** formulation (classification error and subset size):

| Algorithm | Accuracy | Features | Note |
|-----------|----------|----------|------|
| GA | 99.60% | 34 | Highest accuracy; lowest FPR at 0.39% |
| **GWO** | 99.50% | 22 | **Best accuracy/subset balance — 65.08% reduction** |
| PSO | 99.58% | 32 | 49.21% reduction |
| ACO | 97.65% | 7 | Fastest; most aggressive sparsity — 88.89% reduction |

**This paper is directly relevant** despite being a different domain: it is a multi-objective
feature-selection comparison using the same bi-objective structure this project extends to three
objectives. Cite it as methodological precedent, **not** as a phishing benchmark.

This is exactly the error the verification notice above exists to catch.

### What the table supports

- **PSO and GA land in the same accuracy band.** In the multi-objective study the gap was
  0.02 percentage points — GA 99.60% with 34 features, PSO 99.58% with 32. The differentiator is
  cost, not accuracy.
- **Roughly 50% feature reduction is the norm**, not an outlier result.
- **Accuracy in the 97–99% range is expected** for this problem.

---

## 3. Findings by theme

### 3.1 PSO versus GA

The recurring conclusion across sources: **PSO matches or slightly exceeds GA performance while
requiring less model-building time.** PSO also has fewer parameters to tune — no crossover
operator, no selection scheme, no mutation rate.

This directly supports the choice of PSO as primary method and GA as comparison.

### 3.2 Why not PCA

One source notes that PCA-based reduction selects components by eigenvalue, and that
**components with the highest eigenvalues do not necessarily provide optimal sensitivity for the
classifier.** Variance and predictive usefulness are different properties.

This is a citable version of the argument in [`03-methodology.md`](03-methodology.md) section 6, and it is worth
locating the primary source for it — it strengthens the methodology justification considerably.

### 3.3 Hybrid approaches

Hybrids combining PSO with GA (sometimes "GPSO") report faster convergence and better avoidance
of premature convergence than either alone. Related work enhances PSO with orthogonal
initialization and crossover operators.

**Relevance:** a natural "future work" paragraph. Not recommended for implementation here — the
core PSO/GA comparison must be completed and done well first.

### 3.4 Transfer function variants

Beyond the standard S-shaped sigmoid, the literature proposes V-shaped, Z-shaped and
time-varying transfer functions, aimed at better balancing exploration and exploitation in
binary PSO.

**Relevance:** cite as an extension in the discussion. Demonstrates awareness of the current
state of the field.

### 3.5 Deep learning and transformers

Recent work applies CNN/LSTM/GAN architectures and transformer models to phishing detection,
with BERT and RoBERTa reported around 98.99% and 99.08%.

**Relevance — and the honest framing:** these are largely *text/sequence* models operating on
raw URLs, a different problem formulation from tabular feature selection. They are not a fair
comparison for this project, and should be positioned as related-but-distinct work. Claiming to
compete with them would be a mistake; acknowledging them shows awareness.

Note also that a large transformer defeats the deployability argument entirely — it is far
heavier than the 25-feature Random Forest this project aims to produce. That contrast is worth
making explicitly in the discussion.

### 3.6 Cross-dataset generalization — an open problem

At least one paper directly asks whether phishing-detection features can be trusted across
diverse datasets, using explainable AI to investigate. Others examine correlation-based
optimization for phishing URL detection.

**Relevance:** this validates the optional cross-dataset validation experiment as a *live
research question*, not merely a student exercise. If Phase 07 is attempted, this is the work to
cite for motivation.

---

## 4. Primary sources

### Dataset papers

> **[D1]** G. Vrbančič, I. Fister Jr., V. Podgorelec. *Datasets for Phishing Websites Detection.*
> Data in Brief, Vol. 33, 2020. DOI: `10.1016/j.dib.2020.106438`
> — The primary dataset. Also available via Mendeley DOI `10.17632/72ptz43s9v.1`.

> **[D2]** R. Mohammad, F. Thabtah, L. McCluskey. *Phishing Websites.* UCI Machine Learning
> Repository, 2012. DOI: `10.24432/C51W2X`
> — The secondary dataset and the field's long-standing benchmark.

### Method sources to obtain

Located during search; retrieve full texts before citing specifics.

| Topic | Where to find it |
|-------|------------------|
| PSO feature selection for phishing | *Intelligent feature selection model based on particle swarm optimization to detect phishing websites*, Multimedia Tools and Applications |
| GA feature selection for phishing URLs | *Enhanced Feature Selection Using Genetic Algorithm for Machine-Learning-Based Phishing URL Detection*, Applied Sciences 14(14), 6081 |
| Cross-dataset feature trust | *Can Features for Phishing URL Detection Be Trusted Across Diverse Datasets? A Case Study with Explainable AI* (arXiv) |
| Binary PSO transfer functions | *Z-Shaped Transfer Functions for Binary Particle Swarm Optimization*, Computational Intelligence and Neuroscience, 2020 |
| PSO survey and background | *Particle Swarm Optimization: A survey of historical and recent developments with hybridization perspectives* (arXiv) |
| Phishing detection overview | *Phishing Attacks and Websites Classification Using Machine Learning and Multiple Datasets* (arXiv) |

### Foundational references — required

Standard citations that must appear, retrieved from the original publications:

| Reference | For |
|-----------|-----|
| **Zitzler, Laumanns & Thiele (2001).** *SPEA2: Improving the Strength Pareto Evolutionary Algorithm.* TIK Report 103, ETH Zurich | **The primary algorithm** |
| **Deb, Pratap, Agarwal & Meyarivan (2002).** *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.* IEEE Trans. Evolutionary Computation | The comparison algorithm |
| Coello Coello & Lechuga (2002). *MOPSO: A Proposal for Multiple Objective Particle Swarm Optimization* | The swarm comparison |
| Kennedy & Eberhart (1995) — original PSO; (1997) — discrete binary PSO | The weighted-sum baseline |
| Blank & Deb (2020). *pymoo: Multi-Objective Optimization in Python.* IEEE Access | The framework |

### Multi-objective sources located

| Topic | Source |
|-------|--------|
| **SPEA2 outperforming NSGA-II** | *Proven Approximation Guarantees in Multi-Objective Optimization: SPEA2 Beats NSGA-II.* IJCAI 2025 |
| NSGA-II limitations at high dimensionality | *Compact NSGA-II for Multi-objective Feature Selection* (arXiv 2402.12625); *Enhancing Diversity in Multi-objective Feature Selection* (arXiv 2407.17795) |
| Duplicate solutions degrading diversity | *Enhanced NSGA-II-based feature selection method for high-dimensional classification*, Information Sciences |
| Multi-objective FS precedent | *Multi-Objective Feature Selection for Intrusion Detection Systems*, Sensors 25(19), 2025 |
| Knee-point selection | *Knee Point-Guided Multiobjective Optimization*, Complexity, 2020 |

> **Framing note for the report.** The high-dimensionality criticism concerns datasets with
> thousands of features. At 111, this project sits inside the effective range of standard
> dominance-based MOEAs. Cite the criticism *and* explain why it does not apply — that reads as
> rigour rather than omission.

---

## 5. The gap this project addresses

Useful framing for the introduction. Existing work tends to:

- report accuracy while giving **prediction latency little or no attention**, despite real-time
  operation being the actual deployment constraint;
- treat all features as equally costly, ignoring that **some require network calls and most do
  not**;
- evaluate on a single dataset.

This project's distinguishing angles:

1. **Cost-aware analysis.** Explicitly separating the 96 instantly-computable features from the
   15 requiring network lookups, and reporting how many of each survive selection. If the
   expensive features are largely discarded, the reduced model becomes deployable in-browser —
   a qualitative change, not merely a quantitative one.
2. **Latency as a reported metric**, not just accuracy.
3. **Optional cross-dataset validation** against a live open question in the field.

None of these require novel algorithms. They require asking a slightly better question of a
standard method — which is what makes them achievable within a course project.

---

## 6. How to use this in the report

Suggested structure for the literature chapter:

1. **Phishing detection background** — the problem, why URL-based detection is attractive
   (works before page load).
2. **Feature selection taxonomy** — filter / wrapper / embedded, citing standard sources; state
   where this project sits.
3. **Metaheuristics for feature selection** — PSO and GA foundations, then applications to
   phishing specifically.
4. **Comparative results** — the benchmark table from section 2, verified.
5. **Gap statement** — section 5 above, leading into the objectives.

Aim for 8–15 references. Prioritize: the two dataset papers, the original PSO papers, two or
three phishing-specific feature-selection papers, and one recent survey.
