# 03 — Methodology

Multi-objective feature selection with SPEA2. The algorithms, their mechanics, and the
justification for each choice.

---

## 1. Problem formulation

**Given:** 111 candidate features describing a URL.
**Find:** subsets of those features that are simultaneously cheap to compute, rarely miss a
phishing site, and rarely block a legitimate one.

**Decision variable:** a binary vector **x ∈ {0,1}¹¹¹**, where `x_i = 1` means feature *i* is
included.

**Objectives — all minimised:**

| | Objective | Definition | Corresponds to |
|---|---|---|---|
| **f₁** | Detection cost (ms) | Sum of latencies of distinct lookup tiers required — see [`02-cost-model.md`](02-cost-model.md) | *"Minimise time to detect a phishing website"* |
| **f₂** | False negative rate | `FN / (FN + TP)` = `1 − recall` | *"Make sure a scam site isn't classified as safe"* |
| **f₃** | False positive rate | `FP / (FP + TN)` | Keeps precision honest |

**Search space:** 2¹¹¹ ≈ 2.6 × 10³³ subsets.

### Why three objectives rather than one

f₂ and f₃ are in direct conflict. Minimise f₂ alone and the optimal model flags everything as
phishing — perfect recall, worthless precision. Minimise f₃ alone and it flags nothing. f₁
conflicts with both, since the most informative features are often the expensive ones.

There is no single best solution — only a **set of best trade-offs.** That set is what a
multi-objective algorithm returns.

### Why f₃ must be present

Without it, the optimiser has a trivial cheat available: classify everything as phishing, and
f₂ = 0. The optimisation would collapse into a degenerate solution. f₃ closes that door.

---

## 2. Why not a weighted sum

The single-objective alternative combines objectives with weights:

```
fitness = w₁·f₁ + w₂·f₂ + w₃·f₃
```

Three problems, in increasing severity:

1. **The weights are arbitrary.** There is no principled basis for choosing them, and results
   are driven by the choice.
2. **It returns one solution.** One point on the trade-off curve. A changed priority means a
   full re-run.
3. **It is mathematically incapable of reaching parts of the front.** A weighted sum can only
   find solutions on *convex* regions of the Pareto front. Solutions in a concave region are
   unreachable for **any** weight vector. This is a property of the method, not a tuning issue.

Point 3 is the decisive one, and it belongs in the report — it is the formal justification for
the entire approach.

> A single-objective PSO with a weighted-sum fitness is still implemented in this project, as a
> **baseline that demonstrates this limitation empirically.** Showing that it lands on one
> arbitrary point while SPEA2 recovers the whole front is a stronger argument than asserting it.

---

## 3. Pareto concepts

Three definitions the report must state precisely.

**Dominance.** Solution **a** dominates **b** if `a` is no worse than `b` on *every* objective,
and strictly better on *at least one*.

```
a ≺ b   ⟺   ∀i: fᵢ(a) ≤ fᵢ(b)   ∧   ∃j: f_j(a) < f_j(b)
```

**Pareto-optimal set.** All solutions not dominated by any other. No objective can be improved
without worsening another.

**Pareto front.** The image of that set in objective space — the trade-off surface itself.

For this project the front is a surface in 3D (cost × FNR × FPR).

---

## 4. SPEA2 — the primary algorithm

**Strength Pareto Evolutionary Algorithm 2** (Zitzler, Laumanns & Thiele, 2001).

### 4.1 Fitness assignment — fine-grained

Unlike rank-based approaches, SPEA2 measures *how badly* a solution is dominated, not merely
that it is.

**Strength.** For each individual *i*, `S(i)` = the number of solutions *i* dominates.

**Raw fitness.** For each individual *i*, `R(i)` = the sum of the strengths of all solutions
that dominate *i*.

```
R(i) = Σ  S(j)
      j ≺ i
```

`R(i) = 0` means non-dominated. A high `R(i)` means *i* is dominated by solutions that are
themselves strong — a much richer signal than "you are in front 4."

### 4.2 Density estimation — global

To break ties among non-dominated solutions (all of which have `R = 0`), a density term is added
based on the distance to the **k-th nearest neighbour in objective space**:

```
D(i) = 1 / (σᵢᵏ + 2)
```

where `σᵢᵏ` is the distance from *i* to its k-th nearest neighbour. The `+2` keeps `D(i) < 1`,
so density never overrides dominance.

**Final fitness:**

```
F(i) = R(i) + D(i)          — lower is better
```

### 4.3 Environmental selection and archive truncation

SPEA2 maintains a **fixed-size external archive** of the best individuals across generations.

- If there are **fewer** non-dominated solutions than the archive size, the best dominated
  solutions fill the remainder.
- If there are **more**, a **truncation operator** iteratively removes the individual with the
  smallest distance to its neighbours — preserving spread, and explicitly retaining boundary
  solutions.

### 4.4 Why SPEA2 for *this* problem specifically

Three reasons, in order of importance.

**1. Duplicate handling matches the data's structure.**

In SPEA2, identical individuals receive **identical fitness**. In NSGA-II they can receive
*different* fitness, because crowding distance is computed from position among neighbours.

This dataset generates objective-space duplicates constantly: the same 17 characters are counted
across five URL sections, so vast numbers of different masks yield identical accuracy. The
feature-selection literature identifies duplicate solutions as a specific driver of diversity
collapse and premature convergence in NSGA-II. SPEA2's consistent treatment is a structural fit.

**2. The usual objection to SPEA2 does not apply here.**

SPEA2 is computationally heavier — O(N² log N) archive truncation versus NSGA-II's O(MN²).
In a **wrapper** setting that is irrelevant: each fitness evaluation trains a Random Forest
(~300 ms), while the selection machinery runs in microseconds. Roughly 99.9% of runtime goes to
scikit-learn. The algorithmic overhead difference is invisible.

**3. Recent evidence favours it.**

A 2025 IJCAI paper — *"Proven Approximation Guarantees in Multi-Objective Optimization: SPEA2
Beats NSGA-II"* — shows a steady-state SPEA2 computing optimal Pareto-front approximations in
polynomial time on a benchmark where comparable NSGA-II variants only guarantee approximation
ratios around a factor of two. Empirical comparisons report SPEA2 ahead on roughly 68% of tested
space.

### 4.5 The scalability criticism, and why it does not apply

A 2024–25 research strand argues NSGA-II and similar MOEAs degrade on **high-dimensional**
feature selection — premature convergence, collapsing decision-space diversity — and proposes
variants (AF-NSGA-II, Compact NSGA-II) reporting large hypervolume gains.

**"High-dimensional" there means thousands to tens of thousands of features** — gene expression,
text corpora. At **111 features** this problem sits comfortably inside the effective operating
range of standard dominance-based MOEAs.

State this explicitly in the report. Acknowledging the criticism and reasoning about its
applicability is stronger than ignoring it, and stronger than adopting a research-grade variant
that has no library implementation.

---

## 5. Comparison algorithms

| Role | Algorithm | Purpose |
|------|-----------|---------|
| **Primary** | **SPEA2** | The project's result |
| Co-primary | **NSGA-II** | The field's reference point. Same operators, one-line swap |
| Comparison | **MOPSO** | Swarm-based multi-objective — evolutionary vs swarm |
| Baseline | **PSO (weighted sum)** | Demonstrates the scalarisation limitation |
| Control | All 111 features | No selection at all |

### On NSGA-II as co-primary

NSGA-II differs from SPEA2 in exactly two mechanisms:

| | NSGA-II | SPEA2 |
|---|---|---|
| Fitness | Front rank (integer) | `R(i) + D(i)` — strength-based |
| Diversity | Crowding distance — 2 neighbours per objective, **local** | k-NN density over all solutions, **global** |
| Elitism | Combined parent+offspring (μ+λ) | Fixed-size external archive with truncation |

Because both use identical sampling, crossover and mutation operators in `pymoo`, the comparison
isolates precisely these mechanisms. That is a clean experiment, and the hypothesis was
pre-registered *before running it*: **SPEA2 should handle this dataset's duplicate-heavy
structure better.**

> ### ⚠ Result — the hypothesis was NOT confirmed
>
> Across 5 seeds each (pop=100, gen=50): **NSGA-II achieved a higher mean hypervolume
> (1.1863 vs 1.1546) and was roughly 8× more consistent across seeds** (std 0.0080 vs
> 0.0665). Full data in `results/tables/README.md`.
>
> **The likely reason:** both algorithms already run with `eliminate_duplicates=True` at
> the *population* level, which removes exact duplicate individuals before they ever
> reach the fitness function. That may already capture most of the benefit this
> hypothesis was pointing at, leaving little room for SPEA2's finer-grained
> strength/density mechanism to add further value — and one SPEA2 seed (seed 3)
> converged to a visibly worse front (HV 1.0225, vs 1.18+ for every NSGA-II seed),
> which drags down SPEA2's mean and inflates its variance.
>
> **At the level that actually matters — the deployable solutions — the two are nearly
> identical.** Their pooled Pareto fronts overlap almost completely
> (`results/figures/f2_pareto_projections.png`), and the deployment-profile numbers
> (browser/email-gateway recall and FPR) differ by only 0.1–0.3 percentage points. Both
> algorithms find essentially the same trade-off surface; NSGA-II simply finds it more
> *reliably* on a per-run basis.
>
> **This is reported as a negative result, not adjusted after the fact.** A
> mechanistically-motivated hypothesis that fails, and is reported as such, is stronger
> evidence of rigour than a hypothesis quietly tuned until it succeeds — see
> `docs/06-evaluation-protocol.md` §8. SPEA2 remains the project's primary algorithm
> because it was chosen and justified *before* this result was known, and its output is
> what the deployment recommendations are drawn from; NSGA-II's stronger showing here is
> disclosed as a limitation of that choice, not concealed.

### Algorithms deliberately excluded

| Algorithm | Why not |
|-----------|---------|
| **NSGA-III** | Built for 4+ objectives; requires reference directions. Unnecessary machinery at 3 |
| **MOEA/D** | Decomposes using weight vectors — philosophically close to the scalarisation this project rejects. Justifying the distinction is a subtle argument not worth the risk |
| **AF-NSGA-II, Compact NSGA-II** | Research-grade variants with no library implementation; solve a high-dimensional problem this project does not have |
| **MOGWO** | Interesting (a 2025 IDS study found Grey Wolf gave the best accuracy/subset balance) but thin library support and a smaller literature base |

---

## 6. Classifier — the measuring instrument

The classifier is not the subject of the project. It must be held constant across every method.

| Classifier | Role |
|------------|------|
| **Random Forest** | **Primary.** Strong on tabular data, no scaling required, parallelises with `n_jobs=-1`, tolerates the `−1` sentinel values sensibly |
| SVM | Baseline only. Requires scaling; sensitive to the `−1` sentinels |
| KNN | Baseline only. Distance-based, so most damaged by irrelevant features — makes reduction benefits most visible |

**During search**, use a deliberately light configuration (`n_estimators=30–50`). Thousands of
evaluations make model cost the binding constraint. Validate the final Pareto front at full
strength.

> **Limitation to state:** the selected subsets are optimal *for a Random Forest*. A different
> classifier family might prefer different features. This conditioning should be acknowledged
> rather than glossed over.

---

## 7. Feature selection taxonomy — where this sits

| Family | Mechanism | Sees interactions? | Examples |
|--------|-----------|--------------------|----------|
| Filter | Statistics only, no model | **No** | chi², mutual information |
| **Wrapper** | Trains a model per candidate subset | **Yes** | **SPEA2**, NSGA-II, PSO, GA |
| Embedded | Selection during training | Partly | LASSO, tree importances |

**This project uses a wrapper method**, and the dataset's structure justifies it: since the same
characters are counted across five URL sections, redundancy is guaranteed, and only a method that
evaluates subsets *as subsets* can detect it.

Anticipated question — *"why not Random Forest feature importance?"* — it produces a ranking, not
a subset, and still requires an arbitrary cutoff. It also cannot optimise for cost at all.

---

## 8. Justification summary

For the methodology chapter, condensed:

1. The problem has **three genuinely conflicting objectives**, so it is inherently
   multi-objective; a weighted sum would impose arbitrary weights and is provably unable to reach
   concave regions of the front.
2. **Wrapper methods** are indicated because the dataset's repeated character counts across URL
   sections guarantee redundancy that filters cannot detect.
3. **SPEA2** is selected for its consistent treatment of duplicate solutions, which directly
   matches this dataset's structure; its computational overhead is irrelevant in a wrapper
   setting where classifier training dominates runtime.
4. **NSGA-II** is run under identical operators and budget, isolating the fitness-assignment and
   diversity mechanisms as the only differences — a controlled comparison with a
   mechanistically-motivated hypothesis.
5. At **111 features**, documented MOEA scalability limitations do not apply.
