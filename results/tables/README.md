# Results Index

Running log of what has been produced, with the headline numbers. Full detail is in the
individual JSON/txt files in this directory and in `results/fronts/`.

---

## Phase 01 — Data audit

`01_data_audit.txt` (full output of `src/audit_data.py`)

- Shape confirmed: 58,645 × 112, matches docs exactly
- Class balance: 47.7% legitimate / 52.3% phishing
- **`−1` sentinel finding:** WHOIS features (`time_domain_activation`,
  `time_domain_expiration`) missing in 32–35% of rows. DNS-family failures
  (`ttl_hostname`, `qty_ip_resolved`, `time_response`, `asn_ip`) show lift ≈ 0.5 —
  a failed lookup predicts *legitimate*, the opposite of intuition. Documented as a
  likely collection artifact in `docs/01-dataset.md` §6.
- 18 of 111 features are constant or near-constant (dead weight)
- `time_response` calibration: median 477ms, mean 939ms — far above the modelled
  T1 = 30ms, because the dataset was built with cold caches at scale. Our cost model
  deliberately targets the warm/production regime, not bulk-extraction conditions.

## Phase 03 — Baselines, all 111 features

`baselines_all_111_features.json`

| Classifier | Accuracy | Recall | FNR | FPR | Latency (ms/sample) |
|---|---|---|---|---|---|
| **Random Forest** | 0.9571 | **0.9639** | **0.0361** | **0.0504** | 0.0046 |
| SVM | 0.9162 | 0.9323 | 0.0677 | 0.1014 | 0.7909 |
| KNN | 0.9309 | 0.9338 | 0.0662 | 0.0721 | 0.0431 |

**This is the control point.** Random Forest wins on every metric and is used as the
classifier inside the SPEA2/NSGA-II search. All-111-features detection cost = 1,320.01 ms
(dominates model latency by ~4 orders of magnitude — feature extraction, not inference,
is the real bottleneck).

## Phase 05 — SPEA2 search

`../fronts/spea2_pop100_gen50_seed*.{json,npz}` — 5 seeds, pop=100, gen=50

Seed 1 (representative; full multi-seed summary pending):

- Hypervolume: 1.1827
- Front size: 100 non-dominated solutions
- Features per solution: 23–60 (median 36) — down from 111
- Cost: 30.01 – 370.01 ms — down from 1,320.01 ms baseline
- FNR: 4.4% – 9.1% (recall 90.9% – 95.6%)
- FPR: 5.3% – 12.3%

**Tier composition (seed 1, averaged across the 100-solution front):**

| Tier | What | Available | Avg. kept | % kept |
|---|---|---|---|---|
| T1 | DNS A record | 3 | 2.88 | **96%** |
| T3 | DNS MX record | 1 | 0.58 | 58% |
| T5 | WHOIS | 2 | 0.63 | 31.5% |
| T9 | IP→ASN | 1 | 0.29 | 29% |
| T4 | DNS TXT/SPF | 1 | 0.05 | 5% |
| **T6** | **TLS handshake** | 1 | 0.00 | **0%** |
| **T7** | **HTTP redirects** | 1 | 0.00 | **0%** |
| **T8** | **Google index** | 2 | 0.00 | **0%** |
| T0 | URL text (98 features) | 98 | 31.09 | 31.7% of the 98 |

**Reading this:** DNS resolution (T1) is almost always worth its 30ms — it survives in
96% of Pareto-optimal solutions. TLS, redirects, and Google indexing are *never* worth
their cost (300–500ms) — dropped in 100% of solutions. WHOIS is a genuine trade-off,
kept in about a third of solutions (consistent with the `−1` audit finding that a third
of WHOIS values are missing anyway).

**No network-free (T0-only) solution appears on this front.** The best available
solution at any budget uses 41 features and 4 tiers (T0, T1, T3, T5) at 260ms, with
95.58% recall / 6.86% FPR — beats the recall target (≥95%) but misses the tight FPR
target from literature (≤2%). This tension is a real, reportable finding, not an error.

**Feature-count floor improved with the epsilon fix:** 20-gen pilot without epsilon
kept 45–67 features; with the per-feature tiebreak epsilon (§2 of cost model) and the
full 50-generation budget, the floor dropped to 23–60. Confirms the tiebreak is working
as designed — see `docs/02-cost-model.md` §6.1.

## Classical reduction baselines

`classical_baselines.json` — what SPEA2/NSGA-II have to beat

| Method | Features/components | Accuracy | Recall |
|---|---|---|---|
| PCA (95% variance) | 40 components | 0.9447 | 0.9525 |
| Mutual-information filter | top 25 | 0.8793 | **0.8590** |
| *(all 111 features, RF)* | *111* | *0.9571* | *0.9639* |

**This is a strong, clean result for the project's core argument.** The filter method —
which scores each feature independently and cannot see interactions — loses ~10.5
points of recall relative to the full-feature baseline at just 25 features. PCA holds
up much better (only ~1 point lost at 40 components) but its output is 40 *blended*
combinations of all 111 original features — uninterpretable, cannot be turned into a
list of "checks worth computing."

**Compare to SPEA2 (seed 1, single representative solution):** 41 features, 95.58%
recall — beats the filter by **~10 points of recall** using roughly the same feature
count, and is competitive with PCA's 95.25% while remaining fully interpretable (named
features, not blended components). This is exactly the argument
`docs/03-methodology.md` §7 predicted: wrapper methods matter because this dataset's
heavy redundancy (same character counts across 5 URL sections) hides real interactions
that a filter is blind to.

Top 10 features by mutual information alone (for reference — note how different this
list is from SPEA2's high-frequency features in the table above):
`directory_length, qty_slash_url, qty_slash_directory, qty_dot_directory, file_length,
qty_dot_file, qty_hyphen_directory, length_url, qty_underline_file, qty_space_file`
— entirely T0 (URL-text) features. The filter never gets the chance to consider that
T1 (DNS) is worth its cost, because mutual information doesn't know what anything costs.

## SPEA2 — full 5-seed results (COMPLETE)

`spea2_pareto_analysis.json`, `spea2_summary.json`, figures in `../figures/`

**Stability across seeds:**

| | Mean | Std | Min | Max |
|---|---|---|---|---|
| Hypervolume | 1.1546 | 0.0665 | 1.0225 (seed 3) | 1.2032 (seed 5) |
| Median features/solution | 39.8 | 3.5 | — | — |
| Runtime per seed | 355.9s | 18.0s | — | — |

Seed 3 is a mild outlier (smaller front, 87 vs ~100). Normal stochastic variation for a
population-based search — report honestly rather than cherry-picking the best seed.

**Pooled non-dominated front (5 seeds combined, re-filtered):** 201 solutions from 487 raw.

**Tier composition (pooled, 201 solutions):**

| Tier | What | % kept |
|---|---|---|
| T1 (DNS A record) | 77.8% | Near-essential |
| T3 (DNS MX) | 46.8% | Moderate value |
| T5 (WHOIS) | 37.8% | Moderate, despite 32–35% missingness |
| T2 (DNS NS) | 44.8% | Moderate |
| T9 (ASN) | 25.9% | Some value |
| T6 (TLS) | 7.5% | Low value |
| T7 (redirects) | 5.0% | Low value |
| T4 (SPF) | 3.0% | Negligible |
| **T8 (Google index)** | **0.2%** | **Essentially never worth it** |

**Most robust features** (present in most of the 201 solutions): `directory_length`
(94.0%), `domain_length` (85.1%), `qty_dot_domain` (84.1%), `qty_ip_resolved` (83.6%,
tier T1), `ttl_hostname` (82.6%, tier T1). Both of the top-5 network features are T1 —
DNS resolution info is clearly the highest-value lookup in the dataset.

**Deployment profiles (pooled front):**

| Profile | Features | Cost | Recall | FPR | Meets both targets? |
|---|---|---|---|---|---|
| **browser_extension** (≤1ms) | 48 | 0.01 ms | 89.70% | 10.04% | No — recall target ≥92%, FPR target ≤5% |
| **email_gateway** (≤500ms) | 37 | 490.01 ms | 95.76% | 6.14% | Recall met (≥95%); FPR missed (≤2%) |
| **offline_audit** (unlimited) | 37 | 490.01 ms | 95.76% | 6.14% | Recall close (target ≥96%); FPR missed (≤1%) |

**Honest finding — the FPR targets from published literature are not met at any
budget on this front.** Recall targets are achievable (and exceeded at the email-gateway
tier), but FPR stays around 6–10% rather than the 0.1–2% range reported by some cited
papers. Two plausible explanations to discuss in the report: (1) those papers may use
different classifiers/datasets not directly comparable, or (2) an unweighted Pareto
search naturally lands where FNR and FPR are jointly minimised, not where FPR
specifically is minimised — a solution further down the front trades some recall for
much lower FPR, and that trade is available on request (see feature-frequency table)
even though it isn't the "best recall" profile picked by default.

**Network-free (browser-deployable) solutions: 27 of 201 (13.4%).** Best one: 48
features, 89.70% recall, 10.04% FPR, effectively 0ms cost. This is the ceiling for
detection using only URL text with zero network calls — falls short of the browser
target but is a genuine, reportable number, not a failure.

## Required figures — all 4 generated

`../figures/`: `f1_hypervolume_convergence.png`, `f2_pareto_projections.png`,
`f6_feature_frequency_spea2.png`, `f7_cost_recall_tradeoff.png`

- **F1** shows convergence plateauing around generation 25–30 — confirms 50 generations
  was a sufficient budget, not wasted compute.
- **F2** shows three visually distinct cost clusters (≈0.01ms / ≈30ms / ≈200–1300ms),
  a direct visual consequence of the tier structure.
- **F7** shows recall plateauing right around the email-gateway budget line — paying
  more than ~500ms buys almost nothing further.

## NSGA-II — full 5-seed results (COMPLETE)

`nsga2_pareto_analysis.json`, `nsga2_summary.json`

### ⚠ Pre-registered hypothesis NOT confirmed — reported honestly

`docs/03-methodology.md` §4.4 predicted SPEA2 would outperform NSGA-II here because its
consistent treatment of duplicate individuals should suit this dataset's redundant
feature structure. **The data does not support this.**

| | SPEA2 | NSGA-II | 
|---|---|---|
| Hypervolume (mean) | 1.1546 | **1.1863** |
| Hypervolume (std) | 0.0665 | **0.0080** — ~8× tighter |
| Worst seed HV | 1.0225 | 1.1813 |
| Pooled front size | 201 | 241 |
| Network-free solutions | 27 (13.4%) | 32 (13.3%) |
| Browser-ext. recall / FPR | 89.70% / 10.04% | 89.77% / 10.21% |
| Email-gateway recall / FPR | 95.76% / 6.14% (37 feat., 490ms) | 95.77% / 6.43% (46 feat., 290ms) |

**NSGA-II wins on hypervolume and is dramatically more consistent across seeds.**
`f1_hypervolume_convergence.png` shows this clearly: NSGA-II (red) converges faster,
plateaus higher, and its std band is visibly tighter than SPEA2's (blue) throughout
all 50 generations.

**At the level that matters for deployment, though, the two are nearly identical** —
`f2_pareto_projections.png` shows their pooled fronts overlapping almost completely,
and the deployment-profile numbers above differ by around 0.1–0.3 percentage points.
Both algorithms independently discover essentially the same underlying trade-off
surface; NSGA-II just does so more *reliably* run-to-run.

**Why the hypothesis likely failed:** both algorithms already use
`eliminate_duplicates=True` at the population level, which removes exact duplicate
*individuals* before they reach the fitness function. That may already capture most of
the benefit the duplicate-handling argument was pointing at, leaving little room for
SPEA2's finer-grained strength/density mechanism to add further advantage — and
possibly costing it some stability, since seed 3 was a clear underperformer
(HV=1.0225, front size only 87) pulling down SPEA2's mean and inflating its std.

**This is being reported as a negative result, not adjusted or hidden.** Per
`docs/06-evaluation-protocol.md` §8: "a pre-registered hypothesis that fails, reported
honestly with analysis, is worth more than a tuned result." The methodology doc has
been updated to reflect this outcome rather than the untested prediction.

**Tier composition and feature frequency are consistent with SPEA2's findings** — T1
(DNS) kept ~78%, T8 (Google index) kept <2%, `domain_length`/`ttl_hostname`/
`qty_ip_resolved` all appear in >75% of NSGA-II's front too. The trade-off structure
itself is robust across algorithms; only the search reliability differs.

## Weighted-sum PSO baseline (COMPLETE) — demonstrates the scalarisation limitation

`pso_weighted_baseline.json` — 4 weight settings, each a single isolated point:

| Weighting | Features | Cost | Recall | FPR |
|---|---|---|---|---|
| balanced (1/3, 1/3, 1/3) | 51 | 30.01 ms | 91.76% | 8.55% |
| cost_dominant | 50 | 0.01 ms | 89.54% | 9.95% |
| recall_dominant | 60 | 260.01 ms | 95.12% | 6.39% |
| fpr_dominant | 54 | 280.01 ms | 95.04% | 5.91% |

**The argument, made concrete:** 4 separate optimisation runs (each requiring the
weights to be guessed in advance) produced 4 isolated points. SPEA2/NSGA-II's pooled
fronts produced 201–241 non-dominated solutions **in a single run each**, spanning the
same cost range these 4 points only sample sparsely. Every one of the 4 weighted points
lands on or very near the SPEA2/NSGA-II front (compare to `f2_pareto_projections.png`)
— they are valid solutions, just an impractically sparse way to map the trade-off
compared to letting a multi-objective algorithm return the whole curve at once.

*(Note: this run's wall-clock time in the raw log is inflated — the machine slept
partway through, and `time.time()` counted the sleep duration. The actual computational
cost is consistent with ~1,500 evaluations per weight setting, same order as one
SPEA2/NSGA-II seed.)*

## Required figures — all done, now with both algorithms overlaid

`../figures/`: `f1_hypervolume_convergence.png` (SPEA2 vs NSGA-II), `f2_pareto_projections.png`
(both overlaid), `f6_feature_frequency_spea2.png`, `f6_feature_frequency_nsga2.png`,
`f7_cost_recall_tradeoff.png` (both overlaid)

## Still running / not yet done

- [ ] Cost sensitivity analysis (optimistic/baseline/pessimistic) — launching next
- [ ] Final report update with these results
- [ ] Git commit + push

## Final validation on the sealed test set (COMPLETE)

`final_validation_test_set.json` — every method's chosen subset, retrained with a
200-tree Random Forest on the full training split and evaluated once on the sealed
20% test set (11,729 websites). This is the table used in the report (§7).

| Method | Features | Cost | Accuracy | Recall | FPR |
|---|---|---|---|---|---|
| All features | 111 | 1,320 ms | 95.71% | 96.39% | 5.04% |
| **SPEA2 (email gateway)** | **37** | **490 ms** | **95.85%** | **96.52%** | **4.89%** |
| SPEA2 (no lookups) | 48 | 0.01 ms | 90.51% | 89.75% | 8.66% |
| NSGA-II (email gateway) | 46 | 290 ms | 95.61% | 96.26% | 5.11% |
| NSGA-II (no lookups) | 53 | 0.01 ms | 90.73% | 90.23% | 8.71% |
| PSO weighted (recall) | 60 | 260 ms | 95.37% | 95.86% | 5.16% |
| PSO weighted (cost) | 50 | 0.01 ms | 90.59% | 89.98% | 8.75% |
| PCA (95% variance) | 40 comp. | 1,320 ms | 94.47% | 95.25% | 6.39% |
| MI filter (top 25) | 25 | 0.01 ms | 87.93% | 85.90% | 9.86% |

**Headline:** SPEA2's gateway subset matches the full-feature baseline on unseen data
(recall 96.52% vs 96.39%, FPR 4.89% vs 5.04%) with 67% fewer features and 63% lower
detection cost.

## Cost sensitivity analysis (COMPLETE)

`sensitivity_analysis.json` — SPEA2, 3 seeds per scenario, network latencies scaled ÷3 / ×1 / ×3.

Best recall found: optimistic 95.7–95.8%, baseline 95.6–95.7%, pessimistic 95.6–95.8%.
**The achievable recall is stable across a 9× range of latency assumptions** — the result
does not depend on the exact tier latencies chosen.

Exact feature sets vary a lot between seeds (within-scenario Jaccard ≈ 0.15), because the
dataset's redundant features allow many near-equivalent subsets. The union of features
chosen under optimistic vs pessimistic costs overlaps at Jaccard 0.73.
