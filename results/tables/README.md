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

## Still running / not yet done

- [ ] SPEA2 seeds 2–5 (in progress)
- [ ] NSGA-II, 5 seeds — comparison algorithm
- [ ] Multi-seed Pareto analysis (pooled front, stability table)
- [ ] Required figures (F1 hypervolume convergence, F2 projections, F6 feature
      frequency, F7 cost/recall trade-off)
- [ ] Cost sensitivity analysis (optimistic/baseline/pessimistic)
