# 02 — The Detection Cost Model

How objective f₁ (detection time) is defined. This is the project's main methodological
contribution, so it needs to be defensible.

---

## 1. Why not just count features

Almost every feature-selection paper uses **number of features** as the proxy for cost. For this
dataset that proxy is not merely imprecise — it is **actively wrong**, and can rank two subsets
in the opposite order to reality.

Consider two candidate subsets:

| Subset | Features | Actual detection time |
|--------|----------|-----------------------|
| **A** | 40 features, all URL-derived | **~0.01 ms** |
| **B** | 5 features, three requiring WHOIS and DNS | **~250 ms** |

Counting features says B is eight times better. Reality says A is roughly **25,000× faster.**

Since objective 1 is explicitly *"minimise the time required to detect a phishing website"*,
using feature count would optimise the wrong quantity entirely.

---

## 2. Cost is driven by lookups, not features

The key structural insight: **features do not each carry their own cost — they inherit the cost
of the network lookup they depend on, and they share it.**

If a subset already requires a DNS A-record query for `qty_ip_resolved`, then adding
`ttl_hostname` costs approximately nothing — the same query returns both.

So cost is **not additive over features.** It is additive over the *distinct lookup types* the
subset requires:

```
detection_cost(subset) = Σ  latency(t)
                        t ∈ distinct lookup types required by subset
```

This is more realistic than per-feature costing, and explaining why is a genuine contribution to
the write-up.

---

## 3. Lookup tiers

Each of the 111 features is assigned to exactly one tier, by what it physically requires.

| Tier | Operation | Latency | Features served |
|------|-----------|---------|-----------------|
| **T0** | URL string parsing | **0.01 ms** | All 96 URL-derived features, plus `email_in_url` and `url_shortened` |
| **T1** | DNS — A record | 30 ms | `qty_ip_resolved`, `time_response`, `ttl_hostname` |
| **T2** | DNS — NS record | 30 ms | `qty_nameservers` |
| **T3** | DNS — MX record | 30 ms | `qty_mx_servers` |
| **T4** | DNS — TXT / SPF | 30 ms | `domain_spf` |
| **T5** | WHOIS | 200 ms | `time_domain_activation`, `time_domain_expiration` |
| **T6** | TLS handshake | 150 ms | `tls_ssl_certificate` |
| **T7** | HTTP fetch / follow redirects | 300 ms | `qty_redirects` |
| **T8** | Search-index API | 500 ms | `url_google_index`, `domain_google_index` |
| **T9** | IP→ASN lookup | 50 ms | `asn_ip` |
| | **All tiers — full feature set** | **≈ 1,320 ms** | |

### Two important refinements

**`email_in_url` and `url_shortened` are T0, not external.** Despite sitting in the external
group in the dataset's own documentation, both are string operations — checking for an email
pattern, and matching against a known shortener list. Placing them in T0 is more accurate. Note
this reclassification explicitly in the report; it is the kind of detail that shows the cost
model was reasoned about rather than assumed.

**T0 is charged once, not per feature.** The URL is parsed a single time; all 96 counts are
then trivial array operations over the parsed components. Charging 0.01 ms per feature would
overstate cheap subsets by two orders of magnitude.

---

## 4. Calibrating against real data

The tier latencies above are representative estimates, and estimates need justification.

**One of them can be measured directly from the dataset itself.** The `time_response` column
records the actual domain-lookup time for every row — values in your data range from roughly
0.2 to 1.7 seconds.

This gives an empirical anchor:

1. Compute the distribution of `time_response` across the dataset (median, mean, p95).
2. Use the **median** as the calibration point for the DNS tiers.
3. Report the distribution in the methodology section as evidence the cost model is grounded in
   the data rather than invented.

> Note the tension worth discussing: the observed `time_response` values are far higher than the
> 30 ms assigned to T1. Those measurements were taken during bulk dataset construction — likely
> without caching, from a single location, at scale. A production detector with a warm DNS
> cache would be much faster. **State which regime you are modelling.** Either choice is
> defensible; leaving it unstated is not.

---

## 5. Sensitivity analysis — required

Because the tier latencies are chosen rather than measured, the results must be shown to be
robust to them.

**Run the full SPEA2 optimisation under three cost scenarios:**

| Scenario | Description | Effect |
|----------|-------------|--------|
| **Optimistic** | Warm caches, fast network — all external tiers ÷ 3 | External features become more affordable |
| **Baseline** | The table in section 3 | The headline result |
| **Pessimistic** | Cold cache, slow network — all external tiers × 3 | External features heavily penalised |

If the selected feature subsets remain broadly stable across all three, the conclusion is robust
and you can say so. If they shift dramatically, that is *also* a finding — it means the
cost/accuracy trade-off is genuinely sensitive to deployment conditions, which is worth
reporting honestly.

Either outcome strengthens the report. Skipping this step leaves an obvious line of attack.

---

## 6. Implementation shape

Conceptually, in pseudocode:

```
FEATURE_TIER = {
    'qty_dot_url':            'T0',
    ... all 96 URL features:  'T0',
    'email_in_url':           'T0',      # reclassified
    'url_shortened':          'T0',      # reclassified
    'qty_ip_resolved':        'T1',
    'time_response':          'T1',
    'ttl_hostname':           'T1',
    'qty_nameservers':        'T2',
    'qty_mx_servers':         'T3',
    'domain_spf':             'T4',
    'time_domain_activation': 'T5',
    'time_domain_expiration': 'T5',
    'tls_ssl_certificate':    'T6',
    'qty_redirects':          'T7',
    'url_google_index':       'T8',
    'domain_google_index':    'T8',
    'asn_ip':                 'T9',
}

TIER_LATENCY_MS = {
    'T0': 0.01,  'T1': 30,  'T2': 30,  'T3': 30,  'T4': 30,
    'T5': 200,   'T6': 150, 'T7': 300, 'T8': 500, 'T9': 50,
}

FUNCTION detection_cost(mask):
    selected      = feature names WHERE mask == 1
    tiers_needed  = SET of FEATURE_TIER[f] for f in selected
    tier_cost     = SUM of TIER_LATENCY_MS[t] for t in tiers_needed
    RETURN tier_cost + PER_FEATURE_EPSILON_MS * COUNT(selected)
```

**This requires no model training.** Objective f₁ is computed directly from the bit mask in
microseconds, which means multi-objective optimisation costs no more per evaluation than
single-objective did — the classifier fit still dominates entirely.

### 6.1 The per-feature tiebreak — added after a pilot run, and why

**Implemented.** `PER_FEATURE_EPSILON_MS = 1e-5` — two orders of magnitude below the smallest
tier latency (T0 = 0.01 ms), so it can only break ties between subsets requiring the *same*
tiers; it never reorders subsets that differ in which tiers they need.

**Why it was necessary.** Tier cost is charged once per tier regardless of how many features of
that tier are kept — a 98-feature T0 subset and a 1-feature T0 subset both cost 0.01 ms. An
early pilot run confirmed this in practice: SPEA2 correctly minimised network-tier cost
(1,320 ms → 60 ms) but kept 45–67 of the 111 features, since nothing punished the redundant
extras once a tier's flat cost was already being paid.

This is not an artificial penalty — every feature genuinely costs *something* nonzero to
compute (one more counter read from the parsed URL) — but the coarse tier granularity hid that
cost entirely. The epsilon restores it at a scale that cannot distort the tier-level trade-off
the project is actually about.

**What the epsilon can and cannot do.** It reliably drops features whose removal changes
*nothing* about model accuracy — chiefly the 18 constant/near-constant columns identified in the
data audit (`docs/01-dataset.md` §2), since a Random Forest cannot split on a constant feature
and the confusion matrix is identical with or without it. It cannot force the removal of a
feature that genuinely moves accuracy, even slightly — which is the correct behaviour: the
tiebreak should never override real predictive signal.

---

## 7. Why this makes the project's best result possible

With this cost model, a specific and dramatic outcome becomes reportable:

> **If SPEA2 finds a Pareto-optimal solution containing only T0 features, that solution requires
> zero network calls.** Detection cost drops from ~1,320 ms to ~0.01 ms — a factor of roughly
> 130,000 — and the model becomes deployable directly inside a browser extension, evaluating a
> link before the user clicks it.

That is a qualitative change in what the system can do, not just a smaller number in a table.

Counting features could never have surfaced that finding. It is visible only because cost was
modelled by what the features physically require.

---

## 8. Deployment profiles for reporting results

A Pareto front is a set of solutions, and readers need help interpreting it. Rather than
presenting an undifferentiated cloud of points, extract the best solution under three realistic
latency budgets:

| Profile | Budget | Scenario |
|---------|--------|----------|
| **Browser extension** | ≤ 1 ms | Must evaluate a link instantly, before click. Forces T0-only solutions |
| **Email gateway** | ≤ 500 ms | Scanning inbound mail; some lookups affordable |
| **Offline audit** | unlimited | Batch analysis; accuracy is all that matters |

For each profile, report the front's best-recall solution within budget, along with its feature
count, FNR, FPR and tier composition.

This turns the Pareto front from an abstract plot into a set of concrete engineering
recommendations — and it directly demonstrates the value of multi-objective optimisation over a
single-objective run, which could only ever have produced one of these three answers.

---

## 9. Limitations to state

- Tier latencies are **representative estimates**, not measurements taken on production
  infrastructure — hence the mandatory sensitivity analysis in section 5.
- Real-world lookups can be **parallelised**, so a subset needing four independent DNS queries
  might cost one round-trip rather than four. The additive model is therefore a *conservative
  upper bound* on external cost. Say so.
- Caching is not modelled. A detector seeing many URLs from the same domain would amortise
  lookups heavily.
- Classifier inference time is excluded, as it is negligible (microseconds) beside network
  latency and near-identical across subsets.
