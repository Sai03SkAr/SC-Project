# 08 — Objective Benchmarks

Published values for each of this project's three objectives, and the targets they imply.

Where [`04-literature-review.md`](04-literature-review.md) surveys the field, this document
answers one operational question: **what numbers should we be aiming for, and how will we know
if our result is any good?**

---

> ## ⚠ Same verification rule applies
>
> Figures below were gathered by literature search, several from abstracts and summaries.
> **Verify against the primary source before any number enters a submitted report.** They are
> reliable enough to set targets against; they are not yet reliable enough to cite.

---

## Objective 1 — Detection time (minimise f₁)

### The feature taxonomy this field uses

Published work classifies phishing features into three families. Ours map cleanly onto it —
worth adopting the standard vocabulary in the report:

| Standard term | What it needs | Our tiers | Our features |
|---------------|---------------|-----------|--------------|
| **Lexical** | The URL string only | **T0** | **96** |
| **Host-based** | WHOIS, DNS, hosting, server details | **T1–T9** | **15** |
| Content-based | Download and parse the page — HTML, JS | — | **0** — not in this dataset |

That the dataset contains **no content-based features** is a genuine advantage worth stating:
detection can occur *before the page loads*, since nothing needs to be fetched.

### Published latencies

| Approach | Latency | Note |
|----------|---------|------|
| Boolean-algebra method | **~1 ms** | Among the fastest reported |
| Lightweight hybrid MLP | **1.2 ms/URL** | 4,200 URLs/sec throughput |
| Some ensemble classifiers | **sub-millisecond** | Described as suitable for real-time browser deployment |
| Browser-extension implementation | **< 250 ms** | End-to-end, user-facing |
| UI response budget | **< 100 ms** | Interface latency target |
| ADUIN (deep URL model) | **210 ms** | Under high load |

### Directly comparable prior result

One reported optimisation achieved a **28% speedup and 19% less memory while cutting features
from 52 to 31** — a ~40% reduction.

**This is the closest published analogue to what this project does.** It is the number to beat,
and a fair comparison point in the results chapter.

### Confirmation of the cost model's premise

Published work states explicitly that retrieving WHOIS information "involves querying WHOIS
servers, which in turn increase the feature processing time." The premise behind
[`02-cost-model.md`](02-cost-model.md) is therefore not an assumption we invented — it is the
field's own stated reasoning.

### The caching caveat — must be acknowledged

At least one system reaches ~210 ms by **caching host-based indicators in in-memory lookup
tables, eliminating repeated external WHOIS and DNS queries at inference time.**

This matters for honesty about our cost model:

- Our tier latencies model the **cold-cache** case — the first time a domain is seen.
- A production system with a warm cache pays far less for host-based features.
- Our cost model is therefore a **conservative upper bound** on external cost.

State this explicitly. It does not undermine the approach — a browser extension encountering a
freshly-registered phishing domain has, by definition, a cold cache, which is exactly the case
that matters most.

### Targets for this project

| Deployment profile | Budget | Justified by |
|--------------------|--------|--------------|
| **Browser extension** | **≤ 1 ms** | Matches the fastest published methods; forces T0-only solutions |
| **Email gateway** | **≤ 500 ms** | Comfortably inside the <250 ms extension figure plus headroom |
| **Offline audit** | unlimited | Accuracy-only regime |

Our all-features baseline of ~1,320 ms sits **above every published latency figure** — which is
itself a finding: using all 111 features is not deployable in real time at all.

---

## Objective 2 — Recall / false negatives (minimise f₂)

### Published recall and FNR

| Approach | Recall | FNR | Note |
|----------|--------|-----|------|
| Random Forest (PILFER) | ~96% | **4%** | >96% accuracy, 0.1% FPR |
| 2025 Scientific Reports study | **95.4%** | 4.6% | 97.2% precision |
| RL / Deep Q-Network | **94%** | 6% | 2% FPR, 5,000 emails |
| Classical ML + feature extraction | ~92% | 8% | 0.1% FPR |
| Profile-based methods | ~70% | 30% | Markedly weaker |

### What this implies

**Competitive FNR sits in the 4–6% band; recall of 94–96% is the norm.** Anything below ~90%
recall is not competitive. Anything above ~99% should prompt a leakage check before celebration.

### Target for this project

| | Target |
|---|---|
| **Offline profile** (unlimited cost) | **FNR ≤ 5%** (recall ≥ 95%) — matching published work |
| **Browser profile** (≤ 1 ms, T0 only) | **FNR ≤ 8%** would be a strong result |

That second row is the interesting one. If a **URL-only, zero-lookup** subset reaches recall
comparable to published methods that use host-based features, the result is not merely "we
reduced features" — it is *"host-based lookups can be dropped entirely at modest recall cost,
making real-time in-browser detection viable."*

---

## Objective 3 — Precision / false positives (minimise f₃)

### Published FPR

| Approach | FPR |
|----------|-----|
| Random Forest (PILFER) | **0.1%** |
| Classical ML + feature extraction | **0.1%** |
| RL / Deep Q-Network | **2%** |
| Optimal-threshold study | **3.5%** |
| Content-based method (improved) | 14.0% → **7.6%** |

**The best published systems achieve FPR around 0.1%.** That is a demanding bar — one false
alarm per thousand legitimate sites.

### ⚠ A finding that challenges this project's stated priority

Read carefully, because it affects how the results should be framed.

Published work states that false positives are "the greatest concern in terms of preserving user
confidence," and that **commercial phishing detection products typically prioritise users'
sensitivity to false positives — aspiring to high precision while maintaining good recall.**

**That is the opposite emphasis to this project's stated objective 2** ("make sure the site
doesn't get classified as safe"), which prioritises recall.

Neither is wrong; they reflect different deployment contexts:

| Priority | Reasoning | Fits |
|----------|-----------|------|
| **Recall first** (this project) | A missed phishing site can cost credentials and money. Asymmetric harm | High-risk users; banking; a warning-rather-than-blocking UI |
| **Precision first** (commercial) | Blocking real sites destroys trust; users disable the tool, leaving them worse protected | Mass-market products where abandonment is the real risk |

**Handle this in the report by naming the tension explicitly.** Something like:

> *Commercial systems prioritise precision to preserve user trust. This project prioritises
> recall on the grounds that the harm from a missed phishing site is asymmetric. The
> multi-objective formulation makes this a selection decision rather than an assumption — the
> Pareto front contains solutions for both stances, and the deployment profiles report each.*

Volunteering this reads as command of the literature. Being asked about it unprepared does not.

### Target for this project

| | Target |
|---|---|
| Baseline expectation | **FPR ≤ 5%** — usable |
| Competitive | **FPR ≤ 2%** — in line with modern published work |
| Excellent | **FPR ≤ 0.5%** — approaching best-reported |

---

## The strongest argument this research provides

One line from the literature is worth the whole search:

> There is **no single universally accepted industry threshold** for the false-positive /
> false-negative balance — it is a **context-dependent** trade-off.

And separately: every model has an inherent FPR/FNR trade-off, and tuning it "impacts the user
acceptance rate of any phishing detection model deployment."

**This is the formal justification for the entire multi-objective approach.**

If no universal threshold exists, then baking one in as a fixed weight — which is exactly what a
single-objective weighted sum does — is **methodologically wrong, not merely inelegant.** The
correct response is to generate the full trade-off surface and let the deployment context select
from it.

That argument belongs in the introduction, not buried in the discussion. It converts "we used a
multi-objective algorithm" into "a single-objective formulation could not have answered this
question."

---

## Consolidated targets

| Profile | f₁ cost | f₂ FNR | f₃ FPR | Recall | Interpretation |
|---------|---------|--------|--------|--------|----------------|
| **Browser extension** | ≤ 1 ms | ≤ 8% | ≤ 5% | ≥ 92% | Zero network calls; deployable in-page |
| **Email gateway** | ≤ 500 ms | ≤ 5% | ≤ 2% | ≥ 95% | Competitive with published work |
| **Offline audit** | unlimited | ≤ 4% | ≤ 1% | ≥ 96% | Matches best published results |
| *All 111 features* | *~1,320 ms* | *baseline* | *baseline* | *baseline* | *Too slow for real-time — a finding in itself* |

---

## How to use this in the results chapter

1. **Report our numbers beside these.** "Our email-gateway profile achieves X% recall at Y% FPR,
   comparable to published values of 94–96% recall and 0.1–3.5% FPR."
2. **Lead with the latency comparison**, since it is where this project is most distinctive.
   Most papers report accuracy and stay silent on cost.
3. **Use the 52→31 feature result as the direct comparison** for the reduction itself.
4. **Name the precision/recall priority tension** rather than waiting to be asked.
5. **Frame the absence of a universal threshold** as the justification for multi-objective
   optimisation, in the introduction.

---

## Remaining gaps

Honest about what this research did not settle:

- **No published multi-objective phishing feature-selection results were found.** The closest
  precedent is the intrusion-detection study in [`04-literature-review.md`](04-literature-review.md).
  This is a gap in the literature — and therefore a genuine contribution available to this
  project, worth stating as such.
- **No hypervolume figures exist for phishing feature selection** to compare against, so
  hypervolume can only be compared between our own algorithms, not against published work.
- Reported latencies mix end-to-end system time with model-inference time. Be explicit that ours
  measures **feature-extraction cost**, which is the dominant term but not the only one.
