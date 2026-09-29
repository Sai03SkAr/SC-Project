# 01 — Dataset Reference

Complete reference for the data used in this project.

---

## 1. Primary dataset — Vrbančič Phishing Dataset

### Source

| | |
|---|---|
| **Repository** | https://github.com/GregaVrbancic/Phishing-Dataset |
| **Interactive preview** | https://gregavrbancic.github.io/Phishing-Dataset/ |
| **Mendeley DOI** | `10.17632/72ptz43s9v.1` |
| **Paper** | Data in Brief, Vol. 33, 2020 — DOI `10.1016/j.dib.2020.106438` |
| **Licence** | Open — free download, no account required |

### Citation

> G. Vrbančič, I. Fister Jr., V. Podgorelec. *Datasets for Phishing Websites Detection.*
> Data in Brief, Vol. 33, 2020. DOI: 10.1016/j.dib.2020.106438

### Direct download URLs

```
https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_small.csv
https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv
```

### The two variants

| | `dataset_small.csv` | `dataset_full.csv` |
|---|---|---|
| **Total rows** | 58,645 | 88,647 |
| **Legitimate** (`0`) | 27,998 | 58,000 |
| **Phishing** (`1`) | 30,647 | 30,647 |
| **Class balance** | ~48% / 52% — near-balanced | ~65% / 35% — imbalanced |
| **Features** | 111 | 111 |
| **Total columns** | 112 | 112 |
| **Role in this project** | **Primary** — all main experiments | Secondary — optional class-imbalance study |

Note the phishing count is identical in both files (30,647). The "small" variant is the full
variant with legitimate rows down-sampled to produce balance.

### Column structure

**112 columns total = 111 features + 1 target.** There is no index column.

| Position | Column | Notes |
|---|---|---|
| 1–111 | Feature columns | All numeric |
| 112 | `phishing` | Target — `0` = legitimate, `1` = phishing |

> **Encoding warning.** The target is `0`/`1`, *not* `−1`/`1`. This differs from the UCI dataset
> described in section 3. If both datasets are used, harmonize the encoding early.

---

## 2. The 111 features

Features are organized by **which part of the URL they measure**. Most are simple character
counts — the same set of characters counted within different URL components.

### Structural summary

| Group | Count | Computed from | Cost |
|-------|-------|---------------|------|
| Whole URL | 19 | URL string | Instant |
| Domain | 21 | URL string | Instant |
| Directory | 18 | URL string | Instant |
| File | 18 | URL string | Instant |
| Parameters | 20 | URL string | Instant |
| **Subtotal — URL-derived** | **96** | | **Instant** |
| External / network | 15 | DNS, WHOIS, TLS, search index | **Slow — network calls** |
| **Total** | **111** | | |

This 96 / 15 split matches the source paper's own description and is **the most important
structural fact about this dataset for our purposes** — see section 5.

### Group A — Whole URL (19 features)

Character counts across the entire URL string, plus length measures.

```
qty_dot_url             count of  .  in URL
qty_hyphen_url          count of  -  in URL
qty_underline_url       count of  _  in URL
qty_slash_url           count of  /  in URL
qty_questionmark_url    count of  ?  in URL
qty_equal_url           count of  =  in URL
qty_at_url              count of  @  in URL
qty_and_url             count of  &  in URL
qty_exclamation_url     count of  !  in URL
qty_space_url           count of ' ' in URL
qty_tilde_url           count of  ~  in URL
qty_comma_url           count of  ,  in URL
qty_plus_url            count of  +  in URL
qty_asterisk_url        count of  *  in URL
qty_hashtag_url         count of  #  in URL
qty_dollar_url          count of  $  in URL
qty_percent_url         count of  %  in URL
qty_tld_url             top-level-domain length
length_url              total URL length
```

### Group B — Domain (21 features)

Same character counts restricted to the domain, plus three domain-specific signals.

```
qty_dot_domain          qty_hyphen_domain       qty_underline_domain
qty_slash_domain        qty_questionmark_domain qty_equal_domain
qty_at_domain           qty_and_domain          qty_exclamation_domain
qty_space_domain        qty_tilde_domain        qty_comma_domain
qty_plus_domain         qty_asterisk_domain     qty_hashtag_domain
qty_dollar_domain       qty_percent_domain
qty_vowels_domain       count of vowels in domain
domain_length           domain length
domain_in_ip            domain is in IP-address format
server_client_domain    domain contains "server" or "client"
```

### Group C — Directory / path (18 features)

```
qty_dot_directory       qty_hyphen_directory    qty_underline_directory
qty_slash_directory     qty_questionmark_directory  qty_equal_directory
qty_at_directory        qty_and_directory       qty_exclamation_directory
qty_space_directory     qty_tilde_directory     qty_comma_directory
qty_plus_directory      qty_asterisk_directory  qty_hashtag_directory
qty_dollar_directory    qty_percent_directory
directory_length        directory length
```

### Group D — File (18 features)

```
qty_dot_file            qty_hyphen_file         qty_underline_file
qty_slash_file          qty_questionmark_file   qty_equal_file
qty_at_file             qty_and_file            qty_exclamation_file
qty_space_file          qty_tilde_file          qty_comma_file
qty_plus_file           qty_asterisk_file       qty_hashtag_file
qty_dollar_file         qty_percent_file
file_length             file name length
```

### Group E — Parameters / query string (20 features)

```
qty_dot_params          qty_hyphen_params       qty_underline_params
qty_slash_params        qty_questionmark_params qty_equal_params
qty_at_params           qty_and_params          qty_exclamation_params
qty_space_params        qty_tilde_params        qty_comma_params
qty_plus_params         qty_asterisk_params     qty_hashtag_params
qty_dollar_params       qty_percent_params
params_length           query-string length
tld_present_params      a TLD appears inside the parameters
qty_params              number of parameters
```

### Group F — External lookups (15 features) — the expensive ones

These require network operations. They are the features whose removal has real deployment value.

```
email_in_url             an email address appears in the URL
time_response            domain lookup response time
domain_spf               domain publishes an SPF record
asn_ip                   autonomous system number
time_domain_activation   days since domain registration
time_domain_expiration   days until domain expiry
qty_ip_resolved          number of resolved IP addresses
qty_nameservers          number of nameservers
qty_mx_servers           number of MX (mail) servers
ttl_hostname             DNS time-to-live for the hostname
tls_ssl_certificate      valid TLS/SSL certificate present
qty_redirects            number of redirects
url_google_index         URL is indexed by Google
domain_google_index      domain is indexed by Google
url_shortened            URL uses a shortening service
```

---

## 3. Secondary dataset — UCI Phishing Websites

Retained for cross-dataset validation and because published benchmarks exist for it.

| | |
|---|---|
| **UCI page** | https://archive.ics.uci.edu/dataset/327/phishing+websites |
| **Kaggle mirror (CSV)** | https://www.kaggle.com/datasets/akashkr/phishing-website-dataset |
| **Rows** | 11,055 |
| **Columns** | **31** in the official `.arff` (30 features + `Result`) |
| **Target** | `Result` — `−1` = phishing, `1` = legitimate |
| **Feature values** | Ternary `{−1, 0, 1}` — already bucketed |
| **Feature groups** | Address-bar, Abnormal, HTML/JavaScript, Domain |

**Citation:** Mohammad, R. & McCluskey, L. (2012). *Phishing Websites.* UCI Machine Learning
Repository. DOI: `10.24432/C51W2X`

> **Column-count discrepancy, resolved.** The official UCI `.arff` has **31** columns
> (verified by direct inspection). Kaggle CSV re-uploads commonly add a row-number `index`
> column, giving **32**. Both describe the same 30 features.
>
> **If using the Kaggle CSV, drop the index column before training.** If rows happen to be
> sorted by class, the row number becomes a near-perfect predictor and produces fake accuracy.

---

## 4. Dataset comparison

| | Vrbančič | UCI |
|---|---|---|
| Rows | 58,645 / 88,647 | 11,055 |
| Features | 111 | 30 |
| Feature values | Raw counts and continuous | Bucketed `−1 / 0 / 1` |
| Target encoding | `0` / `1` | `−1` / `1` |
| Collected | ~2020 | ~2012 |
| Search space | 2¹¹¹ ≈ 2.6 × 10³³ | 2³⁰ ≈ 1.07 × 10⁹ |
| Requires page content | No — URL only | Partly (HTML/JS group) |
| Peer-reviewed data paper | Yes | Yes |

**On direction of conversion:** Vrbančič raw values *can* be bucketed into UCI-style ternary
values using the thresholds documented in the UCI paper. The reverse is impossible, since UCI
already discarded the raw measurements. Any cross-dataset transfer experiment must therefore run
in the Vrbančič → UCI direction.

---

## 5. Why the 96 / 15 split matters

The 96 URL-derived features are computed from the URL string in microseconds. The 15 external
features each require a network round-trip — DNS resolution, WHOIS lookup, a search-index query.
In a deployed detector those 15 dominate the latency budget entirely.

**This gives the project its strongest possible finding.** If PSO independently discards most of
the external-lookup features while retaining accuracy, the conclusion is not merely "we used
fewer features" but:

> *The reduced model requires no network calls at all, making it viable for real-time,
> in-browser phishing detection.*

Analysis of which groups survive selection should therefore be treated as a **primary result**,
not an afterthought. Report the surviving-feature breakdown by group alongside the headline
accuracy numbers.

---

## 6. The `−1` sentinel — a confirmed data-quality issue

**Observed directly in the data.** The external-lookup columns contain `−1` in a substantial
number of rows — visible in `time_domain_activation`, `time_domain_expiration` and
`qty_ip_resolved`, among others.

**`−1` is not a measurement. It is a sentinel meaning "the lookup failed."** A domain cannot be
−1 days old, and cannot resolve to −1 IP addresses.

This means the dataset contains missing values that do not appear as missing — they are encoded
as an in-range-looking number, so `df.isnull().sum()` reports zero.

### Why it matters, by model family

| Model | Behaviour |
|-------|-----------|
| **Random Forest** | Handles it. Splits at `−1` and may correctly learn that "lookup failed" is itself a signal — a domain with no WHOIS record is mildly suspicious |
| **SVM / KNN** | **Misled.** Reads `−1` as "slightly less than zero days old", placing failed lookups immediately adjacent to brand-new domains in feature space. Those are entirely different things |

### Investigation results — COMPLETED

Run via `src/audit_data.py`; full output in `results/tables/01_data_audit.txt`.

| Feature | Tier | `−1` count | `−1` rate | P(phish \| fail) | Lift |
|---------|------|-----------|-----------|------------------|------|
| `time_domain_expiration` | T5 | 20,698 | **35.3%** | 54.5% | 1.04 |
| `time_domain_activation` | T5 | 18,910 | **32.2%** | 57.4% | 1.10 |
| `domain_spf` | T4 | 11,153 | 19.0% | 61.8% | 1.18 |
| `qty_redirects` | T7 | 5,519 | 9.4% | 28.2% | **0.54** |
| `time_response` | T1 | 4,925 | 8.4% | 28.3% | **0.54** |
| `asn_ip` | T9 | 4,767 | 8.1% | 32.5% | **0.62** |
| `ttl_hostname` | T1 | 3,648 | 6.2% | 25.8% | **0.49** |
| `qty_ip_resolved` | T1 | 3,645 | 6.2% | 25.9% | **0.50** |
| `qty_nameservers`, `qty_mx_servers`, `tls_ssl_certificate` | — | 0 | 0.0% | — | — |

Baseline P(phishing) = 52.3%. Lift = P(phishing \| fail) ÷ baseline; 1.00 is uninformative.

### Two findings, one of them a concern

**1. The WHOIS features are missing for roughly a third of rows.**
`time_domain_activation` and `time_domain_expiration` are `−1` in 32% and 35% of rows
respectively. These are among the most predictive features in the phishing literature, yet a
third of the time the value is simply absent. Worth stating in the report.

**2. Failed DNS lookups predict *legitimate*, not phishing — and this is backwards.**

The DNS-family features (`ttl_hostname`, `qty_ip_resolved`, `time_response`, `asn_ip`,
`qty_redirects`) all show lift ≈ 0.5: when the lookup fails, the row is about **twice as likely
to be legitimate**.

That is the opposite of the intuitive expectation — a domain that will not resolve ought to look
*more* suspicious, not less. The likely explanation is a **collection artifact**: the legitimate
and phishing URLs were gathered from different sources, possibly at different times, and the
extraction pipeline behaved differently across them.

> **Why this matters.** A model can learn "DNS lookup failed → legitimate" and score well on this
> dataset while having learned nothing about phishing. This should be named as a limitation, and
> it is a strong additional argument for the cross-dataset validation described in
> [`07-pitfalls-and-risks.md`](07-pitfalls-and-risks.md) — a feature that only works because of how
> one dataset was built will not transfer.

Possible mitigation if time allows: re-run the search with the DNS-family features excluded, and
compare. If performance barely moves, the artifact was not load-bearing.

### Handling decision — must be documented

| Option | Trade-off |
|--------|-----------|
| **(a)** Leave as-is, rely on tree models | Simplest; restricts safe use of SVM/KNN |
| **(b)** Add explicit binary "lookup failed" indicator columns | Cleaner semantics; adds features, and each indicator must inherit its parent's cost tier — you only learn a lookup failed by attempting it |

Either is defensible. Choosing silently is not.

---

## 7. Feature cost

Not all features cost the same to compute, and the difference spans five orders of magnitude.
This is central enough to the project to have its own document.

**See [`02-cost-model.md`](02-cost-model.md)** for the tier system, latency assignments,
calibration against the `time_response` column, and the sensitivity analysis.

Summary: 96 features derive from parsing the URL string (~0.01 ms total), while 15 require
network lookups (DNS, WHOIS, TLS, search index) totalling roughly 1,320 ms. Cost is driven by
which *lookup types* a subset requires, not by how many features it contains.

---

## 8. Preprocessing notes

Points to verify at load time. None of this has been executed yet — treat as a checklist.

1. **Confirm shape** — expect 58,645 × 112 for `dataset_small.csv`.
2. **Confirm target** — column named `phishing`, values `{0, 1}` only.
3. **Missing values** — the source reports none, and `isnull()` will agree. **This is
   misleading — see section 6.** The real missing values are encoded as `−1`.
4. **Constant / zero-variance columns** — several rare-character counts (e.g. `qty_space_*`,
   `qty_asterisk_*`) may be zero for nearly every row. Worth identifying: they are free wins for
   any reduction method and worth mentioning in the analysis.
5. **Scaling** — Random Forest does not require it. SVM and KNN do. Fit the scaler on the
   **training split only**, never on the full dataset.
6. **Class balance** — near-balanced in `dataset_small.csv`, so stratified splitting is
   sufficient; no resampling needed.
7. **Feature magnitudes vary widely** — counts range from 0 to small integers, while
   `time_domain_expiration` and `asn_ip` can be large. Relevant to distance-based classifiers.

---

## 7. Open questions

To resolve when the data is first loaded:

- How many features are constant or near-constant?
- What is the correlation structure within each group? Character counts across URL sections are
  likely to be highly correlated — this is precisely the redundancy PSO should exploit.
- Are the 15 external features individually strong predictors, or is their value marginal? This
  determines whether the headline "no network calls needed" finding is achievable.
