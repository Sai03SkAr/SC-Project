# Ransomware Detection via Multi-Objective Feature Selection

The phishing project's method — SPEA2-based multi-objective feature selection, compared
against NSGA-II, weighted-sum PSO, PCA and a filter — applied to three ransomware datasets.

## The question

Ransomware does its damage the moment it runs. Static analysis (reading the file) is
milliseconds; dynamic analysis (running it in a sandbox) takes minutes. So the cost
objective becomes: **can ransomware be caught before it executes, and what does skipping
the sandbox cost in accuracy?**

## Objectives (all minimised)

| | Objective | Definition |
|---|---|---|
| f₁ | Detection cost | Sum of the distinct analysis tiers needed: static ≈ 50 ms, dynamic ≈ 120 s (Cuckoo default). UGRansome: number of features |
| f₂ | False negative rate | Ransomware let through (1 − recall) |
| f₃ | False positive rate | Benign software wrongly flagged |

## Datasets

| Dataset | Source | Samples | Features | Task |
|---|---|---|---|---|
| **MLRan** | Onwuegbuche et al., JNCA 2026 — [Zenodo 23403681](https://zenodo.org/records/23403681) (CC-BY-4.0) | 4,880 | 483 binary (authors' RFE set): 223 static strings, 260 dynamic | ransomware vs goodware |
| **RD2024** | Hussain, *Ransomware Dataset 2024* — [Zenodo 13890887](https://zenodo.org/records/13890887) (CC-BY-4.0) | 15,638 (of 21,752) | 148: 135 static PE-header, 13 dynamic | 12 ransomware families vs benign |
| **UGRansome** | Nkongolo et al., 2021 — [Kaggle](https://www.kaggle.com/datasets/nkongolo/ugransome-dataset) (login) | 108,941 (of 149,043; SS dropped) | 13 network-traffic attributes | zero-day (A) vs known-signature (S) |

Only processed feature tables are used — no malware samples are downloaded.

## Data decisions (from the audit — `src/audit.py`)

- **MLRan:** a third of the test vectors (320/975) also appear in training. The authors' split
  is kept (their RFE features were selected on it), and every result is reported on all test
  rows *and* on the 655 unseen rows.
- **RD2024:** `processes_malicious` alone separates the classes at AUC 0.996 — it is the
  sandbox's own verdict, not behaviour. All verdict counts (`*_malicious`, `*_suspicious`,
  `network_threats`) are excluded. Other malware categories (RAT, Stealer, Trojan) are dropped.
- **UGRansome:** no benign class exists; the task is zero-day vs known-signature traffic.
  With 13 features every subset can be evaluated, giving the true Pareto front. Text attributes
  get category codes. No shortcut feature (max single-feature AUC 0.85); 208 duplicate rows
  carry conflicting labels. `Family` and `Threats` are analyst annotations - the best detector
  without them (BTC, USD, Netflow_Bytes) is in `results/ugransome/no_annotation_best.json`.

## Method decisions

- **Sparse initial population** (`problem.SparseBinarySampling`). With the standard 50%-density
  start, MLRan runs never reached a sandbox-free subset (0 in 6 runs, kept in
  `results/mlran/ablation_dense_init/`). Each starting candidate now gets a density drawn
  log-uniformly from 1/n to 0.5. It uses no tier knowledge, and SPEA2, NSGA-II and PSO share it.
- **Deployment points** are chosen per cost tier by lowest balanced error (FNR+FPR)/2 on validation
  data. "Highest recall" picked flag-everything detectors, which are Pareto-optimal on recall.

## Headline results (sealed test set)

| | Detector | Features | Cost | Recall | FPR |
|---|---|---|---|---|---|
| RD2024 | SPEA2 pre-execution | 7 | 50 ms | 96.12% | 1.33% |
| RD2024 | All features (sandbox) | 148 | 120 s | 98.74% | 0.64% |
| MLRan | SPEA2 pre-execution | 73 | 50 ms | 93.33% | 4.90% |
| MLRan | NSGA-II full analysis | 158 | 120 s | 98.06% | 3.33% |
| MLRan | All features (sandbox) | 483 | 120 s | 96.99% | 3.33% |
| UGRansome | SPEA2 = NSGA-II compact | 4 | 4 features | 98.66% | 0.91% |
| UGRansome | All features | 13 | 13 features | 99.30% | 0.46% |

SPEA2 vs NSGA-II hypervolume: no significant difference (Mann–Whitney p = 0.68 MLRan, 1.00 RD2024,
0.15 UGRansome). UGRansome exhaustive search (all 8,191 subsets): true front = 34 detectors; pooled
SPEA2 and NSGA-II runs each recover all 34 (per run: SPEA2 32–34, NSGA-II 33–34).
Full write-up: `ransomware-report.pdf`.

## Pipeline

```bash
python src/audit.py        --dataset <d>
python src/run_baselines.py --dataset <d>
python src/run_multiseed.py --dataset <d> --algorithm spea2     # 10 seeds
python src/run_multiseed.py --dataset <d> --algorithm nsga2
python src/run_pso.py       --dataset <d>
python src/analyze.py       --dataset <d>                       # pooled fronts, test-set evaluation
python src/exhaustive.py    --dataset ugransome                 # true Pareto front
python src/figures.py
```

`<d>` ∈ `mlran`, `rd2024`, `ugransome`. Data files go in `data/<d>/` (not committed).
