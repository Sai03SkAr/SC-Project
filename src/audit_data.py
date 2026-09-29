"""
Phase 01 - data audit.

Answers the open questions recorded in docs/01-dataset.md before any modelling:

  1. Does the file match what the documents claim (58,645 x 112)?
  2. How many `-1` sentinels are in each external-lookup column, and does a
     FAILED LOOKUP itself predict phishing?
  3. Which columns are constant or near-constant (free wins for reduction)?
  4. Does the cost model's feature->tier map line up with the real header?

Uses only the standard library, so it runs before pandas/sklearn are installed.

Run:  python3 src/audit_data.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cost_model import (  # noqa: E402
    EXTERNAL_FEATURE_TIERS,
    RECLASSIFIED_AS_T0,
    TARGET_COLUMN,
    TIER_DESCRIPTION,
    build_feature_tiers,
    max_detection_cost_ms,
    validate_feature_tiers,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset_small.csv"

SENTINEL = -1.0


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def load(path: Path):
    """Read the CSV into column-major float lists."""
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        columns: list[list[float]] = [[] for _ in header]
        n_rows = 0
        for row in reader:
            n_rows += 1
            for i, value in enumerate(row):
                columns[i].append(float(value))
    return header, columns, n_rows


def main() -> int:
    if not DATA.exists():
        print(f"ERROR: {DATA} not found. Download it first.")
        return 1

    header, columns, n_rows = load(DATA)
    col = dict(zip(header, columns))

    # ---------------------------------------------------------------- 1
    rule("1. SHAPE AND TARGET")

    print(f"rows                 : {n_rows:,}")
    print(f"columns              : {len(header)}")
    print(f"expected             : 58,645 rows x 112 columns")
    shape_ok = (n_rows == 58_645 and len(header) == 112)
    print(f"match                : {'YES' if shape_ok else 'NO  <-- INVESTIGATE'}")

    print(f"\nlast column name     : {header[-1]!r}")
    if TARGET_COLUMN not in col:
        print(f"ERROR: target column {TARGET_COLUMN!r} not found")
        return 1

    target = col[TARGET_COLUMN]
    counts = Counter(target)
    n_phish = counts.get(1.0, 0)
    n_legit = counts.get(0.0, 0)
    print(f"target values        : {sorted(counts)}")
    print(f"legitimate (0)       : {n_legit:,}  ({n_legit / n_rows:.1%})")
    print(f"phishing   (1)       : {n_phish:,}  ({n_phish / n_rows:.1%})")
    print(f"expected             : 27,998 legitimate / 30,647 phishing")

    base_rate = n_phish / n_rows

    # ---------------------------------------------------------------- 2
    rule("2. THE -1 SENTINEL INVESTIGATION")

    print("`-1` marks a FAILED lookup, not a measurement. A domain cannot be")
    print("-1 days old. These are missing values that isnull() will not report.\n")

    print(f"{'feature':<26}{'tier':<6}{'-1 count':>10}{'-1 rate':>9}"
          f"{'P(phish|fail)':>15}{'lift':>8}")
    print("-" * 74)

    leaky_signals = []
    for name, tier in sorted(EXTERNAL_FEATURE_TIERS.items(), key=lambda kv: kv[1]):
        if name not in col:
            print(f"{name:<26}{tier:<6}{'MISSING FROM DATASET':>42}")
            continue

        values = col[name]
        fail_idx = [i for i, v in enumerate(values) if v == SENTINEL]
        n_fail = len(fail_idx)

        if n_fail == 0:
            print(f"{name:<26}{tier:<6}{0:>10}{'0.0%':>9}{'-':>15}{'-':>8}")
            continue

        p_phish_given_fail = sum(target[i] for i in fail_idx) / n_fail
        lift = p_phish_given_fail / base_rate
        print(f"{name:<26}{tier:<6}{n_fail:>10,}{n_fail / n_rows:>8.1%}"
              f"{p_phish_given_fail:>14.1%}{lift:>8.2f}")

        if n_fail / n_rows > 0.01 and (lift > 1.25 or lift < 0.75):
            leaky_signals.append((name, n_fail / n_rows, p_phish_given_fail, lift))

    print(f"\nbaseline P(phishing) : {base_rate:.1%}")
    print("lift = P(phishing | lookup failed) / baseline.  1.00 means uninformative.")

    if leaky_signals:
        print("\nFAILED LOOKUPS CARRY SIGNAL in these columns:")
        for name, rate, p, lift in sorted(leaky_signals, key=lambda x: -abs(x[3] - 1)):
            direction = "phishing" if lift > 1 else "legitimate"
            print(f"  {name:<26} {rate:>6.1%} of rows fail, and those rows are "
                  f"{lift:.2f}x more likely to be {direction}")
        print("\n  => `-1` is informative, not just noise. Tree models will exploit")
        print("     this correctly. Distance-based models (SVM, KNN) will NOT: they")
        print("     read -1 as 'slightly less than zero days old', placing failed")
        print("     lookups next to brand-new domains. Handle before using those.")
    else:
        print("\nNo external column shows a strong failed-lookup signal.")

    # ---------------------------------------------------------------- 3
    rule("3. CONSTANT AND NEAR-CONSTANT COLUMNS")

    print("Free wins for any reduction method - they carry no information.\n")

    constant, near_constant = [], []
    for name in header:
        if name == TARGET_COLUMN:
            continue
        values = col[name]
        modal_count = Counter(values).most_common(1)[0][1]
        share = modal_count / n_rows
        if share == 1.0:
            constant.append(name)
        elif share > 0.999:
            near_constant.append((name, share))

    print(f"constant columns      : {len(constant)}")
    for name in constant:
        print(f"    {name}")
    print(f"\nnear-constant (>99.9%): {len(near_constant)}")
    for name, share in sorted(near_constant, key=lambda x: -x[1]):
        print(f"    {name:<28} {share:.3%} identical")

    total_dead = len(constant) + len(near_constant)
    print(f"\n=> {total_dead} of 111 features are effectively dead weight "
          f"({total_dead / 111:.0%}).")
    print("   SPEA2 should discard these immediately; if it does not, that is a bug.")

    # ---------------------------------------------------------------- 4
    rule("4. COST MODEL CROSS-CHECK")

    feature_names = [h for h in header if h != TARGET_COLUMN]
    feature_tiers = build_feature_tiers(header)

    try:
        validate_feature_tiers(feature_tiers)
        print("cost model validates against the real header: OK")
    except ValueError as exc:
        print(f"COST MODEL MISMATCH: {exc}")
        return 1

    tier_counts = Counter(feature_tiers.values())
    print(f"\n{'tier':<6}{'features':>10}  description")
    print("-" * 56)
    for tier in sorted(tier_counts):
        print(f"{tier:<6}{tier_counts[tier]:>10}  {TIER_DESCRIPTION[tier]}")

    n_t0 = tier_counts["T0"]
    n_net = len(feature_names) - n_t0
    print(f"\nURL-text features    : {n_t0}   (cost ~0.01 ms in total)")
    print(f"network features     : {n_net}   (each needs a lookup)")
    print(f"reclassified to T0   : {sorted(RECLASSIFIED_AS_T0)}")
    print(f"all-features cost    : {max_detection_cost_ms():,.2f} ms")

    # ---------------------------------------------------------------- 5
    rule("5. CALIBRATION - time_response DISTRIBUTION")

    print("The one tier latency measurable directly from the data.\n")

    valid = sorted(v for v in col["time_response"] if v != SENTINEL)
    n_valid = len(valid)
    if n_valid:
        def pct(p):
            return valid[min(int(p * n_valid), n_valid - 1)]
        print(f"valid measurements   : {n_valid:,} of {n_rows:,}")
        print(f"median               : {pct(0.50):.3f} s   ({pct(0.50) * 1000:,.0f} ms)")
        print(f"mean                 : {sum(valid) / n_valid:.3f} s")
        print(f"p95                  : {pct(0.95):.3f} s   ({pct(0.95) * 1000:,.0f} ms)")
        print(f"max                  : {valid[-1]:.3f} s")
        print(f"\nOur T1 (DNS) estimate: 30 ms")
        print("The measured values are far higher because the dataset was built by")
        print("bulk extraction with cold caches. State in the report which regime")
        print("is being modelled - see docs/02-cost-model.md section 4.")

    rule("AUDIT COMPLETE")
    print("Next: install pandas / scikit-learn / pymoo, then Phase 03 baselines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
