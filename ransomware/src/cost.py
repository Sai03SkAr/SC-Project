"""
Detection cost model - objective f1.

Ransomware features fall into analysis tiers with very different costs:

    static   ~50 ms      read and parse the file (PE header, printable strings)
    dynamic  ~120,000 ms run the sample in a sandbox (Cuckoo's default 2-minute analysis)

As in the phishing project, features SHARE the cost of their tier: one sandbox run yields
every dynamic feature, so cost sums over the DISTINCT tiers a subset needs, plus a tiny
per-feature tiebreak so that, among subsets needing the same tiers, fewer features win.

With only two tiers the Pareto set does not depend on the exact latencies: any values with
static < dynamic produce the same dominance ordering. The figures above set the scale only.

Datasets whose features all come from one source (UGRansome: network traffic) use
feature count instead - see CountCost.
"""

from __future__ import annotations

import numpy as np

TIER_LATENCY_MS = {"static": 50.0, "dynamic": 120_000.0}
PER_FEATURE_EPSILON_MS = 1e-5


class TierCost:
    kind = "tier"

    def __init__(self, feature_names, feature_tier, latency=None, epsilon=PER_FEATURE_EPSILON_MS):
        self.latency = dict(latency or TIER_LATENCY_MS)
        self.epsilon = epsilon
        tiers = [feature_tier[n] for n in feature_names]
        self.tier_masks = {t: np.array([x == t for x in tiers]) for t in set(tiers)}
        self.n = len(feature_names)
        self.max_cost = sum(self.latency[t] for t in self.tier_masks) + epsilon * self.n

    def __call__(self, mask) -> float:
        mask = np.asarray(mask, dtype=bool)
        tier_cost = sum(self.latency[t] for t, m in self.tier_masks.items() if (mask & m).any())
        return float(tier_cost + self.epsilon * mask.sum())

    def is_static_only(self, mask) -> bool:
        mask = np.asarray(mask, dtype=bool)
        dyn = self.tier_masks.get("dynamic")
        return bool(mask.any() and (dyn is None or not (mask & dyn).any()))


class CountCost:
    """Uniform per-feature cost: f1 is the number of features used."""
    kind = "count"

    def __init__(self, feature_names, feature_tier=None):
        self.n = len(feature_names)
        self.max_cost = float(self.n)

    def __call__(self, mask) -> float:
        return float(np.asarray(mask, dtype=bool).sum())

    def is_static_only(self, mask) -> bool:
        return False


def cost_model_for(ds):
    tiers = set(ds.feature_tier.values())
    if tiers <= {"static", "dynamic"} and len(tiers) == 2:
        return TierCost(ds.feature_names, ds.feature_tier)
    return CountCost(ds.feature_names)
