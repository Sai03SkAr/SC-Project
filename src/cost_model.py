"""
Detection cost model for the phishing feature-selection project.

Objective f1 is the time needed to compute a selected feature subset for one URL.

The central idea is that cost is NOT proportional to the number of features.
Features inherit the cost of the network lookup they depend on, and they SHARE it:
if a subset already needs a DNS A-record query for `qty_ip_resolved`, then adding
`ttl_hostname` costs essentially nothing, because the same query returns both.

So cost sums over the DISTINCT lookup tiers a subset requires, not over features.

See docs/02-cost-model.md for the reasoning and the sensitivity analysis plan.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Lookup tiers
# --------------------------------------------------------------------------
# T0 is charged once for the whole subset: the URL is parsed a single time and
# all 96 text features are then trivial operations over the parsed components.

TIER_LATENCY_MS: dict[str, float] = {
    "T0": 0.01,   # URL string parsing (charged once)
    "T1": 30.0,   # DNS - A record
    "T2": 30.0,   # DNS - NS record
    "T3": 30.0,   # DNS - MX record
    "T4": 30.0,   # DNS - TXT / SPF
    "T5": 200.0,  # WHOIS
    "T6": 150.0,  # TLS handshake
    "T7": 300.0,  # HTTP fetch / follow redirects
    "T8": 500.0,  # Search-index API
    "T9": 50.0,   # IP -> ASN lookup
}

TIER_DESCRIPTION: dict[str, str] = {
    "T0": "URL string parsing",
    "T1": "DNS A record",
    "T2": "DNS NS record",
    "T3": "DNS MX record",
    "T4": "DNS TXT / SPF record",
    "T5": "WHOIS registry query",
    "T6": "TLS handshake",
    "T7": "HTTP fetch / redirects",
    "T8": "Search-index API",
    "T9": "IP to ASN lookup",
}

# --------------------------------------------------------------------------
# Feature -> tier, for the features that need a network lookup
# --------------------------------------------------------------------------
# Every feature NOT listed here is T0 (derived from the URL text alone).
#
# Two deliberate reclassifications, documented in docs/02-cost-model.md:
#   - `email_in_url`  : the dataset groups it with the external features, but
#                       detecting an email pattern is a string operation -> T0
#   - `url_shortened` : matching against a known-shortener list -> T0
# Note this in the report; it shows the cost model was reasoned about rather
# than copied from the dataset's own grouping.

EXTERNAL_FEATURE_TIERS: dict[str, str] = {
    # --- DNS A record: resolve the host ---
    "qty_ip_resolved":        "T1",
    "time_response":          "T1",
    "ttl_hostname":           "T1",
    # --- other DNS record types ---
    "qty_nameservers":        "T2",
    "qty_mx_servers":         "T3",
    "domain_spf":             "T4",
    # --- registry ---
    "time_domain_activation": "T5",
    "time_domain_expiration": "T5",
    # --- transport / content ---
    "tls_ssl_certificate":    "T6",
    "qty_redirects":          "T7",
    # --- third-party services ---
    "url_google_index":       "T8",
    "domain_google_index":    "T8",
    "asn_ip":                 "T9",
}

# Listed in the dataset's external group but reclassified as T0 (see above).
RECLASSIFIED_AS_T0: frozenset[str] = frozenset({"email_in_url", "url_shortened"})

TARGET_COLUMN = "phishing"


# --------------------------------------------------------------------------
# Building the map
# --------------------------------------------------------------------------

def build_feature_tiers(feature_names) -> dict[str, str]:
    """Map every feature name to its lookup tier.

    Anything not in EXTERNAL_FEATURE_TIERS is T0, i.e. readable from the URL text.
    """
    return {
        name: EXTERNAL_FEATURE_TIERS.get(name, "T0")
        for name in feature_names
        if name != TARGET_COLUMN
    }


def validate_feature_tiers(feature_tiers: dict[str, str]) -> None:
    """Fail loudly if the dataset does not look the way the documents describe.

    Guards against a renamed column or a different dataset variant silently
    producing a wrong cost model.
    """
    n_features = len(feature_tiers)
    if n_features != 111:
        raise ValueError(f"expected 111 features, found {n_features}")

    n_external = sum(1 for t in feature_tiers.values() if t != "T0")
    if n_external != len(EXTERNAL_FEATURE_TIERS):
        missing = set(EXTERNAL_FEATURE_TIERS) - set(feature_tiers)
        raise ValueError(
            f"expected {len(EXTERNAL_FEATURE_TIERS)} network features, "
            f"found {n_external}. Missing from dataset: {sorted(missing)}"
        )


# --------------------------------------------------------------------------
# The cost function - this is objective f1
# --------------------------------------------------------------------------

# A tiny per-feature tiebreak added on top of the tier cost.
#
# Without this, the tier model alone gives the optimiser NO reason to drop a
# T0 (URL-text) feature: tier cost is charged once per tier regardless of how
# many features of that tier are kept, so a 98-feature T0 subset and a
# 1-feature T0 subset cost the same 0.01 ms. In practice, an early pilot run
# confirmed this: SPEA2 correctly minimised network-tier cost (1320ms -> 60ms)
# but kept 45-67 of the 111 features, since nothing punished the extra ones.
#
# Each feature genuinely does cost *something* nonzero to compute (one more
# counter over the parsed URL), so this is not an artificial penalty - it is
# a real cost that the coarse tier granularity otherwise hides. The value is
# set two orders of magnitude below the smallest tier latency (T0 = 0.01 ms)
# so it can only break ties between equal-tier-cost subsets; it never
# reorders subsets that differ in which tiers they need.
PER_FEATURE_EPSILON_MS = 1e-5


def detection_cost_ms(
    mask,
    feature_names,
    feature_tiers: dict[str, str],
    tier_latency: dict[str, float] | None = None,
    feature_epsilon: float = PER_FEATURE_EPSILON_MS,
) -> float:
    """Estimated time to compute the selected features for a single URL.

    Args:
        mask:            binary sequence, 1 = keep that feature
        feature_names:   feature names, in the same order as `mask`
        feature_tiers:   mapping from build_feature_tiers()
        tier_latency:    override latencies, used for the sensitivity analysis
        feature_epsilon: per-feature tiebreak; set to 0.0 to disable

    Returns:
        Estimated latency in milliseconds.

    Cost is summed over DISTINCT tiers, not over features, because features
    sharing a lookup only pay for it once. A small per-feature term is added
    on top so that, among subsets requiring the same tiers, fewer features
    is preferred - see PER_FEATURE_EPSILON_MS above.
    """
    latency = tier_latency if tier_latency is not None else TIER_LATENCY_MS

    selected = [name for bit, name in zip(mask, feature_names) if bit]
    tiers_needed = {feature_tiers[name] for name in selected}
    tier_cost = sum(latency[t] for t in tiers_needed)
    return tier_cost + feature_epsilon * len(selected)


def max_detection_cost_ms(
    tier_latency: dict[str, float] | None = None,
    feature_epsilon: float = PER_FEATURE_EPSILON_MS,
    n_features: int = 111,
) -> float:
    """Cost when every tier AND every feature is required - the baseline."""
    latency = tier_latency if tier_latency is not None else TIER_LATENCY_MS
    return sum(latency.values()) + feature_epsilon * n_features


# --------------------------------------------------------------------------
# Sensitivity scenarios (docs/02-cost-model.md, section 5)
# --------------------------------------------------------------------------
# Tier latencies are estimates, so results must be shown robust to them.
# T0 is left unscaled: local string parsing does not vary with network conditions.

def scaled_tier_latency(factor: float) -> dict[str, float]:
    """Scale every network tier by `factor`, leaving T0 unchanged."""
    return {
        tier: (ms if tier == "T0" else ms * factor)
        for tier, ms in TIER_LATENCY_MS.items()
    }


SENSITIVITY_SCENARIOS: dict[str, dict[str, float]] = {
    "optimistic":  scaled_tier_latency(1 / 3),  # warm caches, fast network
    "baseline":    TIER_LATENCY_MS,
    "pessimistic": scaled_tier_latency(3.0),    # cold cache, slow network
}


# --------------------------------------------------------------------------
# Reporting helper
# --------------------------------------------------------------------------

def tier_composition(mask, feature_names, feature_tiers: dict[str, str]) -> dict[str, int]:
    """Count selected features per tier.

    Used for the results table showing how many of the 15 expensive
    network features survived selection.
    """
    counts: dict[str, int] = {}
    for bit, name in zip(mask, feature_names):
        if bit:
            tier = feature_tiers[name]
            counts[tier] = counts.get(tier, 0) + 1
    return counts


def is_network_free(mask, feature_names, feature_tiers: dict[str, str]) -> bool:
    """True if the subset needs no network calls - i.e. browser-deployable.

    A solution satisfying this is the project's headline result.
    """
    return all(
        feature_tiers[name] == "T0"
        for bit, name in zip(mask, feature_names)
        if bit
    )
