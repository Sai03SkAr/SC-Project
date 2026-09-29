"""
The multi-objective feature-selection problem, as a pymoo Problem.

Decision variable : binary vector of length 111 (1 = keep that feature)
Objectives, all minimised:
    f1  detection cost in ms   - from the cost model; needs NO model training
    f2  false negative rate    - 1 - recall  (Objective 2: maximise recall)
    f3  false positive rate    - stops the optimiser labelling everything phishing

f2 and f3 both come from a SINGLE confusion matrix, and f1 needs no model at all,
so evaluating three objectives costs the same as evaluating one.

Data discipline (docs/07-pitfalls-and-risks.md section 1):
    The test set is split off once and never touched here. Every fitness
    evaluation uses only an internal validation split carved out of the
    training data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from cost_model import (  # noqa: E402
    TARGET_COLUMN,
    build_feature_tiers,
    detection_cost_ms,
    max_detection_cost_ms,
    validate_feature_tiers,
)

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_dataset(filename: str = "dataset_small.csv"):
    """Load the CSV and return (X, y, feature_names) with X as float32."""
    import pandas as pd

    path = ROOT / "data" / filename
    df = pd.read_csv(path)

    feature_names = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int8)
    return X, y, feature_names


def make_splits(X, y, search_subsample: int | None = 15_000):
    """Split into sealed test, and the search-time train/validation pair.

    Returns a dict with:
        X_train_full / y_train_full : the full 80% training portion
        X_test       / y_test       : sealed; used ONCE at the very end
        X_fit / y_fit               : subsample used to train during search
        X_val / y_val               : held-out slice used to score during search

    `search_subsample` trades fidelity for speed during the search only. The
    final selected subsets are re-validated on the full training data.
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Internal split for fitness evaluation - the test set stays sealed.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25, stratify=y_train_full, random_state=RANDOM_STATE,
    )

    if search_subsample and search_subsample < len(X_fit):
        idx, _ = train_test_split(
            np.arange(len(X_fit)),
            train_size=search_subsample,
            stratify=y_fit,
            random_state=RANDOM_STATE,
        )
        X_fit, y_fit = X_fit[idx], y_fit[idx]

    return {
        "X_train_full": X_train_full, "y_train_full": y_train_full,
        "X_test": X_test, "y_test": y_test,
        "X_fit": X_fit, "y_fit": y_fit,
        "X_val": X_val, "y_val": y_val,
    }


# --------------------------------------------------------------------------
# The problem
# --------------------------------------------------------------------------

class PhishingFeatureSelection(ElementwiseProblem):
    """Minimise (detection cost, false negative rate, false positive rate)."""

    def __init__(
        self,
        splits: dict,
        feature_names: list[str],
        n_estimators: int = 30,
        tier_latency: dict[str, float] | None = None,
        use_cache: bool = True,
    ):
        self.feature_names = feature_names
        self.feature_tiers = build_feature_tiers(feature_names)
        validate_feature_tiers(self.feature_tiers)

        self.X_fit = splits["X_fit"]
        self.y_fit = splits["y_fit"]
        self.X_val = splits["X_val"]
        self.y_val = splits["y_val"]

        self.n_estimators = n_estimators
        self.tier_latency = tier_latency
        self.max_cost = max_detection_cost_ms(tier_latency)

        # Duplicate masks are common here: the same characters are counted
        # across five URL sections, so many different subsets behave alike.
        # Caching turns those into free lookups - and the hit rate is itself
        # evidence for the redundancy that motivated choosing SPEA2.
        self.use_cache = use_cache
        self._cache: dict[bytes, tuple[float, float, float]] = {}
        self.n_evaluations = 0
        self.n_cache_hits = 0

        super().__init__(n_var=len(feature_names), n_obj=3, xl=0, xu=1, vtype=bool)

    # ---------------------------------------------------------------- core
    def _evaluate(self, x, out, *args, **kwargs):
        mask = np.asarray(x, dtype=bool)

        if self.use_cache:
            key = mask.tobytes()
            hit = self._cache.get(key)
            if hit is not None:
                self.n_cache_hits += 1
                out["F"] = list(hit)
                return

        result = self._objectives(mask)

        if self.use_cache:
            self._cache[key] = result
        out["F"] = list(result)

    def _objectives(self, mask) -> tuple[float, float, float]:
        self.n_evaluations += 1

        # An empty subset must be worst on EVERY objective. It does have zero
        # cost, and returning that truthfully would make it non-dominated and
        # pollute the entire front.
        if not mask.any():
            return (self.max_cost, 1.0, 1.0)

        f1 = detection_cost_ms(
            mask, self.feature_names, self.feature_tiers, self.tier_latency
        )

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        clf.fit(self.X_fit[:, mask], self.y_fit)
        pred = clf.predict(self.X_val[:, mask])

        tn, fp, fn, tp = confusion_matrix(self.y_val, pred, labels=[0, 1]).ravel()
        f2 = fn / (fn + tp) if (fn + tp) else 1.0   # FNR = 1 - recall
        f3 = fp / (fp + tn) if (fp + tn) else 1.0   # FPR

        return (float(f1), float(f2), float(f3))

    # ------------------------------------------------------------- helpers
    @property
    def cache_hit_rate(self) -> float:
        total = self.n_evaluations + self.n_cache_hits
        return self.n_cache_hits / total if total else 0.0

    def normalise(self, F):
        """Scale objectives to [0,1] so hypervolume is not dominated by cost.

        Cost spans 0.01-1320 ms while the two rates span 0-1; without this,
        cost would account for essentially all of the hypervolume.
        """
        F = np.asarray(F, dtype=float).reshape(-1, 3)
        return np.column_stack([F[:, 0] / self.max_cost, F[:, 1], F[:, 2]])
