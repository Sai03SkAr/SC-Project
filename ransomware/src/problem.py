"""
The multi-objective feature-selection problem for any of the ransomware datasets.

Decision variable: a bit per feature (1 = keep). Objectives, all minimised:
    f1  detection cost      - from cost.py (analysis tiers, or feature count)
    f2  false negative rate - ransomware let through (1 - recall)
    f3  false positive rate - benign software wrongly flagged

Same design as the phishing project: f1 needs no model; f2 and f3 come from one confusion
matrix on an internal validation split carved from the TRAINING data. The sealed test set
is never used here.
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from cost import cost_model_for
from datasets import RANDOM_STATE


class SparseBinarySampling(Sampling):
    """Initial population spread across a range of subset sizes.

    Standard binary sampling switches each feature on with probability 0.5, so every starting
    individual holds about half the features. In a large feature space that makes small
    subsets - and subsets that avoid an entire analysis tier, such as a sandbox-free detector
    that must switch off all 260 dynamic MLRan features - effectively unreachable: bit-flip
    mutation changes about one bit per generation. Here individual i switches features on with
    a density drawn log-uniformly between 1/n (about one feature) and 0.5, the usual remedy in
    large-scale feature selection (cf. SparseEA). It uses no knowledge of feature tiers, and
    both algorithms receive exactly the same sampler.
    """

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        n = problem.n_var
        dens = np.exp(random_state.uniform(np.log(1.0 / n), np.log(0.5), size=n_samples))
        X = random_state.random((n_samples, n)) < dens[:, None]
        empty = ~X.any(axis=1)
        X[empty, random_state.integers(0, n, size=empty.sum())] = True
        return X


def search_splits(ds, subsample: int | None = 15_000):
    X_fit, X_val, y_fit, y_val = train_test_split(
        ds.X_train, ds.y_train, test_size=0.25, stratify=ds.y_train, random_state=RANDOM_STATE)
    if subsample and subsample < len(X_fit):
        idx, _ = train_test_split(np.arange(len(X_fit)), train_size=subsample,
                                  stratify=y_fit, random_state=RANDOM_STATE)
        X_fit, y_fit = X_fit[idx], y_fit[idx]
    return X_fit, y_fit, X_val, y_val


class RansomwareFeatureSelection(ElementwiseProblem):
    def __init__(self, ds, n_estimators: int = 30, cost=None):
        self.ds = ds
        self.cost = cost or cost_model_for(ds)
        self.max_cost = self.cost.max_cost
        self.X_fit, self.y_fit, self.X_val, self.y_val = search_splits(ds)
        self.n_estimators = n_estimators
        self._cache: dict[bytes, tuple] = {}
        self.n_evaluations = 0
        self.n_cache_hits = 0
        super().__init__(n_var=ds.n_features, n_obj=3, xl=0, xu=1, vtype=bool)

    def _evaluate(self, x, out, *args, **kwargs):
        mask = np.asarray(x, dtype=bool)
        key = mask.tobytes()
        if key in self._cache:
            self.n_cache_hits += 1
            out["F"] = list(self._cache[key]); return
        res = self.objectives(mask)
        self._cache[key] = res
        out["F"] = list(res)

    def objectives(self, mask):
        self.n_evaluations += 1
        if not mask.any():                       # worst on every objective
            return (self.max_cost, 1.0, 1.0)
        clf = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1,
                                     random_state=RANDOM_STATE)
        clf.fit(self.X_fit[:, mask], self.y_fit)
        pred = clf.predict(self.X_val[:, mask])
        tn, fp, fn, tp = confusion_matrix(self.y_val, pred, labels=[0, 1]).ravel()
        fnr = fn / (fn + tp) if (fn + tp) else 1.0
        fpr = fp / (fp + tn) if (fp + tn) else 1.0
        return (self.cost(mask), float(fnr), float(fpr))

    @property
    def cache_hit_rate(self):
        t = self.n_evaluations + self.n_cache_hits
        return self.n_cache_hits / t if t else 0.0

    def normalise(self, F):
        F = np.asarray(F, float).reshape(-1, 3)
        return np.column_stack([F[:, 0] / self.max_cost, F[:, 1], F[:, 2]])
