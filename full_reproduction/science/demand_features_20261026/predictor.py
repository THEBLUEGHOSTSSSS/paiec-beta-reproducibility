"""Visible-only predictor using the frozen additive demand kernel."""
from __future__ import annotations

import math
import numpy as np
from scipy.special import expit, logit

from science.affine_calibration_20261015.core import offset, visible_key
from science.online_acquisition_20261015.stream import visible_group
from .core import SCALE, centered_correction

EPS = 1e-6


class Predictor:
    def __init__(self, prior, semantic_features, demand_features):
        self.prior = prior
        self.semantic_features = semantic_features
        self.demand_features = demand_features

    @staticmethod
    def _lookup(table, value, label):
        content = visible_key(value)[8]
        try:
            vector = np.asarray(table[content], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"visible item_content lacks frozen {label} vector") from exc
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"frozen {label} vector must be finite and one-dimensional")
        return vector

    def evaluate(self, queries, labeled):
        queries = list(queries)
        groups = {visible_group(query): [] for query in queries}
        # Isolation occurs before any prior or feature lookup.
        for value, outcome in labeled:
            group = visible_group(value)
            if group in groups:
                if isinstance(outcome, (bool, np.bool_)) or outcome not in (0, 1):
                    raise ValueError("binary purchased evidence required")
                groups[group].append((visible_key(value), int(outcome), value))
        states, corrections = {}, {}
        for group, rows in groups.items():
            rows.sort(key=lambda row: (row[0], row[1]))
            observations = [(row[2], row[1]) for row in rows]
            priors = [float(self.prior(value)) for value, _ in observations]
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in priors):
                raise ValueError("invalid frozen prior")
            outcomes = [outcome for _, outcome in observations]
            delta = float(offset(priors, outcomes)) if rows else 0.
            beta = float((sum(outcomes)+1)/(len(outcomes)+2))
            states[group] = (delta, beta, len(rows))
            if len(rows) < 2:
                corrections[group] = None
                continue
            probability = np.clip(np.asarray(priors), EPS, 1-EPS)
            residual = np.asarray(outcomes, dtype=np.float64) - expit(logit(probability)+delta)
            se = np.stack([self._lookup(self.semantic_features, value, "semantic")
                           for value, _ in observations])
            sd = np.stack([self._lookup(self.demand_features, value, "demand")
                           for value, _ in observations])
            indices = [i for i, query in enumerate(queries) if visible_group(query) == group]
            qe = np.stack([self._lookup(self.semantic_features, queries[i], "semantic")
                           for i in indices])
            qd = np.stack([self._lookup(self.demand_features, queries[i], "demand")
                           for i in indices])
            corrections[group] = (dict(zip(indices, range(len(indices)))),
                                  centered_correction(qe, se, qd, sd, residual))
        probabilities, betas = [], []
        for index, query in enumerate(queries):
            group = visible_group(query)
            delta, beta, _ = states[group]
            prior = float(self.prior(query))
            if not math.isfinite(prior) or not 0 <= prior <= 1:
                raise ValueError("invalid frozen prior")
            base = .5*float(expit(logit(np.clip(prior, EPS, 1-EPS))+delta)) + .5*beta
            item = corrections[group]
            correction = 0. if item is None else float(item[1][item[0][index]])
            probabilities.append(float(np.clip(base + SCALE*correction, EPS, 1-EPS)))
            betas.append(beta)
        return probabilities, betas, states

    def predict(self, query, labeled):
        return self.evaluate([query], labeled)[0][0]

