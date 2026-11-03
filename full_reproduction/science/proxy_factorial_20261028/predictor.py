"""Visible-only prediction; isolation precedes prior and feature access."""
from __future__ import annotations

import math
import numpy as np
from scipy.special import expit, logit

from science.affine_calibration_20261015.core import offset, visible_key
from science.online_acquisition_20261015.stream import visible_group
from .core import MODES, SCALE, auxiliary, projections, solve_centered

EPS = 1e-6
METHODS = ("full", "phrase", "structure", "masked_phrase", "masked_structure",
           "separable", "structure_linear", "structure_shuffle", "linear_shuffle",
           "kernel_headtail", "headtail", "beta11")


class Predictor:
    def __init__(self, prior, semantic_features, raw_features, mode):
        if mode not in MODES:
            raise ValueError("unknown frozen mode")
        self.prior = prior
        self.semantic_features = semantic_features
        self.raw_features = raw_features
        self.mode = mode

    @staticmethod
    def _lookup(table, value):
        try:
            result = np.asarray(table[visible_key(value)[8]], dtype=np.float64)
        except KeyError as error:
            raise ValueError("unknown visible item_content") from error
        if result.ndim != 1 or not np.isfinite(result).all():
            raise ValueError("finite feature vector required")
        return result

    def evaluate(self, queries, labeled):
        values, states = evaluate_modes(self.prior, self.semantic_features, self.raw_features,
                                       None, queries, labeled, (self.mode,))
        return values[self.mode].tolist(), values["beta11"].tolist(), states

    def predict(self, query, labeled):
        return self.evaluate([query], labeled)[0][0]


def evaluate_modes(prior, semantic, raw, shuffled, queries, labeled, modes=None):
    queries = list(queries)
    modes = METHODS[:10] if modes is None else modes
    groups = {visible_group(query): [] for query in queries}
    for value, outcome in labeled:
        group = visible_group(value)
        if group in groups:
            if isinstance(outcome, (bool, np.bool_)) or outcome not in (0, 1):
                raise ValueError("binary purchased evidence required")
            groups[group].append((visible_key(value), int(outcome), value))
    output = {mode: np.empty(len(queries)) for mode in (*modes, "headtail", "beta11")}
    states = {}

    def priors(values):
        result = np.asarray([prior(value) for value in values], dtype=np.float64)
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError("invalid frozen prior")
        return result

    def features(table, values):
        return np.stack([Predictor._lookup(table, value) for value in values])

    for group, rows in groups.items():
        rows.sort(key=lambda row: (row[0], row[1]))
        observations = [row[2] for row in rows]
        outcomes = np.asarray([row[1] for row in rows], dtype=np.float64)
        support_prior = priors(observations)
        delta = float(offset(support_prior, outcomes)) if rows else 0.
        beta = float((math.fsum(outcomes)+1)/(len(rows)+2))
        states[group] = (delta, beta, len(rows))
        positions = [index for index, query in enumerate(queries) if visible_group(query) == group]
        group_queries = [queries[index] for index in positions]
        qe = features(semantic, group_queries)
        qr = features(raw, group_queries)
        query_prior = priors(group_queries)
        base = .5*expit(logit(np.clip(query_prior, EPS, 1-EPS))+delta)+.5*beta
        output["headtail"][positions] = base
        output["beta11"][positions] = beta
        if len(rows) < 2:
            for mode in modes:
                output[mode][positions] = np.clip(base, EPS, 1-EPS)
            continue
        se = features(semantic, observations)
        sr = features(raw, observations)
        query_projection, support_projection = projections(qr), projections(sr)
        semantic_cross = np.clip(qe@se.T, -1., 1.)**4
        semantic_gram = np.clip(se@se.T, -1., 1.)**4
        residual = outcomes-expit(logit(np.clip(support_prior, EPS, 1-EPS))+delta)
        if shuffled is not None:
            shuffle_query = projections(features(shuffled, group_queries))
            shuffle_support = projections(features(shuffled, observations))
        for mode in modes:
            actual = "none" if mode == "kernel_headtail" else mode
            qp, sp = query_projection, support_projection
            if mode in ("structure_shuffle", "linear_shuffle"):
                actual = "structure" if mode == "structure_shuffle" else "structure_linear"
                qp, sp = shuffle_query, shuffle_support
            cross = semantic_cross+auxiliary(qp, sp, actual)
            gram = semantic_gram+auxiliary(sp, sp, actual)
            correction = solve_centered(cross, gram, residual)
            output[mode][positions] = np.clip(base+SCALE*correction, EPS, 1-EPS)
    return output, states
