"""Gaussian-offset posterior predictive integration."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import expit, logsumexp


@lru_cache(maxsize=4)
def _quadrature(nodes):
    return np.polynomial.hermite.hermgauss(nodes)


def _values(values, name):
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(name + " must be finite and one-dimensional")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(name + " must be in [0, 1]")
    return array


def posterior_predict(acquired_priors, labels, target_priors, strength=2, nodes=121):
    """Return posterior predictive probabilities under a shared Gaussian offset.

    The prior is ``delta ~ N(0, 1 / strength)`` and the observed responses
    use Bernoulli likelihoods at ``logit(prior) + delta``.  Gauss-Hermite
    quadrature is evaluated in log space, including the empty-observation B0
    case rather than returning the unintegrated target prior.
    """
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(float(strength)) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    if isinstance(nodes, bool) or not isinstance(nodes, (int, np.integer)) or nodes < 3:
        raise ValueError("nodes must be an integer >= 3")
    acquired = _values(acquired_priors, "acquired_priors")
    target = _values(target_priors, "target_priors")
    label_values = list(labels)
    labels_array = np.asarray(label_values)
    if labels_array.ndim != 1 or labels_array.size != acquired.size:
        raise ValueError("labels and acquired_priors must have equal length")
    if labels_array.size and (
        not np.isfinite(labels_array.astype(np.float64)).all()
        or np.any((labels_array != 0) & (labels_array != 1))
        or any(isinstance(value, (bool, np.bool_)) for value in label_values)
    ):
        raise ValueError("labels must be exact finite 0/1 values")
    labels_array = labels_array.astype(np.float64)

    quadrature_x, quadrature_weights = _quadrature(int(nodes))
    offsets = math.sqrt(2.0 / float(strength)) * quadrature_x
    log_weights = np.log(quadrature_weights) - 0.5 * math.log(math.pi)
    acquired = np.clip(acquired, 1e-6, 1.0 - 1e-6)
    target = np.clip(target, 1e-6, 1.0 - 1e-6)
    prior_logits = np.log(acquired / (1.0 - acquired))
    target_logits = np.log(target / (1.0 - target))
    if acquired.size:
        likelihood = np.sum(
            labels_array[:, None] * (-np.logaddexp(0.0, -(prior_logits[:, None] + offsets[None, :])))
            + (1.0 - labels_array[:, None]) * (-np.logaddexp(0.0, prior_logits[:, None] + offsets[None, :])),
            axis=0,
        )
    else:
        likelihood = np.zeros_like(offsets)
    log_posterior_weights = log_weights + likelihood
    log_posterior_weights -= logsumexp(log_posterior_weights)
    return np.asarray(
        np.exp(logsumexp(log_posterior_weights[None, :] + np.log(expit(target_logits[:, None] + offsets[None, :])), axis=1)),
        dtype=np.float64,
    )
