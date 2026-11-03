"""Small, dependency-free probability and calibration primitives."""
from __future__ import annotations

import math


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _probability(value: object) -> float:
    if not _finite(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    return float(value)


def _component(component: dict) -> tuple[float, float, float]:
    if not isinstance(component, dict):
        raise ValueError("mixture components must be dictionaries")
    try:
        weight, alpha, beta = component["weight"], component["alpha"], component["beta"]
    except KeyError as exc:
        raise ValueError("mixture component is missing a field") from exc
    if not all(_finite(value) for value in (weight, alpha, beta)) or weight < 0 or alpha <= 0 or beta <= 0:
        raise ValueError("weights must be non-negative and beta parameters positive")
    return float(weight), float(alpha), float(beta)


def beta_mixture_predict(s: int, n: int, components: list[dict] | None = None) -> float:
    """Return the posterior predictive mean of a finite Beta mixture.

    Component weights are prior weights and are updated using the exact
    Binomial-count likelihood, with the common binomial coefficient cancelling.
    """
    if isinstance(s, bool) or isinstance(n, bool) or not isinstance(s, int) or not isinstance(n, int):
        raise ValueError("counts must be integers")
    if n < 0 or s < 0 or s > n:
        raise ValueError("successes must satisfy 0 <= s <= n")
    components = components if components is not None else [{"weight": 1.0, "alpha": 1.0, "beta": 1.0}]
    parsed = [_component(component) for component in components]
    if not parsed or not any(weight > 0 for weight, _, _ in parsed):
        raise ValueError("mixture must have positive total weight")
    log_weights = []
    posterior_means = []
    for weight, alpha, beta in parsed:
        if weight == 0:
            continue
        log_weights.append(math.log(weight) + math.lgamma(alpha + s) + math.lgamma(beta + n - s)
                           - math.lgamma(alpha + beta + n) + math.lgamma(alpha + beta)
                           - (math.lgamma(alpha) + math.lgamma(beta)))
        posterior_means.append((alpha + s) / (alpha + beta + n))
    max_log = max(log_weights)
    normalized = [math.exp(value - max_log) for value in log_weights]
    denominator = math.fsum(normalized)
    return math.fsum(weight * mean for weight, mean in zip(normalized, posterior_means)) / denominator


def fit_logit_offset(priors, labels, strength: float) -> float:
    """Solve the regularized one-dimensional logistic calibration equation."""
    priors = list(priors)
    labels = list(labels)
    if len(priors) != len(labels):
        raise ValueError("priors and labels must have equal length")
    if not _finite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    if not priors:
        return 0.0
    clipped = [_probability(value) for value in priors]
    if any(not _finite(value) or value not in (0, 1) for value in labels):
        raise ValueError("labels must be exact finite 0/1 values")
    logits = [math.log(min(max(value, 1e-6), 1 - 1e-6) / (1 - min(max(value, 1e-6), 1 - 1e-6))) for value in clipped]
    target = [int(value) for value in labels]

    def sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def equation(delta: float) -> float:
        return math.fsum(sigmoid(logit + delta) - label for logit, label in zip(logits, target)) + strength * delta

    if equation(0.0) == 0.0:
        return 0.0
    low, high = -1.0, 1.0
    while equation(low) > 0:
        low *= 2.0
    while equation(high) < 0:
        high *= 2.0
    for _ in range(160):
        middle = (low + high) / 2.0
        if equation(middle) < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0
