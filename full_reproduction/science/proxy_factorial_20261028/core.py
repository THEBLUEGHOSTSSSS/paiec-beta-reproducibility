"""Pure frozen kernels, centered correction, and material gate."""
from __future__ import annotations

import numpy as np

from science.demand_features_20261026.core import material_gate

MODES = ("full", "phrase", "structure", "masked_phrase", "masked_structure",
         "separable", "structure_linear", "none")
RIDGE = 4.0
SCALE = 0.5


def matrix(value, width, label):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != width or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite matrix of width {width}")
    return result


def projections(raw):
    raw = matrix(raw, 24, "raw24")
    full = np.column_stack((np.ones(len(raw)), raw))
    phrase = full[:, :17]
    structure = np.column_stack((np.ones(len(raw)), raw[:, 16:]))
    denominator = np.linalg.norm(full, axis=1, keepdims=True)
    return dict(full=full/denominator,
                phrase=phrase/np.linalg.norm(phrase, axis=1, keepdims=True),
                structure=structure/np.linalg.norm(structure, axis=1, keepdims=True),
                masked_phrase=phrase/denominator,
                masked_structure=structure/denominator)


def auxiliary(query, support, mode):
    if mode == "none":
        return 0.0
    if mode == "separable":
        return (.5*np.clip(query["phrase"]@support["phrase"].T, -1., 1.)**4 +
                .5*np.clip(query["structure"]@support["structure"].T, -1., 1.)**4)
    key = "structure" if mode == "structure_linear" else mode
    return np.clip(query[key]@support[key].T, -1., 1.)**(1 if mode == "structure_linear" else 4)


def kernel(qe, se, qr, sr, mode):
    if mode not in MODES:
        raise ValueError("unknown frozen kernel mode")
    qe, se = matrix(qe, 384, "query semantic"), matrix(se, 384, "support semantic")
    qr, sr = matrix(qr, 24, "query raw"), matrix(sr, 24, "support raw")
    if len(qe) != len(qr) or len(se) != len(sr):
        raise ValueError("unaligned feature rows")
    return np.clip(qe@se.T, -1., 1.)**4 + auxiliary(projections(qr), projections(sr), mode)


def solve_centered(cross, gram, residuals):
    if not len(gram):
        return np.zeros(len(cross), dtype=np.float64)
    centered = gram-gram.mean(0, keepdims=True)-gram.mean(1, keepdims=True)+gram.mean()
    alpha = np.linalg.solve(centered+RIDGE*np.eye(len(gram)), residuals-np.mean(residuals))
    return (cross-cross.mean(1, keepdims=True)-gram.mean(0, keepdims=True)+gram.mean())@alpha


def centered_correction(qe, se, qr, sr, residuals, mode):
    cross = kernel(qe, se, qr, sr, mode)
    gram = kernel(se, se, sr, sr, mode)
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.shape != (len(gram),) or not np.isfinite(residuals).all():
        raise ValueError("finite aligned residuals required")
    return solve_centered(cross, gram, residuals)
