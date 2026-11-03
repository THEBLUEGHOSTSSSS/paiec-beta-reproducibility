"""Pure additive centered-kernel math and frozen gates."""
from __future__ import annotations

import numpy as np

RIDGE = 4.0
SCALE = 0.5


def _matrix(value, name, width=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty-width matrix")
    if width is not None and result.shape[1] != width:
        raise ValueError(f"{name} has wrong width")
    if len(result) and not np.allclose(np.linalg.norm(result, axis=1), 1., rtol=0., atol=1e-6):
        raise ValueError(f"{name} must be unit normalized")
    return result


def centered_correction(query_semantic, support_semantic, query_demand,
                        support_demand, residuals):
    """Return the unscaled centered correction for the additive kernel."""
    qe = _matrix(query_semantic, "query_semantic")
    se = _matrix(support_semantic, "support_semantic", qe.shape[1])
    qd = _matrix(query_demand, "query_demand")
    sd = _matrix(support_demand, "support_demand", qd.shape[1])
    if len(qe) != len(qd) or len(se) != len(sd):
        raise ValueError("semantic and demand rows differ")
    residual = np.asarray(residuals, dtype=np.float64)
    if residual.shape != (len(se),) or not np.isfinite(residual).all():
        raise ValueError("residuals must be finite and support-aligned")
    if not len(se):
        return np.zeros(len(qe), dtype=np.float64)
    gram = (np.clip(se @ se.T, -1., 1.) ** 4 +
            np.clip(sd @ sd.T, -1., 1.) ** 4)
    cross = (np.clip(qe @ se.T, -1., 1.) ** 4 +
             np.clip(qd @ sd.T, -1., 1.) ** 4)
    n = len(se)
    centered_residual = residual - residual.mean()
    centered_gram = (gram - gram.mean(axis=0, keepdims=True) -
                     gram.mean(axis=1, keepdims=True) + gram.mean())
    alpha = np.linalg.solve(centered_gram + RIDGE * np.eye(n), centered_residual)
    centered_cross = (cross - cross.mean(axis=1, keepdims=True) -
                      gram.mean(axis=0, keepdims=True) + gram.mean())
    result = centered_cross @ alpha
    if not np.isfinite(result).all():
        raise FloatingPointError("kernel correction is nonfinite")
    return result


def material_gate(main, headtail, beta, seed_triplets, stress_triplets):
    """The unchanged public material gate (smaller risk is better)."""
    epsilon = 1e-12
    gain = ((headtail - main) >= .002 - epsilon and
            (beta - main) >= .002 - epsilon)
    wins = sum(row[0] < row[1] - epsilon and row[0] < row[2] - epsilon
               for row in seed_triplets)
    stable = all(row[0] <= row[1] + epsilon and row[0] <= row[2] + epsilon
                 for row in stress_triplets)
    return dict(passed=bool(gain and wins >= 4 and stable), gain_pass=bool(gain),
                joint_seed_wins=int(wins), stress_pass=bool(stable),
                gain_vs_headtail=float(headtail-main), gain_vs_beta=float(beta-main),
                epsilon=epsilon)


def structural_gate(main, oldkernel, shuffled, seed_triplets, stress_triplets):
    """Frozen structural gate against old kernel and shuffled demand."""
    epsilon = 1e-12
    mean_pass = main < oldkernel - epsilon and main < shuffled - epsilon
    wins = sum(row[0] < row[1] - epsilon and row[0] < row[2] - epsilon
               for row in seed_triplets)
    stable = all(row[0] <= row[1] + epsilon and row[0] <= row[2] + epsilon
                 for row in stress_triplets)
    return dict(passed=bool(mean_pass and wins >= 4 and stable),
                mean_pass=bool(mean_pass), joint_seed_wins=int(wins),
                stress_pass=bool(stable), gain_vs_oldkernel=float(oldkernel-main),
                gain_vs_shuffled=float(shuffled-main), epsilon=epsilon)

