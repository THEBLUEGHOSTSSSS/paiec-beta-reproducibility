"""Pure fixed affine objective and stateless visible-interface prediction.

No filesystem, dataset, source identity, split, or scoring access is used here.
"""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

SUBJECT_FIELDS = ('normalized_name', 'provider', 'release_date', 'access_date',
                  'harness', 'harness_version', 'reasoning_effort', 'subject_features_extra')
ITEM_FIELDS = ('item_content', 'item_features', 'interactors', 'benchmark_id')


def probabilities(q):
    q = np.asarray(q, dtype=np.float64)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('expected finite probabilities in [0,1]')
    return q


def calibrate(q, a, t):
    q = probabilities(q)
    if not math.isfinite(a) or not math.isfinite(t):
        raise ValueError('nonfinite affine parameters')
    return expit(float(a) + float(t) * logit(np.clip(q, 1e-6, 1-1e-6)))


def fit_affine(q, successes, trials, pair_ids):
    q = probabilities(q)
    s, m = np.asarray(successes, float), np.asarray(trials, float)
    groups = np.asarray(pair_ids)
    if (q.ndim != 1 or not len(q) or s.shape != q.shape or m.shape != q.shape
            or groups.shape != q.shape or not np.isfinite(s).all()
            or not np.isfinite(m).all() or np.any(m <= 0) or np.any(s < 0)
            or np.any(s > m)):
        raise ValueError('invalid aligned counts/groups')
    _, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, weights=m)
    weight = 1 / totals[inverse]
    logits = logit(np.clip(q, 1e-6, 1-1e-6))

    def objective(x):
        z = x[0] + x[1]*logits
        value = np.sum(weight*(m*np.logaddexp(0, z)-s*z)) + 2*(x[0]**2+(x[1]-1)**2)
        residual = weight*(m*expit(z)-s)
        gradient = np.array([residual.sum()+4*x[0], residual@logits+4*(x[1]-1)])
        return value, gradient

    result = minimize(objective, [0., 1.], jac=True, method='L-BFGS-B',
                      bounds=[(-2., 2.), (.25, 4.)],
                      options={'ftol': 1e-15, 'gtol': 1e-9, 'maxiter': 1000, 'maxls': 50})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('affine optimizer did not converge: '+str(result.message))
    return tuple(float(v) for v in result.x)


def visible_key(value):
    subject, item = value
    if not isinstance(subject, dict) or not isinstance(item, dict):
        raise ValueError('visible dictionaries required')
    if set(subject) != set(SUBJECT_FIELDS) or set(item) != set(ITEM_FIELDS):
        raise ValueError('exact eight/four visible fields required')
    values = tuple(subject[f] for f in SUBJECT_FIELDS) + tuple(item[f] for f in ITEM_FIELDS)
    if any(not isinstance(v, str) for v in values) or not item['benchmark_id']:
        raise ValueError('visible fields must be strings; alias nonempty')
    return values


def matching(value, labeled):
    key = visible_key(value)
    result = []
    for observed, y in labeled:
        other = visible_key(observed)
        if other[:8] == key[:8] and other[-1] == key[-1]:
            if isinstance(y, (bool, np.bool_)) or y not in (0, 1):
                raise ValueError('binary observations required')
            result.append((other, observed, int(y)))
    # Stable accumulation under evidence permutations; retain repeated entries.
    return sorted(result, key=lambda x: (x[0], x[2]))


def offset(q, outcomes):
    logits = logit(np.clip(probabilities(q), 1e-6, 1-1e-6))
    y = np.asarray(outcomes, float)
    if logits.shape != y.shape or np.any((y != 0) & (y != 1)):
        raise ValueError('unaligned binary evidence')
    d = 0.
    for _ in range(60):
        p = expit(logits+d)
        step = (math.fsum(p-y)+4*d)/(math.fsum(p*(1-p))+4)
        d -= step
        if abs(step) < 1e-13:
            return float(d)
    raise RuntimeError('MAP4 solver did not converge')


def predict_visible(queries, labeled, prior, a=0., t=1., method='affine_v6'):
    """All queries share a visible subject/alias; only current evidence is accepted.

    prior is a callable of the visible [subject,item] only. No indices or counts
    other than purchased binary outcomes enter this function.
    """
    if method not in ('affine_v6', 'v6', 'beta11') or not queries:
        raise ValueError('known method and nonempty queries required')
    first = visible_key(queries[0])
    for query in queries:
        key = visible_key(query)
        if key[:8] != first[:8] or key[-1] != first[-1]:
            raise ValueError('queries must share visible subject/alias')
    observations = matching(queries[0], labeled)
    y = [row[2] for row in observations]
    beta = (sum(y)+1)/(len(y)+2)
    if method == 'beta11':
        return np.full(len(queries), beta)
    if method == 'v6':
        a, t = 0., 1.
    q = calibrate([prior(row[1]) for row in observations], a, t)
    delta = offset(q, y)
    target = calibrate([prior(query) for query in queries], a, t)
    return .5*expit(logit(np.clip(target, 1e-6, 1-1e-6))+delta)+.5*beta
