from __future__ import annotations
import hashlib,json,math,statistics,itertools
from pathlib import Path
from collections import defaultdict,OrderedDict,Counter
import numpy as np
from scipy.special import expit,logit
BUDGETS=[0,1,3,7,15,31]
SEEDS=[0,1,2,3,4]
WEIGHTS=[.1,.2,.2,.2,.2,.1]
from science.scaling import heads

def config(seed):
    return dict(kind='bilinear', rank=8, hidden=32, lr=.01, weight_decay=.03,
                steps=256, batch_size=512, seed=seed, max_seconds=60)

def prepare_training(records, matrix):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError('Nonempty records required')
    if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float32 or matrix.shape != (len(records), 512) or not np.isfinite(matrix).all():
        raise ValueError('Finite float32 N x 512 matrix required')
    sentinel = '__export_v6_unlabeled_sentinel__'
    if sentinel in {r['benchmark'] for r in records}:
        raise ValueError('Sentinel collision')
    augmented = [dict(row) for row in records] + [{'benchmark': sentinel}]
    features = np.concatenate([matrix, np.zeros((1, 512), np.float32)], axis=0)
    indices, rates, weights = heads.training_distribution(augmented, {sentinel})
    if not np.array_equal(indices, np.arange(len(records))):
        raise RuntimeError('Full-development bridge index mismatch')
    return augmented, features, {sentinel}
