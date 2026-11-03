from __future__ import annotations
import hashlib,json,math,statistics,itertools
from pathlib import Path
from collections import defaultdict,OrderedDict,Counter
import numpy as np
from scipy.special import expit,logit
BUDGETS=[0,1,3,7,15,31]
SEEDS=[0,1,2,3,4]
WEIGHTS=[.1,.2,.2,.2,.2,.1]


def numpy_forward(features, state):
    interaction = (features[:, :384] @ state["text_factor"]) * (features[:, 384:] @ state["metadata_factor"])
    logits = features @ state["linear_weight"] + state["linear_bias"]
    logits = logits + interaction.sum(axis=1, dtype=np.float32) / np.float32(math.sqrt(8))
    exponential = np.exp(-np.abs(logits))
    return np.where(logits >= 0, 1 / (1 + exponential), exponential / (1 + exponential)).astype(np.float32)
