from __future__ import annotations
import hashlib,json,math,statistics,itertools
from pathlib import Path
from collections import defaultdict,OrderedDict,Counter
import numpy as np
from scipy.special import expit,logit
BUDGETS=[0,1,3,7,15,31]
SEEDS=[0,1,2,3,4]
WEIGHTS=[.1,.2,.2,.2,.2,.1]
from science.autopipeline.maths import fit_logit_offset
from science.optimization.posterior import posterior_predict

def score_pair(pair, stats, priors, job, nodes):
    target = np.array([priors[i] for i in stats['evaluation_items']])
    counts = np.array([len(pair.by_item[i]) for i in stats['evaluation_items']])
    successes = np.array([sum(y for _, y in pair.by_item[i]) for i in stats['evaluation_items']])
    curve = []
    for budget in BUDGETS:
        acquired = [priors[i] for i in stats['acquired_items'][:budget]]
        labels = stats['acquired_labels'][:budget]
        if job['kind'] == 'posterior':
            calibrated = posterior_predict(acquired, labels, target, strength=job['strength'], nodes=nodes)
        else:
            delta = fit_logit_offset(acquired, labels, job['strength'])
            calibrated = expit(logit(np.clip(target, 1e-6, 1-1e-6))+delta)
        prediction = (1-job['blend'])*calibrated + job['blend']*(sum(labels)+1)/(budget+2)
        curve.append(float(np.sum(successes*(1-prediction)**2+(counts-successes)*prediction**2)/counts.sum()))
    return math.fsum(w*b for w, b in zip(WEIGHTS, curve)), curve
