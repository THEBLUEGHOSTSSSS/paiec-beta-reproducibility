from __future__ import annotations
import hashlib,json,math,statistics,itertools
from pathlib import Path
from collections import defaultdict,OrderedDict,Counter
import numpy as np
from scipy.special import expit,logit
BUDGETS=[0,1,3,7,15,31]
SEEDS=[0,1,2,3,4]
WEIGHTS=[.1,.2,.2,.2,.2,.1]
from science.optimization.evaluator import score_pair
from science.scaling.head_campaign import aggregate_rows
MODES={"all":set(BUDGETS),"early01":{0,1},"early0137":{0,1,3,7}}

def compose_rows(learned_rows, reference_rows, mode):
    """Splice complete matched trajectories, without resampling or mutation."""
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError('unknown predeclared mode')
    aggregate_rows(learned_rows)
    aggregate_rows(reference_rows)
    learned = {(row['pair_id'], row['seed'], row['budget']): row for row in learned_rows}
    reference = {(row['pair_id'], row['seed'], row['budget']): row for row in reference_rows}
    if learned.keys() != reference.keys():
        raise ValueError('different pair/seed/budget universes')
    output = []
    for row in reference_rows:
        key = row['pair_id'], row['seed'], row['budget']
        if learned[key]['benchmark'] != row['benchmark']:
            raise ValueError('benchmark mismatch')
        chosen = learned[key] if row['budget'] in MODES[mode] else row
        output.append(dict(chosen, config_id='ensemble_' + mode))
    aggregate_rows(output)
    return output

def config(seed):
    return dict(kind='bilinear', rank=8, hidden=32, lr=.01, weight_decay=.03,
                steps=256, batch_size=512, seed=seed, max_seconds=60)

def validate_rows(data, rows):
    summary = aggregate_rows(rows)
    wanted = {pair.pair_id: pair.benchmark for pair in data.pairs}
    if len(rows) != 4410 or {row['pair_id']: row['benchmark'] for row in rows} != wanted:
        raise RuntimeError('unexpected scoring universe')
    return summary

def score(data, priors, identifier, strength=4.):
    rows = []
    for pair in data.pairs:
        for seed in SEEDS:
            _, curve = score_pair(pair, data.splits[(pair.pair_id, seed)],
                                  priors[pair.pair_id], dict(kind='map', strength=strength, blend=.5), 121)
            rows.extend(dict(pair_id=pair.pair_id, benchmark=pair.benchmark, seed=seed,
                             budget=budget, brier=float(value), config_id=identifier)
                        for budget, value in zip(BUDGETS, curve))
    validate_rows(data, rows)
    return rows
