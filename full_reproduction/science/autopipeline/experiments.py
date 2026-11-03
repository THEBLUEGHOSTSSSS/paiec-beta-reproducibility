from __future__ import annotations
import hashlib,json,math,statistics,itertools
from pathlib import Path
from collections import defaultdict,OrderedDict,Counter
import numpy as np
from scipy.special import expit,logit
BUDGETS=[0,1,3,7,15,31]
SEEDS=[0,1,2,3,4]
WEIGHTS=[.1,.2,.2,.2,.2,.1]
FIELDS=("normalized_name","provider","release_date","access_date","harness","harness_version","reasoning_effort","subject_features_extra")

def summarize(rows):
    if not rows:
        return {}
    curves = defaultdict(dict)
    for row in rows:
        key = row["pair_id"], row["benchmark"], row["seed"]
        if row["budget"] in curves[key]:
            raise ValueError("duplicate score")
        curves[key][row["budget"]] = row["brier"]
    pair_seed, benchmark, seeds = [], defaultdict(list), defaultdict(list)
    for (_, name, seed), curve in curves.items():
        if set(curve) != set(BUDGETS):
            raise ValueError("incomplete scoring curve")
        value = math.fsum(weight * curve[budget] for weight, budget in zip(WEIGHTS, BUDGETS))
        pair_seed.append(value)
        benchmark[name].append(value)
        seeds[str(seed)].append(value)
    return {"alc": statistics.mean(pair_seed), "macro_benchmark_alc": statistics.mean(statistics.mean(values) for values in benchmark.values()),
            "per_benchmark": {name: statistics.mean(values) for name, values in benchmark.items()},
            "per_seed": {name: statistics.mean(values) for name, values in seeds.items()},
            "per_budget": {str(budget): statistics.mean(row["brier"] for row in rows if row["budget"] == budget) for budget in BUDGETS},
            "score_rows": len(rows), "pair_seed_count": len(pair_seed)}
