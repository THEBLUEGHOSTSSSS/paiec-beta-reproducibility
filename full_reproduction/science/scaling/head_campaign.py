from __future__ import annotations
import hashlib,json,math,statistics,itertools
from pathlib import Path
from collections import defaultdict,OrderedDict,Counter
import numpy as np
from scipy.special import expit,logit
BUDGETS=[0,1,3,7,15,31]
SEEDS=[0,1,2,3,4]
WEIGHTS=[.1,.2,.2,.2,.2,.1]
from science.autopipeline.experiments import summarize

def aggregate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty metric rows required')
    groups, benchmarks, config_ids = {}, {}, set()
    expected = set(itertools.product(SEEDS, BUDGETS))
    for row in rows:
        pair, benchmark = row['pair_id'], row['benchmark']
        if not isinstance(pair, str) or not isinstance(benchmark, str):
            raise ValueError('invalid pair/benchmark')
        if pair in benchmarks and benchmarks[pair] != benchmark:
            raise ValueError('pair benchmark mismatch')
        benchmarks[pair] = benchmark
        key = (row['seed'], row['budget'])
        value = row['brier']
        if (type(row['seed']) is not int or type(row['budget']) is not int
                or key not in expected or type(value) not in (int, float)
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('invalid metric value or trajectory')
        if key in groups.setdefault(pair, set()):
            raise ValueError('duplicate metric row')
        groups[pair].add(key)
        config_ids.add(row.get('config_id'))
    if any(keys != expected for keys in groups.values()) or len(config_ids) != 1:
        raise ValueError('incomplete trajectory or mixed configuration')
    return summarize(rows)
