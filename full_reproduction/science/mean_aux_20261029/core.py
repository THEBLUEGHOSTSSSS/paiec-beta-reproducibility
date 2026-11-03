"""Source-excluded relational sampling and probability-difference objective."""
from collections import defaultdict
import math

import numpy as np
import torch

from science.affine_calibration_20261015.core import visible_key
from science.scaling.heads import training_distribution

CONFIG = dict(kind="bilinear", rank=8, hidden=64, lr=.01, weight_decay=.03,
              steps=256, batch_size=512, seed=0)
ARMS = ("pooled", "shuffled", "mean")


def relations(records, inputs, excluded):
    cells = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        source = record["benchmark"]
        if source in excluded:
            continue
        key = visible_key(inputs[index])
        cells[source, key[8]][key[:8]].append(index)
    return [dict(source=source, content=content,
                 subjects=[dict(subject=list(subject), indices=sorted(indices))
                           for subject, indices in sorted(subjects.items())])
            for (source, content), subjects in sorted(cells.items())
            if len(subjects) >= 2]


def schedule(records, inputs, excluded, seed):
    indices, _, weights = training_distribution(records, excluded)
    main = np.random.default_rng(seed).choice(indices, size=(256, 512),
                                               replace=True, p=weights)
    by_source = defaultdict(list)
    for cell in relations(records, inputs, excluded):
        by_source[cell["source"]].append(cell)
    sources = sorted(by_source)
    if not sources:
        raise ValueError("no source has a same-content cross-subject relation")
    generator = np.random.default_rng(38000 + seed)
    endpoints = np.empty((65536, 2), dtype=np.int64)
    for position in range(len(endpoints)):
        source = sources[int(generator.integers(len(sources)))]
        contents = by_source[source]
        cell = contents[int(generator.integers(len(contents)))]
        left_subject, right_subject = generator.choice(len(cell["subjects"]), 2, replace=False)
        left = cell["subjects"][left_subject]["indices"]
        right = cell["subjects"][right_subject]["indices"]
        endpoints[position] = (left[int(generator.integers(len(left)))],
                               right[int(generator.integers(len(right)))])
    strata = defaultdict(list)
    for index in indices:
        key = visible_key(inputs[index])
        strata[records[index]["benchmark"], key[:8], int(records[index]["trials"])].append(
            (key[8], int(index)))
    permutation = np.full(len(records), -1, dtype=np.int64)
    generator = np.random.default_rng(38100 + seed)
    for stratum in sorted(strata):
        ordered = np.asarray([index for _, index in sorted(strata[stratum])], dtype=np.int64)
        shift = int(generator.integers(1, len(ordered))) if len(ordered) > 1 else 0
        permutation[ordered] = np.roll(ordered, -shift)
    return dict(main_indices=main, left_indices=endpoints[:, 0].reshape(256, 256),
                right_indices=endpoints[:, 1].reshape(256, 256), shuffle_map=permutation)


def auxiliary_loss(left_logits, right_logits, left_rates, right_rates):
    return ((torch.sigmoid(left_logits) - torch.sigmoid(right_logits)
             - (left_rates - right_rates)) ** 2).mean()


def mean_targets(records, inputs, excluded):
    strata = defaultdict(list)
    result = np.full(len(records), np.nan, dtype=np.float64)
    for index, record in enumerate(records):
        if record["benchmark"] in excluded:
            continue
        key = visible_key(inputs[index])
        strata[record["benchmark"], key[:8], int(record["trials"])].append(index)
    for indices in strata.values():
        rates = [records[index]["successes"] / records[index]["trials"] for index in indices]
        total = math.fsum(rates)
        for index, rate in zip(indices, rates):
            result[index] = (total - rate) / (len(indices) - 1) if len(indices) > 1 else rate
    return result
