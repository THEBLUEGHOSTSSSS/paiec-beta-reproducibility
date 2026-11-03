"""Frozen legal streaming policies; no state, data access, or fitted parameters."""
import hashlib
import json
import math
from numbers import Integral, Real

MAX_LABELS = 31
THRESHOLD = .8


def counts(context):
    if not isinstance(context, dict):
        raise TypeError('context must be a dictionary')
    values = []
    for name in ('labels_acquired', 'labels_remaining', 'max_labels', 'items_remaining'):
        value = context[name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError('integer counts required: '+name)
        if value < 0:
            raise ValueError('nonnegative counts required: '+name)
        values.append(int(value))
    acquired, remaining, maximum, items = values
    if maximum != MAX_LABELS or acquired+remaining != maximum:
        raise ValueError('inconsistent total-31 budget')
    if remaining and items == 0:
        raise ValueError('current candidate must be included in items_remaining')
    return acquired, remaining, items


def acquisition_function(input, prediction=None, labeled=None, context=None):
    acquired, remaining, items = counts(context)
    if acquired == MAX_LABELS:
        return False
    if isinstance(prediction, bool) or not isinstance(prediction, Real):
        raise TypeError('finite real prediction required')
    prediction = float(prediction)
    if not math.isfinite(prediction) or not 0 <= prediction <= 1:
        raise ValueError('prediction must be finite and in [0,1]')
    return bool(4*prediction*(1-prediction) >= THRESHOLD or items <= remaining)


def default_uniform(input, context):
    key = json.dumps([context, input], sort_keys=True, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')/2**64


def default_random(input, context):
    acquired, remaining, items = counts(context)
    if acquired == MAX_LABELS:
        return False
    return bool(default_uniform(input, context) < min(1, remaining/items))
