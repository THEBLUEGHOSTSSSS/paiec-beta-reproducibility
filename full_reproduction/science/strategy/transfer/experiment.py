from __future__ import annotations
import hashlib,json,math,statistics,itertools
from pathlib import Path
from collections import defaultdict,OrderedDict,Counter
import numpy as np
from scipy.special import expit,logit
BUDGETS=[0,1,3,7,15,31]
SEEDS=[0,1,2,3,4]
WEIGHTS=[.1,.2,.2,.2,.2,.1]
import numbers,random
from dataclasses import dataclass
from science.autopipeline.experiments import FIELDS as SUBJECT_FIELDS
DATA_ROOT=None

def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            hasher.update(block)
    return hasher.hexdigest()

def finite_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)

def binary_labels(labels):
    values = list(labels)
    if any(not finite_number(value) or value not in (0, 1) for value in values):
        raise ValueError('Labels must be exact finite numeric 0/1; no coercion')
    return values

def probability(value):
    if not finite_number(value) or not 0 <= value <= 1:
        raise ValueError('Probability must be finite and in [0,1]')
    return float(value)

def subject_key(subject):
    def normalize(value):
        if value is None or (isinstance(value, numbers.Real) and math.isnan(value)):
            return ''
        return str(value)
    return tuple(normalize(subject.get(field)) for field in SUBJECT_FIELDS)

def seeded(seed, pair_id, phase):
    payload = json.dumps([seed, pair_id, phase], separators=(',', ':')).encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), 'big'))

@dataclass
class Pair:
    benchmark: str
    subject_id: str
    key: tuple
    by_item: dict
    rate: float

    @property
    def pair_id(self):
        return self.benchmark + ':' + self.subject_id

def validate_references(benchmark, subjects, items, responses):
    for table, key in [(subjects, 'subject_id'), (items, 'item_id'), (responses, 'response_id')]:
        ids = [row[key] for row in table]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError('Invalid or duplicate ' + key)
    subject_ids = {row['subject_id'] for row in subjects}
    item_ids = {row['item_id'] for row in items}
    if any(row['benchmark_id'] != benchmark for row in items + responses):
        raise ValueError('Cross-benchmark reference')
    for row in responses:
        if row['subject_id'] not in subject_ids or row['item_id'] not in item_ids:
            raise ValueError('Unknown subject/item reference')

def load_data(selection):
    import pandas as pd
    pairs, exclusions, audits, hashes = [], [], [], {}
    for benchmark in selection['benchmarks']:
        tables = {}
        for name in ['benchmarks', 'subjects', 'items', 'responses']:
            path = DATA_ROOT / benchmark / (name + '.parquet')
            hashes[str(path.relative_to(DATA_ROOT))] = digest(path)
            tables[name] = pd.read_parquet(path).to_dict('records')
        if len(tables['benchmarks']) != 1:
            raise ValueError('Expected exactly one benchmark record')
        meta = tables['benchmarks'][0]
        if meta['benchmark_id'] != benchmark or (meta['release'], meta['response_type'], meta['granularity']) != ('public', 'binary', 'item'):
            raise ValueError('Ineligible benchmark metadata')
        scale = json.loads(meta['response_scale'])
        if scale['direction'] != 'higher_is_better' or binary_labels(scale['values']) != [0, 1]:
            raise ValueError('Unsupported label direction/scale')
        validate_references(benchmark, tables['subjects'], tables['items'], tables['responses'])
        subjects = {row['subject_id']: row for row in tables['subjects']}
        grouped = defaultdict(list)
        for row in tables['responses']:
            grouped[row['subject_id']].append(row)
        included = 0
        for subject_id in sorted(subjects):
            rows = grouped[subject_id]
            item_count = len({row['item_id'] for row in rows})
            invalid = sum(not finite_number(row['response']) or row['response'] not in (0, 1) for row in rows)
            reasons = []
            if benchmark == 'igakuqa' and subject_id == '0816aca9fae59781':
                reasons.append('explicit_student_majority_human_comparator')
            if invalid:
                reasons.append('invalid_binary_labels_whole_pair')
            if item_count < 80:
                reasons.append('fewer_than_80_distinct_observed_items')
            if reasons:
                exclusions.append(dict(benchmark=benchmark, subject_id=subject_id, reasons=reasons,
                                       observed_items=item_count, response_count=len(rows), invalid_labels=invalid))
                continue
            by_item = defaultdict(list)
            for row in sorted(rows, key=lambda record: record['response_id']):
                by_item[row['item_id']].append((row['response_id'], int(row['response'])))
            values = binary_labels(row['response'] for row in rows)
            pairs.append(Pair(benchmark, subject_id, subject_key(subjects[subject_id]), dict(by_item), statistics.mean(values)))
            included += 1
        audits.append(dict(benchmark=benchmark, eligible_pairs=included, raw_subjects=len(subjects),
                           raw_items=len(tables['items']), raw_responses=len(tables['responses']),
                           invalid_labels=sum(not finite_number(row['response']) or row['response'] not in (0, 1) for row in tables['responses']),
                           referential_validation='PASS', exact_binary_check='PASS_FOR_INCLUDED_PAIRS',
                           metadata_scale=scale, license=meta.get('license', 'unknown')))
    return pairs, exclusions, audits, hashes

def split_pair(pair, seed):
    item_ids = sorted(pair.by_item)
    if len(item_ids) < 80:
        raise ValueError('At least 80 distinct observed items required')
    for trials in pair.by_item.values():
        if not trials:
            raise ValueError('Observed item without trials')
        binary_labels(label for _, label in trials)
    response_ids = [response_id for trials in pair.by_item.values() for response_id, _ in trials]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError('Duplicate response_id')
    seeded(seed, pair.pair_id, 'split').shuffle(item_ids)
    acquisition = item_ids[:len(item_ids) // 2]
    evaluation = sorted(item_ids[len(item_ids) // 2:])
    order = sorted(acquisition)
    generator = seeded(seed, pair.pair_id, 'acquisition')
    generator.shuffle(order)
    chosen = [generator.choice(sorted(pair.by_item[item])) for item in order[:max(BUDGETS)]]
    outcomes = [label for item in evaluation for _, label in pair.by_item[item]]
    labels = [label for _, label in chosen]
    return dict(pair_id=pair.pair_id, benchmark=pair.benchmark, seed=seed,
                acquisition_items=sorted(acquisition), evaluation_items=evaluation,
                acquired_items=order[:max(BUDGETS)], selected_response_ids=[identifier for identifier, _ in chosen],
                acquired_labels=labels, acquired_sums={budget: sum(labels[:budget]) for budget in BUDGETS},
                eval_count=len(outcomes), eval_successes=sum(outcomes), eval_item_count=len(evaluation))
