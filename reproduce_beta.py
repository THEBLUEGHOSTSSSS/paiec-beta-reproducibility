"""Standalone deterministic replay of the frozen two-source Beta diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from model import predict
import argparse

HERE = Path(__file__).resolve().parent
REVISION = '54074453da7b9911be0f48e4c560be547bc84d21'
DATASETS = ['real_webagents', 'researchcodebench']
BUDGETS = [0, 1, 3, 7, 15, 31]
METHODS = ['constant_05', 'official_mean', 'beta11']


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_binary(values, allow_empty):
    if not values and not allow_empty:
        raise ValueError('Empty evaluation vector')
    if any(v not in (0, 1) for v in values):
        raise ValueError('Nonbinary or missing response')


def score_labels(acquired, evaluation):
    """Predict with fixed formulas and score binary responses independently."""
    check_binary(acquired, True)
    check_binary(evaluation, False)
    n = len(acquired)
    k = sum(acquired)
    prevalence = sum(evaluation) / len(evaluation)
    predictions = {'constant_05': .5, 'official_mean': k/n if n else .5,
                   'beta11': (k+1)/(n+2)}
    return {name: {'prediction': p,
                   'brier': p*p + (1-2*p)*prevalence}
            for name, p in predictions.items()}


def rng_for(seed, pair_id, phase):
    raw = json.dumps([seed, pair_id, phase], separators=(',', ':')).encode('utf-8')
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest(), 'big'))


def partition_and_acquire(rows, seed, pair_id):
    """Select using identities only; changing outcomes cannot change selection."""
    ids = set()
    grouped = defaultdict(list)
    for row in rows:
        if row['response_id'] in ids:
            raise ValueError('Duplicate response identifier')
        ids.add(row['response_id'])
        grouped[row['item_id']].append(row)
    check_binary([r['response'] for r in rows], False)
    for values in grouped.values():
        values.sort(key=lambda r: r['response_id'])
    items = sorted(grouped)
    if len(items) < 80:
        raise ValueError('Fewer than 80 distinct items')
    rng_for(seed, pair_id, 'split').shuffle(items)
    midpoint = len(items)//2
    pool = sorted(items[:midpoint])
    evaluation = sorted(items[midpoint:])
    order = list(pool)
    generator = rng_for(seed, pair_id, 'acquisition')
    generator.shuffle(order)
    selected = [generator.choice(grouped[item]) for item in order[:31]]
    outcomes = [r['response'] for item in evaluation for r in grouped[item]]
    return {'acquisition_items': pool, 'evaluation_items': evaluation,
            'selected_response_ids': [r['response_id'] for r in selected],
            'acquired_outcomes': [r['response'] for r in selected],
            'eval_outcomes': outcomes}


def simulate(rows, seed, pair_id, subject=None, items=None):
    evidence = partition_and_acquire(rows, seed, pair_id)
    metrics = []
    for n in BUDGETS:
        scores = score_labels(evidence['acquired_outcomes'][:n], evidence['eval_outcomes'])
        if subject is not None:
            lookup = {r['response_id']: r for r in rows}
            labels = [[[subject, items[lookup[rid]['item_id']]], lookup[rid]['response']]
                      for rid in evidence['selected_response_ids'][:n]]
            p = predict([subject, items[evidence['evaluation_items'][0]]], labels)
            if p != predict([subject, items[evidence['evaluation_items'][-1]]], labels):
                raise ValueError('Item-dependent candidate unsupported')
            if abs(p - scores['beta11']['prediction']) > 1e-15:
                raise ValueError('Candidate differs from declared Beta formula')
            scores['beta11'] = {'prediction': p, 'brier': math.fsum(
                (p-y)**2 for y in evidence['eval_outcomes']) / len(evidence['eval_outcomes'])}
        for method in METHODS:
            metrics.append({'pair_id': pair_id, 'seed': seed, 'budget': n,
                            'method': method, **scores[method],
                            'n_eval_responses': len(evidence['eval_outcomes']),
                            'n_eval_items': len(evidence['evaluation_items']),
                            'label_response_ids': evidence['selected_response_ids'][:n]})
    return evidence, metrics


def trapezoid(curve):
    if sorted(curve) != BUDGETS:
        raise ValueError('Incomplete budget curve')
    values = [curve[n] for n in BUDGETS]
    return math.fsum((left+right)/2 for left, right in zip(values, values[1:]))/5


def aggregate(rows):
    curves = defaultdict(dict)
    for r in rows:
        key = (r['pair_id'], r['seed'], r['method'])
        if r['budget'] in curves[key]:
            raise ValueError('Duplicate budget record')
        curves[key][r['budget']] = r['brier']
    grouped = defaultdict(list)
    for (pair, seed, method), curve in curves.items():
        grouped[seed, method].append(trapezoid(curve))
    result = {}
    for method in sorted({r['method'] for r in rows}):
        per_seed = {str(seed): statistics.mean(grouped[seed, method])
                    for seed in sorted({r['seed'] for r in rows})}
        values = list(per_seed.values())
        result[method] = {'alc_mean': statistics.mean(values),
                          'seed_sd': statistics.stdev(values) if len(values)>1 else 0.,
                          'seed_min': min(values), 'seed_max': max(values),
                          'seed_alc': per_seed}
    return result


SUBJECT_FIELDS = ['normalized_name', 'provider', 'release_date', 'access_date',
                  'harness', 'harness_version', 'reasoning_effort', 'subject_features_extra']


def text(value):
    if value is None:
        return ''
    if not isinstance(value, str):
        raise ValueError('Expected string attribute')
    return value


def verify_inputs(data_root):
    manifest = json.loads((HERE/'data_sources.json').read_text())
    expected = {(d, f'{name}.parquet') for d in DATASETS
                for name in ['benchmarks', 'subjects', 'items', 'responses']}
    seen = set()
    for entry in manifest['files']:
        parts = Path(entry['local_path']).parts
        key = (parts[-2], parts[-1])
        if key not in expected or key in seen:
            raise ValueError('Invalid source inventory')
        seen.add(key)
        if file_hash(data_root/key[0]/key[1]) != entry['expected_sha256']:
            raise ValueError('Source hash mismatch: ' + '/'.join(key))
    if seen != expected or manifest['revision'] != REVISION:
        raise ValueError('Incomplete source inventory or wrong revision')


def load_sources(data_root):
    import pyarrow.parquet as pq
    verify_inputs(data_root)
    records = {}
    for dataset in DATASETS:
        tables = {name: pq.read_table(data_root/dataset/(name+'.parquet')).to_pylist()
                  for name in ['benchmarks', 'subjects', 'items', 'responses']}
        if len(tables['benchmarks']) != 1:
            raise ValueError('Expected one benchmark')
        meta = tables['benchmarks'][0]
        if (meta['release'], meta['response_type'], meta['granularity']) != ('public', 'binary', 'item'):
            raise ValueError('Ineligible metadata')
        subjects = {s['subject_id']: {k: text(s[k]) for k in SUBJECT_FIELDS} for s in tables['subjects']}
        items = {i['item_id']: i for i in tables['items']}
        responses = tables['responses']
        if len(subjects) != len(tables['subjects']) or len(items) != len(tables['items']):
            raise ValueError('Duplicate primary key')
        if len({json.dumps(s, sort_keys=True) for s in subjects.values()}) != len(subjects):
            raise ValueError('Ambiguous subject identity')
        if len({i['content'] for i in items.values()}) != len(items):
            raise ValueError('Duplicate item content')
        if len({r['response_id'] for r in responses}) != len(responses):
            raise ValueError('Duplicate response key')
        check_binary([r['response'] for r in responses], False)
        groups = defaultdict(list)
        for r in responses:
            if r['subject_id'] not in subjects or r['item_id'] not in items:
                raise ValueError('Broken foreign key')
            if r['benchmark_id'] != meta['benchmark_id'] or items[r['item_id']]['benchmark_id'] != meta['benchmark_id']:
                raise ValueError('Benchmark mismatch')
            if text(r['test_condition']):
                raise ValueError('Unsupported condition mapping')
            groups[r['subject_id']].append(r)
        for sid, rows in groups.items():
            if len({r['item_id'] for r in rows}) < 80:
                continue
            visible_items = {}
            for r in rows:
                i = items[r['item_id']]
                visible = {'item_content': text(i['content']), 'item_features': text(i['item_features']),
                           'interactors': text(r['interactors']), 'benchmark_id': 'public_' + dataset}
                if r['item_id'] in visible_items and visible_items[r['item_id']] != visible:
                    raise ValueError('Varying interactors')
                visible_items[r['item_id']] = visible
            records[dataset+':'+sid] = (rows, subjects[sid], visible_items)
    if len(records) != 57:
        raise ValueError('Expected 57 eligible pairs')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-inputs', action='store_true')
    args = parser.parse_args()
    verify_inputs(args.data_root)
    if args.verify_inputs:
        print('PASS: all eight pinned input hashes')
        return
    if args.output is None:
        parser.error('--output is required for replay')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('--output must be new or empty')
    groups = load_sources(args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with (args.output/'splits.jsonl').open('x') as handle:
        for pair in sorted(groups):
            rows, subject, items = groups[pair]
            for seed in range(100):
                split, measured = simulate(rows, seed, pair, subject, items)
                kept = {k: split[k] for k in ['acquisition_items', 'evaluation_items', 'selected_response_ids']}
                handle.write(json.dumps({'pair_id': pair, 'seed': seed, **kept}, sort_keys=True)+'\n')
                for r in measured:
                    r['benchmark'] = pair.split(':')[0]
                    records.append(r)
    with (args.output/'metrics.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for r in records:
            writer.writerow({**r, 'label_response_ids': json.dumps(r['label_response_ids'])})
    summary = {'seeds': list(range(100)), 'pair_count': len(groups), 'datasets': DATASETS,
               'data_revision': REVISION, 'overall': aggregate(records),
               'by_benchmark': {d: aggregate([r for r in records if r['benchmark']==d]) for d in DATASETS},
               'additional_seeds20_99': aggregate([r for r in records if r['seed']>=20])}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v['alc_mean'] for k,v in summary['overall'].items()}, indent=2))


if __name__ == '__main__':
    main()


