from __future__ import annotations
import hashlib,json,math,statistics,itertools
from pathlib import Path
from collections import defaultdict,OrderedDict,Counter
import numpy as np
from scipy.special import expit,logit
BUDGETS=[0,1,3,7,15,31]
SEEDS=[0,1,2,3,4]
WEIGHTS=[.1,.2,.2,.2,.2,.1]
from science.affine_calibration_20261015.core import visible_key
from science.strategy.transfer.experiment import seeded
from science.online_acquisition_20261015.stream import Group
from engine import coordinator as Coordinator, setup
from safety import reference_path
ROOT=None
CONTEXTS=("restricted_mixed_balanced","restricted_same_alias","singlepair")
def read(path):return json.loads(reference_path(str(path)).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

class VisiblePrior:
    @staticmethod
    def feature_key(value):
        key = visible_key(value)
        return key[:8]+(key[8],)

    def __init__(self, inputs, probabilities):
        self.values = {}
        maximum = 0.
        for value, probability in zip(inputs, probabilities):
            key = self.feature_key(value)
            if key in self.values:
                maximum = max(maximum, abs(self.values[key]-float(probability)))
            else:
                self.values[key] = float(probability)
        if maximum > 1e-7:
            raise RuntimeError('identical model features have inconsistent priors')
        self.max_visible_error = maximum

    def __call__(self, value):
        return self.values[self.feature_key(value)]

class Catalog:
    def __init__(self):
        self.co = Coordinator(setup())
        self.data = self.co.data
        self.inputs = self.co.inputs
        self.index = self.co.index
        self.pairs = self.co.pairs
        self.targets = sorted(pid for pid in self.pairs if self.co.cost[pid] <= 1000)
        self.excluded = sorted(pid for pid in self.pairs if self.co.cost[pid] > 1000)
        if len(self.targets) != 130 or len(self.excluded) != 17:
            raise RuntimeError('accepted target membership changed')
        self.accepted_panels = {(row['target'], row['seed'], row['context']): row['members']
                                for row in read('panels.json')['panels']}
        self.streams, self.responses, self.response_ids = {}, {}, {}
        for pid in self.targets:
            pair = self.pairs[pid]
            for seed in range(5):
                split = self.co.splits[pid, seed]
                items = sorted(split['acquisition_items'])
                generator = seeded(seed, pid, 'acquisition')
                generator.shuffle(items)
                chosen = [generator.choice(sorted(pair.by_item[item])) for item in items]
                if items[:31] != split['acquired_items'] or [row[0] for row in chosen[:31]] != split['selected_response_ids'] or [row[1] for row in chosen[:31]] != split['acquired_labels']:
                    raise RuntimeError('accepted one-response prefix changed')
                tokens = tuple(self.index[pid, item] for item in items)
                self.streams[pid, seed] = tokens
                for token, (response_id, outcome) in zip(tokens, chosen):
                    self.responses[seed, token] = int(outcome)
                    self.response_ids[seed, token] = response_id

    def panel(self, target, seed, context):
        if context == 'singlepair':
            return [target]
        members = self.co.panel(target, seed, context)
        if members != self.accepted_panels[target, seed, context]:
            raise RuntimeError('accepted target-first panel membership changed')
        return members

    def order(self, members, seed):
        return sorted(members, key=lambda pid: (digest(['online-group-order-v1', seed, pid]), pid))

    def groups(self, members, seed):
        groups = []
        for pid in self.order(members, seed):
            pair = self.pairs[pid]
            subject_alias = 'subject_'+digest(['online-subject-v1', pair.subject_id])[:16]
            alias = self.inputs[self.streams[pid, seed][0]][1]['benchmark_id']
            groups.append(Group(pid, subject_alias, alias, self.streams[pid, seed]))
        return groups

    def evaluation(self, members, seed):
        return [self.index[pid, item] for pid in members for item in self.co.splits[pid, seed]['evaluation_items']]

    def outcomes(self, target, seed):
        pair = self.pairs[target]
        items = self.co.splits[target, seed]['evaluation_items']
        return ([sum(outcome for _, outcome in pair.by_item[item]) for item in items],
                [len(pair.by_item[item]) for item in items])

