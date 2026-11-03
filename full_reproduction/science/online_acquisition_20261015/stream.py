"""Visible-only predictor and trusted, checkpoint-barrier streaming coordinator."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import expit, logit
from science.affine_calibration_20261015.core import visible_key, offset
from .core import acquisition_function, default_random, default_uniform

CAPS = (0, 1, 3, 7, 15, 31)


def visible_group(value):
    key = visible_key(value)
    return key[:8]+(key[-1],)


def copied(value):
    return copy.deepcopy(value)


class VisiblePredictor:
    def __init__(self, prior):
        self.prior = prior

    def states(self, queries, labeled):
        groups = {visible_group(query): [] for query in queries}
        for value, outcome in labeled:
            group = visible_group(value)
            if group in groups:
                if isinstance(outcome, bool) or outcome not in (0, 1):
                    raise ValueError('binary purchased evidence required')
                groups[group].append((visible_key(value), int(outcome), value))
        result = {}
        for group, observations in groups.items():
            observations.sort(key=lambda row: (row[0], row[1]))
            outcomes = [row[1] for row in observations]
            priors = [self.prior(row[2]) for row in observations]
            delta = offset(priors, outcomes)
            beta = (sum(outcomes)+1)/(len(outcomes)+2)
            result[group] = (float(delta), float(beta), len(outcomes))
        return result

    def evaluate(self, queries, labeled):
        states = self.states(queries, labeled)
        probabilities, betas = [], []
        for query in queries:
            delta, beta, count = states[visible_group(query)]
            prior = float(self.prior(query))
            if not math.isfinite(prior) or not 0 <= prior <= 1:
                raise ValueError('invalid frozen prior')
            probabilities.append(float(.5*expit(logit(np.clip(prior, 1e-6, 1-1e-6))+delta)+.5*beta))
            betas.append(beta)
        return probabilities, betas, states

    def predict(self, input, labeled):
        return self.evaluate([input], labeled)[0][0]


@dataclass(frozen=True)
class Group:
    name: str
    subject_alias: str
    benchmark_alias: str
    stream: tuple


class Replay:
    """Only this trusted coordinator resolves purchased labels or tracks caps.

    Group streams hold opaque coordinator tokens, not future outcomes. Predictors
    receive copied visible inputs and the complete current purchased label list.
    Evaluation outcomes are neither accepted nor inspected by this class.
    """
    def __init__(self, groups, visible, resolve, prior, policy='uncertainty',
                 predictor_factory=VisiblePredictor, candidate=None):
        if policy not in ('uncertainty', 'random'):
            raise ValueError('unknown policy')
        self.groups = tuple(groups)
        self.visible = visible
        self.resolve = resolve
        self.prior = prior
        self.policy = policy
        self.predictor_factory = predictor_factory
        self.candidate = candidate or acquisition_function
        self.positions = [0]*len(groups)
        self.acquired = [0]*len(groups)
        self.labeled = []
        self.purchased = []
        self.decisions = []
        self.barriers = []
        self.last_cap = -1

    def advance(self, cap):
        if cap not in CAPS or cap <= self.last_cap:
            raise ValueError('strictly increasing frozen checkpoints required')
        predictor = self.predictor_factory(self.prior)
        for group_index, group in enumerate(self.groups):
            while self.acquired[group_index] < cap and self.positions[group_index] < len(group.stream):
                position = self.positions[group_index]
                token = group.stream[position]
                current = self.visible(token)
                context = dict(subject_id=group.subject_alias, benchmark_id=group.benchmark_alias,
                               labels_remaining=31-self.acquired[group_index],
                               labels_acquired=self.acquired[group_index], max_labels=31,
                               items_remaining=len(group.stream)-position)
                if self.policy == 'random':
                    probability = None
                    diagnostic = default_uniform(current, context)
                    decision = default_random(copied(current), copied(context))
                else:
                    probability = predictor.predict(copied(current), copied(self.labeled))
                    diagnostic = probability
                    decision = self.candidate(copied(current), prediction=probability,
                                              labeled=copied(self.labeled), context=copied(context))
                if type(decision) is not bool:
                    raise ValueError('native bool required')
                self.positions[group_index] += 1
                self.decisions.append([group_index, position, cap, diagnostic, int(decision)])
                if decision:
                    outcome = self.resolve(token)
                    if isinstance(outcome, bool) or outcome not in (0, 1):
                        raise ValueError('binary acquired outcome required')
                    self.labeled.append([copied(current), int(outcome)])
                    self.purchased.append(token)
                    self.acquired[group_index] += 1
        self.last_cap = cap
        self.barriers.append(dict(cap=cap, positions=list(self.positions), acquired=list(self.acquired),
                                  global_labels=len(self.labeled), decisions=len(self.decisions)))

    def evaluate(self, tokens):
        if self.last_cap < 0:
            raise ValueError('checkpoint required')
        unique, owners, restored = [], {}, []
        for token in tokens:
            query = self.visible(token)
            key = visible_key(query)
            if key not in owners:
                owners[key] = len(unique)
                unique.append(copied(query))
            restored.append(owners[key])
        predictor = self.predictor_factory(self.prior)
        probabilities, betas, states = predictor.evaluate(unique, copied(self.labeled))
        return dict(v6=[probabilities[index] for index in restored],
                    beta11=[betas[index] for index in restored], states=states,
                    unique_count=len(unique), evaluation_count=len(tokens),
                    restored=restored)
