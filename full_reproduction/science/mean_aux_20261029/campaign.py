from __future__ import annotations
import hashlib,json,math,statistics,itertools
from pathlib import Path
from collections import defaultdict,OrderedDict,Counter
import numpy as np
from scipy.special import expit,logit
BUDGETS=[0,1,3,7,15,31]
SEEDS=[0,1,2,3,4]
WEIGHTS=[.1,.2,.2,.2,.2,.1]
from science.online_acquisition_20261015.stream import CAPS,Replay,VisiblePredictor,visible_group
from science.affine_calibration_20261015.core import visible_key
from science.demand_features_20261026.core import material_gate
CONTEXTS=("restricted_mixed_balanced","restricted_same_alias","singlepair")
WEIGHTS=dict(zip(CAPS,(.1,.2,.2,.2,.2,.1)))
from scipy.optimize import brentq
from scipy.linalg import cho_factor,cho_solve
FULL=("mean_ensemble","shuffled_ensemble","pooled_ensemble","old_full","headtail","beta11")
INDIVIDUAL=tuple(arm+"_s"+str(seed) for seed in range(3) for arm in ("mean","shuffled","pooled"))

def methods(context):
    return FULL+INDIVIDUAL if context == CONTEXTS[0] else FULL

def vectors(priors, semantic, demand, queries, labeled, selected):
    unique, owners, restored = [], {}, []
    for query in queries:
        key = visible_key(query)
        if key not in owners:
            owners[key] = len(unique)
            unique.append(query)
        restored.append(owners[key])
    group = visible_group(unique[0])
    if any(visible_group(query) != group for query in unique):
        raise ValueError('mixed query groups')
    allowed = []
    for value, outcome in labeled:
        if visible_group(value) == group:
            if isinstance(outcome, (bool, np.bool_)) or outcome not in (0, 1):
                raise ValueError('nonbinary own-group evidence')
            allowed.append((visible_key(value), int(outcome), value))
    allowed.sort(key=lambda row: (row[0], row[1]))
    support = [value for _, _, value in allowed]
    outcomes = np.asarray([outcome for _, outcome, _ in allowed], dtype=np.float64)
    count = len(outcomes)
    beta = (float(outcomes.sum())+1)/(count+2)
    order = tuple(method for method in selected if method not in ('headtail', 'beta11'))
    width = len(order)
    support_prior = np.asarray([[priors[arm](value) for arm in order] for value in support], dtype=np.float64).reshape(count, width)
    query_prior = np.asarray([[priors[arm](value) for arm in order] for value in unique], dtype=np.float64)
    support_logits = logit(np.clip(support_prior, 1e-6, 1-1e-6))
    deltas = np.zeros(width)
    for column in range(width):
        if count:
            deltas[column] = brentq(lambda delta: float(np.sum(expit(support_logits[:, column]+delta)-outcomes))+4*delta,
                                    -32., 32., xtol=5e-15, rtol=1e-15)
    base = .5*expit(logit(np.clip(query_prior, 1e-6, 1-1e-6))+deltas)+.5*beta
    correction = np.zeros_like(base)
    if count >= 2:
        support_semantic = np.stack([semantic[value[1]['item_content']] for value in support])
        support_demand = np.stack([demand[value[1]['item_content']] for value in support])
        query_semantic = np.stack([semantic[value[1]['item_content']] for value in unique])
        query_demand = np.stack([demand[value[1]['item_content']] for value in unique])
        gram = np.clip(support_semantic@support_semantic.T, -1, 1)**4+np.clip(support_demand@support_demand.T, -1, 1)**4
        cross = np.clip(query_semantic@support_semantic.T, -1, 1)**4+np.clip(query_demand@support_demand.T, -1, 1)**4
        centered = gram-gram.mean(axis=0, keepdims=True)-gram.mean(axis=1, keepdims=True)+gram.mean()
        residual = outcomes[:, None]-expit(support_logits+deltas)
        coefficient = cho_solve(cho_factor(centered+4*np.eye(count), lower=True), residual-residual.mean(axis=0, keepdims=True))
        cross = cross-cross.mean(axis=1, keepdims=True)-gram.mean(axis=0, keepdims=True)+gram.mean()
        correction = cross@coefficient
    prediction = np.clip(base+.5*correction, 1e-6, 1-1e-6)[restored]
    result = {arm: prediction[:, column] for column, arm in enumerate(order)}
    result['headtail'] = base[restored, order.index('old_full')]
    result['beta11'] = np.full(len(queries), beta)
    return result, count

def loss(vector, successes, trials):
    return float(np.sum(successes*(1-vector)**2+(trials-successes)*vector*vector, dtype=np.float64)/np.sum(trials))

def aggregate(rows):
    alc, budget = defaultdict(float), defaultdict(list)
    for row in rows:
        alc[row['context'], row['target_pair'], row['seed'], row['method']] += WEIGHTS[row['cap']]*row['loss']
        budget[row['context'], row['cap'], row['method']].append(row['loss'])
    result = {}
    for context in CONTEXTS:
        targets = sorted({key[1] for key in alc if key[0] == context})
        def means(selected, seeds):
            return {method: float(np.mean([alc[context, target, seed, method] for target in selected for seed in seeds])) for method in methods(context)}
        sources = sorted({target.split(':')[0] for target in targets})
        groups = dict(mean=means(targets, range(5)), seeds={str(seed): means(targets, [seed]) for seed in range(5)},
            sources={source: means([target for target in targets if target.split(':')[0] == source], range(5)) for source in sources},
            targets={target: means([target], range(5)) for target in targets},
            caps={str(cap): {method: float(np.mean(budget[context, cap, method])) for method in methods(context)} for cap in CAPS})
        groups['source_equal_diagnostic'] = {method: float(np.mean([groups['sources'][source][method] for source in sources])) for method in methods(context)}
        groups['pair_seed_comparison'] = {control: dict(
            wins=sum(alc[context, target, seed, FULL[0]] < alc[context, target, seed, control]-1e-12 for target in targets for seed in range(5)),
            ties=sum(abs(alc[context, target, seed, FULL[0]]-alc[context, target, seed, control]) <= 1e-12 for target in targets for seed in range(5)),
            losses=sum(alc[context, target, seed, FULL[0]] > alc[context, target, seed, control]+1e-12 for target in targets for seed in range(5))) for control in methods(context) if control != FULL[0]}
        result[context] = groups
    return result

def gates(groups):
    main = groups[CONTEXTS[0]]
    means = main['mean']
    candidate = FULL[0]
    controls = ('pooled_ensemble', 'shuffled_ensemble')
    joint = sum(all(main['seeds'][str(seed)][candidate] < main['seeds'][str(seed)][control]-1e-12 for control in controls) for seed in range(5))
    stress = all(groups[context]['mean'][candidate] <= groups[context]['mean'][control]+1e-12 for context in CONTEXTS[1:] for control in controls)
    mean_pass = all(means[candidate] < means[control]-1e-12 for control in controls)
    training = {str(seed): dict(passed=all(means['mean_s'+str(seed)] < means[arm+'_s'+str(seed)]-1e-12 for arm in ('pooled', 'shuffled')),
        gain_vs_pooled=means['pooled_s'+str(seed)]-means['mean_s'+str(seed)],
        gain_vs_shuffled=means['shuffled_s'+str(seed)]-means['mean_s'+str(seed)]) for seed in range(3)}
    training_wins = sum(row['passed'] for row in training.values())
    structural = dict(passed=bool(mean_pass and joint >= 4 and stress and training_wins >= 2), mean_pass=bool(mean_pass),
        joint_evaluation_seed_wins=int(joint), stress_pass=bool(stress), training_seed_wins=int(training_wins),
        training_seed_details=training, gain_vs_pooled=means['pooled_ensemble']-means[candidate],
        gain_vs_shuffled=means['shuffled_ensemble']-means[candidate], epsilon=1e-12)
    material = material_gate(means[candidate], means['headtail'], means['beta11'],
        [(main['seeds'][str(seed)][candidate], main['seeds'][str(seed)]['headtail'], main['seeds'][str(seed)]['beta11']) for seed in range(5)],
        [(groups[context]['mean'][candidate], groups[context]['mean']['headtail'], groups[context]['mean']['beta11']) for context in CONTEXTS[1:]])
    return dict(structural=structural, material=material, continue_recipe=bool(structural['passed'] and material['passed']),
                promotion=False, independent_review='PENDING/BLOCKED until Commander native review', repeated_development=True)
