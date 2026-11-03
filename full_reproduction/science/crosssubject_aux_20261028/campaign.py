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
from science.demand_features_20261026.predictor import Predictor
METHODS=("intact","shuffled","pooled","old_full","headtail","beta11")

def unique_evaluate(predictor, queries, labeled):
    unique, owners, restored = [], {}, []
    for query in queries:
        key = visible_key(query)
        if key not in owners:
            owners[key] = len(unique)
            unique.append(query)
        restored.append(owners[key])
    values, beta, states = predictor.evaluate(unique, labeled)
    return np.asarray(values, dtype=np.float64)[restored], np.asarray(beta, dtype=np.float64)[restored], states

def vectors(priors, semantic, demand, queries, labeled):
    result = {}
    for arm in ("intact", "shuffled", "pooled", "old_full"):
        result[arm] = unique_evaluate(Predictor(priors[arm], semantic, demand), queries, labeled)[0]
    base, beta, states = unique_evaluate(VisiblePredictor(priors["old_full"]), queries, labeled)
    result.update(headtail=base, beta11=beta)
    return result, states

def aggregate(rows):
    alc = defaultdict(float)
    budget = defaultdict(list)
    for row in rows:
        alc[row["context"], row["target_pair"], row["seed"], row["method"]] += WEIGHTS[row["cap"]]*row["loss"]
        budget[row["context"], row["cap"], row["method"]].append(row["loss"])
    result = {}
    for context in CONTEXTS:
        targets = sorted({key[1] for key in alc if key[0] == context})
        def means(selected, seeds):
            return {method: float(np.mean([alc[context, target, seed, method]
                                           for target in selected for seed in seeds])) for method in METHODS}
        sources = sorted({target.split(":")[0] for target in targets})
        result[context] = dict(mean=means(targets, range(5)),
            seeds={str(seed): means(targets, [seed]) for seed in range(5)},
            sources={source: means([target for target in targets if target.split(":")[0]==source], range(5))
                     for source in sources}, targets={target: means([target], range(5)) for target in targets},
            caps={str(cap): {method: float(np.mean(budget[context, cap, method])) for method in METHODS}
                  for cap in CAPS}, pair_seed_comparison={
                control: dict(wins=sum(alc[context,target,seed,"intact"] < alc[context,target,seed,control]-1e-12
                                      for target in targets for seed in range(5)),
                              ties=sum(abs(alc[context,target,seed,"intact"]-alc[context,target,seed,control]) <= 1e-12
                                       for target in targets for seed in range(5)),
                              losses=sum(alc[context,target,seed,"intact"] > alc[context,target,seed,control]+1e-12
                                         for target in targets for seed in range(5))) for control in METHODS[1:]})
    return result

def gates(aggregates):
    main = aggregates[CONTEXTS[0]]
    mean = main["mean"]
    seed_wins = sum(main["seeds"][str(seed)]["intact"] < main["seeds"][str(seed)]["pooled"]-1e-12 and
                    main["seeds"][str(seed)]["intact"] < main["seeds"][str(seed)]["shuffled"]-1e-12
                    for seed in range(5))
    stress = all(aggregates[context]["mean"]["intact"] <= aggregates[context]["mean"][control]+1e-12
                 for context in CONTEXTS[1:] for control in ("pooled", "shuffled"))
    main_pass = all(mean["intact"] < mean[control]-1e-12 for control in ("pooled", "shuffled"))
    structural = dict(passed=bool(main_pass and seed_wins >= 4 and stress), mean_pass=bool(main_pass),
        joint_seed_wins=int(seed_wins), stress_pass=bool(stress), epsilon=1e-12,
        gain_vs_pooled=mean["pooled"]-mean["intact"], gain_vs_shuffled=mean["shuffled"]-mean["intact"])
    material = material_gate(mean["intact"], mean["headtail"], mean["beta11"],
        [(main["seeds"][str(seed)]["intact"], main["seeds"][str(seed)]["headtail"],
          main["seeds"][str(seed)]["beta11"]) for seed in range(5)],
        [(aggregates[context]["mean"]["intact"], aggregates[context]["mean"]["headtail"],
          aggregates[context]["mean"]["beta11"]) for context in CONTEXTS[1:]])
    passed = bool(structural["passed"] and material["passed"])
    return dict(structural=structural, material=material, recipe_gates_passed=passed,
                decision="PILOT_GATES_PASS_NOT_PROMOTION" if passed else "REJECT_FROZEN_RECIPE",
                promotion=False, commander_acceptance=False, full_ensemble_superseded=False)
