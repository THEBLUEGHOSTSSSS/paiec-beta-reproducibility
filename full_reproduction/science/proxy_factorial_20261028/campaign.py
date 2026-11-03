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
from science.proxy_factorial_20261028.predictor import METHODS,evaluate_modes
CANDIDATES=METHODS[:7]

class Evaluator:
    def __init__(self, data):
        self.data = data
        self.cache = OrderedDict()
        self.hits = self.computations = 0

    def evaluate(self, bench, point):
        data = self.data
        query_tokens = point["query_tokens"]
        query_groups = {data.visible_groups[token] for token in query_tokens}
        observations = [(token, outcome) for token, outcome in zip(point["labeled_tokens"], point["labeled_outcomes"])
                        if data.visible_groups[token] in query_groups]
        signature = (bench, tuple(query_tokens), tuple(sorted(observations)))
        if signature in self.cache:
            self.hits += 1
            self.cache.move_to_end(signature)
            return self.cache[signature]
        queries = [data.catalog.inputs[token] for token in query_tokens]
        labeled = [(data.catalog.inputs[token], outcome) for token, outcome in observations]
        values, states = evaluate_modes(data.priors[bench], data.semantic, data.raw_map,
                                       data.shuffled_map, queries, labeled)
        counts = {state[2] for state in states.values()}
        if len(counts) != 1:
            raise RuntimeError("target queries unexpectedly span distinct evidence counts")
        result = (values, counts.pop())
        self.cache[signature] = result
        self.computations += 1
        if len(self.cache) > 512:
            self.cache.popitem(last=False)
        return result

def aggregate(rows):
    alc, budgets = defaultdict(float), defaultdict(list)
    for row in rows:
        alc[row["context"], row["target_pair"], row["seed"], row["method"]] += WEIGHTS[row["cap"]]*row["loss"]
        budgets[row["context"], row["cap"], row["method"]].append(row["loss"])
    output = {}
    for context in CONTEXTS:
        targets = sorted({key[1] for key in alc if key[0] == context})
        def means(which, seeds):
            return {method: float(np.mean([alc[context, target, seed, method] for target in which for seed in seeds]))
                    for method in METHODS}
        benches = sorted({target.split(":")[0] for target in targets})
        output[context] = dict(mean=means(targets, range(5)),
            seeds={str(seed): means(targets, [seed]) for seed in range(5)},
            benchmarks={bench: means([target for target in targets if target.split(":")[0] == bench], range(5)) for bench in benches},
            pairs={target: means([target], range(5)) for target in targets},
            budgets={str(cap): {method: float(np.mean(budgets[context, cap, method])) for method in METHODS} for cap in CAPS})
    return output

def gates(aggregates):
    main = aggregates[CONTEXTS[0]]
    material = {}
    for candidate in CANDIDATES:
        material[candidate] = material_gate(main["mean"][candidate], main["mean"]["headtail"], main["mean"]["beta11"],
            [tuple(main["seeds"][str(seed)][method] for method in (candidate, "headtail", "beta11")) for seed in range(5)],
            [tuple(aggregates[context]["mean"][method] for method in (candidate, "headtail", "beta11")) for context in CONTEXTS[1:]])
    def structural(candidate, shuffle):
        controls = ("full", "kernel_headtail", shuffle)
        epsilon = 1e-12
        mean_pass = all(main["mean"][candidate] < main["mean"][control]-epsilon for control in controls)
        wins = sum(all(main["seeds"][str(seed)][candidate] < main["seeds"][str(seed)][control]-epsilon for control in controls)
                   for seed in range(5))
        stress = all(aggregates[context]["mean"][candidate] <= aggregates[context]["mean"][control]+epsilon
                     for context in CONTEXTS[1:] for control in controls)
        return dict(passed=bool(mean_pass and wins >= 4 and stress), mean_pass=bool(mean_pass),
                    joint_seed_wins=int(wins), stress_pass=bool(stress), controls=list(controls), epsilon=epsilon,
                    gains={control: main["mean"][control]-main["mean"][candidate] for control in controls})
    primary = dict(candidate="structure", material=material["structure"], structural=structural("structure", "structure_shuffle"))
    secondary = dict(candidate="structure_linear", material=material["structure_linear"],
                     structural=structural("structure_linear", "linear_shuffle"), prespecified_secondary=True)
    for result in (primary, secondary):
        result["joint_pass"] = bool(result["material"]["passed"] and result["structural"]["passed"])
    return dict(material=material, primary=primary, secondary=secondary, winner_substitution=False,
                accepted_original_target=primary["joint_pass"])

def contrast_groups(aggregates):
    contrasts, negative = [], []
    for context, metrics in aggregates.items():
        for dimension in ("seeds", "benchmarks", "pairs", "budgets"):
            for group, values in metrics[dimension].items():
                for method in METHODS:
                    for control in ("full", "kernel_headtail"):
                        row = dict(context=context, dimension=dimension, group=group, method=method,
                                   control=control, gain=float(values[control]-values[method]))
                        contrasts.append(row)
                        if row["gain"] <= 0:
                            negative.append(row)
    return contrasts, negative
