"""Pure supplied-input fitting; orchestration reserves every actual fit."""
from collections import defaultdict
import copy
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional

from science.affine_calibration_20261015.core import visible_key
from science.scaling.heads import _Head, _state, training_distribution
from .core import ARMS, CONFIG, auxiliary_loss, schedule, mean_targets


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tables(records, matrix, inputs, excluded):
    indices, rates, weights = training_distribution(records, excluded)
    features = np.asarray(matrix[indices], dtype=np.float32).copy()
    if features.shape != (len(indices), 512) or not np.isfinite(features).all():
        raise ValueError("allowed training features must be finite N x 512")
    labels = [[int(index), records[index]["benchmark"], records[index]["pair_id"],
               int(records[index]["successes"]), int(records[index]["trials"])]
              for index in indices]
    identity = [list(visible_key(inputs[index])) for index in indices]
    receipt = dict(train_records=len(indices), indices_sha256=array_sha(indices),
                   rates_sha256=array_sha(rates), weights_sha256=array_sha(weights),
                   features_sha256=array_sha(features), label_sha256=digest(labels),
                   visible_inputs_sha256=digest(identity),
                   training_sources=sorted({records[index]["benchmark"] for index in indices}))
    return dict(indices=indices, rates=rates, weights=weights, features=features,
                receipt=receipt, mean_targets=mean_targets(records, inputs, excluded)[indices])


def local_schedule(tab, planned):
    indices = tab["indices"]
    result = {}
    for name in ("main_indices", "left_indices", "right_indices"):
        values = planned[name]
        local = np.searchsorted(indices, values)
        if np.any(local >= len(indices)) or not np.array_equal(indices[local], values):
            raise ValueError("excluded endpoint in schedule")
        result[name] = local
    mapped = planned["shuffle_map"][indices]
    local = np.searchsorted(indices, mapped)
    if np.any(local >= len(indices)) or not np.array_equal(indices[local], mapped):
        raise ValueError("excluded endpoint in shuffle")
    result["shuffle_map"] = local
    return result


def batch_loss(model, features, rates, local, step, arm, means=None):
    selected = local["main_indices"][step]
    main = functional.binary_cross_entropy_with_logits(model(features[selected]), rates[selected])
    if arm == "pooled":
        return main, main, torch.zeros((), dtype=main.dtype)
    left = local["left_indices"][step]
    right = local["right_indices"][step]
    mapped_left = local["shuffle_map"][left] if arm == "shuffled" else left
    mapped_right = local["shuffle_map"][right] if arm == "shuffled" else right
    auxiliary = auxiliary_loss(model(features[left]), model(features[right]),
                               (means if arm == "mean" else rates)[mapped_left],
                               (means if arm == "mean" else rates)[mapped_right])
    return main + auxiliary, main, auxiliary


def initial_gradient(records, matrix, inputs, excluded, planned):
    endpoint = int(planned["left_indices"][0, 0])
    endpoint_key = visible_key(inputs[endpoint])[:8]
    selected = next(index for index, record in enumerate(records)
                    if record["benchmark"] not in excluded and index != endpoint
                    and record["benchmark"] == records[endpoint]["benchmark"]
                    and record["trials"] == records[endpoint]["trials"]
                    and visible_key(inputs[index])[:8] == endpoint_key)
    modified = list(records)
    record = dict(records[selected])
    record["successes"] = 0 if record["successes"] > 0 else record["trials"]
    modified[selected] = record
    results = {}
    gradients = []
    for supplied in (records, modified):
        tab = tables(supplied, matrix, inputs, excluded)
        local = local_schedule(tab, planned)
        model = _Head(CONFIG, torch.Generator(device="cpu").manual_seed(0))
        loss, main, auxiliary = batch_loss(model, torch.from_numpy(tab["features"]),
            torch.from_numpy(tab["rates"].astype(np.float32)), local, 0, "mean",
            torch.from_numpy(tab["mean_targets"].astype(np.float32)))
        auxiliary.backward()
        gradient = np.concatenate([parameter.grad.numpy().ravel() for parameter in model.parameters()])
        gradients.append(gradient)
        results["original" if supplied is records else "modified"] = dict(
            loss=float(loss.detach()), main_loss=float(main.detach()),
            auxiliary_loss=float(auxiliary.detach()), gradient_sha256=array_sha(gradient),
            gradient_l2=float(np.linalg.norm(gradient.astype(np.float64))), training=tab["receipt"])
    difference = float(np.max(np.abs(gradients[0]-gradients[1])))
    if difference == 0 or results["original"]["auxiliary_loss"] == results["modified"]["auxiliary_loss"]:
        raise AssertionError("allowed-label mean auxiliary positive control failed")
    return dict(selected_record=selected, endpoint_record=endpoint, source=record["benchmark"],
                pair_id=record["pair_id"], old_successes=int(records[selected]["successes"]),
                new_successes=int(record["successes"]), trials=int(record["trials"]),
                gradient_max_abs_change=difference, full_fits=0, **results)


def fit(records, matrix, inputs, excluded, arm, seed=0, expected_schedule=None, resource_check=None):
    if arm not in ARMS or seed not in (0, 1, 2):
        raise ValueError("only frozen arms and training seeds0/1/2 are allowed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    planned = schedule(records, inputs, excluded, seed)
    if expected_schedule is not None:
        for name in planned:
            if not np.array_equal(planned[name], expected_schedule[name]):
                raise RuntimeError("actual supplied-input schedule differs from prefit freeze")
    tab = tables(records, matrix, inputs, excluded)
    local = local_schedule(tab, planned)
    features = torch.from_numpy(tab["features"])
    rates = torch.from_numpy(tab["rates"].astype(np.float32))
    means = torch.from_numpy(tab["mean_targets"].astype(np.float32))
    model = _Head(CONFIG, torch.Generator(device="cpu").manual_seed(seed))
    initial = _state(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.03, foreach=False)
    losses, main_losses, auxiliary_losses, norms = [], [], [], []
    started = time.monotonic()
    for step in range(256):
        if resource_check is not None and step % 32 == 0:
            resource_check()
        optimizer.zero_grad(set_to_none=True)
        loss, main, auxiliary = batch_loss(model, features, rates, local, step, arm, means)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise FloatingPointError("nonfinite fitted parameter")
        losses.append(float(loss.detach()))
        main_losses.append(float(main.detach()))
        auxiliary_losses.append(float(auxiliary.detach()))
        norms.append(float(norm))
    receipt = dict(training=tab["receipt"], steps=256, losses=losses,
                   main_losses=main_losses, auxiliary_losses=auxiliary_losses,
                   preclip_gradient_norms=norms, seconds=time.monotonic()-started,
                   schedule_hashes={name: array_sha(value) for name, value in planned.items()})
    return model, initial, receipt


def forward(model, matrix):
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(matrix[start:start+4096].copy()))).numpy()
                               for start in range(0, len(matrix), 4096)])


def poisoned(records, matrix, inputs, outer):
    modified = copy.deepcopy(records)
    visible = list(inputs)
    features = matrix.copy()
    changed = []
    for index, record in enumerate(modified):
        if record["benchmark"] != outer:
            continue
        record["successes"] = int(record["trials"]) + 101
        record["trials"] = -17
        record["content"] = "EXCLUDED_RAW_SOURCE_POISON_" + str(index)
        record["subject"] = {"poison": "excluded original metadata replaced"}
        visible[index] = object()
        features[index] = np.nan
        changed.append(index)
    return modified, features, visible, changed


def coverage(records, inputs, planned, relation_cells):
    indices = np.flatnonzero(planned["shuffle_map"] >= 0)
    strata = defaultdict(list)
    sources = defaultdict(lambda: dict(rows=0, visible_cells=set(), visible_subjects=set()))
    changed = np.zeros(len(records), dtype=bool)
    for index in indices:
        key = visible_key(inputs[index])
        source = records[index]["benchmark"]
        strata[source, key[:8], int(records[index]["trials"])].append(int(index))
        sources[source]["rows"] += 1
        sources[source]["visible_cells"].add(key[:9])
        sources[source]["visible_subjects"].add(key[:8])
        changed[index] = key[8] != visible_key(inputs[planned["shuffle_map"][index]])[8]
    endpoints = np.concatenate([planned["left_indices"].ravel(), planned["right_indices"].ravel()])
    eligible_sources = sorted({cell["source"] for cell in relation_cells})
    return dict(train_rows=len(indices), relation_contents=len(relation_cells),
        relation_visible_cells=sum(len(cell["subjects"]) for cell in relation_cells),
        eligible_aux_sources=eligible_sources,
        source_counts={source: dict(rows=value["rows"], visible_cells=len(value["visible_cells"]),
                                    visible_subjects=len(value["visible_subjects"]))
                       for source, value in sorted(sources.items())},
        shuffle_strata=len(strata), singleton_strata=sum(len(value)==1 for value in strata.values()),
        no_content_change_strata=sum(not any(changed[index] for index in value) for value in strata.values()),
        mapped_rows_changed_content=int(changed[indices].sum()),
        mapped_rows_changed_content_fraction=float(changed[indices].mean()),
        sampled_endpoints_changed_content=int(changed[endpoints].sum()),
        sampled_endpoints_changed_content_fraction=float(changed[endpoints].mean()),
        auxiliary_pairs=65536, independent_evidence_count=False,
        weighting="uniform source/content/distinct-subject-pair/original-record; main BCE pair-equal-trial")
