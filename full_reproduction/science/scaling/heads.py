"""Bounded CPU-only heads on immutable 384-text + 128-metadata features."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Integral, Real
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


def _partition(records, excluded):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("records must be a nonempty list or tuple")
    if not isinstance(excluded, (list, tuple, set, frozenset)):
        raise TypeError("excluded must contain one or two benchmark IDs")
    if not 1 <= len(excluded) <= 2 or any(
            not isinstance(benchmark, str) or not benchmark for benchmark in excluded):
        raise ValueError("excluded must contain one or two nonempty benchmark IDs")
    if len(set(excluded)) != len(excluded):
        raise ValueError("excluded benchmark IDs must be distinct")
    excluded = set(excluded)
    benchmarks = []
    for row in records:
        if not isinstance(row, Mapping):
            raise TypeError("records must contain mappings")
        benchmark = row.get("benchmark")
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError("invalid benchmark ID")
        benchmarks.append(benchmark)
    if not excluded <= set(benchmarks):
        raise ValueError("excluded benchmark does not exist")
    train = [index for index, benchmark in enumerate(benchmarks) if benchmark not in excluded]
    target = [index for index, benchmark in enumerate(benchmarks) if benchmark in excluded]
    if not train or not target:
        raise ValueError("training and target partitions must both be nonempty")
    return train, target


def training_distribution(records, excluded):
    """Return indices, binomial rates, and pair-equal sampling probabilities.

    Each pair has mass 1 / number_of_pairs; a record's share within its pair
    is trials / total_pair_trials. Excluded successes/trials are never accessed.
    """
    train, _ = _partition(records, excluded)
    totals, owners, counts = defaultdict(int), {}, []
    for index in train:
        row = records[index]
        pair = row.get("pair_id")
        if not isinstance(pair, str) or not pair:
            raise ValueError("invalid training pair ID")
        if owners.setdefault(pair, row["benchmark"]) != row["benchmark"]:
            raise ValueError("pair ID spans benchmarks")
        successes, trials = row.get("successes"), row.get("trials")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               for value in (successes, trials)):
            raise ValueError("training counts must be integers, not booleans")
        successes, trials = int(successes), int(trials)
        if not 0 <= successes <= trials or trials <= 0:
            raise ValueError("invalid training binomial counts")
        counts.append((pair, successes, trials))
        totals[pair] += trials
    targets = np.asarray([successes / trials for _, successes, trials in counts], np.float64)
    weights = np.asarray([trials / totals[pair] / len(totals)
                          for pair, _, trials in counts], np.float64)
    weights /= weights.sum()
    if not np.isfinite(targets).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("unrepresentable training distribution")
    return np.asarray(train, dtype=np.int64), targets, weights


def _configuration(config):
    required = {"kind", "rank", "hidden", "lr", "weight_decay", "steps",
                "batch_size", "seed", "max_seconds"}
    if not isinstance(config, Mapping) or set(config) != required:
        raise ValueError("config must contain exactly the documented keys")
    config = dict(config)
    if config["kind"] not in ("linear", "bilinear", "mlp"):
        raise ValueError("unknown head kind")
    for name, lower, upper in (("rank", 1, 64), ("hidden", 1, 256),
                               ("steps", 1, 4096), ("batch_size", 1, 4096),
                               ("seed", 0, 2**63 - 1)):
        value = config[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(name + " must be an integer, not boolean")
        if not lower <= value <= upper:
            raise ValueError(name + " out of bounds")
        config[name] = int(value)
    for name in ("lr", "weight_decay", "max_seconds"):
        value = config[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(name + " must be a real number")
        value = float(value)
        if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
            raise ValueError(name + " invalid")
        config[name] = value
    return config


class _Head(nn.Module):
    def __init__(self, config, generator):
        super().__init__()
        self.kind, self.rank = config["kind"], config["rank"]
        self.linear_weight = self._weight((512,), 512, generator)
        self.linear_bias = nn.Parameter(torch.zeros((), dtype=torch.float32, device="cpu"))
        if self.kind == "bilinear":
            self.text_factor = self._weight((384, self.rank), 384, generator)
            self.metadata_factor = self._weight((128, self.rank), 128, generator)
        elif self.kind == "mlp":
            hidden = config["hidden"]
            self.hidden_weight = self._weight((hidden, 512), 512, generator)
            self.hidden_bias = nn.Parameter(torch.zeros(hidden, dtype=torch.float32, device="cpu"))
            self.output_weight = self._weight((1, hidden), hidden, generator)
            self.output_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cpu"))

    @staticmethod
    def _weight(shape, fan_in, generator):
        value = torch.empty(shape, dtype=torch.float32, device="cpu")
        nn.init.uniform_(value, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in), generator=generator)
        return nn.Parameter(value)

    def forward(self, features):
        logits = features @ self.linear_weight + self.linear_bias
        if self.kind == "bilinear":
            interaction = (features[:, :384] @ self.text_factor) * (
                features[:, 384:] @ self.metadata_factor)
            logits = logits + interaction.sum(dim=1) / math.sqrt(self.rank)
        elif self.kind == "mlp":
            hidden = functional.gelu(functional.linear(features, self.hidden_weight, self.hidden_bias))
            logits = logits + functional.linear(hidden, self.output_weight, self.output_bias).squeeze(1)
        return logits


def _finite(value, name):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError("nonfinite " + name)


def _state(model):
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def fit_head(records, matrix, excluded, config):
    """Fit a new head; return aligned probabilities, copied states, and evidence.

    AdamW minimizes sampled BCE (no second weighting); evidence losses are the
    full pair-equal binomial probability objective, excluding weight decay.
    No encoder parameters, files, global RNG state, or input arrays are changed.
    A deadline overrun raises TimeoutError and never returns a completed fit.
    """
    started = time.monotonic()
    config = _configuration(config)
    steps_completed = 0

    def deadline():
        if time.monotonic() - started >= config["max_seconds"]:
            raise TimeoutError(f"head deadline exceeded; steps_completed={steps_completed}/{config['steps']}")

    train, targets, weights = training_distribution(records, excluded)
    _, target = _partition(records, excluded)
    if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float32:
        raise TypeError("matrix must be a float32 numpy array")
    if matrix.shape != (len(records), 512) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite N x 512")
    label_rows = [[int(index), records[index]["benchmark"], records[index]["pair_id"],
                   int(records[index]["successes"]), int(records[index]["trials"])] for index in train]
    label_hash = hashlib.sha256(json.dumps(label_rows, separators=(",", ":"),
                                          ensure_ascii=False).encode("utf-8")).hexdigest()
    deadline()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        generator = torch.Generator(device="cpu").manual_seed(config["seed"])
        sampler = np.random.default_rng(config["seed"])
        model = _Head(config, generator)
        initial_state = _state(model)
        features = torch.from_numpy(matrix[train].copy())
        target_rates = torch.from_numpy(targets.astype(np.float32))
        probabilities = torch.from_numpy(weights)

        def objective():
            total = 0.0
            with torch.no_grad():
                for start in range(0, len(train), 4096):
                    deadline()
                    logits = model(features[start:start + 4096])
                    _finite(logits, "objective logits")
                    losses = functional.binary_cross_entropy_with_logits(
                        logits, target_rates[start:start + 4096], reduction="none")
                    _finite(losses, "objective loss")
                    total += (losses.double() * probabilities[start:start + 4096]).sum().item()
            return total

        initial_loss = objective()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                      weight_decay=config["weight_decay"], foreach=False)
        for _ in range(config["steps"]):
            deadline()
            selected = sampler.choice(len(train), config["batch_size"], replace=True, p=weights)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[selected])
            _finite(logits, "training logits")
            loss = functional.binary_cross_entropy_with_logits(logits, target_rates[selected])
            _finite(loss, "training loss")
            loss.backward()
            for parameter in model.parameters():
                _finite(parameter.grad, "gradient")
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            steps_completed += 1
            for parameter in model.parameters():
                _finite(parameter, "parameter")
            deadline()
        final_loss = objective()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(target), 4096):
                deadline()
                logits = model(torch.from_numpy(matrix[target[start:start + 4096]].copy()))
                _finite(logits, "target logits")
                predictions.append(torch.sigmoid(logits).numpy().copy())
        state = _state(model)
        delta = math.sqrt(sum(float(np.square(state[name].astype(np.float64) -
                                              initial_state[name].astype(np.float64)).sum()) for name in state))
        deadline()
        return {"target_indices": target, "predictions": np.concatenate(predictions),
                "state": state, "initial_state": initial_state, "evidence": {
                    "training_benchmarks": sorted({records[index]["benchmark"] for index in train}),
                    "training_label_hash": label_hash, "steps_completed": steps_completed,
                    "initial_loss": initial_loss, "final_loss": final_loss,
                    "parameter_delta_l2": delta, "seconds": time.monotonic() - started,
                    "model_kind": config["kind"], "train_records": len(train),
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters())}}
    finally:
        torch.set_num_threads(previous_threads)
