"""Offline five-bilinear-head prior; call-local strength-4/Beta(1,1) calibration."""
import hashlib
import json
import math
import os
from functools import lru_cache
from numbers import Real
from pathlib import Path
from threading import RLock

FIELDS = ('normalized_name', 'provider', 'release_date', 'access_date', 'harness',
          'harness_version', 'reasoning_effort', 'subject_features_extra')
ENCODER_SHA = {
    'config.json': '69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959',
    'model.safetensors': '1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477',
    'sentencepiece.bpe.model': 'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865',
    'special_tokens_map.json': 'd05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7',
    'tokenizer.json': '0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39',
    'tokenizer_config.json': 'a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b',
}
ROOT = Path(__file__).resolve().parent
_runtime = None
_lock = RLock()


def _asset_path(name):
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise RuntimeError('Unsafe asset name')
    directory = ROOT / 'assets'
    target = directory / name
    if directory.is_symlink() or target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError('Asset symlink or escape')
    return target


def _verify_assets():
    manifest_path = _asset_path('manifest.json')
    manifest = json.loads(manifest_path.read_text())
    expected_names = set(ENCODER_SHA) | {f'head_seed{seed}.npz' for seed in range(5)}
    if set(manifest) != expected_names or any(manifest[n] != h for n, h in ENCODER_SHA.items()):
        raise RuntimeError('Invalid asset inventory or encoder identity')
    for name, expected in manifest.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError('Invalid asset SHA')
        try:
            with _asset_path(name).open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        except OSError as error:
            raise RuntimeError('Missing frozen asset: ' + name) from error
        if actual != expected:
            raise RuntimeError('Corrupt frozen asset: ' + name)


def _validate_state(state):
    import numpy as np
    shapes = {'linear_weight': (512,), 'linear_bias': (),
              'text_factor': (384, 8), 'metadata_factor': (128, 8)}
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError('Invalid bilinear state keys')
    for name, shape in shapes.items():
        value = state[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid bilinear parameter: ' + name)


def ensemble_prior(features, states):
    """Each head evaluates in float32; sigmoid precedes float64 probability mean."""
    import numpy as np
    if not isinstance(features, np.ndarray) or features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 512 or not np.isfinite(features).all():
        raise ValueError('Expected finite float32 N x 512 features')
    if not isinstance(states, (list, tuple)) or len(states) != 5:
        raise ValueError('Exactly five states required')
    probabilities = []
    for state in states:
        _validate_state(state)
        with np.errstate(over='raise', invalid='raise'):
            interaction = (features[:, :384] @ state['text_factor']) * (features[:, 384:] @ state['metadata_factor'])
            logits = features @ state['linear_weight'] + state['linear_bias']
            logits = logits + interaction.sum(axis=1, dtype=np.float32) / np.float32(math.sqrt(8))
            exponential = np.exp(-np.abs(logits))
            probability = np.where(logits >= 0, 1 / (1 + exponential), exponential / (1 + exponential))
        if logits.dtype != np.float32 or not np.isfinite(probability).all():
            raise ValueError('Nonfinite head prediction')
        probabilities.append(probability)
    return np.mean(np.stack(probabilities).astype(np.float64), axis=0)


def _load():
    global _runtime
    with _lock:
        if _runtime is not None:
            return _runtime
        _verify_assets()
        for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_IMPLICIT_TOKEN'):
            os.environ[name] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import numpy as np
        import torch
        from sklearn.feature_extraction import FeatureHasher
        from sklearn.preprocessing import normalize
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(2)
        states = []
        for seed in range(5):
            with np.load(_asset_path(f'head_seed{seed}.npz'), allow_pickle=False) as head:
                state = {name: head[name].copy() for name in head.files}
            _validate_state(state)
            for value in state.values():
                value.setflags(write=False)
            states.append(state)
        directory = str(ROOT / 'assets')
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False, token=False)
        encoder = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                           use_safetensors=True, token=False).to(device='cpu', dtype=torch.float32).eval()
        encoder.requires_grad_(False)
        hasher = FeatureHasher(n_features=128, input_type='string', alternate_sign=False, dtype=np.float32)
        _runtime = (np, torch, normalize, tokenizer, encoder, hasher, states)
        return _runtime


@lru_cache(maxsize=4096)
def _embedding(text):
    np, torch, normalize, tokenizer, encoder, hasher, states = _load()
    with _lock, torch.inference_mode():
        batch = tokenizer(['query: ' + text], padding=True, truncation=True, max_length=512, return_tensors='pt')
        last = encoder(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).to(last.dtype)
        pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
        vector = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()[0]
    if vector.shape != (384,) or not np.isfinite(vector).all():
        raise RuntimeError('Invalid E5 embedding')
    vector = vector.astype(np.float32)
    vector.setflags(write=False)
    return vector


def _pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Input must be [subject, item]')
    subject, item = value
    if not isinstance(subject, dict) or not isinstance(item, dict):
        raise ValueError('Subject and item must be dictionaries')
    if any(not isinstance(subject.get(field, ''), str) for field in FIELDS):
        raise ValueError('Visible subject fields must be strings')
    if not isinstance(item.get('item_content'), str):
        raise ValueError('item_content must be a string')
    return subject, item


def _features(subject, item):
    _pair([subject, item])
    np, torch, normalize, tokenizer, encoder, hasher, states = _load()
    metadata = hasher.transform([[field + '=' + subject.get(field, '') for field in FIELDS]])
    metadata = normalize(metadata, norm='l2').toarray()
    return normalize(np.hstack([_embedding(item['item_content'])[None, :], metadata]), norm='l2').astype(np.float32)


def _prior(subject, item):
    features = _features(subject, item)
    return float(ensemble_prior(features, _load()[-1])[0])


def _sigmoid(value):
    exponential = math.exp(-abs(value))
    return 1 / (1 + exponential) if value >= 0 else exponential / (1 + exponential)


def _logit(probability):
    if not isinstance(probability, Real) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Prior must be a finite probability')
    clipped = min(max(float(probability), 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


def predict(input, labeled=None):
    """Full subject+benchmark filter; preserve repeats and isolate every call."""
    subject, item = _pair(input)
    if labeled is not None and not isinstance(labeled, (list, tuple)):
        raise ValueError('labeled must be a sequence')
    observations = []
    if labeled:
        benchmark = item.get('benchmark_id')
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError('Nonempty benchmark_id required with labeled evidence')
        for entry in labeled:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError('Malformed labeled entry')
            pair, outcome = entry
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not isinstance(pair[0], dict):
                raise ValueError('Malformed labeled input')
            if pair[0] != subject:
                continue
            if not isinstance(pair[1], dict) or not isinstance(pair[1].get('benchmark_id'), str) or not pair[1]['benchmark_id']:
                raise ValueError('Matching-subject evidence requires benchmark_id')
            if pair[1]['benchmark_id'] != benchmark:
                continue
            observed_subject, observed_item = _pair(pair)
            if isinstance(outcome, bool) or not isinstance(outcome, Real) or outcome not in (0, 1):
                raise ValueError('Matching outcomes must be binary 0/1')
            observations.append((observed_subject, observed_item, int(outcome)))
    logits = [_logit(_prior(s, i)) for s, i, outcome in observations]
    outcomes = [outcome for s, i, outcome in observations]
    delta = 0.0
    if observations:
        low, high = -len(observations) / 4, len(observations) / 4
        for iteration in range(160):
            middle = (low + high) / 2
            equation = math.fsum(_sigmoid(logit + middle) - outcome for logit, outcome in zip(logits, outcomes)) + 4 * middle
            if equation < 0:
                low = middle
            else:
                high = middle
        delta = (low + high) / 2
    calibrated = _sigmoid(_logit(_prior(subject, item)) + delta)
    return float(.5 * calibrated + .5 * (sum(outcomes) + 1) / (len(outcomes) + 2))
