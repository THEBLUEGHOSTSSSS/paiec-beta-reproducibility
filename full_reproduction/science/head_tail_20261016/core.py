"""Pure, fixed transformations for the head-tail intervention."""
from __future__ import annotations

from collections.abc import Sequence
import numbers

import numpy as np


def tail_body_ids(fullbody, prefix_length, body_limit=510):
    """Keep a fixed prefix and the final content IDs within ``body_limit``.

    ``fullbody`` must contain tokenizer output before special tokens are added.
    The caller is responsible for asserting that its leading IDs are the frozen
    query prefix. Short inputs are returned unchanged.
    """
    if not isinstance(fullbody, Sequence) or isinstance(fullbody, (str, bytes)):
        raise TypeError("fullbody must be a sequence of integer token IDs")
    if isinstance(prefix_length, bool) or not isinstance(prefix_length, numbers.Integral):
        raise TypeError("prefix_length must be an integer")
    if isinstance(body_limit, bool) or not isinstance(body_limit, numbers.Integral):
        raise TypeError("body_limit must be an integer")
    prefix_length, body_limit = int(prefix_length), int(body_limit)
    if prefix_length <= 0:
        raise ValueError("prefix_length must be nonempty")
    if body_limit <= prefix_length:
        raise ValueError("body_limit must leave positive tail capacity")
    if prefix_length > len(fullbody):
        raise ValueError("prefix_length exceeds fullbody")
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in fullbody):
        raise TypeError("token IDs must be integers")
    values = tuple(int(value) for value in fullbody)
    if len(values) <= body_limit:
        return values
    capacity = body_limit - prefix_length
    return values[:prefix_length] + values[-capacity:]


def fuse_text_blocks(original_text_blocks, tail_vectors, affected_mask):
    """Fuse unit tail vectors with frozen text blocks without mutating inputs."""
    if not isinstance(original_text_blocks, np.ndarray) or not isinstance(tail_vectors, np.ndarray):
        raise TypeError("text blocks and tail vectors must be numpy arrays")
    if original_text_blocks.ndim != 2 or not original_text_blocks.shape[1]:
        raise ValueError("original_text_blocks must be nonempty N x D")
    if tail_vectors.shape != original_text_blocks.shape:
        raise ValueError("tail_vectors must match original_text_blocks")
    if original_text_blocks.dtype not in (np.float32, np.float64):
        raise TypeError("original_text_blocks must use float32 or float64")
    if tail_vectors.dtype != original_text_blocks.dtype:
        raise TypeError("tail_vectors must have the same dtype as original_text_blocks")
    if not isinstance(affected_mask, np.ndarray) or affected_mask.dtype != np.bool_:
        raise TypeError("affected_mask must be a boolean numpy array")
    if affected_mask.shape != (len(original_text_blocks),):
        raise ValueError("affected_mask must have shape (N,)")
    if not np.isfinite(original_text_blocks).all() or not np.isfinite(tail_vectors).all():
        raise ValueError("text vectors must be finite")
    result = original_text_blocks.copy()
    if not affected_mask.any():
        return result
    selected = np.flatnonzero(affected_mask)
    original = original_text_blocks[selected].astype(np.float64)
    tails = tail_vectors[selected].astype(np.float64)
    original_norm = np.linalg.norm(original, axis=1)
    tail_norm = np.linalg.norm(tails, axis=1)
    if np.any(original_norm <= 0):
        raise ValueError("affected original text blocks must have positive norm")
    if not np.allclose(tail_norm, 1.0, rtol=0.0, atol=2e-4):
        raise ValueError("affected tail vectors must be unit normalized")
    heads = original / original_norm[:, None]
    combined = heads + tails
    combined_norm = np.linalg.norm(combined, axis=1)
    fused = heads.copy()
    usable = combined_norm > 1e-12
    fused[usable] = combined[usable] / combined_norm[usable, None]
    fused *= original_norm[:, None]
    result[selected] = fused.astype(result.dtype)
    if not np.isfinite(result).all():
        raise FloatingPointError("fusion produced non-finite values")
    return result

