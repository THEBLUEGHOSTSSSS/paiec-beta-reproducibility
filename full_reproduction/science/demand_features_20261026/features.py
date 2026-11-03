"""Frozen, label-free full-text demand/structure proxy features."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np

_SPEC = json.loads((Path(__file__).with_name("schema.json")).read_text())
NAMES = tuple(_SPEC["names"])
_PATTERNS = tuple(re.compile(_SPEC["patterns"][name], re.I) for name in NAMES[:16])
_SCALES = tuple(_SPEC["scales"])


def raw(text: str) -> np.ndarray:
    """Return the ordered 24 frozen proxy values for *text*."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    value = unicodedata.normalize("NFKC", text)
    binary = [float(pattern.search(value) is not None) for pattern in _PATTERNS]
    counts = (
        len(value),
        len(value.split()),
        value.count("\n"),
        len(re.findall(r"\d+(?:\.\d+)?", value)),
        len(re.findall(r"[+*/=<>−×÷]", value)),
        len(re.findall(r"^\s*(?:[-*]|\d+[.)])\s+", value, re.M)),
        value.count("```"),
        len(re.findall(r"[?？]", value)),
    )
    structural = [min(1.0, math.log1p(count) / math.log1p(scale))
                  for count, scale in zip(counts, _SCALES)]
    return np.asarray(binary + structural, dtype=np.float64)


def extract(text: str) -> np.ndarray:
    """Return the unit-normalized bias-plus-proxy 25-vector."""
    value = np.concatenate((np.ones(1, dtype=np.float64), raw(text)))
    return value / np.linalg.norm(value)

