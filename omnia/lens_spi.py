# omnia/lens_spi.py
# SPI — Structural Persistence Index
# OMNIA lens: pure structural measurement
# Deterministic, ASCII-only, standalone

from __future__ import annotations

import math
from typing import Iterable, List, Callable

import numpy as np


# -------------------------------------------------
# Core idea
# -------------------------------------------------
# SPI(x) measures how much a signal x "persists"
# under a set of simultaneous references.
#
# - No privileged unit
# - No privileged base
# - No semantics
# - Only dispersion under transformations
#
# Low SPI  -> structure persists (rigid)
# High SPI -> structure fragile (reference-dependent)
# -------------------------------------------------


# -------------------------------------------------
# Reference transforms (numeric, minimal)
# -------------------------------------------------

def ref_identity(x: int) -> float:
    return float(x)


def ref_log(x: int) -> float:
    if x <= 0:
        return 0.0
    return math.log(float(x))


def ref_sqrt(x: int) -> float:
    if x < 0:
        return 0.0
    return math.sqrt(float(x))


def ref_digit_sum(x: int) -> float:
    return float(sum(int(c) for c in str(abs(int(x)))))


# Default reference set (can be overridden)
DEFAULT_REFS: List[Callable[[int], float]] = [
    ref_identity,
    ref_log,
    ref_sqrt,
    ref_digit_sum,
]


# -------------------------------------------------
# SPI computation
# -------------------------------------------------

def spi(x: int, refs: Iterable[Callable[[int], float]] = DEFAULT_REFS) -> float:
    """
    Structural Persistence Index for a single value x.

    SPI(x) = variance of responses under all references,
             normalized to be scale-stable.

    Deterministic.
    """
    vals: List[float] = []

    for r in refs:
        try:
            v = float(r(int(x)))
        except Exception:
            v = 0.0
        if math.isnan(v) or math.isinf(v):
            v = 0.0
        vals.append(v)

    if len(vals) <= 1:
        return 0.0

    arr = np.array(vals, dtype=float)

    # center (no privileged origin)
    arr = arr - float(arr.mean())

    # variance as dispersion proxy
    var = float(arr.var(ddof=0))

    return var


# -------------------------------------------------
# Series utilities
# -------------------------------------------------

def spi_series(n_min: int, n_max: int,
               refs: Iterable[Callable[[int], float]] = DEFAULT_REFS) -> np.ndarray:
    """
    Compute SPI for n in [n_min, n_max].
    """
    out = []
    for n in range(int(n_min), int(n_max) + 1):
        out.append(spi(n, refs))
    return np.array(out, dtype=float)


def local_minima_indices(values: np.ndarray, window: int) -> List[int]:
    """
    Indices of local minima with a symmetric window.
    """
    w = int(window)
    idx: List[int] = []
    for i in range(w, len(values) - w):
        if values[i] == np.min(values[i - w:i + w + 1]):
            idx.append(i)
    return idx


# -------------------------------------------------
# Null model: block shuffle
# -------------------------------------------------

def block_shuffle(arr: np.ndarray, block: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    block = max(1, int(block))

    n = len(arr)
    blocks = [arr[i:i + block] for i in range(0, n, block)]
    order = np.arange(len(blocks))
    rng.shuffle(order)

    return np.concatenate([blocks[i] for i in order])


def block_shuffle_null_zscore(values: np.ndarray,
                             window: int,
                             block: int,
                             trials: int,
                             seed: int = 0) -> float:
    """
    Z-score of real minima count vs block-shuffle null.
    """
    real_min = len(local_minima_indices(values, window))

    null_counts = []
    for t in range(int(trials)):
        shuffled = block_shuffle(values, block, seed + t)
        null_counts.append(len(local_minima_indices(shuffled, window)))

    null_arr = np.array(null_counts, dtype=float)
    mu = float(null_arr.mean())
    sd = float(null_arr.std(ddof=0))

    if sd <= 1e-12:
        return 0.0

    return (float(real_min) - mu) / sd