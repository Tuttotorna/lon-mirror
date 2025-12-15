# omnia/metrics.py
# OMNIA — Metrics (TruthΩ / Δ / κ / ε) — MB-X.01
# Massimiliano Brighindi
#
# Goal of this module:
# - Provide a stable, minimal, testable API for OMNIA metrics.
# - Ensure:
#   (1) identical vectors across bases => Δ=0, ε=0, κ=1, TruthΩ=1
#   (2) length mismatch across bases => raises ValueError
#   (3) unstable injections (e.g., 999 in one base) reduce κ and TruthΩ

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import math

import numpy as np


@dataclass(frozen=True)
class Metrics:
    truth_omega: float
    delta_coherence: float
    kappa_alignment: float
    epsilon_drift: float


def _as_float_array(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(list(v), dtype=float)
    if a.ndim != 1:
        raise ValueError("Each signature vector must be 1D.")
    return a


def _matrix_from_signatures(signatures: Mapping[int, Sequence[float]]) -> Tuple[np.ndarray, List[int]]:
    if not signatures or len(signatures) < 2:
        raise ValueError("signatures must contain at least 2 bases.")

    bases = sorted(int(b) for b in signatures.keys())
    vecs = [_as_float_array(signatures[b]) for b in bases]

    L = int(vecs[0].shape[0])
    for i, v in enumerate(vecs[1:], start=1):
        if int(v.shape[0]) != L:
            raise ValueError("All signature vectors must have the same length across bases.")

    M = np.stack(vecs, axis=0)  # shape: (B, L)
    return M, bases


def delta_coherence(signatures: Mapping[int, Sequence[float]]) -> float:
    """
    Δ-coherence: mean standard deviation across bases per component.
    Identical vectors across bases => 0.
    """
    M, _ = _matrix_from_signatures(signatures)
    std_per_pos = np.std(M, axis=0, ddof=0)
    return float(np.mean(np.abs(std_per_pos)))


def epsilon_drift(signatures: Mapping[int, Sequence[float]]) -> float:
    """
    ε-drift: mean range (max-min) across bases per component.
    Identical vectors across bases => 0.
    """
    M, _ = _matrix_from_signatures(signatures)
    rng_per_pos = np.max(M, axis=0) - np.min(M, axis=0)
    return float(np.mean(np.abs(rng_per_pos)))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation with deterministic handling of zero-variance vectors:
    - If both vectors are identical constants => corr = 1
    - If constants but different => corr = 0
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    sa = float(np.std(a))
    sb = float(np.std(b))

    if sa == 0.0 and sb == 0.0:
        return 1.0 if float(np.max(np.abs(a - b))) == 0.0 else 0.0
    if sa == 0.0 or sb == 0.0:
        return 0.0

    c = float(np.corrcoef(a, b)[0, 1])
    if math.isnan(c):
        return 0.0
    return c


def kappa_alignment(signatures: Mapping[int, Sequence[float]]) -> float:
    """
    κ-alignment: average pairwise correlation between base vectors, mapped to [0,1].
    - Identical vectors across bases => 1
    - Divergent/injected vectors => decreases
    """
    M, bases = _matrix_from_signatures(signatures)
    B = M.shape[0]

    corrs: List[float] = []
    for i in range(B):
        for j in range(i + 1, B):
            c = _safe_corr(M[i], M[j])  # in [-1,1]
            corrs.append(c)

    if not corrs:
        return 0.0

    avg = float(np.mean(corrs))  # [-1,1]
    # map to [0,1]
    k = 0.5 * (avg + 1.0)
    # clamp
    if k < 0.0:
        k = 0.0
    if k > 1.0:
        k = 1.0
    return k


def truth_omega(delta: float, epsilon: float) -> float:
    """
    TruthΩ in (0,1], equals 1 iff delta=0 and epsilon=0.
    Chosen stable form: exp(-( |Δ| + |ε| )).
    """
    t = math.exp(-(abs(float(delta)) + abs(float(epsilon))))
    # numeric safety
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return t


def compute_metrics(signatures: Mapping[int, Sequence[float]]) -> Metrics:
    """
    Main entrypoint used by tests and higher layers.
    """
    d = delta_coherence(signatures)
    e = epsilon_drift(signatures)
    k = kappa_alignment(signatures)
    t = truth_omega(d, e)
    return Metrics(
        truth_omega=t,
        delta_coherence=d,
        kappa_alignment=k,
        epsilon_drift=e,
    )
```0