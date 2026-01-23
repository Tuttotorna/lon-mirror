"""
OMNIA — Unified Structural Metrics (deterministic, semantics-free)

This module centralizes the core scalar metrics used across the repo.
It is intentionally minimal: no I/O, no randomness, no model calls.

Notes:
- TruthΩ is treated as a measured scalar in [0,1].
- Co⁺ and Score⁺ are monotone transforms used for reporting/ranking only.
- Δ-coherence, κ-alignment, ε-drift are generic structural deltas.

Author: Massimiliano Brighindi (MB-X.01)
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def truth_omega(omega: float) -> float:
    """
    TruthΩ: structural invariance/coherence proxy.
    For now it is a strict clamp; the measurement is produced elsewhere (lenses/engine).
    """
    return _clamp01(omega)


def co_plus(truth_omega_value: float) -> float:
    """
    Co⁺: coherence-positive mapping.
    Kept as identity to avoid hidden semantics; future versions may apply a monotone calibration.
    """
    return _clamp01(truth_omega_value)


def score_plus(co_plus_value: float) -> float:
    """
    Score⁺: reporting score.
    Identity by design: OMNIA measures; downstream may decide how to use the score.
    """
    return _clamp01(co_plus_value)


def delta_coherence(omega_a: float, omega_b: float) -> float:
    """
    Δ-coherence: signed structural delta.
    Positive means B is more coherent than A.
    """
    return float(_clamp01(omega_b) - _clamp01(omega_a))


def kappa_alignment(omega_a: float, omega_b: float) -> float:
    """
    κ-alignment: similarity-like alignment in [0,1].
    1.0 when equal, decreasing linearly with absolute difference.
    """
    a = _clamp01(omega_a)
    b = _clamp01(omega_b)
    return _clamp01(1.0 - abs(a - b))


def epsilon_drift(omega_prev: float, omega_next: float) -> float:
    """
    ε-drift: absolute drift magnitude in [0,1].
    """
    p = _clamp01(omega_prev)
    n = _clamp01(omega_next)
    return abs(n - p)


@dataclass(frozen=True)
class MetricPack:
    truth_omega: float
    co_plus: float
    score_plus: float
    delta: float
    kappa: float
    eps_drift: float


def pack(
    omega_a: float,
    omega_b: Optional[float] = None,
) -> MetricPack:
    """
    Create a consistent metric bundle.

    If omega_b is None, deltas are computed against omega_a itself (zero deltas).
    """
    a = truth_omega(omega_a)
    b = truth_omega(omega_b) if omega_b is not None else a

    t = a
    c = co_plus(t)
    s = score_plus(c)

    d = delta_coherence(a, b)
    k = kappa_alignment(a, b)
    e = epsilon_drift(a, b)

    return MetricPack(
        truth_omega=t,
        co_plus=c,
        score_plus=s,
        delta=d,
        kappa=k,
        eps_drift=e,
    )


def pack_pair(omega_a: float, omega_b: float) -> Tuple[MetricPack, MetricPack]:
    """
    Convenience: returns (pack(A vs B), pack(B vs A)).
    """
    return (pack(omega_a, omega_b), pack(omega_b, omega_a))
```0