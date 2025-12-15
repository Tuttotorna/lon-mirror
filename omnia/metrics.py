# omnia/metrics.py
# OMNIA metrics — MB-X.01 / OMNIA_TOTALE
# Stable ASCII API (no Unicode identifiers)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


EPS = 1e-12


def _clip01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def truth_omega(coherence: float, *, eps: float = EPS) -> float:
    """
    TruthΩ = -log( clamp(coherence, eps, 1) )

    coherence:
      - expected range: [0, 1]
      - 1 -> perfectly coherent (TruthΩ=0)
      - 0 -> maximally incoherent (TruthΩ large)
    """
    c = max(eps, min(1.0, float(coherence)))
    return -math.log(c)


def co_plus(truth_omega_value: float) -> float:
    """
    Co⁺ = exp(-TruthΩ)
    (inverse mapping; stable in [0,1])
    """
    return _clip01(math.exp(-float(truth_omega_value)))


def delta_coherence(values: Iterable[float], *, eps: float = EPS) -> float:
    """
    Δ-coherence: dispersion proxy (0 = identical, higher = more spread).
    Uses normalized mean absolute deviation around mean.
    """
    xs = [float(v) for v in values]
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    mad = sum(abs(x - m) for x in xs) / len(xs)
    denom = abs(m) + eps
    return mad / denom


def kappa_alignment(a: float, b: float, *, eps: float = EPS) -> float:
    """
    κ-alignment: similarity score in [0,1] between two positive signals.
    1 = equal, 0 = maximally different (relative).
    """
    a = float(a)
    b = float(b)
    denom = max(eps, abs(a) + abs(b))
    return _clip01(1.0 - (abs(a - b) / denom))


def epsilon_drift(prev: float, curr: float, *, eps: float = EPS) -> float:
    """
    ε-drift: relative change magnitude >=0.
    """
    prev = float(prev)
    curr = float(curr)
    denom = abs(prev) + eps
    return abs(curr - prev) / denom


@dataclass(frozen=True)
class OmegaMetrics:
    truth_omega: float
    co_plus: float
    delta: float
    kappa: float
    epsilon: float


def omega_metrics(
    coherence: float,
    *,
    lens_values: Optional[Iterable[float]] = None,
    prev_score: Optional[float] = None,
    curr_score: Optional[float] = None,
    align_a: Optional[float] = None,
    align_b: Optional[float] = None,
) -> OmegaMetrics:
    """
    Convenience bundle:
    - TruthΩ from coherence
    - Co⁺ from TruthΩ
    - Δ from lens_values (if provided)
    - κ from align_a/align_b (if provided)
    - ε from prev_score/curr_score (if provided)
    """
    t = truth_omega(coherence)
    c = co_plus(t)
    d = delta_coherence(lens_values or [])
    k = 0.0 if (align_a is None or align_b is None) else kappa_alignment(align_a, align_b)
    e = 0.0 if (prev_score is None or curr_score is None) else epsilon_drift(prev_score, curr_score)
    return OmegaMetrics(truth_omega=t, co_plus=c, delta=d, kappa=k, epsilon=e)


__all__ = [
    "EPS",
    "truth_omega",
    "co_plus",
    "delta_coherence",
    "kappa_alignment",
    "epsilon_drift",
    "OmegaMetrics",
    "omega_metrics",
]