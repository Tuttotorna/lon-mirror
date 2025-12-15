# omnia/metrics.py
# OMNIA metrics — MB-X.01
# Provides a stable public API used by tests and by omnia/__init__.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Union, Any
import math

Number = Union[int, float]


@dataclass(frozen=True)
class Metrics:
    truth_omega: float
    co_plus: float
    delta_coherence: float
    kappa_alignment: float
    epsilon_drift: float


def _as_vector(x: Any) -> Sequence[float]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    # single number
    if isinstance(x, (int, float)):
        return [float(x)]
    # fallback iterable
    try:
        return [float(v) for v in x]
    except Exception:
        return []


def _mean(v: Sequence[float]) -> float:
    if not v:
        return 0.0
    return sum(v) / float(len(v))


def _std(v: Sequence[float]) -> float:
    n = len(v)
    if n <= 1:
        return 0.0
    m = _mean(v)
    var = sum((a - m) ** 2 for a in v) / float(n)
    return math.sqrt(var)


def truth_omega(signatures_by_base: Mapping[int, Sequence[Number]]) -> float:
    """
    TruthΩ: structural inconsistency across bases.
    Higher => less coherent. Lower => more coherent.
    Defined as log(1 + dispersion_of_base_means).
    """
    if not signatures_by_base:
        return 0.0

    base_means = []
    for _, sig in signatures_by_base.items():
        v = _as_vector(sig)
        base_means.append(_mean(v))

    disp = _std(base_means)
    return float(math.log1p(abs(disp)))


def co_plus(truth_omega_value: float) -> float:
    """
    Co⁺: monotonic inverse of TruthΩ in (0, 1].
    """
    # exp(-TruthΩ) is stable and monotone
    return float(math.exp(-max(0.0, truth_omega_value)))


def delta_coherence(signatures_by_base: Mapping[int, Sequence[Number]]) -> float:
    """
    Δ-coherence: mean intra-base spread averaged across bases.
    Higher => more internal variance => less coherent.
    """
    if not signatures_by_base:
        return 0.0

    spreads = []
    for _, sig in signatures_by_base.items():
        v = _as_vector(sig)
        spreads.append(_std(v))

    return float(_mean(spreads))


def kappa_alignment(signatures_by_base: Mapping[int, Sequence[Number]]) -> float:
    """
    κ-alignment: penalizes a base whose mean deviates strongly from the global mean.
    Higher => worse alignment.
    """
    if not signatures_by_base:
        return 0.0

    means = []
    for _, sig in signatures_by_base.items():
        v = _as_vector(sig)
        means.append(_mean(v))

    g = _mean(means)
    dev = [abs(m - g) for m in means]
    return float(_mean(dev))


def epsilon_drift(signatures_by_base: Mapping[int, Sequence[Number]]) -> float:
    """
    ε-drift: relative instability proxy.
    Here: normalized dispersion of base means.
    """
    if not signatures_by_base:
        return 0.0

    means = []
    for _, sig in signatures_by_base.items():
        v = _as_vector(sig)
        means.append(_mean(v))

    m = _mean(means)
    s = _std(means)
    denom = abs(m) + 1e-12
    return float(s / denom)


def compute_metrics(signatures_by_base: Mapping[int, Sequence[Number]]) -> Metrics:
    """
    Public helper used by tests.
    Returns a Metrics dataclass (attribute access expected by tests).
    """
    t = truth_omega(signatures_by_base)
    c = co_plus(t)
    d = delta_coherence(signatures_by_base)
    k = kappa_alignment(signatures_by_base)
    e = epsilon_drift(signatures_by_base)
    return Metrics(
        truth_omega=t,
        co_plus=c,
        delta_coherence=d,
        kappa_alignment=k,
        epsilon_drift=e,
    )