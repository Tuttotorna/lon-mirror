# omnia/metrics.py
# OMNIA metrics — MB-X.01 / OMNIA_TOTALE
# Stable ASCII API (no Unicode identifiers)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any

try:
    import numpy as np
except ImportError:
    np = None


EPS = 1e-12


def _clip01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def truth_omega(coherence: float, *, eps: float = EPS) -> float:
    """
    TruthOmega = -log( clamp(coherence, eps, 1) )

    coherence:
      - expected range: [0, 1]
      - 1 -> perfectly coherent (TruthOmega=0)
      - 0 -> maximally incoherent (TruthOmega large)
    """
    c = max(eps, min(1.0, float(coherence)))
    return -math.log(c)


def co_plus(truth_omega_value: float) -> float:
    """
    CoPlus = exp(-TruthOmega)
    (inverse mapping; stable in [0,1])
    """
    return _clip01(math.exp(-float(truth_omega_value)))


def score_plus(co_plus_value: float, *, bias: float = 0.0, info: float = 1.0) -> float:
    """
    ScorePlus: simple composite in [0,1].

    Design (reviewer-friendly):
      - base = clip01(co_plus_value + bias)
      - info <= 0 => 0 (no usable information)
      - score = clip01(base * info)

    This keeps the output bounded and deterministic.
    """
    base = _clip01(float(co_plus_value) + float(bias))
    info = float(info)
    if info <= 0.0:
        return 0.0
    return _clip01(base * info)


def delta_coherence(values: Iterable[float] | Any, *, eps: float = EPS) -> float:
    """
    Delta-coherence: dispersion proxy (0 = identical, higher = more spread).
    Uses normalized mean absolute deviation around mean.

    Optimized for numpy arrays if numpy is available.
    """
    if np is not None and isinstance(values, (np.ndarray, list, tuple)):
        # Attempt vectorized path
        try:
            arr = np.asanyarray(values, dtype=float)
            if arr.size == 0:
                return 0.0
            m = np.mean(arr)
            mad = np.mean(np.abs(arr - m))
            denom = abs(m) + eps
            return float(mad / denom)
        except Exception:
            # Fallback if numpy fails for any reason
            pass

    # Standard python fallback
    xs = [float(v) for v in values]
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    mad = sum(abs(x - m) for x in xs) / len(xs)
    denom = abs(m) + eps
    return mad / denom


def kappa_alignment(a: float, b: float, *, eps: float = EPS) -> float:
    """
    Kappa-alignment: similarity score in [0,1] between two signals.
    1 = equal, 0 = maximally different (relative).
    """
    a = float(a)
    b = float(b)
    denom = max(eps, abs(a) + abs(b))
    return _clip01(1.0 - (abs(a - b) / denom))


def epsilon_drift(prev: float, curr: float, *, eps: float = EPS) -> float:
    """
    Epsilon-drift: relative change magnitude >= 0.
    """
    prev = float(prev)
    curr = float(curr)
    denom = abs(prev) + eps
    return abs(curr - prev) / denom


@dataclass(frozen=True)
class OmegaMetrics:
    truth_omega: float
    co_plus: float
    score_plus: float
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
    bias: float = 0.0,
    info: float = 1.0,
) -> OmegaMetrics:
    """
    Convenience bundle:
    - TruthOmega from coherence
    - CoPlus from TruthOmega
    - ScorePlus from CoPlus (+bias, *info)
    - Delta from lens_values (if provided)
    - Kappa from align_a/align_b (if provided)
    - Epsilon from prev_score/curr_score (if provided)
    """
    t = truth_omega(coherence)
    c = co_plus(t)
    s = score_plus(c, bias=bias, info=info)
    d = delta_coherence(lens_values or [])
    k = 0.0 if (align_a is None or align_b is None) else kappa_alignment(align_a, align_b)
    e = 0.0 if (prev_score is None or curr_score is None) else epsilon_drift(prev_score, curr_score)
    return OmegaMetrics(truth_omega=t, co_plus=c, score_plus=s, delta=d, kappa=k, epsilon=e)


def compute_metrics(
    coherence: float,
    *,
    bias: float = 0.0,
    info: float = 1.0,
    kappa_ref: float = 1.0,
    eps_ref: float = 0.0,
) -> Dict[str, float]:
    """
    Convenience wrapper expected by CI tests.

    Returns dict keys:
      truth_omega, co_plus, score_plus, delta_coherence, kappa_alignment, epsilon_drift
    """
    t = truth_omega(coherence)
    c = co_plus(t)
    s = score_plus(c, bias=bias, info=info)

    # With a single scalar coherence, dispersion is defined as 0 by construction.
    d = delta_coherence([coherence])

    k = kappa_alignment(coherence, kappa_ref)
    e = epsilon_drift(eps_ref, coherence)

    return {
        "truth_omega": float(t),
        "co_plus": float(c),
        "score_plus": float(s),
        "delta_coherence": float(d),
        "kappa_alignment": float(k),
        "epsilon_drift": float(e),
    }


__all__ = [
    "EPS",
    "truth_omega",
    "co_plus",
    "score_plus",
    "delta_coherence",
    "kappa_alignment",
    "epsilon_drift",
    "OmegaMetrics",
    "omega_metrics",
    "compute_metrics",
]
