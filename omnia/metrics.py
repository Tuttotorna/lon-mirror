# omnia/metrics.py
# OMNIA · Core metric utilities
# MB-X.01
# License: MIT

from __future__ import annotations
from typing import Dict, Any, Iterable


def _is_seq(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _mean(seq: Iterable[float]) -> float:
    s = 0.0
    n = 0
    for v in seq:
        s += float(v)
        n += 1
    return 0.0 if n == 0 else (s / n)


def compute_metrics(metrics: Dict[Any, Any]) -> Dict[str, float]:
    """
    Canonical metrics entry point (stable API).

    Accepts:
      - {key: float}
      - {key: [float, float, ...]}
      - {key: (float, float, ...)}
    Returns:
      - {str(key): float} where list/tuple values are reduced by mean.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")

    out: Dict[str, float] = {}
    for k, v in metrics.items():
        key = str(k)

        if _is_seq(v):
            out[key] = _mean(v)
        else:
            out[key] = float(v)

    return out


def truth_omega(metrics: Dict[Any, Any]) -> float:
    """
    TruthΩ — minimal stable definition.

    Uses compute_metrics() normalization, then returns the arithmetic mean
    of the resulting scalar metric values.
    """
    m = compute_metrics(metrics)
    if not m:
        return 0.0
    return _mean(m.values())


def delta_coherence(metrics: Dict[Any, Any]) -> float:
    """
    Δ-coherence — compatibility API.
    Minimal definition: identical to TruthΩ until a richer formulation is introduced.
    """
    return truth_omega(metrics)


def epsilon_drift(metrics: Dict[Any, Any]) -> float:
    """
    ε-drift — compatibility API.
    Minimal definition: mean absolute deviation from the mean, computed on
    normalized scalar metrics.
    """
    m = compute_metrics(metrics)
    if not m:
        return 0.0
    vals = list(m.values())
    mu = _mean(vals)
    return _mean([abs(v - mu) for v in vals])


def kappa_alignment(metrics: Dict[Any, Any]) -> float:
    """
    κ-alignment — compatibility API.
    Minimal definition: squashed inverse of ε-drift, bounded in (0, 1].
    """
    e = epsilon_drift(metrics)
    return 1.0 / (1.0 + float(e))