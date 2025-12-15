# omnia/metrics.py
# OMNIA · Core metric utilities
# MB-X.01
# License: MIT

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


def compute_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Canonical metrics entry point (stable API).
    Validates input and casts values to float.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    return {str(k): float(v) for k, v in metrics.items()}


def truth_omega(metrics: Dict[str, float]) -> float:
    """
    TruthΩ (minimal stable definition).
    Current: arithmetic mean of metric values.
    """
    if not metrics:
        return 0.0
    vals = [float(v) for v in metrics.values()]
    return sum(vals) / len(vals)


# --- Compatibility layer: symbols expected by omnia/__init__.py and tests --- #

def delta_coherence(metrics: Dict[str, float]) -> float:
    """
    Δ-coherence (minimal stable definition).
    Current: same as TruthΩ until a richer formulation is introduced.
    """
    return truth_omega(metrics)


def kappa_alignment(metrics: Dict[str, float]) -> float:
    """
    κ-alignment (minimal stable definition).
    Current: 1 - normalized dispersion proxy; bounded in [0,1].
    """
    if not metrics:
        return 1.0
    vals = [float(v) for v in metrics.values()]
    m = sum(vals) / len(vals)
    # mean absolute deviation
    mad = sum(abs(v - m) for v in vals) / len(vals)
    # simple squashing to [0,1]
    return 1.0 / (1.0 + mad)


def epsilon_drift(metrics: Dict[str, float]) -> float:
    """
    ε-drift (minimal stable definition).
    Current: dispersion proxy (MAD).
    """
    if not metrics:
        return 0.0
    vals = [float(v) for v in metrics.values()]
    m = sum(vals) / len(vals)
    return sum(abs(v - m) for v in vals) / len(vals)