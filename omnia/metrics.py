# omnia/metrics.py
# OMNIA · Core metric utilities
# MB-X.01
# License: MIT

from typing import Dict


def compute_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Canonical metrics entry point.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")

    return {k: float(v) for k, v in metrics.items()}


def truth_omega(metrics: Dict[str, float]) -> float:
    """
    TruthΩ — base invariant aggregation.

    Current minimal definition:
    arithmetic mean of metric values.

    This is a placeholder implementation
    that defines a stable API boundary.
    """
    if not metrics:
        return 0.0

    values = [float(v) for v in metrics.values()]
    return sum(values) / len(values)