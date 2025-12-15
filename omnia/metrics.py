# omnia/metrics.py
# OMNIA · Core metric utilities
# MB-X.01
# License: MIT

from typing import Dict


def compute_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Canonical metrics entry point.

    This function defines the stable public API expected by tests
    and higher-level aggregators (Ω-TOTAL).

    Current behavior:
    - validates input
    - casts values to float
    - returns metrics unchanged

    Future versions may:
    - normalize
    - clamp
    - weight
    """

    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")

    out: Dict[str, float] = {}
    for k, v in metrics.items():
        out[k] = float(v)

    return out