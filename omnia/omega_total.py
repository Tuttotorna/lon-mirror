# omnia/omega_total.py
# OMNIA · Ω-TOTAL (canonical structural aggregator)
# MB-X.01 / OMNIA
# Author: Massimiliano Brighindi
# License: MIT

from typing import Dict, Optional


def omega_total(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Canonical Ω-TOTAL aggregator.

    - metrics: dict of structural metrics (already computed)
    - weights: optional weights per metric (default = uniform)

    Returns a dict with:
    - omega_total: aggregated Ω score
    - components: weighted components
    """

    if not metrics:
        return {"omega_total": 0.0, "components": {}}

    if weights is None:
        weights = {k: 1.0 for k in metrics.keys()}

    total_weight = sum(weights.get(k, 0.0) for k in metrics.keys())
    if total_weight == 0:
        return {"omega_total": 0.0, "components": {}}

    components = {}
    acc = 0.0

    for k, v in metrics.items():
        w = weights.get(k, 0.0)
        c = w * float(v)
        components[k] = c
        acc += c

    omega = acc / total_weight

    return {
        "omega_total": omega,
        "components": components,
    }