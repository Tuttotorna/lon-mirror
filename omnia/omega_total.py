# omnia/omega_total.py
# OMNIA · Ω-TOTAL (canonical structural aggregator)
# MB-X.01 / OMNIA
# Author: Massimiliano Brighindi
# License: MIT

from typing import Dict, Optional, Callable, Any

from omnia.inevitability import omega_inevitability


def omega_total(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    *,
    include_inev: bool = False,
    inevitability_input: Optional[Any] = None,
    inevitability_signature: Optional[Callable[[Any], float]] = None,
    inevitability_perturbations: Optional[list] = None,
    inevitability_tolerance: float = 1e-6,
) -> Dict[str, float]:
    """
    Canonical Ω-TOTAL aggregator.

    - metrics: dict of structural metrics (already computed)
    - weights: optional weights per metric (default = uniform)

    Optional Ω-INEV integration (default OFF):
    - include_inev: enable inevitability
    - inevitability_input: base structure
    - inevitability_signature: signature function
    - inevitability_perturbations: list of perturbations
    """

    if not metrics:
        base = {"omega_total": 0.0, "components": {}}
    else:
        if weights is None:
            weights = {k: 1.0 for k in metrics.keys()}

        total_weight = sum(weights.get(k, 0.0) for k in metrics.keys())
        if total_weight == 0:
            base = {"omega_total": 0.0, "components": {}}
        else:
            components = {}
            acc = 0.0

            for k, v in metrics.items():
                w = weights.get(k, 0.0)
                c = w * float(v)
                components[k] = c
                acc += c

            omega = acc / total_weight
            base = {"omega_total": omega, "components": components}

    # --- Ω-INEV (optional, non-invasive) ---
    if include_inev:
        if (
            inevitability_input is not None
            and inevitability_signature is not None
            and inevitability_perturbations is not None
        ):
            inev = omega_inevitability(
                base_signal=inevitability_input,
                perturbations=inevitability_perturbations,
                signature_fn=inevitability_signature,
                tolerance=inevitability_tolerance,
            )
            base["omega_inev"] = inev.omega_inev
            base["inev_curve"] = inev.stability_curve
        else:
            base["omega_inev"] = None
            base["inev_curve"] = None

    return base