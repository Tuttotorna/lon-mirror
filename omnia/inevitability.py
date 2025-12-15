# omnia/inevitability.py
# OMNIA · Ω-INEV — Inevitability Lens
# MB-X.01 / OMNIA
# Author: Massimiliano Brighindi
# License: MIT

from typing import Callable, List
import numpy as np


class InevitabilityResult:
    def __init__(self, omega_inev: float, stability_curve: List[float]):
        self.omega_inev = omega_inev          # [0,1] inevitability score
        self.stability_curve = stability_curve


def omega_inevitability(
    base_signal: np.ndarray,
    perturbations: List[Callable[[np.ndarray], np.ndarray]],
    signature_fn: Callable[[np.ndarray], float],
    tolerance: float = 1e-6
) -> InevitabilityResult:
    """
    Measures how inevitable a structural signature is under arbitrary perturbations.

    - base_signal: original structure
    - perturbations: independent deviation functions
    - signature_fn: structural signature extractor
    - tolerance: stability threshold

    Returns Ω-INEV score and per-perturbation stability.
    """

    base_sig = signature_fn(base_signal)
    stability = []

    for perturb in perturbations:
        altered = perturb(base_signal)
        sig = signature_fn(altered)
        delta = abs(sig - base_sig)
        stability.append(1.0 if delta <= tolerance else 0.0)

    omega_inev = float(np.mean(stability))

    return InevitabilityResult(
        omega_inev=omega_inev,
        stability_curve=stability
    )