# omnia/inev_integration.py
# OMNIA · Ω-INEV integration helper (text perturbation stability)
# MB-X.01 / OMNIA
# Author: Massimiliano Brighindi
# License: MIT

from dataclasses import dataclass
from typing import Callable, List
import re


@dataclass
class InevitabilityResult:
    omega_inev: float
    stability_curve: List[float]


def _perturbations_text(x: str) -> List[str]:
    """Independent, low-impact perturbations. Structural, not semantic."""
    if x is None:
        x = ""
    p0 = x
    p1 = re.sub(r"\s+", " ", x).strip()                 # normalize whitespace
    p2 = x.lower()                                      # lowercasing
    p3 = re.sub(r"[^\w\s]", "", x)                      # remove punctuation
    p4 = re.sub(r"\s+", "", x)                          # remove all spaces
    return [p0, p1, p2, p3, p4]


def omega_inev_from_text(
    text: str,
    score_fn: Callable[[str], float],
    tolerance: float = 1e-3,
) -> InevitabilityResult:
    """
    Ω-INEV: fraction of perturbations whose score remains within tolerance
    from the base score. Pure measurement layer.
    """
    variants = _perturbations_text(text)
    base = score_fn(variants[0])

    curve: List[float] = []
    for v in variants[1:]:
        s = score_fn(v)
        curve.append(1.0 if abs(s - base) <= tolerance else 0.0)

    omega_inev = sum(curve) / float(len(curve)) if curve else 1.0
    return InevitabilityResult(omega_inev=float(omega_inev), stability_curve=curve)