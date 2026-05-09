# MB-X.01 / OMNIA — Omega-set Residue Estimator
# Massimiliano Brighindi
#
# Purpose:
#   Estimate the invariant residue that survives multiple structural measurements.
#   - post-hoc
#   - deterministic
#   - model-agnostic
#   - semantics-free
#
# Minimal operational definition:
#   Given scalar Omega observations {omega_i}:
#     omega_hat = median({omega_i})
#     dispersion = MAD({omega_i})
#     invariance = 1 / (1 + dispersion)
#
# Boundary:
#   measurement != inference != decision

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class OmegaSetResult:
    omega_hat: float
    dispersion_mad: float
    invariance: float
    n: int
    min_omega: float
    max_omega: float
    method: str


def _median(xs: List[float]) -> float:
    values = sorted(float(x) for x in xs)
    n = len(values)

    if n == 0:
        return 0.0

    mid = n // 2

    if n % 2 == 1:
        return values[mid]

    return 0.5 * (values[mid - 1] + values[mid])


def _mad(xs: List[float], center: Optional[float] = None, eps: float = 0.0) -> float:
    values = [float(x) for x in xs]

    if not values:
        return 0.0

    c = _median(values) if center is None else float(center)
    deviations = [abs(x - c) for x in values]

    return float(_median(deviations) + eps)


def omega_set(
    omegas: Iterable[float],
    eps: float = 0.0,
    method: str = "median+mad",
) -> OmegaSetResult:
    """
    Estimate omega_hat as the robust invariant residue of scalar Omega observations.

    Inputs:
      omegas: scalar Omega observations under different representations / transforms
      eps: optional numerical stabilizer for MAD
      method: currently only 'median+mad'

    Returns:
      OmegaSetResult
    """
    values = [float(x) for x in omegas]

    if len(values) == 0:
        raise ValueError("omegas must contain at least one value")

    if method != "median+mad":
        raise ValueError("Only method='median+mad' is supported")

    center = _median(values)
    dispersion = _mad(values, center=center, eps=eps)
    invariance = 1.0 / (1.0 + dispersion)

    return OmegaSetResult(
        omega_hat=float(center),
        dispersion_mad=float(dispersion),
        invariance=float(invariance),
        n=len(values),
        min_omega=float(min(values)),
        max_omega=float(max(values)),
        method=method,
    )


class OmegaSet:
    """
    Compatibility wrapper for legacy tests.

    Expected usage:
        from omnia.omega_set import OmegaSet
        OmegaSet(values).estimate()

    Returned keys preserve legacy names:
        median
        mad
        invariance
        inv
        n
        min
        max

    Boundary:
        measurement != inference != decision
    """

    def __init__(self, values: Iterable[float], eps: float = 0.0) -> None:
        self.values = [float(x) for x in values]
        self.eps = float(eps)

        if not self.values:
            raise ValueError("OmegaSet requires at least one value")

    def estimate(self) -> dict:
        result = omega_set(self.values, eps=self.eps)

        return {
            "median": result.omega_hat,
            "mad": result.dispersion_mad,
            "invariance": result.invariance,
            "inv": result.invariance,
            "n": result.n,
            "min": result.min_omega,
            "max": result.max_omega,
        }