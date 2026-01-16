# MB-X.01 / OMNIA — Ω̂ (Omega-set) Residue Estimator
# Massimiliano Brighindi
#
# Purpose:
#   Formalize "Ω is deduced by subtraction":
#   given multiple structural measurements under different representations/transforms,
#   estimate the invariant residue (Ω̂) that survives them.
#
# Properties:
#   - post-hoc
#   - deterministic
#   - semantics-free
#   - model-agnostic
#   - does not decide, does not optimize
#
# Minimal operational definition (v0.1):
#   Given a set of scalar Omega observations {ω_i} produced under independent transforms,
#   define:
#     Ω̂ = robust_center({ω_i})          (median or trimmed-mean)
#     dispersion = MAD({ω_i})           (robust spread)
#     invariance = 1 / (1 + dispersion) (monotone map, higher = more invariant)
#
# This is a minimal, defensible "residue" estimator for scalar Ω.
# Higher-order residues (vector / per-lens) can be added without breaking v0.1.

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
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _mad(xs: List[float], center: Optional[float] = None, eps: float = 1e-12) -> float:
    if not xs:
        return 0.0
    c = _median(xs) if center is None else center
    dev = [abs(x - c) for x in xs]
    m = _median(dev)
    return float(m + eps)


def omega_set(
    omegas: Iterable[float],
    eps: float = 1e-12,
    method: str = "median+mad",
) -> OmegaSetResult:
    """
    Estimate Ω̂ (omega-hat) as the robust invariant residue of scalar Ω observations.

    Inputs:
      omegas: iterable of ω_i produced under different representations/transforms
      eps: numerical stability
      method: currently only 'median+mad' is supported (frozen v0.1)

    Returns:
      OmegaSetResult:
        - omega_hat: robust center (median)
        - dispersion_mad: robust dispersion (MAD)
        - invariance: 1 / (1 + dispersion_mad)
        - range and count for audit
    """
    xs = [float(x) for x in omegas]
    if len(xs) == 0:
        raise ValueError("omegas must contain at least one value")

    if method != "median+mad":
        raise ValueError("Only method='median+mad' is supported in v0.1")

    center = _median(xs)
    disp = _mad(xs, center=center, eps=eps)
    inv = 1.0 / (1.0 + disp)

    return OmegaSetResult(
        omega_hat=float(center),
        dispersion_mad=float(disp),
        invariance=float(inv),
        n=len(xs),
        min_omega=float(min(xs)),
        max_omega=float(max(xs)),
        method=method,
    )