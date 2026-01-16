# MB-X.01 / OMNIA — IRI (Irreversibility / Hysteresis Index)
# Massimiliano Brighindi
#
# Purpose:
#   Measure structural hysteresis: loss of recoverable stability after a cycle A -> B -> A'
#   - post-hoc
#   - deterministic
#   - model-agnostic
#   - semantics-free
#
# Core idea:
#   If output quality can remain similar, but structural stability changes after returning,
#   then we have irreversible structural damage (hysteresis).
#
# Minimal definition:
#   IRI = max(0, Ω(A) - Ω(A'))
#
# Optional extras:
#   - normalize by Ω(A) to obtain relative IRI
#   - report cycle diagnostics

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IRIResult:
    iri_abs: float
    iri_rel: Optional[float]
    omega_a: float
    omega_a_prime: float
    omega_b: Optional[float]
    note: str


def iri_cycle(
    omega_a: float,
    omega_a_prime: float,
    omega_b: float | None = None,
    eps: float = 1e-12,
) -> IRIResult:
    """
    Compute IRI for a hysteresis cycle A -> B -> A'.

    Inputs:
      omega_a:       Ω at baseline state A
      omega_a_prime: Ω after returning (A')
      omega_b:       optional Ω at intermediate state B (for reporting only)

    Returns:
      IRIResult:
        - iri_abs = max(0, Ω(A) - Ω(A'))
        - iri_rel = iri_abs / (|Ω(A)| + eps)  (if Ω(A) != 0)
    """
    iri_abs = omega_a - omega_a_prime
    if iri_abs < 0:
        iri_abs = 0.0

    denom = abs(omega_a) + eps
    iri_rel = iri_abs / denom if denom > eps else None

    note = "OK"
    if iri_abs > 0:
        note = "HYSTERESIS_DETECTED"

    return IRIResult(
        iri_abs=float(iri_abs),
        iri_rel=(float(iri_rel) if iri_rel is not None else None),
        omega_a=float(omega_a),
        omega_a_prime=float(omega_a_prime),
        omega_b=(float(omega_b) if omega_b is not None else None),
        note=note,
    )