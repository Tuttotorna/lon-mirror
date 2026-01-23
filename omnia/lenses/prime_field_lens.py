# omnia/lenses/prime_field_lens.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Optional

from omnia.lenses.prime_field import compute_pld, PLDResult


@dataclass(frozen=True)
class PrimeFieldLensResult:
    """Lens-level output for OMNIA aggregation."""
    omega_score: float
    details: Dict[str, Any]


class PrimeFieldLens:
    """
    OMNIA lens: Prime Field / Local Density.
    Input: sequence of primes (ints).
    Output: an omega_score in [0,1] + details for debugging/inspection.

    Aggregation rule (minimal, deterministic):
    - Use the mean PLD across primes, clipped to [0,1].
    Rationale:
    - PLD is already [0,1] per prime; mean is stable and order-free.
    """

    def __init__(self, k: int = 2, eps: float = 1e-12) -> None:
        self.k = int(k)
        self.eps = float(eps)

    def measure(self, primes: Sequence[int]) -> PrimeFieldLensResult:
        res: PLDResult = compute_pld(primes, k=self.k, eps=self.eps)

        pld = res.pld_by_prime
        if not pld:
            omega = 0.0
        else:
            omega = sum(pld.values()) / float(len(pld))

        # enforce [0,1]
        if omega < 0.0:
            omega = 0.0
        elif omega > 1.0:
            omega = 1.0

        details: Dict[str, Any] = {
            "k": res.k,
            "eps": res.eps,
            "count": len(pld),
            "omega_prime_field": omega,
            "pld_by_prime": pld,
            "dk_by_prime": res.dk_by_prime,
            "rho_by_prime": res.rho_by_prime,
        }

        return PrimeFieldLensResult(omega_score=omega, details=details)