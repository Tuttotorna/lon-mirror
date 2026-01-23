# omnia/engine/prime_kernel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from omnia.lenses.prime_field_lens import PrimeFieldLens, PrimeFieldLensResult


@dataclass(frozen=True)
class PrimeOMNIAReport:
    """Minimal OMNIA-style report for prime-domain measurements."""
    omega_prime: float
    per_lens: Dict[str, Dict[str, Any]]


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class PrimeKernel:
    """
    Minimal deterministic OMNIA kernel for prime-domain lenses.
    - Runs registered lenses on the same prime sequence.
    - Aggregates lens omega_scores into a single Ω_prime.

    Aggregator: simple mean over lens omega_scores (stable, deterministic).
    """

    def __init__(self, lenses: Sequence[Tuple[str, Any]] | None = None) -> None:
        if lenses is None:
            lenses = [("prime_field", PrimeFieldLens(k=2))]
        self.lenses = list(lenses)

    def measure(self, primes: Sequence[int]) -> PrimeOMNIAReport:
        per_lens: Dict[str, Dict[str, Any]] = {}
        scores: List[float] = []

        for name, lens in self.lenses:
            out = lens.measure(primes)

            # support objects that expose omega_score + details
            omega = float(getattr(out, "omega_score"))
            details = getattr(out, "details", {})

            omega = _clip01(omega)
            scores.append(omega)

            per_lens[name] = {
                "omega": omega,
                "details": details,
            }

        omega_prime = sum(scores) / float(len(scores)) if scores else 0.0
        omega_prime = _clip01(omega_prime)

        return PrimeOMNIAReport(
            omega_prime=omega_prime,
            per_lens=per_lens,
        )