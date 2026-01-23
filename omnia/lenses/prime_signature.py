"""
OMNIA — Prime Signature Lens (Φ)

This lens builds a deterministic structural signature Φ(p)
for a prime (or integer) using multi-base residue geometry.

Goal:
- Not to "predict primes"
- To embed prime events into a structural space where regimes can be measured

Φ(p) is a compact vector:
- residues across small moduli (bases)
- simple gap-local features (optional upstream)
- normalized into [0,1]

This is a measurement lens: no semantics, no oracle.

Author: Massimiliano Brighindi (MB-X.01)
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass(frozen=True)
class PrimeSignature:
    """
    Φ(p): multi-base residue signature.

    components: list of floats in [0,1]
    moduli: the bases/moduli used
    """

    p: int
    moduli: List[int]
    components: List[float]

    def l2(self, other: "PrimeSignature") -> float:
        """Euclidean distance in Φ-space."""
        if self.moduli != other.moduli:
            raise ValueError("Signature moduli mismatch")
        s = 0.0
        for a, b in zip(self.components, other.components):
            d = a - b
            s += d * d
        return s ** 0.5


def prime_phi(p: int, moduli: Iterable[int] = (2, 3, 5, 7, 11, 13, 17)) -> PrimeSignature:
    """
    Compute Φ(p) using normalized residues:

        r_m = (p mod m) / (m-1)

    This yields a stable embedding in [0,1]^k.

    Note:
    - This is not number theory magic.
    - It is a structural coordinate system for OMNIA regimes.
    """
    if p <= 1:
        raise ValueError("p must be >= 2")

    mods = list(moduli)
    comps: List[float] = []

    for m in mods:
        if m <= 1:
            raise ValueError("Invalid modulus")
        r = (p % m) / float(m - 1)
        comps.append(_clamp01(r))

    return PrimeSignature(p=p, moduli=mods, components=comps)


def phi_batch(primes: List[int], moduli: Iterable[int]) -> List[PrimeSignature]:
    """Compute Φ for a list of primes/integers."""
    return [prime_phi(p, moduli=moduli) for p in primes]


def nearest_signature(
    target: PrimeSignature,
    pool: List[PrimeSignature],
    k: int = 5,
) -> List[PrimeSignature]:
    """
    Deterministic KNN in Φ-space.
    Returns the k closest signatures.
    """
    if k <= 0:
        return []

    scored = [(s.l2(target), s) for s in pool]
    scored.sort(key=lambda x: x[0])
    return [s for _, s in scored[:k]]