from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from omnia.features.meaning_blind import meaning_blind_features

Transform = Callable[[str], str]


def _jaccard_like(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Continuous Jaccard over sparse feature maps: sum(min)/sum(max).
    Meaning-blind: operates on structural tokens only.
    """
    if not a and not b:
        return 1.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 1.0
    num = 0.0
    den = 0.0
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, 0.0))
        num += min(av, bv)
        den += max(av, bv)
    return 0.0 if den <= 0.0 else max(0.0, min(1.0, num / den))


@dataclass(frozen=True)
class IrreversibilityResult:
    """
    Irreversibility Invariance (IIv-1.0)

    Measures structural hysteresis: A -> B -> A' where A' is not structurally
    recoverable even if surface form looks similar.

    - omega_A: Ω(A, A) = 1 by definition (baseline self-similarity)
    - omega_AB: Ω(A, B) structural similarity after forward transform
    - omega_AAprime: Ω(A, A') similarity after round-trip
    - iri: max(0, omega_A - omega_AAprime) = 1 - omega_AAprime
    - is_irreversible: iri > eps
    """
    omega_AB: float
    omega_AAprime: float
    iri: float
    is_irreversible: bool
    diagnostics: Dict[str, float]


class IrreversibilityInvariance:
    """
    Irreversibility Invariance Lens (IIv-1.0)

    Inputs:
    - forward: A -> B
    - backward: B -> A'  (not required to be a true inverse)

    Measurement:
      fA  = F(A)
      fB  = F(B)
      fAp = F(A')
      Ω(A,B)  = J(fA,fB)
      Ω(A,A') = J(fA,fAp)
      IRI     = max(0, 1 - Ω(A,A'))

    Notes:
    - No semantics: F is meaning-blind features.
    - Deterministic: if transforms are stochastic, seed-lock them.
    """

    def __init__(self, forward: Transform, backward: Transform, *, eps: float = 1e-6):
        self.forward = forward
        self.backward = backward
        self.eps = float(eps)

    def measure(self, a: str) -> IrreversibilityResult:
        fA = meaning_blind_features(a)

        b = self.forward(a)
        fB = meaning_blind_features(b)

        a_prime = self.backward(b)
        fAp = meaning_blind_features(a_prime)

        omega_AB = _jaccard_like(fA, fB)
        omega_AAprime = _jaccard_like(fA, fAp)

        iri = max(0.0, 1.0 - omega_AAprime)
        is_irreversible = iri > self.eps

        diag = {
            "omega_AB": omega_AB,
            "omega_AAprime": omega_AAprime,
            "eps": self.eps,
            "len_A": float(len(a)),
            "len_B": float(len(b)),
            "len_Aprime": float(len(a_prime)),
        }

        return IrreversibilityResult(
            omega_AB=omega_AB,
            omega_AAprime=omega_AAprime,
            iri=iri,
            is_irreversible=is_irreversible,
            diagnostics=diag,
        )


# -------------------------
# Minimal example transforms
# -------------------------

def forward_drop_vowels(s: str) -> str:
    import re
    return re.sub(r"[aeiouAEIOUàèéìòùÀÈÉÌÒÙ]", "", s)


def backward_identity(s: str) -> str:
    # Not a true inverse: demonstrates hysteresis (information loss stays lost)
    return s


if __name__ == "__main__":
    lens = IrreversibilityInvariance(
        forward=forward_drop_vowels,
        backward=backward_identity,
        eps=1e-6,
    )

    A = """
    OMNIA measures structure only. Removing vowels is lossy.
    If we "return" with identity, A' cannot recover A structurally.
    2026 2025 2024
    """

    r = lens.measure(A)
    print("Irreversibility Invariance (IIv-1.0)")
    print("Ω(A,B):", round(r.omega_AB, 4))
    print("Ω(A,A'):", round(r.omega_AAprime, 4))
    print("IRI:", round(r.iri, 6))
    print("is_irreversible:", r.is_irreversible)