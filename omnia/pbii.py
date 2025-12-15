from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

from .omniabase import omni_signature
from .metrics import delta_coherence


def pbii_index(
    n: int,
    bases: Iterable[int] = (2, 3, 4, 5, 7, 8, 10, 12, 16),
) -> float:
    """
    PBII — Prime Base Instability Index

    Definition:
      PBII(n) = log(1 + Δ(n)) / log(1 + Δ_ref)

    where:
      Δ(n)     = delta_coherence of omni_signature(n)
      Δ_ref    = delta_coherence of a regular reference (nearest power of two)

    Properties:
    - PBII ≈ 0   -> structurally regular (e.g., powers)
    - PBII ~ 1+  -> high cross-base tension (prime-like / irregular)
    - dimensionless, deterministic, comparable across n
    """

    # structural signature of n
    sig_n = omni_signature(n, bases=bases)
    delta_n = delta_coherence(sig_n)

    # reference: nearest power of two
    if n <= 0:
        ref = 1
    else:
        k = max(0, int(round(math.log(n, 2))))
        ref = 1 << k

    sig_ref = omni_signature(ref, bases=bases)
    delta_ref = delta_coherence(sig_ref)

    # numerical safety
    eps = 1e-12
    num = math.log(1.0 + max(0.0, delta_n))
    den = math.log(1.0 + max(eps, delta_ref))

    return float(num / den)