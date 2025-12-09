"""
omnia.core.omniabase — multi-base numeric lens (PBII, signatures)

Provides:
- digits_in_base_np: integer → digits in base b (NumPy, MSB first)
- normalized_entropy_base: Shannon entropy of digits, normalized [0,1]
- sigma_b: base symmetry score (low entropy + divisibility bonus)
- OmniabaseSignature: dataclass with per-base scores and means
- omniabase_signature: compute multi-base signature for n
- pbii_index: Prime Base Instability Index (PBII)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable
import math

import numpy as np


# =========================
# 1. DIGITS + ENTROPY
# =========================

def digits_in_base_np(n: int, b: int) -> np.ndarray:
    """Return digits of n in base b as numpy array (MSB first)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if b <= 1:
        raise ValueError("base must be >= 2")
    if n == 0:
        return np.array([0], dtype=int)
    digits = []
    while n > 0:
        digits.append(n % b)
        n //= b
    return np.array(digits[::-1], dtype=int)


def normalized_entropy_base(n: int, b: int) -> float:
    """Normalized Shannon entropy of digits of n in base b."""
    digits = digits_in_base_np(n, b)
    L = len(digits)
    if L == 0:
        return 0.0
    counts = np.bincount(digits, minlength=b).astype(float)
    probs = counts[counts > 0] / L
    if probs.size == 0:
        return 0.0
    H = -np.sum(probs * np.log2(probs))
    Hmax = math.log2(b)
    return float(H / Hmax) if Hmax > 0 else 0.0


# =========================
# 2. SIGMA_b + SIGNATURE
# =========================

def sigma_b(
    n: int,
    b: int,
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> float:
    """
    Base Symmetry Score (NumPy version).

    sigma_b(n) = length_weight * (1 - H_norm) / L^length_exponent
                 + divisibility_bonus * I[n % b == 0]
    """
    digits = digits_in_base_np(n, b)
    L = len(digits)
    if L == 0:
        return 0.0

    counts = np.bincount(digits, minlength=b).astype(float)
    probs = counts[counts > 0] / L
    if probs.size == 0:
        Hn = 0.0
    else:
        H = -np.sum(probs * np.log2(probs))
        Hmax = math.log2(b)
        Hn = float(H / Hmax) if Hmax > 0 else 0.0

    length_term = length_weight * (1.0 - Hn) / (L ** length_exponent)
    div_term = divisibility_bonus * (1.0 if n % b == 0 else 0.0)
    return float(length_term + div_term)


@dataclass
class OmniabaseSignature:
    n: int
    bases: List[int]
    sigmas: Dict[int, float]
    entropy: Dict[int, float]
    sigma_mean: float
    entropy_mean: float

    def to_dict(self) -> Dict:
        return asdict(self)


def omniabase_signature(
    n: int,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> OmniabaseSignature:
    """Compute multi-base signature for integer n."""
    bases = list(bases)
    sigmas: Dict[int, float] = {}
    entropy: Dict[int, float] = {}
    for b in bases:
        sig = sigma_b(
            n,
            b,
            length_weight=length_weight,
            length_exponent=length_exponent,
            divisibility_bonus=divisibility_bonus,
        )
        Hn = normalized_entropy_base(n, b)
        sigmas[b] = sig
        entropy[b] = Hn
    sigma_vals = np.array(list(sigmas.values()), dtype=float)
    ent_vals = np.array(list(entropy.values()), dtype=float)
    return OmniabaseSignature(
        n=n,
        bases=bases,
        sigmas=sigmas,
        entropy=entropy,
        sigma_mean=float(sigma_vals.mean()) if sigma_vals.size else 0.0,
        entropy_mean=float(ent_vals.mean()) if ent_vals.size else 0.0,
    )


# =========================
# 3. PBII INDEX
# =========================

def pbii_index(
    n: int,
    composite_window: Iterable[int] = (4, 6, 8, 9, 10, 12, 14, 15),
    bases: Iterable[int] = (2, 3, 5,