# omnia/omniabase_pure.py
# OMNIABASE — PURE FORM (unitless, base-agnostic dispersion)
#
# Principle:
# - No privileged unit (no "1" anchor)
# - No privileged zero (E0 is estimated)
# - Measurement = invariance under transformation (dispersion across representations)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import math


# -----------------------------
# Transforms (minimal set)
# -----------------------------

def to_base_digits(n: int, b: int) -> List[int]:
    """Unsigned base expansion digits for n in base b."""
    if b < 2:
        raise ValueError("Base must be >= 2")
    if n == 0:
        return [0]
    n_abs = abs(n)
    digits: List[int] = []
    while n_abs:
        digits.append(n_abs % b)
        n_abs //= b
    return digits[::-1]


def digit_count_descriptor(n: int, b: int) -> List[int]:
    """
    Unitless structural descriptor (raw digit counts) for n in base b.
    Not normalized to a unit; it captures representation shape.
    """
    digits = to_base_digits(n, b)
    counts = [0] * b
    for d in digits:
        counts[d] += 1
    return counts


# -----------------------------
# Pure dispersion (Omega)
# -----------------------------

def l1_distance(a: List[int], b: List[int]) -> int:
    """L1 distance with automatic padding."""
    m = max(len(a), len(b))
    if len(a) < m:
        a = a + [0] * (m - len(a))
    if len(b) < m:
        b = b + [0] * (m - len(b))
    return sum(abs(x - y) for x, y in zip(a, b))


def omega_number(n: int, bases: Iterable[int] = range(2, 33)) -> float:
    """
    Omniabase-Pure Ω(n): dispersion of structural descriptors across bases.
    - No normalization to '1'
    - No semantic assumptions
    Returns a non-negative scalar (0 = perfectly invariant under chosen transforms).
    """
    bases_list = list(bases)
    descs = [digit_count_descriptor(n, b) for b in bases_list]

    # Pairwise dispersion (median of pairwise L1 distances)
    dists: List[int] = []
    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            dists.append(l1_distance(descs[i], descs[j]))

    if not dists:
        return 0.0

    dists.sort()
    mid = len(dists) // 2
    if len(dists) % 2 == 1:
        return float(dists[mid])
    return 0.5 * (dists[mid - 1] + dists[mid])


# -----------------------------
# E0 (zero not assumed; estimated)
# -----------------------------

@dataclass(frozen=True)
class E0Result:
    e0: int
    omega_min: float
    omega_at_n: float
    delta0: int


def estimate_e0(n: int, search_radius: int = 200, bases: Iterable[int] = range(2, 33)) -> E0Result:
    """
    Estimate equilibrium point E0 by minimizing Ω(n - E) over E in [-R, +R].
    This replaces privileged '0' with a measured offset.
    """
    omega_at_n = omega_number(n, bases=bases)
    best_e = 0
    best_val = float("inf")

    for e in range(-search_radius, search_radius + 1):
        val = omega_number(n - e, bases=bases)
        if val < best_val:
            best_val = val
            best_e = e

    return E0Result(
        e0=best_e,
        omega_min=best_val,
        omega_at_n=omega_at_n,
        delta0=n - best_e,
    )


# -----------------------------
# Quick self-check (optional)
# -----------------------------

def _demo():
    for x in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 97, 99, 101]:
        om = omega_number(x)
        e0 = estimate_e0(x, search_radius=50)
        print(f"n={x:>3} Ω={om:>6.1f} | E0={e0.e0:>4} Ωmin={e0.omega_min:>6.1f} Δ0={e0.delta0:>4}")


if __name__ == "__main__":
    _demo()