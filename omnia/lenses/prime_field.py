# omnia/lenses/prime_field.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import math


@dataclass(frozen=True)
class PLDResult:
    k: int
    eps: float
    pld_by_prime: Dict[int, float]
    rho_by_prime: Dict[int, float]
    dk_by_prime: Dict[int, float]


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _mad(xs: List[float], med: float) -> float:
    dev = [abs(x - med) for x in xs]
    return _median(dev)


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def compute_pld(primes: Sequence[int], k: int = 2, eps: float = 1e-12) -> PLDResult:
    """
    Prime Local Density (PLD)
    - Order-free: works on set/sequence without assuming adjacency.
    - Aperspective: uses log-space + robust normalization (median/MAD).
    Returns PLD in [0,1] per prime.
    """
    ps = [int(p) for p in primes if int(p) > 1]
    ps = sorted(set(ps))
    n = len(ps)
    if n < 2:
        # Degenerate: cannot define neighborhood
        pld = {ps[0]: 0.0} if n == 1 else {}
        return PLDResult(k=k, eps=eps, pld_by_prime=pld, rho_by_prime={}, dk_by_prime={})

    xs = [math.log(p) for p in ps]

    dk: Dict[int, float] = {}
    rho: Dict[int, float] = {}

    kk = max(1, min(k, n - 1))

    for i, p in enumerate(ps):
        dists = [abs(xs[i] - xs[j]) for j in range(n) if j != i]
        dists.sort()
        d_k = dists[kk - 1]
        dk[p] = d_k
        rho[p] = 1.0 / (eps + d_k)

    rho_vals = list(rho.values())
    med = _median(rho_vals)
    mad = _mad(rho_vals, med)

    denom = mad + eps

    pld_by_prime: Dict[int, float] = {}
    for p, r in rho.items():
        z = (r - med) / denom  # robust z
        # map z -> [0,1] with simple saturating transform
        # z<=0 -> 0, z>=1 -> 1, linear in between
        pld_by_prime[p] = _clip01(z)

    return PLDResult(k=kk, eps=eps, pld_by_prime=pld_by_prime, rho_by_prime=rho, dk_by_prime=dk)