from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _safe_log1p(x: int) -> float:
    return math.log1p(float(x))


def _validate_primes(primes: Sequence[int]) -> None:
    if len(primes) < 3:
        raise ValueError("Need at least 3 primes")
    for p in primes:
        if p < 2:
            raise ValueError("primes must be >= 2")
        if p % 2 == 0 and p != 2:
            raise ValueError(f"non-prime detected (even): {p}")


def compute_Ck(n: int, primes: Sequence[int]) -> float:
    """
    C_k(n) = (1/N_k) * sum_{i<j<m} log(1 + gcd(d_ij, d_im, d_jm))

    with:
      R[i] = n mod p[i]
      d_ij = gcd(|R[i]-R[j]|, p[i]*p[j])
      N_k = binom(k,3)

    Interpretation:
      - purely structural, modular residue geometry
      - no semantics, no "prime prediction"
    """
    k = len(primes)
    if k < 3:
        raise ValueError("Need at least 3 primes to compute C_k")

    R = [n % p for p in primes]
    Nk = _binom(k, 3)
    if Nk == 0:
        return 0.0

    acc = 0.0
    for i in range(k - 2):
        pi = primes[i]
        Ri = R[i]
        for j in range(i + 1, k - 1):
            pj = primes[j]
            Rj = R[j]
            dij = math.gcd(abs(Ri - Rj), pi * pj)
            for m in range(j + 1, k):
                pm = primes[m]
                Rm = R[m]
                dim = math.gcd(abs(Ri - Rm), pi * pm)
                djm = math.gcd(abs(Rj - Rm), pj * pm)
                Dijm = math.gcd(dij, math.gcd(dim, djm))
                acc += _safe_log1p(Dijm)

    return acc / float(Nk)


def curve_Ck(n: int, primes: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """
    Returns (k_values, Ck_values) for k = 3..K.
    """
    _validate_primes(primes)
    K = len(primes)
    k_values = tuple(range(3, K + 1))
    Ck = []
    for k in k_values:
        Ck.append(compute_Ck(n, primes[:k]))
    return k_values, tuple(Ck)


def discrete_diffs(vals: Sequence[float]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Δ and Δ² with alignment to vals length.
    Δ[0]=0, Δ²[0]=Δ²[1]=0
    """
    n = len(vals)
    d = [0.0] * n
    for i in range(1, n):
        d[i] = vals[i] - vals[i - 1]
    d2 = [0.0] * n
    for i in range(2, n):
        d2[i] = d[i] - d[i - 1]
    return tuple(d), tuple(d2)


@dataclass(frozen=True)
class PrimeCKLensResult:
    n: int
    primes: Tuple[int, ...]
    residues: Tuple[int, ...]
    k_values: Tuple[int, ...]
    Ck: Tuple[float, ...]
    dCk: Tuple[float, ...]
    d2Ck: Tuple[float, ...]
    details: Dict[str, float]


class PrimeCKLens:
    """
    OMNIA lens: Prime Residue Geometry (CK-lens)

    Measures structural geometry of n across a prime basis P
    by aggregating triple-wise gcd couplings in residue space.

    Output is a deterministic measurement bundle.
    """

    def __init__(self, primes: Sequence[int]) -> None:
        _validate_primes(primes)
        self._primes = tuple(int(p) for p in primes)

    @property
    def primes(self) -> Tuple[int, ...]:
        return self._primes

    def measure(self, n: int) -> PrimeCKLensResult:
        if not isinstance(n, int):
            raise TypeError("n must be int")
        if n < 0:
            raise ValueError("n must be non-negative")

        residues = tuple(n % p for p in self._primes)
        k_values, Ck = curve_Ck(n, self._primes)
        dCk, d2Ck = discrete_diffs(Ck)

        # simple stable scalars for downstream OMNIA aggregation
        C_last = Ck[-1]
        dC_last = dCk[-1]
        d2C_last = d2Ck[-1]

        details = {
            "C_last": float(C_last),
            "dC_last": float(dC_last),
            "d2C_last": float(d2C_last),
            "K": float(len(self._primes)),
        }

        return PrimeCKLensResult(
            n=int(n),
            primes=self._primes,
            residues=residues,
            k_values=k_values,
            Ck=Ck,
            dCk=dCk,
            d2Ck=d2Ck,
            details=details,
        )