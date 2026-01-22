from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _safe_log1p(x: int) -> float:
    # x is non-negative int
    return math.log1p(float(x))


@dataclass(frozen=True)
class PrimeLensResult:
    n: int
    primes: Tuple[int, ...]
    residues: Tuple[int, ...]
    Ck_curve: Tuple[float, ...]          # C_3 .. C_k (aligned with k_values)
    dCk_curve: Tuple[float, ...]         # ΔC_k (same length, first is 0.0)
    d2Ck_curve: Tuple[float, ...]        # Δ²C_k (same length, first two are 0.0)
    k_values: Tuple[int, ...]            # 3..k


def compute_Ck(n: int, primes: Sequence[int]) -> float:
    """
    C_k(n) = (1 / N_k) * sum_{i<j<m} log(1 + gcd(d_ij, d_im, d_jm))
    where:
      R[i] = n mod p[i]
      d_ij = gcd(|R[i]-R[j]|, p[i]*p[j])
      N_k = binom(k,3)
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


def prime_lens_curve(n: int, primes: Sequence[int]) -> PrimeLensResult:
    """
    Returns C_k for k=3..K and discrete derivatives:
      ΔC_k = C_k - C_{k-1}  (aligned; first Δ is 0)
      Δ²C_k = ΔC_k - ΔC_{k-1} (aligned; first two are 0)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if len(primes) < 3:
        raise ValueError("Need at least 3 primes")

    # basic prime sanity (minimal)
    for p in primes:
        if p < 2:
            raise ValueError("primes must be >= 2")
        if p % 2 == 0 and p != 2:
            raise ValueError(f"non-prime detected (even): {p}")

    K = len(primes)
    k_values = list(range(3, K + 1))

    # residues for full list (useful diagnostics)
    residues = tuple(n % p for p in primes)

    Ck = []
    for k in k_values:
        Ck.append(compute_Ck(n, primes[:k]))

    # ΔC aligned with Ck list (same length)
    dC = [0.0] * len(Ck)
    for idx in range(1, len(Ck)):
        dC[idx] = Ck[idx] - Ck[idx - 1]

    # Δ²C aligned (same length)
    d2C = [0.0] * len(Ck)
    for idx in range(2, len(Ck)):
        d2C[idx] = dC[idx] - dC[idx - 1]

    return PrimeLensResult(
        n=n,
        primes=tuple(primes),
        residues=residues,
        Ck_curve=tuple(Ck),
        dCk_curve=tuple(dC),
        d2Ck_curve=tuple(d2C),
        k_values=tuple(k_values),
    )


def _first_primes(limit: int) -> List[int]:
    """Deterministic tiny helper for demo only."""
    if limit <= 0:
        return []
    out = []
    x = 2
    while len(out) < limit:
        is_p = True
        r = int(math.isqrt(x))
        for p in out:
            if p > r:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            out.append(x)
        x += 1 if x == 2 else 2
    return out


def demo() -> None:
    primes = _first_primes(19)  # 19 primes → k up to 19 (k_values from 3)
    for n in [29, 30, 31, 32, 33, 34, 35]:
        res = prime_lens_curve(n, primes)
        # show last values as "state"
        C_last = res.Ck_curve[-1]
        dC_last = res.dCk_curve[-1]
        d2C_last = res.d2Ck_curve[-1]
        print(f"n={n:>3}  Ck={C_last:.6f}  dC={dC_last:+.6f}  d2C={d2C_last:+.6f}")


if __name__ == "__main__":
    demo()