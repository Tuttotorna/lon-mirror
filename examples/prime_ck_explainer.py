from __future__ import annotations

import math
from typing import Dict, List, Any

from omnia.lenses.prime_ck import PrimeCKLens


def _first_primes(limit: int) -> List[int]:
    if limit <= 0:
        return []
    out: List[int] = []
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


def main(n: int = 0, k: int = 19) -> Dict[str, Any]:
    primes = _first_primes(k)
    lens = PrimeCKLens(primes=primes)
    r = lens.measure(n)

    return {
        "n": r.n,
        "basis": {"type": "prime", "K": len(r.primes), "primes": list(r.primes)},
        "residues": list(r.residues),
        "Ck": {
            "k_values": list(r.k_values),
            "Ck": list(r.Ck),
            "dCk": list(r.dCk),
            "d2Ck": list(r.d2Ck),
        },
        "scalars": r.details,
    }


if __name__ == "__main__":
    # quick deterministic demo (no plotting)
    for n in [29, 30, 31, 32, 33, 34, 35]:
        report = main(n=n, k=19)
        s = report["scalars"]
        print(f"n={n:>3}  C_last={s['C_last']:.6f}  dC_last={s['dC_last']:+.6f}  d2C_last={s['d2C_last']:+.6f}")