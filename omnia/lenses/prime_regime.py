from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math


EPS = 1e-12


def _shannon_entropy(probs: List[float]) -> float:
    s = 0.0
    for p in probs:
        if p > 0.0:
            s -= p * math.log(p + EPS, 2)
    return s


def _normalize(xs: List[float]) -> List[float]:
    s = sum(xs)
    if s <= 0.0:
        return [0.0 for _ in xs]
    return [x / s for x in xs]


def _tau_update(prev_tau: int, T: float, theta: float) -> int:
    return prev_tau + (1 if T > theta else 0)


def _phi(prime: int, mods: List[int]) -> List[float]:
    # normalized residues vector
    out = []
    for m in mods:
        out.append((prime % m) / float(m))
    return out


def _l1(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


@dataclass(frozen=True)
class PrimeState:
    idx: int
    p: int
    g_prev: int
    phi: List[float]
    S: float
    T: float
    tau: int


def prime_state_from_primes(
    primes: List[int],
    idx: int,
    mods: List[int],
    window: int = 512,
    drift_theta: float = 0.05,
    prev_phi: Optional[List[float]] = None,
    prev_tau: int = 0,
) -> PrimeState:
    """
    Deterministic prime regime state.
    - Phi: modular residue vector of p_idx
    - S: gap-distribution stability (entropy-based, normalized)
    - T: drift magnitude between prev_phi and current phi (L1 / |phi|)
    - tau: structural time incremented only when drift exceeds theta
    """
    if idx <= 0 or idx >= len(primes):
        raise ValueError("idx out of range")

    p = primes[idx]
    g_prev = primes[idx] - primes[idx - 1]

    phi = _phi(p, mods)

    # Drift T
    if prev_phi is None:
        T = 0.0
    else:
        if len(prev_phi) != len(phi):
            raise ValueError("prev_phi mismatch")
        T = _l1(prev_phi, phi) / (len(phi) + EPS)

    tau = _tau_update(prev_tau, T, drift_theta)

    # Stability S from recent gaps histogram entropy
    start = max(1, idx - window)
    gaps = [primes[k] - primes[k - 1] for k in range(start, idx + 1)]
    # simple histogram on observed gaps
    counts = {}
    for g in gaps:
        counts[g] = counts.get(g, 0) + 1
    probs = _normalize(list(counts.values()))
    H = _shannon_entropy(probs)
    # normalize entropy by max possible log2(K)
    K = max(1, len(probs))
    Hmax = math.log(K + EPS, 2)
    Hn = (H / (Hmax + EPS)) if Hmax > 0 else 0.0
    # Stability is inverse of normalized entropy
    S = max(0.0, min(1.0, 1.0 - Hn))

    return PrimeState(idx=idx, p=p, g_prev=g_prev, phi=phi, S=S, T=T, tau=tau)