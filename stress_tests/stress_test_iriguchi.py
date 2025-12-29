#!/usr/bin/env python3
"""
OMNIA — Stress Test (Iriguchi)
Minimal, reproducible numeric probe.

Goal:
- Sweep epsilon on a purely numeric object.
- Compare OMNIA-style Δ-coherence / TruthΩ proxies under base transforms.
- Control for two known confounders:
  (1) power-related bases (4, 8, 16, ...)
  (2) digit sums encode n mod (b-1): use mod-normalized digit sum.

This file is intentionally small: one script, one CSV, optional print.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


# -----------------------------
#  Configuration (edit safely)
# -----------------------------

EPS_GRID = [1e-8, 1e-7, 1e-6, 1e-5]  # Iriguchi suggested log grid around ~1e-6

# Base sets:
BASES_ALL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
BASES_NO_POWERS = [2, 3, 5, 6, 7, 9, 10, 12]  # excludes 4, 8, 16

# Numeric object domain (purely numeric, no semantics):
N_MIN = 2
N_MAX = 20000

# Sample sizes (kept modest for phone/colab speed)
SAMPLE_PRIMES = 1200
SAMPLE_CONTROLS = 1200

RNG_SEED = 1337

OUT_CSV = "stress_test_iriguchi_results.csv"


# -----------------------------
#  Helpers: base digits & digit sums
# -----------------------------

DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def to_base_digits(n: int, b: int) -> List[int]:
    """Return digits of n in base b as list of ints (most significant first)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if b < 2:
        raise ValueError("base must be >= 2")
    if n == 0:
        return [0]
    out = []
    while n > 0:
        out.append(n % b)
        n //= b
    return list(reversed(out))


def digit_sum(n: int, b: int) -> int:
    return sum(to_base_digits(n, b))


def digit_sum_mod_norm(n: int, b: int) -> float:
    """
    Control for n mod (b-1):
    digit_sum(n,b) ≡ n (mod b-1)
    Use normalized residue in [0,1).
    """
    m = b - 1
    if m <= 1:
        return 0.0
    return (digit_sum(n, b) % m) / m


def digit_entropy(n: int, b: int) -> float:
    """
    Simple structural feature: Shannon entropy of digit distribution in base b.
    This is NOT a claim of optimality; it's a stable, base-dependent observable.
    """
    ds = to_base_digits(n, b)
    total = len(ds)
    counts: Dict[int, int] = {}
    for d in ds:
        counts[d] = counts.get(d, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p)
    return h


# -----------------------------
#  Prime and control sampling
# -----------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


def primes_in_range(n_min: int, n_max: int) -> List[int]:
    return [n for n in range(n_min, n_max + 1) if is_prime(n)]


def control_set_same_local_exclusions(primes: List[int], n_min: int, n_max: int) -> List[int]:
    """
    Control set: numbers that share the same local exclusions as primes:
    - exclude multiples of small primes (2,3,5) to match the obvious sieve structure.
    This is a cheap proxy for "same local exclusions" without claiming equivalence.
    """
    out = []
    for n in range(n_min, n_max + 1):
        if n < 2:
            continue
        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
            continue
        if not is_prime(n):
            out.append(n)
    return out


# -----------------------------
#  OMNIA-style scalar: Δ-coherence proxy + TruthΩ proxy
# -----------------------------

@dataclass
class CoherenceResult:
    eps: float
    base_mode: str
    group: str
    delta: float
    truth_omega: float


def feature_vector(n: int, bases: List[int], use_mod_norm: bool) -> List[float]:
    """
    Extract a small base-wise feature vector for number n.
    - digit entropy (structural, length-aware)
    - (optional) mod-normalized digit sum residue
    """
    v = []
    for b in bases:
        v.append(digit_entropy(n, b))
        if use_mod_norm:
            v.append(digit_sum_mod_norm(n, b))
        else:
            # raw digit sum normalized by (b-1)*len to keep scale bounded-ish
            ds = digit_sum(n, b)
            L = len(to_base_digits(n, b))
            denom = max(1, (b - 1) * L)
            v.append(ds / denom)
    return v


def add_noise(vec: List[float], eps: float, rng: random.Random) -> List[float]:
    """
    Small anonymous perturbation parameter ε.
    Gaussian noise with std=eps applied per component.
    """
    return [x + rng.gauss(0.0, eps) for x in vec]


def mean_vector(vectors: List[List[float]]) -> List[float]:
    k = len(vectors[0])
    m = [0.0] * k
    for vec in vectors:
        for i in range(k):
            m[i] += vec[i]
    n = len(vectors)
    return [x / n for x in m]


def l2(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def delta_coherence(sample: List[int], bases: List[int], eps: float, use_mod_norm: bool, rng: random.Random) -> float:
    """
    Δ-coherence proxy:
    - compute mean feature vector for clean sample
    - compute mean feature vector for ε-perturbed sample
    - Δ = ||mu_clean - mu_noisy||_2

    If structure is stable under perturbation, Δ stays small.
    """
    feats_clean = [feature_vector(n, bases, use_mod_norm) for n in sample]
    feats_noisy = [add_noise(f, eps, rng) for f in feats_clean]
    mu_clean = mean_vector(feats_clean)
    mu_noisy = mean_vector(feats_noisy)
    return l2(mu_clean, mu_noisy)


def truth_omega_from_delta(delta: float) -> float:
    """
    Minimal TruthΩ proxy:
    TruthΩ = -log(1 + Δ)
    This is monotone decreasing in Δ and bounded.
    """
    return -math.log(1.0 + max(0.0, delta))


# -----------------------------
#  Main run
# -----------------------------

def sample_list(xs: List[int], k: int, rng: random.Random) -> List[int]:
    if len(xs) <= k:
        return xs[:]
    return rng.sample(xs, k)


def run() -> None:
    rng = random.Random(RNG_SEED)

    primes = primes_in_range(N_MIN, N_MAX)
    controls_pool = control_set_same_local_exclusions(primes, N_MIN, N_MAX)

    primes_s = sample_list(primes, SAMPLE_PRIMES, rng)
    controls_s = sample_list(controls_pool, SAMPLE_CONTROLS, rng)

    configs: List[Tuple[str, List[int]]] = [
        ("all_bases", BASES_ALL),
        ("no_power_bases", BASES_NO_POWERS),
    ]

    # Two modes: raw digit-sum scaled vs mod(b-1) normalized control
    mod_modes = [
        ("raw_scaled_digitsum", False),
        ("modnorm_digitsum", True),
    ]

    rows: List[CoherenceResult] = []

    for eps in EPS_GRID:
        for base_mode, bases in configs:
            for mod_mode, use_mod_norm in mod_modes:
                # IMPORTANT: use a fresh RNG stream per condition for determinism
                local_rng = random.Random(RNG_SEED + int(abs(math.log10(eps)) * 1000) + hash(base_mode) % 997)

                d_p = delta_coherence(primes_s, bases, eps, use_mod_norm, local_rng)
                t_p = truth_omega_from_delta(d_p)
                rows.append(CoherenceResult(eps, f"{base_mode}|{mod_mode}", "primes", d_p, t_p))

                # separate RNG stream for controls (still deterministic)
                local_rng2 = random.Random(RNG_SEED + 9999 + int(abs(math.log10(eps)) * 1000) + hash(base_mode) % 997)
                d_c = delta_coherence(controls_s, bases, eps, use_mod_norm, local_rng2)
                t_c = truth_omega_from_delta(d_c)
                rows.append(CoherenceResult(eps, f"{base_mode}|{mod_mode}", "controls", d_c, t_c))

    # Save CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eps", "base_mode", "group", "delta", "truth_omega"])
        for r in rows:
            w.writerow([f"{r.eps:.0e}", r.base_mode, r.group, f"{r.delta:.12g}", f"{r.truth_omega:.12g}"])

    # Console summary (compact)
    print(f"[OK] wrote: {OUT_CSV}")
    # Show a quick ranking: larger separation between primes vs controls
    # for each eps/base_mode we compute |Δp - Δc|
    sep: Dict[Tuple[str, str], float] = {}
    tmp: Dict[Tuple[str, str, str], float] = {}
    for r in rows:
        key = (f"{r.eps:.0e}", r.base_mode, r.group)
        tmp[key] = r.delta
    for eps in [f"{e:.0e}" for e in EPS_GRID]:
        for base_mode, _ in configs:
            for mod_mode, _ in mod_modes:
                bm = f"{base_mode}|{mod_mode}"
                dp = tmp[(eps, bm, "primes")]
                dc = tmp[(eps, bm, "controls")]
                sep[(eps, bm)] = abs(dp - dc)

    print("Top separations |Δ(primes)-Δ(controls)|:")
    for (eps, bm), s in sorted(sep.items(), key=lambda x: x[1], reverse=True)[:6]:
        print(f"  eps={eps}  mode={bm:28s}  sep={s:.6g}")


if __name__ == "__main__":
    run()