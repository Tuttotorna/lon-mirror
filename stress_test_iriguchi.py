#!/usr/bin/env python3
"""
OMNIA × Iriguchi — Stress Test (single-file, reproducible)

This script does TWO things, minimally:

A) ε-sweep stress test (primes vs controls) under agreed constraints
   - bases exclude power-bases (no 4,8,16,...)
   - control for digit-sum mod(b−1) artifacts via residue-normalized channel
   - outputs: stress_test_iriguchi_results.csv

B) Omniabase "regularity space" table (Structural Response Matrix, SRM)
   - builds a per-n table of canonical coordinates where structure can look regular:
     * mu_entropy, sigma_entropy, aniso_entropy  (across bases)
     * delta0 (ε reference)
     * slope_S = d(delta)/d(log ε) over the ε grid
   - outputs: srm_omnia_regular_space.csv

No narrative. No claims. Pure numeric artifact.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ============================================================
# 1) Frozen configuration (do not mutate in-flight)
# ============================================================

# Iriguchi-style probing grid (log-spaced around ~1e-6)
EPS_GRID: List[float] = [1e-8, 1e-7, 1e-6, 1e-5]

# Bases: EXCLUDE power bases. Frozen.
BASES: List[int] = [3, 5, 6, 7, 9, 10, 11, 12]

# Domain
N_MIN = 2
N_MAX = 20000

# Sampling sizes (kept modest, deterministic)
SAMPLE_PRIMES = 1200
SAMPLE_CONTROLS = 1200

# SRM table size (how many n rows to write)
SRM_ROWS = 3000  # increase if you want, but keep phone-friendly

# RNG seed (global determinism)
RNG_SEED = 1337

# Outputs
OUT_STRESS_CSV = "stress_test_iriguchi_results.csv"
OUT_SRM_CSV = "srm_omnia_regular_space.csv"


# ============================================================
# 2) Base utilities
# ============================================================

def to_base_digits(n: int, b: int) -> List[int]:
    """Return digits of n in base b as list of ints (MSD first)."""
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
    Control for n mod (b-1) artifact:
    digit_sum(n,b) ≡ n (mod b-1)
    Use residue normalized to [0,1).
    """
    m = b - 1
    if m <= 1:
        return 0.0
    return (digit_sum(n, b) % m) / m


# ============================================================
# 3) Canonical Omniabase coordinate: entropy + anisotropy
# ============================================================

def digit_entropy(n: int, b: int) -> float:
    """Shannon entropy of digit distribution for n in base b."""
    ds = to_base_digits(n, b)
    total = len(ds)
    counts: Dict[int, int] = {}
    for d in ds:
        counts[d] = counts.get(d, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p)  # natural log
    return h


def entropy_stats(n: int, bases: List[int]) -> Tuple[float, float, float]:
    """
    Returns (mu_entropy, sigma_entropy, aniso_entropy)
    where aniso_entropy = sigma / (abs(mu)+tau)
    """
    tau = 1e-12
    vals = [digit_entropy(n, b) for b in bases]
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / len(vals)
    sigma = math.sqrt(var)
    aniso = sigma / (abs(mu) + tau)
    return mu, sigma, aniso


# ============================================================
# 4) Prime + control construction
# ============================================================

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


def control_set_same_local_exclusions(n_min: int, n_max: int) -> List[int]:
    """
    Cheap "prime-like" control pool:
    - exclude multiples of small primes (2,3,5)
    - keep composites only
    This matches the obvious sieve exclusions without claiming equivalence.
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


def sample_list(xs: List[int], k: int, rng: random.Random) -> List[int]:
    if len(xs) <= k:
        return xs[:]
    return rng.sample(xs, k)


# ============================================================
# 5) Feature vector + Δ-coherence proxy + TruthΩ proxy
# ============================================================

def feature_vector(n: int, bases: List[int]) -> List[float]:
    """
    Minimal per-base feature vector for n.
    Two channels per base:
      (1) digit entropy
      (2) mod(b-1) residue normalized digit sum (artifact-controlled)
    """
    v: List[float] = []
    for b in bases:
        v.append(digit_entropy(n, b))
        v.append(digit_sum_mod_norm(n, b))
    return v


def add_noise(vec: List[float], eps: float, rng: random.Random) -> List[float]:
    """Anonymous perturbation parameter ε: Gaussian noise with std=eps per component."""
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


def delta_coherence(sample: List[int], bases: List[int], eps: float, rng: random.Random) -> float:
    """
    Δ-coherence proxy:
      Δ = || mu_clean - mu_noisy ||_2
    where mu_* are mean feature vectors over the sample.
    """
    feats_clean = [feature_vector(n, bases) for n in sample]
    feats_noisy = [add_noise(f, eps, rng) for f in feats_clean]
    mu_clean = mean_vector(feats_clean)
    mu_noisy = mean_vector(feats_noisy)
    return l2(mu_clean, mu_noisy)


def truth_omega_from_delta(delta: float) -> float:
    """Minimal monotone TruthΩ proxy (bounded, deterministic): TruthΩ = -log(1 + Δ)."""
    return -math.log(1.0 + max(0.0, delta))


def slope_vs_logeps(deltas: List[float], eps_grid: List[float]) -> float:
    """
    Compute slope S = d(Δ)/d(log ε) using least squares over the grid.
    If the response becomes regular in this space, S tends to stabilize by class.
    """
    xs = [math.log(e) for e in eps_grid]
    ys = deltas
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = sum((x - xbar) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


# ============================================================
# 6) Outputs
# ============================================================

@dataclass
class StressRow:
    eps: float
    group: str
    delta: float
    truth_omega: float


@dataclass
class SRMRow:
    n: int
    is_prime: int
    mu_entropy: float
    sigma_entropy: float
    aniso_entropy: float
    delta0: float
    slope_S: float


# ============================================================
# 7) Main run
# ============================================================

def run_stress_test(primes_s: List[int], controls_s: List[int]) -> List[StressRow]:
    """
    A) ε-sweep stress test over two groups
    """
    rows: List[StressRow] = []
    for eps in EPS_GRID:
        # deterministic streams (separate for primes/controls)
        rng_p = random.Random(RNG_SEED + 101 + int(abs(math.log10(eps)) * 10_000))
        rng_c = random.Random(RNG_SEED + 202 + int(abs(math.log10(eps)) * 10_000))

        d_p = delta_coherence(primes_s, BASES, eps, rng_p)
        d_c = delta_coherence(controls_s, BASES, eps, rng_c)

        rows.append(StressRow(eps, "primes", d_p, truth_omega_from_delta(d_p)))
        rows.append(StressRow(eps, "controls", d_c, truth_omega_from_delta(d_c)))
    return rows


def write_stress_csv(rows: List[StressRow]) -> None:
    with open(OUT_STRESS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eps", "group", "delta", "truth_omega"])
        for r in rows:
            w.writerow([f"{r.eps:.0e}", r.group, f"{r.delta:.12g}", f"{r.truth_omega:.12g}"])
    print(f"[OK] wrote: {OUT_STRESS_CSV}")


def build_srm_table(n_list: List[int]) -> List[SRMRow]:
    """
    B) SRM: per-n canonical coordinates to make structure 'regular' in Omniabase space.
    Columns:
      n, is_prime, mu_entropy, sigma_entropy, aniso_entropy, delta0, slope_S
    where delta0 is Δ at eps_ref (middle of grid), slope_S fitted on ε grid.
    """
    eps_ref = EPS_GRID[len(EPS_GRID) // 2]  # reference epsilon (1e-6 here)
    out: List[SRMRow] = []

    for n in n_list:
        mu_e, sig_e, aniso = entropy_stats(n, BASES)

        # Compute per-n deltas by treating "sample" as singleton [n]
        # This is not a claim of optimality; it defines a stable coordinate system.
        deltas = []
        for eps in EPS_GRID:
            rng = random.Random(RNG_SEED + n * 17 + int(abs(math.log10(eps)) * 10_000))
            d = delta_coherence([n], BASES, eps, rng)
            deltas.append(d)

        delta0 = deltas[EPS_GRID.index(eps_ref)]
        S = slope_vs_logeps(deltas, EPS_GRID)

        out.append(
            SRMRow(
                n=n,
                is_prime=1 if is_prime(n) else 0,
                mu_entropy=mu_e,
                sigma_entropy=sig_e,
                aniso_entropy=aniso,
                delta0=delta0,
                slope_S=S,
            )
        )
    return out


def write_srm_csv(rows: List[SRMRow]) -> None:
    with open(OUT_SRM_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "is_prime", "mu_entropy", "sigma_entropy", "aniso_entropy", "delta0", "slope_S"])
        for r in rows:
            w.writerow([
                r.n,
                r.is_prime,
                f"{r.mu_entropy:.12g}",
                f"{r.sigma_entropy:.12g}",
                f"{r.aniso_entropy:.12g}",
                f"{r.delta0:.12g}",
                f"{r.slope_S:.12g}",
            ])
    print(f"[OK] wrote: {OUT_SRM_CSV}")


def main() -> None:
    rng = random.Random(RNG_SEED)

    primes = primes_in_range(N_MIN, N_MAX)
    controls_pool = control_set_same_local_exclusions(N_MIN, N_MAX)

    if len(primes) == 0 or len(controls_pool) == 0:
        raise RuntimeError("Empty primes or controls pool; adjust N_MIN/N_MAX.")

    primes_s = sample_list(primes, SAMPLE_PRIMES, rng)
    controls_s = sample_list(controls_pool, SAMPLE_CONTROLS, rng)

    # A) stress test (group-level)
    stress_rows = run_stress_test(primes_s, controls_s)
    write_stress_csv(stress_rows)

    # B) SRM table (per-n space)
    # Pick a deterministic mix of numbers (not only primes) to reveal manifold/bands.
    # We bias toward the same local exclusions to reduce trivial sieve noise.
    candidates = list(range(N_MIN, N_MAX + 1))
    rng.shuffle(candidates)
    n_list = candidates[:SRM_ROWS]

    srm_rows = build_srm_table(n_list)
    write_srm_csv(srm_rows)

    # Minimal console sanity
    # Show mean Aniso for primes vs non-primes in SRM sample
    p_an = [r.aniso_entropy for r in srm_rows if r.is_prime == 1]
    c_an = [r.aniso_entropy for r in srm_rows if r.is_prime == 0]
    if p_an and c_an:
        print(f"[SRM] mean aniso (prime)   = {sum(p_an)/len(p_an):.6g}   n={len(p_an)}")
        print(f"[SRM] mean aniso (nonprime)= {sum(c_an)/len(c_an):.6g}   n={len(c_an)}")


if __name__ == "__main__":
    main()