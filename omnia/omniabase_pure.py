#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OMNIABASE — PURE (unitless)
MB-X.01 / OMNIA

Goal:
- Remove the implicit "unit = 1" convention from the measurement.
- Work on *relations under transformations* (base changes + scale/offset families),
  not on absolute magnitude.

What this file does:
- For each integer n in a demo range, it computes a unitless instability score Ω(n)
  by scanning multiple numeral bases and using normalization-free, relative features.
- It then searches an equilibrium offset E0 (a calibration offset) that minimizes Ω
  over a bounded window of offsets.
- Reports: n, Ω, E0, Ωmin, Δ0 where Δ0 := n - E0.

Notes:
- This is a "pure lens" artifact: measurement only, no prediction, no learning.
- Deterministic, no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable


# -------------------------
# USER CONTROLS (EDIT HERE)
# -------------------------

# If DEBUG_N is an int, the script prints a detailed debug block for that n.
# Set to None to disable.
DEBUG_N = None  # e.g. 9

# Demo scan range (inclusive)
N_START = 2
N_END = 120

# Bases to scan (2..36 recommended; keep it modest for speed in Colab)
BASE_MIN = 2
BASE_MAX = 16

# Offset search window for E0 (bounded, symmetric)
# We search E0 in [n - OFFSET_WINDOW, n + OFFSET_WINDOW].
OFFSET_WINDOW = 200

# Step size for E0 search (1 = full search; >1 = faster but coarser)
OFFSET_STEP = 1

# If True, prints only a sparse subset (useful if you pipe to grep)
SPARSE_PRINT = False
SPARSE_KEEP = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 97, 99, 101}


# -------------------------
# CORE: base representation
# -------------------------

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def to_base_digits(n: int, base: int) -> List[int]:
    """Return digits of |n| in given base (most-significant first)."""
    if base < 2 or base > 36:
        raise ValueError("base must be in [2, 36]")

    x = abs(n)
    if x == 0:
        return [0]
    out = []
    while x > 0:
        out.append(x % base)
        x //= base
    return list(reversed(out))


def digit_hist(digits: List[int], base: int) -> List[int]:
    h = [0] * base
    for d in digits:
        h[d] += 1
    return h


def shannon_entropy_from_hist(hist: List[int]) -> float:
    total = sum(hist)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in hist:
        if c:
            p = c / total
            ent -= p * math.log(p)
    return ent  # natural log


def normalized_entropy(hist: List[int]) -> float:
    """Entropy normalized to [0,1] regardless of base."""
    base = len(hist)
    if base <= 1:
        return 0.0
    h = shannon_entropy_from_hist(hist)
    hmax = math.log(base)
    return 0.0 if hmax == 0 else (h / hmax)


def runs_complexity(digits: List[int]) -> float:
    """
    A unitless "run" complexity:
    - counts transitions between adjacent digits
    - normalized by max possible transitions
    """
    if len(digits) <= 1:
        return 0.0
    trans = 0
    for i in range(1, len(digits)):
        if digits[i] != digits[i - 1]:
            trans += 1
    return trans / (len(digits) - 1)


def digit_mass_moments(hist: List[int]) -> Tuple[float, float]:
    """
    Compute unitless moments of the digit-mass distribution:
    - mean position (normalized)
    - variance (normalized)
    """
    base = len(hist)
    total = sum(hist)
    if total == 0:
        return 0.0, 0.0
    # positions normalized to [0,1]
    pos = [i / (base - 1) if base > 1 else 0.0 for i in range(base)]
    mu = sum(pos[i] * hist[i] for i in range(base)) / total
    var = sum(((pos[i] - mu) ** 2) * hist[i] for i in range(base)) / total
    return mu, var


# -------------------------
# PURE Ω: instability score
# -------------------------

@dataclass(frozen=True)
class BaseFeatures:
    H: float       # normalized entropy
    R: float       # runs complexity
    MU: float      # normalized mean digit position
    VAR: float     # normalized variance of digit position
    L: float       # length penalty (unitless)


def features_for(n: int, base: int) -> BaseFeatures:
    d = to_base_digits(n, base)
    h = digit_hist(d, base)

    H = normalized_entropy(h)
    R = runs_complexity(d)
    MU, VAR = digit_mass_moments(h)

    # length penalty: compare representation length vs log_base(|n|+1), normalized
    # This avoids "unit=1": it's relative to base and to n itself.
    if abs(n) <= 1:
        L = 0.0
    else:
        expected = math.log(abs(n) + 1) / math.log(base)
        L = abs(len(d) - expected) / max(1.0, expected)

    return BaseFeatures(H=H, R=R, MU=MU, VAR=VAR, L=L)


def omega_pure(n: int, base_min: int = BASE_MIN, base_max: int = BASE_MAX) -> Tuple[float, Dict[int, BaseFeatures]]:
    """
    Ω(n): cross-base instability of features.
    Compute features for each base and measure dispersion.

    We avoid absolute scale by:
    - using only normalized features in [0,1] or unitless ratios.
    - aggregating via variance/mean-dispersion across bases.
    """
    feats: Dict[int, BaseFeatures] = {}
    for b in range(base_min, base_max + 1):
        feats[b] = features_for(n, b)

    # Collect per-feature lists across bases
    Hs = [feats[b].H for b in feats]
    Rs = [feats[b].R for b in feats]
    MUs = [feats[b].MU for b in feats]
    VARs = [feats[b].VAR for b in feats]
    Ls = [min(1.0, feats[b].L) for b in feats]  # cap

    def var(x: List[float]) -> float:
        m = sum(x) / len(x)
        return sum((t - m) ** 2 for t in x) / len(x)

    # Weighted instability (tunable). Keep simple and deterministic.
    Ω = (
        1.00 * var(Hs) +
        0.80 * var(Rs) +
        0.60 * var(MUs) +
        0.60 * var(VARs) +
        0.40 * var(Ls)
    )

    # Scale Ω to a readable range (still unitless).
    # This is purely for presentation; ranking is what matters.
    Ω_scaled = 100.0 * Ω
    return Ω_scaled, feats


# -------------------------
# EQUILIBRIUM OFFSET E0
# -------------------------

def find_E0(n: int) -> Tuple[int, float]:
    """
    Search E0 in a bounded window around n such that Ω(n - E0) is minimized.
    Here we define the "observed" value as x = n - E0 and pick E0 that makes x
    maximally *structurally stable* under Omniabase transformations.

    Δ0 := n - E0 is the inferred offset drift (the residual after calibration).
    """
    best_E0 = n
    best_Ω = float("inf")

    lo = n - OFFSET_WINDOW
    hi = n + OFFSET_WINDOW

    for E0 in range(lo, hi + 1, OFFSET_STEP):
        x = n - E0  # residual
        Ω, _ = omega_pure(x)
        if Ω < best_Ω:
            best_Ω = Ω
            best_E0 = E0

    return best_E0, best_Ω


# -------------------------
# DEBUG PRINT
# -------------------------

def debug_block(n: int) -> None:
    print("\n[DEBUG] OMNIABASE_PURE")
    print(f"n={n}  bases=[{BASE_MIN}..{BASE_MAX}]  OFFSET_WINDOW={OFFSET_WINDOW}  STEP={OFFSET_STEP}")

    Ωn, feats = omega_pure(n)
    print(f"Ω(n)={Ωn:.6f}")

    # show per-base features
    print("\nbase | H(ent) | R(runs) | MU | VAR | L(len)")
    for b in range(BASE_MIN, BASE_MAX + 1):
        f = feats[b]
        print(f"{b:>4} | {f.H:>6.3f} | {f.R:>6.3f} | {f.MU:>5.3f} | {f.VAR:>5.3f} | {min(1.0,f.L):>5.3f}")

    E0, Ωmin = find_E0(n)
    Δ0 = n - E0
    print("\n[E0 search]")
    print(f"E0={E0}  Δ0=n-E0={Δ0}  Ωmin={Ωmin:.6f}\n")


# -------------------------
# MAIN
# -------------------------

def main() -> None:
    if DEBUG_N is not None:
        debug_block(int(DEBUG_N))

    for n in range(N_START, N_END + 1):
        if SPARSE_PRINT and (n not in SPARSE_KEEP):
            continue

        Ω, _ = omega_pure(n)
        E0, Ωmin = find_E0(n)
        Δ0 = n - E0

        # Compact, grep-friendly line
        print(f"n={n:>4}  Ω={Ω:>8.3f} | E0={E0:>6}  Ωmin={Ωmin:>8.3f}  Δ0={Δ0:>6}")


if __name__ == "__main__":
    main()