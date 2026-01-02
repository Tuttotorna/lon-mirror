# omnia/omniabase_pure.py
# OMNIABASE-PURE (standalone)
# Deterministic, ASCII-only, Colab-safe.
#
# Goal:
# - Define a unit-free structural field omega(n) from multi-base representations.
# - Provide omega_pure_v2 with local multi-scale normalization to sharpen minima.
#
# Notes:
# - ASCII-only to avoid Colab invalid character issues.
# - Standalone: does not import other omnia modules.

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import numpy as np

try:
    import sympy as sp
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# -----------------------------
# Primality (deterministic)
# -----------------------------

def is_prime(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if _HAS_SYMPY:
        return bool(sp.isprime(n))
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


# -----------------------------
# Base representation + features
# -----------------------------

def digits_in_base(n: int, base: int) -> List[int]:
    base = int(base)
    if base < 2:
        raise ValueError("base must be >= 2")
    n = int(n)
    if n == 0:
        return [0]
    if n < 0:
        raise ValueError("n must be >= 0")

    out: List[int] = []
    x = n
    while x > 0:
        out.append(x % base)
        x //= base
    out.reverse()
    return out


def shannon_entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    p = counts / total
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def digit_hist_entropy(digs: List[int], base: int) -> float:
    if not digs:
        return 0.0
    base = int(base)
    counts = np.zeros(base, dtype=float)
    for d in digs:
        counts[int(d)] += 1.0
    return shannon_entropy(counts)


def run_length_variability(digs: List[int]) -> float:
    if not digs:
        return 0.0
    runs: List[int] = []
    cur = digs[0]
    ln = 1
    for d in digs[1:]:
        if d == cur:
            ln += 1
        else:
            runs.append(ln)
            cur = d
            ln = 1
    runs.append(ln)
    if len(runs) <= 1:
        return 0.0
    return float(np.std(np.array(runs, dtype=float), ddof=0))


def digit_transition_entropy(digs: List[int], base: int) -> float:
    if len(digs) < 2:
        return 0.0
    base = int(base)
    mat = np.zeros((base, base), dtype=float)
    for a, b in zip(digs[:-1], digs[1:]):
        mat[int(a), int(b)] += 1.0
    return shannon_entropy(mat.ravel())


def base_features(n: int, base: int) -> Tuple[float, float, float]:
    digs = digits_in_base(n, base)
    h = digit_hist_entropy(digs, base)
    rlv = run_length_variability(digs)
    th = digit_transition_entropy(digs, base)
    return h, rlv, th


# -----------------------------
# Omega raw (v1)
# -----------------------------

def omega_raw(n: int, bases: Iterable[int]) -> float:
    """
    Raw omega: dispersion of multi-base structural features.
    Higher -> more cross-base instability.
    """
    n = int(n)
    bases_list = [int(b) for b in bases]

    feats: List[List[float]] = []
    for b in bases_list:
        h, rlv, th = base_features(n, b)
        feats.append([h, rlv, th])

    X = np.array(feats, dtype=float)  # shape (B, 3)
    if X.shape[0] <= 1:
        return 0.0

    # Robust channel normalization via MAD (avoid divide-by-zero)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad = np.where(mad <= 1e-12, 1.0, mad)
    Z = (X - med) / mad

    # Dispersion: sum of variances across channels (ddof=0 deterministic)
    v = np.var(Z, axis=0, ddof=0)
    return float(np.sum(v))


# -----------------------------
# Omega v2: local multi-scale normalization
# -----------------------------

def _local_stats(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    mu = float(arr.mean())
    sd = float(arr.std(ddof=0))
    return mu, sd


def omega_pure_v2(n: int, bases: Iterable[int], scales: Iterable[int] = (1, 2, 4, 8)) -> float:
    """
    Omega v2:
    - compute raw omega(n)
    - compute local z-scores across multiple neighborhood radii
    - aggregate absolute z to sharpen minima and suppress drift
    """
    n = int(n)
    bases_list = [int(b) for b in bases]
    scales_list = [int(s) for s in scales]

    o0 = omega_raw(n, bases_list)

    z_list: List[float] = []
    for s in scales_list:
        lo = max(2, n - s)
        hi = max(lo, n + s)
        neigh = [omega_raw(k, bases_list) for k in range(lo, hi + 1)]
        mu, sd = _local_stats(neigh)
        if sd <= 1e-12:
            z = 0.0
        else:
            z = (o0 - mu) / sd
        z_list.append(abs(float(z)))

    if not z_list:
        return 0.0

    # weights: smaller scales slightly more important, sum to 1
    weights = np.array([1.0 / (1.0 + i) for i in range(len(z_list))], dtype=float)
    weights /= float(weights.sum())
    return float(np.dot(weights, np.array(z_list, dtype=float)))


# Backward-compatible alias (what you import in Colab)
def omega_pure(n: int, bases: Iterable[int], scales: Iterable[int] = (1, 2, 4, 8)) -> float:
    return omega_pure_v2(n, bases=bases, scales=scales)


# -----------------------------
# Utilities: sequences, minima, null tests
# -----------------------------

@dataclass
class MinimaResult:
    window: int
    local_min_primes: List[int]
    local_min_composites: List[int]


def compute_series(n_max: int, mode: str, bmin: int, bmax: int, scales: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    ns = np.arange(2, int(n_max) + 1, dtype=int)
    bases = list(range(int(bmin), int(bmax) + 1))

    if mode == "raw":
        omegas = np.array([omega_raw(int(n), bases) for n in ns], dtype=float)
    elif mode == "v2":
        omegas = np.array([omega_pure_v2(int(n), bases, scales=scales) for n in ns], dtype=float)
    else:
        raise ValueError("mode must be 'raw' or 'v2'")

    return ns, omegas


def find_local_minima(ns: np.ndarray, omegas: np.ndarray, window: int) -> MinimaResult:
    window = int(window)
    local_min_primes: List[int] = []
    local_min_composites: List[int] = []

    for i in range(window, len(ns) - window):
        o = float(omegas[i])
        neigh = omegas[i - window : i + window + 1]
        if o == float(np.min(neigh)):
            n = int(ns[i])
            if is_prime(n):
                local_min_primes.append(n)
            else:
                local_min_composites.append(n)

    return MinimaResult(window=window, local_min_primes=local_min_primes, local_min_composites=local_min_composites)


def block_shuffle_null(ns: np.ndarray, omegas: np.ndarray, block: int, seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    block = max(1, int(block))

    n = len(ns)
    blocks = [(i, min(i + block, n)) for i in range(0, n, block)]
    order = np.arange(len(blocks), dtype=int)
    rng.shuffle(order)

    ns_out = []
    om_out = []
    for bi in order:
        lo, hi = blocks[bi]
        ns_out.append(ns[lo:hi])
        om_out.append(omegas[lo:hi])

    return np.concatenate(ns_out), np.concatenate(om_out)


def delta_pc(min_primes: List[int], min_composites: List[int]) -> int:
    return int(len(min_primes) - len(min_composites))


def local_null_zscore(ns: np.ndarray, omegas: np.ndarray, window: int, block: int, trials: int, seed: int = 123) -> Dict[str, float]:
    window = int(window)
    block = int(block)
    trials = int(trials)

    real = find_local_minima(ns, omegas, window)
    delta_real = delta_pc(real.local_min_primes, real.local_min_composites)

    deltas: List[int] = []
    for t in range(trials):
        ns_s, om_s = block_shuffle_null(ns, omegas, block=block, seed=seed + t)
        res = find_local_minima(ns_s, om_s, window)
        deltas.append(delta_pc(res.local_min_primes, res.local_min_composites))

    arr = np.array(deltas, dtype=float)
    mu = float(arr.mean()) if arr.size else 0.0
    sd = float(arr.std(ddof=0)) if arr.size else 0.0
    if sd <= 1e-12:
        z = 0.0
    else:
        z = (float(delta_real) - mu) / sd

    return {
        "window": float(window),
        "block": float(block),
        "trials": float(trials),
        "delta_real": float(delta_real),
        "p_real": float(len(real.local_min_primes)),
        "c_real": float(len(real.local_min_composites)),
        "n_minima": float(len(real.local_min_primes) + len(real.local_min_composites)),
        "null_mean": mu,
        "null_std": sd,
        "null_min": float(arr.min()) if arr.size else 0.0,
        "null_max": float(arr.max()) if arr.size else 0.0,
        "z_score": float(z),
    }


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Omniabase-Pure omega field (raw + v2 normalized).")
    p.add_argument("--nmax", type=int, default=120)
    p.add_argument("--bmin", type=int, default=2)
    p.add_argument("--bmax", type=int, default=16)
    p.add_argument("--mode", type=str, default="raw", choices=["raw", "v2"])
    p.add_argument("--scales", type=str, default="1,2,4,8", help="comma-separated scales for v2")
    p.add_argument("--print_series", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--primes_overlay", action="store_true")

    p.add_argument("--min_window", type=int, default=3)
    p.add_argument("--min_windows", type=str, default="", help="comma-separated windows, overrides --min_window")
    p.add_argument("--block", type=int, default=12)
    p.add_argument("--trials", type=int, default=400)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--null_test", action="store_true")

    args = p.parse_args()

    scales = tuple(int(x.strip()) for x in args.scales.split(",") if x.strip())
    ns, omegas = compute_series(args.nmax, args.mode, args.bmin, args.bmax, scales)

    if args.print_series:
        for n, o in zip(ns, omegas):
            print(f"n= {int(n):3d} omega= {float(o):8.6f}")

    if args.plot:
        if not _HAS_MPL:
            raise RuntimeError("matplotlib not available in this runtime.")
        plt.figure()
        plt.plot(ns, omegas, ".")
        plt.xlabel("n")
        plt.ylabel("omega")
        plt.title(f"Omniabase-Pure omega(n) [{args.mode}]")
        if args.primes_overlay:
            primes_mask = np.array([is_prime(int(n)) for n in ns], dtype=bool)
            plt.figure()
            plt.plot(ns[~primes_mask], omegas[~primes_mask], ".", label="Composites")
            plt.plot(ns[primes_mask], omegas[primes_mask], ".", label="Primes")
            plt.xlabel("n")
            plt.ylabel("omega")
            plt.title(f"Omniabase-Pure omega(n): primes vs composites [{args.mode}]")
            plt.legend()
        plt.show()

    windows: List[int]
    if args.min_windows.strip():
        windows = [int(x.strip()) for x in args.min_windows.split(",") if x.strip()]
    else:
        windows = [int(args.min_window)]

    for w in windows:
        res = find_local_minima(ns, omegas, w)
        print(f"Local minima at primes (window={w}): {res.local_min_primes}")
        print(f"Local minima at composites (window={w}): {res.local_min_composites}")
        print(f"Counts -> primes: {len(res.local_min_primes)} | composites: {len(res.local_min_composites)}")

    if args.null_test:
        for w in windows:
            out = local_null_zscore(ns, omegas, window=w, block=args.block, trials=args.trials, seed=args.seed)
            print("LOCAL NULL (block-shuffle)")
            print(f"WINDOW = {int(out['window'])} BLOCK = {int(out['block'])} TRIALS = {int(out['trials'])}")
            print(f"delta_real (P-C) = {int(out['delta_real'])} | P_real = {int(out['p_real'])} C_real = {int(out['c_real'])} | n_minima = {int(out['n_minima'])}")
            print(f"null_mean = {out['null_mean']:.6f} std = {out['null_std']:.6f} min = {out['null_min']:.1f} max = {out['null_max']:.1f}")
            print(f"z_score = {out['z_score']:.6f}")
            print("-" * 60)


if __name__ == "__main__":
    main()