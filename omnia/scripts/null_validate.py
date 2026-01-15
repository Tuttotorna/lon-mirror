#!/usr/bin/env python3
"""
OMNIABASE-PURE — base-free Ω metric + null validation (z > 2)
"""

import argparse
import random
import numpy as np

from omnia.omniabase_pure import omega_pure


def block_shuffle(seq, block):
    out = []
    for i in range(0, len(seq), block):
        chunk = seq[i:i+block]
        random.shuffle(chunk)
        out.extend(chunk)
    return out


def count_local_minima(vals, window):
    P = 0
    C = 0
    for i in range(window, len(vals) - window):
        left = vals[i-window:i]
        right = vals[i+1:i+1+window]
        if vals[i] < min(left) and vals[i] < min(right):
            if is_prime(i):
                P += 1
            else:
                C += 1
    return P, C


def is_prime(n):
    if n < 2:
        return False
    for k in range(2, int(n**0.5) + 1):
        if n % k == 0:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=120)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ns = list(range(2, args.nmax + 1))
    omegas = [omega_pure(n)[0] for n in ns]

    P_real, C_real = count_local_minima(omegas, args.window)
    delta_real = P_real - C_real

    deltas = []
    for _ in range(args.trials):
        sh = block_shuffle(omegas.copy(), args.block)
        P, C = count_local_minima(sh, args.window)
        deltas.append(P - C)

    mu = np.mean(deltas)
    sigma = np.std(deltas)
    z = (delta_real - mu) / sigma if sigma > 0 else 0.0

    print("OMNIABASE-PURE null validation")
    print(f"nmax={args.nmax} window={args.window} block={args.block} trials={args.trials}")
    print(f"delta_real (P-C) = {delta_real}")
    print(f"null mean = {mu:.3f}  std = {sigma:.3f}")
    print(f"z-score = {z:.3f}")
    print("SIGNIFICANT =", z > 2)


if __name__ == "__main__":
    main()