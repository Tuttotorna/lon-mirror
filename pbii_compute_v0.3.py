# pbii_compute_v0.3.py
# MBX–Omniabase · PBII computation engine (n ≤ 10,000,000)
# Massimiliano Brighindi — MB-X.01 / Omniabase±

import math
import csv
from sympy import primerange

# ===== PARAMETERS =====
MAX_N = 10_000_000
WINDOW = 100
BASES = list(primerange(2, 200))  # prime bases < 200

# ===== FAST BASE CONVERSION =====
def digits_in_base(n, b):
    res = []
    while n > 0:
        res.append(n % b)
        n //= b
    return res[::-1] if res else [0]

# ===== ENTROPY AND SYMMETRY =====
def sigma_b(n, b):
    d = digits_in_base(n, b)
    L = len(d)
    freq = [0] * b
    for x in d:
        freq[x] += 1
    p = [c / L for c in freq if c > 0]
    H = -sum(pi * math.log(pi) for pi in p)
    Hn = H / math.log(b)
    base_term = (1 - Hn) / L
    div_bonus = 0.5 if (n % b == 0) else 0
    return base_term + div_bonus

def Sigma(n):
    return sum(sigma_b(n, b) for b in BASES) / len(BASES)

# ===== PBII =====
def compute_pbii():
    Sigma_cache = [0.0] * (MAX_N + 1)

    for n in range(2, MAX_N + 1):
        Sigma_cache[n] = Sigma(n)

    def Sat(n):
        start = max(2, n - WINDOW)
        comps = [Sigma_cache[k] for k in range(start, n) if not is_prime[k]]
        if not comps:
            return Sigma_cache[n-1]
        return sum(comps) / len(comps)

    pbii = [0.0] * (MAX_N + 1)
    for n in range(3, MAX_N + 1):
        pbii[n] = Sat(n) - Sigma_cache[n]

    return pbii, Sigma_cache

# ===== SIEVE FOR PRIMES =====
is_prime = [True] * (MAX_N + 1)
is_prime[0] = is_prime[1] = False
for i in range(2, int(MAX_N**0.5) + 1):
    if is_prime[i]:
        for j in range(i*i, MAX_N + 1, i):
            is_prime[j] = False

# ===== MAIN =====
pbii, Sigma_cache = compute_pbii()

with open("PBII_data_10M.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "PBII", "Sigma", "is_prime"])
    for n in range(2, MAX_N + 1):
        w.writerow([n, pbii[n], Sigma_cache[n], int(is_prime[n])])