"""
OMNIABASE — Multi-base numerical signature
Core module for OMNIA_TOTALE package
Author: Massimiliano Brighindi (concepts) + MBX IA (structure)
"""

import math
import numpy as np

def digits_in_base(n: int, b: int) -> np.ndarray:
    if n == 0:
        return np.array([0])
    out = []
    while n > 0:
        out.append(n % b)
        n //= b
    return np.array(out[::-1])

def entropy_norm(n: int, b: int) -> float:
    d = digits_in_base(n, b)
    L = len(d)
    freq = np.bincount(d, minlength=b)
    probs = freq[freq > 0] / L
    if probs.size == 0:
        return 0.0
    H = -np.sum(probs * np.log2(probs))
    return float(H / math.log2(b))

def sigma_b(n: int, b: int) -> float:
    Hn = entropy_norm(n, b)
    d = digits_in_base(n, b)
    L = len(d)
    return (1 - Hn) / L + (0.5 if n % b == 0 else 0.0)

def pbii(n: int, bases=None, W=50) -> float:
    if bases is None:
        bases = [2,3,5,7,11,13,17,19]
    comps = [k for k in range(max(2, n-W), n)
             if any(k % d == 0 for d in range(2, int(math.sqrt(k))+1))]
    if not comps:
        return 0.0
    base_avg = lambda x: np.mean([sigma_b(x, b) for b in bases])
    sat = np.mean([base_avg(c) for c in comps])
    return float(sat - base_avg(n))