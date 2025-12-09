"""
OMNIA_TOTALE — Ω-LENS ENGINE v1.0
Unified engine for stability analysis, causal inference, Ω-correlations,
PBII scoring, token-level Ω-maps, and GSM8K reproducible evaluation hooks.

Author: Massimiliano Brighindi (MB-X.01)
Repo: https://github.com/Tuttotorna/lon-mirror
"""

import math
import numpy as np
import re

# ============================================================
# 1. BASIC UTILITIES
# ============================================================

def digits_in_base(n, b):
    if n == 0:
        return [0]
    res = []
    while n > 0:
        res.append(n % b)
        n //= b
    return res[::-1]

def sigma_b(n, b):
    digits = digits_in_base(n, b)
    L = len(digits)
    freq = [0] * b
    for d in digits:
        freq[d] += 1
    probs = [c / L for c in freq if c > 0]
    H = -sum(p * math.log2(p) for p in probs) if probs else 0
    Hn = H / math.log2(b) if H else 0
    bonus = 0.5 if n % b == 0 else 0
    return (1 - Hn) / L + bonus if L else 0

def sigma_avg(n, bases):
    return sum(sigma_b(n, b) for b in bases) / len(bases)

def prime_test(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# ============================================================
# 2. PBII — PRIME-BASE INSTABILITY INDEX
# ============================================================

def saturation(n, bases, W=100):
    comps = [k for k in range(max(2, n-W), n)
             if any(k % d == 0 for d in range(2, int(math.sqrt(k))+1))]
    if not comps:
        return 0
    values = [sigma_avg(k, bases) for k in comps]
    return sum(values) / len(values)

def pbii(n, bases=None, W=100):
    if bases is None:
        bases = [2,3,5,7,11,13,17,19,23,29]
    return saturation(n, bases, W) - sigma_avg(n, bases)

# ============================================================
# 3. TRUTHΩ — STRUCTURE CONSISTENCY SCORE
# ============================================================

def truth_omega(seq):
    seq = np.array(seq)
    diff = np.abs(np.diff(seq))
    if len(diff) == 0:
        return 0
    return 1 / (1 + np.std(diff))

# ============================================================
# 4. TOKEN-LEVEL Ω-MAPS
# ============================================================

def extract_numbers(text):
    return [int(num) for num in re.findall(r"\b\d+\b", text)]

def token_omega_map(text):
    nums = extract_numbers(text)
    return {n: pbii(n) for n in nums}

# ============================================================
# 5. OMNIATEMPO — REGIME CHANGE DETECTOR
# ============================================================

def regime_changes(seq, window=4, threshold=1.5):
    seq = np.array(seq, dtype=float)
    changes = []
    for i in range(window, len(seq)):
        local = seq[i-window:i]
        if np.std(local) > threshold:
            changes.append(i)
    return changes

# ============================================================
# 6. OMNIACAUSA — LAGGED CAUSAL FINGERPRINTS
# ============================================================

def causal_lag(x, y, max_lag=5):
    lags = {}
    x = np.array(x)
    y = np.array(y)
    for lag in range(1, max_lag+1):
        if len(y)-lag <= 0 or len(x)-lag <= 0:
            continue
        corr = np.corrcoef(x[:-lag], y[lag:])[0,1]
        lags[lag] = corr
    return lags

# ============================================================
# 7. SELF-REVISION LOOP
# ============================================================

def selfrev(text, cycles=2):
    nums = extract_numbers(text)
    rev = nums.copy()
    for _ in range(cycles):
        rev = [n - int(pbii(n)*10) for n in rev]
    return rev

# ============================================================
# 8. GSM8K HOOKS — DETECTION OF HALLUCINATIONS
# ============================================================

def hallucination_flag(chain_text, threshold=0.1):
    nums = extract_numbers(chain_text)
    if not nums:
        return False
    pb = np.mean([pbii(n) for n in nums])
    return pb > threshold

# ============================================================
# 9. Ω-LENS ENGINE — UNIFIED API
# ============================================================

class OmegaLens:

    def instability(self, n):
        return pbii(n)

    def omega_correlation(self, seq):
        return truth_omega(seq)

    def token_map(self, text):
        return token_omega_map(text)

    def tempo(self, seq):
        return regime_changes(seq)

    def causa(self, x, y):
        return causal_lag(x, y)

    def revise(self, text):
        return selfrev(text)

    def hallucination(self, text):
        return hallucination_flag(text)

# ============================================================
# END
# ============================================================