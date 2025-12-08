# OMNIA_TOTALE — Structural Coherence Lenses (Work in Progress)

Author: **Massimiliano Brighindi (MB-X.01)**  
Concepts + Design: Massimiliano  
Formalization Support: MBX-IA

> This repository is a **work in progress**.  
> All claims about performance (e.g. hallucination reduction, AUC on primes) are currently based on **synthetic experiments and internal tests**.  
> Reproducible scripts are being added step by step.

---

## Overview

OMNIA_TOTALE is a set of **structural lenses** for analyzing stability and coherence in numbers, time series, and reasoning chains:

- **Omniabase** → multi-base structure of integers (PBII, σ-scores, entropy).  
- **Omniatempo** → regime-change detection in 1D time series.  
- **Omniacausa** → lagged dependency graph between multiple series.  
- **Ω / Supervisor (early)** → simple instability flags on reasoning chains.

The goal is to provide **deterministic, model-agnostic metrics** that can sit *outside* an LLM and evaluate its outputs.

---

## Current Contents

Right now, the repo contains:

- `OMNIA_TOTALE_v0.2.py`  
  NumPy-based implementation of:
  - Omniabase (PBII, σ, entropy over multiple bases)
  - Omniatempo (temporal regime-change score)
  - Omniacausa (lagged correlation graph)
  - Fused Ω-score combining the three lenses

- `OMNIA_TOTALE_SELFREV_v0.1.py` (if present)  
  Self-review / supervisor-style wrapper (early prototype).

- `OMNIA_TOTALE_TOKENMAP_v0.1.py` (if present)  
  Per-token deviation map for reasoning texts (prototype).

- `OMNIA_TOTALE_INTERNAL_IFACE_v0.1.py` (if present)  
  Internal interface sketch for integration in LLM pipelines.

- `benchmarks/gsm8k_benchmark_demo.py`  
  Synthetic demo showing:
  - How PBII can flag unstable numeric patterns in toy GSM8K-like chains.
  - How to compute a simple prime/composite AUC using PBII.

If a file listed above does **not** exist yet in this repo, it means it is still under construction and will be added in a future commit.

---

## Status of Benchmarks (Honest Version)

- **GSM8K-style hallucinations**  
  - Currently: small synthetic demo in `benchmarks/gsm8k_benchmark_demo.py`.  
  - Real: full evaluation on large GSM8K subsets has been run locally, not yet fully packaged here.

- **Prime vs composite AUC**  
  - Theoretical target: AUC ≈ 0.98 based on extensive local experiments.  
  - In this repo: a smaller reproducible experiment (100–1,000 numbers) is provided in the demo script; AUC may vary depending on range and parameters.

Until the full datasets and pipelines are published, **all numbers should be treated as “claims under active verification”**, not as formally established benchmarks.

---

## Quick Start

Requirements:

```bash
pip install numpy matplotlib

Run the core demo:

python OMNIA_TOTALE_v0.2.py

Run synthetic benchmarks (if the file exists):

python benchmarks/gsm8k_benchmark_demo.py


---

Intended Use

As an external evaluator for LLM reasoning chains.

As a structural lens for number sequences (e.g. primes) and time series.

As a starting point for research into coherence metrics and AI safety tools.


This repository is evolving rapidly.
The priority is structural coherence and transparency: what is here, is real and runnable; what is not yet here, is still work-in-progress and is clearly marked as such.

Questo README:

- Elimina OMNIA_TOTALE_v0.7 se non esiste.  
- Non promette file che non ci sono.  
- Non spaccia per “fatti verificati” le percentuali: le declassa a “claim da verificare”, finché non hai codice + dati.

---

## 2. Script benchmark reale, minimale ma onesto

Se non l’hai ancora caricato, devi creare almeno **un** file che faccia davvero qualcosa di misurabile, anche piccolo.

File: `benchmarks/gsm8k_benchmark_demo.py`

```python
"""
gsm8k_benchmark_demo.py

Minimal synthetic demo for:
- PBII prime/composite separation (AUC on a small sample)
- PBII-based instability flag on toy GSM8K-like chains

Author: Massimiliano Brighindi (MB-X.01)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import re

# ---- PBII core (coherent with OMNIA_TOTALE_v0.2) ----

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
    if L == 0:
        return 0.0
    freq = [0] * b
    for d in digits:
        freq[d] += 1
    probs = [c / L for c in freq if c > 0]
    if not probs:
        H = 0.0
    else:
        H = -sum(p * math.log2(p) for p in probs)
    Hmax = math.log2(b)
    Hn = H / Hmax if Hmax > 0 else 0.0
    bonus = 0.5 if n % b == 0 else 0.0
    return (1.0 - Hn) / L + bonus

def sigma_avg(n, bases):
    return sum(sigma_b(n, b) for b in bases) / len(bases)

def saturation(n, bases, W=100):
    comps = []
    for k in range(max(2, n - W), n):
        # simple composite check
        if any(k % d == 0 for d in range(2, int(math.sqrt(k)) + 1)):
            comps.append(k)
    if not comps:
        return 0.0
    vals = [sigma_avg(k, bases) for k in comps]
    return sum(vals) / len(vals)

def pbii(n, bases=None, W=100):
    if bases is None:
        bases = [2, 3, 5, 7, 11, 13, 17, 19]
    return saturation(n, bases, W) - sigma_avg(n, bases)

# ---- Utility ----

def is_prime(num: int) -> bool:
    if num <= 1:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def compute_auc(labels, scores):
    """
    Simple AUC implementation (ROC AUC for binary labels).
    labels: 1 for positive (prime), 0 for negative (composite)
    scores: higher = more "prime-like"
    """
    labels = np.array(labels)
    scores = np.array(scores)
    # sort by score descending
    idx = np.argsort(scores)[::-1]
    labels = labels[idx]

    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0

    tp = 0
    fp = 0
    prev_fp = 0
    auc = 0.0

    for lab in labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp

    return auc / (pos * neg)

def extract_numbers(text: str):
    return [int(x) for x in re.findall(r"\b\d+\b", text)]

# ---- Benchmark 1: Prime vs Composite AUC ----

def benchmark_primes_sample(n_samples=200, low=10, high=5000):
    np.random.seed(0)
    nums = np.random.randint(low, high, size=n_samples)
    labels = [1 if is_prime(n) else 0 for n in nums]
    scores = [-pbii(n) for n in nums]  # invert so primes -> higher score

    auc = compute_auc(labels, scores)
    print(f"[AUC demo] AUC on {n_samples} random numbers in [{low},{high}]: {auc:.3f}")

    primes_pbii = [pbii(n) for n, l in zip(nums, labels) if l == 1]
    comps_pbii = [pbii(n) for n, l in zip(nums, labels) if l == 0]

    plt.figure(figsize=(8, 4))
    plt.hist(primes_pbii, bins=20, alpha=0.5, label="primes")
    plt.hist(comps_pbii, bins=20, alpha=0.5, label="composites")
    plt.xlabel("PBII score")
    plt.ylabel("count")
    plt.title("PBII distribution — primes vs composites (demo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pbii_distribution_demo.png")
    print("[AUC demo] Saved plot: pbii_distribution_demo.png")

# ---- Benchmark 2: Toy GSM8K-style hallucination detection ----

def detect_hallucination(chain_text: str, threshold=0.1) -> bool:
    nums = extract_numbers(chain_text)
    if not nums:
        return False
    scores = [pbii(n) for n in nums]
    avg_pbii = float(np.mean(scores))
    return avg_pbii > threshold

def benchmark_gsm8k_toy():
    correct_chains = [
        "Sam skipped 16 times per round. Jeff: 15, 13, 20, 8. Total 56, avg 14.",
        "Mark bought 40 cans, Jennifer 60, total 100.",
    ]
    hallucinated_chains = [
        "Sam skipped 17 times per round. Jeff: 16, 14, 21, 9. Total 60, avg 15.",
        "Mark bought 41 cans, Jennifer 77, total 118.",
    ]

    fp = sum(detect_hallucination(c) for c in correct_chains) / len(correct_chains)
    dr = sum(detect_hallucination(c) for c in hallucinated_chains) / len(hallucinated_chains)

    print(f"[GSM8K toy] False positives on correct chains: {fp*100:.1f}%")
    print(f"[GSM8K toy] Detection rate on hallucinated chains: {dr*100:.1f}%")
    print("[GSM8K toy] This is a small synthetic demo, not a full benchmark.")

# ---- Main ----

if __name__ == "__main__":
    print("=== OMNIA_TOTALE — synthetic benchmarks demo ===")
    benchmark_primes_sample()
    benchmark_gsm8k_toy()

