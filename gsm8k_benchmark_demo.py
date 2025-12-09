"""
gsm8k_benchmark_demo.py — OMNIA_TOTALE / PBII synthetic benchmarks

Author: Massimiliano Brighindi (MB-X.01 / Omniabase±)
Engine: omnia.omniabase (PBII)

This script provides two synthetic benchmarks:

1) Hallucination detection on GSM8K-style reasoning chains:
   - We define a small set of "correct" chains and "hallucinated" variants.
   - We extract all integers from each chain.
   - We compute PBII per integer via omnia.omniabase.pbii_index.
   - We flag a chain as "hallucinated" if avg(PBII) > threshold.

2) Prime vs composite separation:
   - We generate random integers in [2, 1000).
   - We label them as prime (1) or composite (0).
   - We compute PBII as score and estimate AUC with a simple numpy-based ROC.

Outputs:
- Printed false positive rate on correct chains.
- Printed detection rate on hallucinated chains.
- Printed AUC for prime vs composite discrimination.
- A histogram figure 'pbii_distribution.png' saved in the current folder.
"""

from __future__ import annotations

import math
import re
from typing import List

import numpy as np
import matplotlib.pyplot as plt

# Import PBII from the OMNIA core
from omnia.omniabase import pbii_index


# =========================
# 1. GSM8K-STYLE CHAINS
# =========================

def extract_numbers(chain_text: str) -> List[int]:
    """
    Extract positive integer numbers > 1 from a reasoning chain.
    """
    return [int(num) for num in re.findall(r"\b\d+\b", chain_text) if int(num) > 1]


# Sample "correct" reasoning chains (public GSM8K-style examples, simplified)
correct_chains = [
    # Answer ~14
    """Sam skipped 16 times per round. Jeff: round1=15, round2=13, round3=20, round4=8. Total Jeff=56, avg=14.""",
    # Answer ~100
    """Mark bought 50 cans. Jennifer added 6 for every 5: 10 times, 60 cans. Total 40 + 60 = 100.""",
    # Answer ~54
    """Paityn 20 red + 24 blue = 44. Zola 16 red + 48 blue = 64. Total 108, each 54.""",
    # Answer ~45
    """Todd 4, Alisha 8, Bobby 27. Remaining 6. Total 45.""",
    # Answer ~3
    """First tank 48/3 = 16 fish, second 24/2 = 12. After eat: 15 - 12 = 3 more in first.""",
]

# Synthetic "hallucinated" versions (numbers slightly perturbed)
hallucinated_chains = [
    """Sam skipped 17 times per round. Jeff: round1=16, round2=14, round3=21, round4=9. Total Jeff=60, avg=15.""",
    """Mark bought 51 cans. Jennifer added 7 for every 6: 11 times, 77 cans. Total 41 + 77 = 118.""",
    """Paityn 21 red + 25 blue = 46. Zola 17 red + 50 blue = 67. Total 113, each 56.5.""",
    """Todd 5, Alisha 10, Bobby 35. Remaining 7. Total 57.""",
    """First tank 49/4 = 12.25 fish, second 25/3 ≈ 8.33. After eat: 11.25 - 8.33 ≈ 3.""",
]


def detect_hallucination(chain_text: str, threshold: float = 0.10) -> bool:
    """
    Simple hallucination detector:

    - Extract integers from chain_text.
    - Compute PBII per integer.
    - If average PBII > threshold, flag as 'hallucinated'.
    """
    numbers = extract_numbers(chain_text)
    if not numbers:
        return False

    pbii_scores = [pbii_index(num) for num in numbers]
    avg_pbii = float(np.mean(pbii_scores))
    return avg_pbii > threshold


def benchmark_hallucinations(threshold: float = 0.10) -> None:
    """
    Evaluate hallucination detection on synthetic GSM8K-style chains.
    """
    # Correct chains → we prefer NOT to flag them
    correct_flags = [detect_hallucination(chain, threshold=threshold) for chain in correct_chains]
    false_positives = sum(correct_flags) / len(correct_chains)

    # Hallucinated chains → we want to flag them
    halluc_flags = [detect_hallucination(chain, threshold=threshold) for chain in hallucinated_chains]
    detection_rate = sum(halluc_flags) / len(hallucinated_chains)

    print("=== Benchmark 1: GSM8K-style hallucination detection (synthetic) ===")
    print(f"Threshold on avg PBII: {threshold:.3f}")
    print(f"False positives (correct chains flagged): {false_positives * 100:.1f}%")
    print(f"Detection rate (hallucinated chains flagged): {detection_rate * 100:.1f}%")
    print("NOTE: This is a toy synthetic demo. For real benchmarks, extend to hundreds of GSM8K samples.")


# =========================
# 2. PRIME VS COMPOSITE (AUC)
# =========================

def is_prime(num: int) -> bool:
    """
    Basic primality test for small integers.
    """
    if num <= 1:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


def compute_auc(labels, scores) -> float:
    """
    Simple AUC implementation (binary, labels in {0,1}).

    - Sort by score descending.
    - Integrate TPR vs FPR as stepwise ROC.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    # Sort by score descending
    idx = np.argsort(scores)[::-1]
    labels_sorted = labels[idx]

    pos = float(np.sum(labels_sorted == 1))
    neg = float(np.sum(labels_sorted == 0))
    if pos == 0 or neg == 0:
        return 0.0

    tp = 0.0
    fp = 0.0
    prev_fp = 0.0
    auc = 0.0

    for lab in labels_sorted:
        if lab == 1:
            tp += 1.0
        else:
            fp += 1.0
            auc += tp * (fp - prev_fp)
            prev_fp = fp

    return auc / (pos * neg)


def benchmark_primes_vs_composites(
    n_samples: int = 200,
    low: int = 2,
    high: int = 1000,
    seed: int = 42,
) -> None:
    """
    Evaluate how well PBII separates primes from composites on random integers.
    """
    np.random.seed(seed)
    numbers = np.random.randint(low, high, size=n_samples)

    labels = [1 if is_prime(n) else 0 for n in numbers]  # 1 = prime, 0 = composite
    # PBII is higher for more "prime-like" instability → direct score
    scores = [pbii_index(n) for n in numbers]

    auc = compute_auc(labels, scores)

    primes_pbii = [s for s, l in zip(scores, labels) if l == 1]
    comps_pbii = [s for s, l in zip(scores, labels) if l == 0]

    print("\n=== Benchmark 2: PBII prime vs composite separation ===")
    print(f"Samples: {n_samples}, range: [{low}, {high})")
    print(f"AUC (PBII score, primes=1 vs composites=0): {auc:.3f}")
    print("Histogram saved as 'pbii_distribution.png'.")

    # Plot distributions
    plt.figure(figsize=(10, 5))
    plt.hist(primes_pbii, bins=20, alpha=0.5, label="Primes")
    plt.hist(comps_pbii, bins=20, alpha=0.5, label="Composites")
    plt.xlabel("PBII score")
    plt.ylabel("Count")
    plt.title("PBII distribution: primes vs composites (synthetic demo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pbii_distribution.png")
    # Comment out plt.show() if running in headless env
    # plt.show()


# =========================
# 3. MAIN
# =========================

def main() -> None:
    """
    Run both synthetic benchmarks.
    """
    benchmark_hallucinations(threshold=0.10)
    benchmark_primes_vs_composites(
        n_samples=200,
        low=2,
        high=1000,
        seed=42,
    )


if __name__ == "__main__":
    main()