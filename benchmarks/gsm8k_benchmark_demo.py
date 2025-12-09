"""
gsm8k_benchmark_demo.py — synthetic benchmarks for OMNIA_TOTALE

Author: Massimiliano Brighindi (MB-X.01 / Omniabase±)
Engine formalization: MBX IA

This script demonstrates:

1) PBII-based hallucination detection on synthetic GSM8K-style chains.
2) AUC for prime vs composite separation using PBII scores.
3) Histogram of PBII for primes vs composites (saved as pbii_distribution.png).

Requirements:
    pip install numpy matplotlib
"""

from __future__ import annotations

import math
import re
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from omnia.core.omniabase import pbii_index


# =========================
# 1. UTILS
# =========================

def extract_numbers(chain_text: str) -> List[int]:
    """
    Extract positive integers > 1 from a reasoning chain (plain text).
    """
    nums = [int(num) for num in re.findall(r"\b\d+\b", chain_text)]
    return [n for n in nums if n > 1]


def detect_hallucination(chain_text: str, threshold: float = 0.1) -> bool:
    """
    Simple PBII-based hallucination detector.

    - Extracts all numbers in the chain.
    - Computes PBII for each.
    - Flags the chain if the average PBII > threshold.

    High PBII ≈ multi-base instability ≈ potential hallucination.
    """
    numbers = extract_numbers(chain_text)
    if not numbers:
        return False
    scores = [pbii_index(n) for n in numbers]
    avg_pbii = float(np.mean(scores))
    return avg_pbii > threshold


def is_prime(num: int) -> bool:
    if num <= 1:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


def compute_auc(labels: List[int], scores: List[float]) -> float:
    """
    Simple ROC AUC implementation (labels: 1=positive, 0=negative).
    """
    labels_arr = np.array(labels, dtype=int)
    scores_arr = np.array(scores, dtype=float)

    # sort by descending score
    idx = np.argsort(scores_arr)[::-1]
    labels_sorted = labels_arr[idx]

    pos = int(labels_sorted.sum())
    neg = len(labels_sorted) - pos
    if pos == 0 or neg == 0:
        return 0.0

    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0

    for lab in labels_sorted:
        if lab == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp

    return float(auc / (pos * neg))


# =========================
# 2. SYNTHETIC CHAINS (GSM8K-STYLE)
# =========================

correct_chains = [
    # Answer 14
    """Sam skipped 16 times per round. Jeff: round1=15, round2=13, round3=20, round4=8. Total Jeff=56, avg=14.""",
    # Answer 100
    """Mark bought 50 cans. Jennifer added 6 for every 5: 10 times, 60 cans. Total 40+60=100.""",
    # Answer 54
    """Paityn 20 red +24 blue=44. Zola 16 red +48 blue=64. Total 108, each 54.""",
    # Answer 45
    """Todd 4, Alisha 8, Bobby 27. Remaining 6. Total 45.""",
    # Answer 3
    """First tank 48/3=16 fish, second 24/2=12. After eat: 15-12=3 more in first.""",
]

hallucinated_chains = [
    # altered numbers
    """Sam skipped 17 times per round. Jeff: round1=16, round2=14, round3=21, round4=9. Total Jeff=60, avg=15.""",
    """Mark bought 51 cans. Jennifer added 7 for every 6: 11 times, 77 cans. Total 41+77=118.""",
    """Paityn 21 red +25 blue=46. Zola 17 red +50 blue=67. Total 113, each 56.5.""",
    """Todd 5, Alisha 10, Bobby 35. Remaining 7. Total 57.""",
    """First tank 49/4=12.25 fish, second 25/3≈8.33. After eat: 11.25-8.33≈3.""",
]


def run_hallucination_benchmark(threshold: float = 0.1) -> None:
    """
    Benchmark 1:
    - false positives on correct chains
    - detection rate on hallucinated chains
    """
    correct_flags = [detect_hallucination(c, threshold=threshold) for c in correct_chains]
    halluc_flags = [detect_hallucination(c, threshold=threshold) for c in hallucinated_chains]

    false_positive_rate = sum(correct_flags) / len(correct_flags)
    detection_rate = sum(halluc_flags) / len(halluc_flags)

    print("=== Benchmark 1: PBII hallucination detection (synthetic GSM8K) ===")
    print(f"Threshold: {threshold:.3f}")
    print(f"False positives on correct chains: {false_positive_rate * 100:.1f}%")
    print(f"Detection rate on hallucinated chains: {detection_rate * 100:.1f}%")
    print("NOTE: This is a small synthetic demo. Full-scale GSM8K tests require real data.\n")


# =========================
# 3. PRIME vs COMPOSITE AUC
# =========================

def run_prime_composite_auc(n_samples: int = 200) -> None:
    """
    Benchmark 2:
    - random integers
    - label: 1 = prime, 0 = composite
    - score: -PBII (lower PBII for primes → higher score for primes)
    - compute AUC
    """
    np.random.seed(42)
    numbers = np.random.randint(2, 2000, n_samples)
    labels = [1 if is_prime(n) else 0 for n in numbers]
    scores = [-pbii_index(n) for n in numbers]

    auc = compute_auc(labels, scores)
    print("=== Benchmark 2: PBII AUC (prime vs composite) ===")
    print(f"Samples: {n_samples}")
    print(f"AUC ≈ {auc:.3f}")
    print("NOTE: This is a synthetic demo; behavior stabilizes with larger samples.\n")

    # histogram of PBII
    primes_pbii = [pbii_index(n) for n, lab in zip(numbers, labels) if lab == 1]
    comps_pbii = [pbii_index(n) for n, lab in zip(numbers, labels) if lab == 0]

    plt.figure(figsize=(10, 5))
    plt.hist(primes_pbii, bins=20, alpha=0.5, label="Primes")
    plt.hist(comps_pbii, bins=20, alpha=0.5, label="Composites")
    plt.xlabel("PBII score")
    plt.ylabel("Count")
    plt.title("PBII distribution: primes vs composites (synthetic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pbii_distribution.png")
    print("Saved histogram to pbii_distribution.png\n")


# =========================
# 4. MAIN
# =========================

def main():
    run_hallucination_benchmark(threshold=0.1)
    run_prime_composite_auc(n_samples=200)


if __name__ == "__main__":
    main()