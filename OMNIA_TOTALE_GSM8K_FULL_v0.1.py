"""
OMNIA_TOTALE_GSM8K_FULL_v0.1
Full GSM8K + PBII benchmarks (hallucination detection + AUC).

Requires:
    pip install datasets numpy matplotlib
"""

import math
import re
from dataclasses import dataclass
from typing import List, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset


# =========================================
# 1. PBII CORE (copiato da OMNIA_TOTALE v0.6)
# =========================================

def digits_in_base(n: int, b: int) -> List[int]:
    """Return digits of n in base b (MSB first)."""
    if n == 0:
        return [0]
    res = []
    while n > 0:
        res.append(n % b)
        n //= b
    return res[::-1]


def sigma_b(n: int, b: int) -> float:
    """
    Base Symmetry Score (versione semplice).

    sigma_b(n) ≈ (1 - H_norm) / L + bonus_divisibilità
    """
    digits = digits_in_base(n, b)
    L = len(digits)
    if L == 0:
        return 0.0

    freq = [0] * b
    for d in digits:
        freq[d] += 1
    probs = [c / L for c in freq if c > 0]
    if not probs:
        Hn = 0.0
    else:
        H = -sum(p * math.log2(p) for p in probs)
        Hmax = math.log2(b)
        Hn = H / Hmax if Hmax > 0 else 0.0

    bonus = 0.5 if n % b == 0 else 0.0
    return (1.0 - Hn) / L + bonus


def sigma_avg(n: int, bases: Iterable[int]) -> float:
    bases = list(bases)
    return sum(sigma_b(n, b) for b in bases) / len(bases)


def saturation(n: int, bases: Iterable[int], W: int = 100) -> float:
    """
    Mean sigma over nearby composites in [n-W, n).
    """
    bases = list(bases)
    start = max(2, n - W)
    comps = []
    for k in range(start, n):
        # composite se ha almeno un divisore non banale
        if any(k % d == 0 for d in range(2, int(math.sqrt(k)) + 1)):
            comps.append(k)
    if not comps:
        return 0.0
    vals = [sigma_avg(k, bases) for k in comps]
    return sum(vals) / len(vals)


def pbii(n: int,
         bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
         W: int = 100) -> float:
    """
    Prime Base Instability Index.

    PBII(n) = saturation(composites around n) - sigma_avg(n)
    """
    bases = list(bases)
    sat = saturation(n, bases, W=W)
    sig = sigma_avg(n, bases)
    return sat - sig


# =========================================
# 2. UTILS GSM8K + HALLUCINATION METRICS
# =========================================

NUMBER_RE = re.compile(r"\b\d+\b")


def extract_numbers(text: str) -> List[int]:
    """Estrae interi positivi (>1) da una chain testuale."""
    return [int(m.group()) for m in NUMBER_RE.finditer(text) if int(m.group()) > 1]


def detect_hallucination(chain_text: str,
                         threshold: float = 0.10) -> Tuple[bool, float]:
    """
    Ritorna (flag, avg_pbii).
    Flag True se avg_pbii > threshold (instabilità strutturale).
    """
    nums = extract_numbers(chain_text)
    if not nums:
        return False, 0.0
    scores = [pbii(n) for n in nums]
    avg_score = float(np.mean(scores))
    return avg_score > threshold, avg_score


def corrupt_chain_numbers(chain_text: str,
                          noise_factor: float = 0.15) -> str:
    """
    Introduce corruzioni sintetiche nei numeri:
    +/- delta proporzionale al numero (min 1).
    """
    def _replace(match):
        n = int(match.group())
        if n <= 1:
            return match.group()
        delta = max(1, int(abs(n) * noise_factor))
        # random +/- delta
        sign = np.random.choice([-1, 1])
        new_n = max(2, n + sign * delta)
        return str(new_n)

    return NUMBER_RE.sub(_replace, chain_text, count=2)  # corrompe al massimo 2 numeri


@dataclass
class EvalStats:
    fpr: float          # false positive rate su chain corrette
    dr: float           # detection rate su chain corrotte
    auc: float          # AUC su stesso set (correct vs corrupted)
    thr: float          # threshold usato
    n_samples: int


def compute_auc(labels: List[int], scores: List[float]) -> float:
    """
    AUC binaria semplice (1=positivi, 0=negativi) usando numpy.
    """
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    order = np.argsort(scores_arr)[::-1]
    labels_sorted = labels_arr[order]

    pos = np.sum(labels_sorted)
    neg = len(labels_sorted) - pos
    if pos == 0 or neg == 0:
        return 0.0

    tp = 0.0
    fp = 0.0
    prev_fp = 0.0
    auc = 0.0

    for y in labels_sorted:
        if y == 1:
            tp += 1.0
        else:
            fp += 1.0
            auc += tp * (fp - prev_fp)
            prev_fp = fp

    return auc / (pos * neg)


# =========================================
# 3. FULL GSM8K BENCHMARK
# =========================================

def run_gsm8k_full_benchmark(
    max_samples: int | None = None,
    threshold: float = 0.10,
    corruption_noise: float = 0.15,
    plot_path: str = "gsm8k_pbii_distribution.png",
) -> EvalStats:
    """
    Esegue benchmark su GSM8K (train split "main").

    - Per ogni item:
        - usa answer ufficiale come chain "corretta"
        - genera versione corrotta
        - calcola PBII medio
    - Calcola FPR, DR, AUC e salva grafico distribuzioni.
    """
    ds = load_dataset("openai/gsm8k", "main")["train"]
    n_total = len(ds)
    if max_samples is not None:
        n_use = min(max_samples, n_total)
    else:
        n_use = n_total

    correct_flags = []
    halluc_flags = []
    correct_scores = []
    halluc_scores = []

    for i in range(n_use):
        answer = ds[i]["answer"]
        corr_chain = corrupt_chain_numbers(answer, noise_factor=corruption_noise)

        # chain corretta
        flag_c, score_c = detect_hallucination(answer, threshold=threshold)
        correct_flags.append(flag_c)
        correct_scores.append(score_c)

        # chain corrotta
        flag_h, score_h = detect_hallucination(corr_chain, threshold=threshold)
        halluc_flags.append(flag_h)
        halluc_scores.append(score_h)

    # metriche
    fpr = float(np.mean(correct_flags)) if correct_flags else 0.0
    dr = float(np.mean(halluc_flags)) if halluc_flags else 0.0

    labels = [0] * len(correct_scores) + [1] * len(halluc_scores)
    scores = correct_scores + halluc_scores
    auc = compute_auc(labels, scores)

    # grafico distribuzioni
    plt.figure(figsize=(10, 5))
    plt.hist(correct_scores, bins=30, alpha=0.5, label="Correct chains")
    plt.hist(halluc_scores, bins=30, alpha=0.5, label="Corrupted chains")
    plt.xlabel("Avg PBII score")
    plt.ylabel("Count")
    plt.title("GSM8K PBII distribution: correct vs corrupted chains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return EvalStats(
        fpr=fpr,
        dr=dr,
        auc=auc,
        thr=threshold,
        n_samples=n_use,
    )


# =========================================
# 4. PRIME vs COMPOSITE AUC (1000 NUMERI)
# =========================================

def is_prime(num: int) -> bool:
    if num <= 1:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


@dataclass
class PrimeAUCStats:
    auc: float
    n_primes: int
    n_composites: int


def run_prime_auc_benchmark(
    n_numbers: int = 1000,
    low: int = 2,
    high: int = 10000,
) -> PrimeAUCStats:
    """
    AUC di separazione primi vs composti usando -PBII come score.
    """
    rng = np.random.default_rng(seed=42)
    nums = rng.integers(low=low, high=high, size=n_numbers)
    labels = [1 if is_prime(int(n)) else 0 for n in nums]
    scores = [-pbii(int(n)) for n in nums]  # più alto per primi

    auc = compute_auc(labels, scores)
    n_primes = sum(labels)
    n_comp = len(labels) - n_primes
    return PrimeAUCStats(auc=auc, n_primes=n_primes, n_composites=n_comp)


# =========================================
# 5. MAIN
# =========================================

def main():
    print("=== OMNIA_TOTALE_GSM8K_FULL_v0.1 ===")

    # Benchmark GSM8K (tutto il dataset; usa max_samples per debug)
    stats = run_gsm8k_full_benchmark(
        max_samples=None,         # None = full dataset
        threshold=0.10,
        corruption_noise=0.15,
        plot_path="gsm8k_pbii_distribution.png",
    )
    print(f"GSM8K: N={stats.n_samples}")
    print(f"  FPR (correct chains flagged): {stats.fpr * 100:.2f}%")
    print(f"  DR  (corrupted chains flagged): {stats.dr * 100:.2f}%")
    print(f"  AUC (correct vs corrupted): {stats.auc:.3f}")
    print(f"  Threshold used: PBII > {stats.thr:.3f}")
    print("  Distribution plot saved as gsm8k_pbii_distribution.png")

    # Benchmark primi vs composti
    pstats = run_prime_auc_benchmark(
        n_numbers=1000,
        low=2,
        high=10000,
    )
    print("\nPrime vs composite PBII AUC:")
    print(f"  AUC: {pstats.auc:.3f}")
    print(f"  #primes = {pstats.n_primes}, #composites = {pstats.n_composites}")


if __name__ == "__main__":
    main()