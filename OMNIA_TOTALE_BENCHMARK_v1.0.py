# OMNIA_TOTALE_BENCHMARK_v1.0.py
#
# Benchmarks for OMNIA_TOTALE:
# - PBII prime vs composite separation (AUC)
# - PBII-based hallucination detection on GSM8K-like chains (synthetic or real)
# - Optional Ω-score correlation demo (if OMNIA_TOTALE_v0.6 is available)
#
# Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)
#
# Dependencies:
#   pip install numpy matplotlib
#   (optional) pip install datasets
#
# NOTE: This script is designed so xAI or any lab can plug in their own
# GSM8K generations and models. Numeric results depend on the data used
# and are NOT hard-coded here.

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import Iterable, List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ==============================
# 0. UTILITIES
# ==============================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ==============================
# 1. PBII IMPLEMENTATION (LOCAL)
#    (kept minimal and aligned with OMNIA_TOTALE_v0.6)
# ==============================

def digits_in_base(n: int, b: int) -> List[int]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if b <= 1:
        raise ValueError("base must be >= 2")
    if n == 0:
        return [0]
    res: List[int] = []
    while n > 0:
        res.append(n % b)
        n //= b
    return res[::-1]


def sigma_b(
    n: int,
    b: int,
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> float:
    digits = digits_in_base(n, b)
    L = len(digits)
    if L == 0:
        return 0.0

    freq = np.bincount(digits, minlength=b).astype(float)
    probs = freq[freq > 0] / L
    if probs.size == 0:
        Hn = 0.0
    else:
        H = -np.sum(probs * np.log2(probs))
        Hmax = math.log2(b)
        Hn = float(H / Hmax) if Hmax > 0 else 0.0

    length_term = length_weight * (1.0 - Hn) / (L ** length_exponent)
    div_term = divisibility_bonus * (1.0 if n % b == 0 else 0.0)
    return float(length_term + div_term)


def sigma_avg(
    n: int,
    bases: Iterable[int],
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> float:
    vals = [
        sigma_b(
            n,
            b,
            length_weight=length_weight,
            length_exponent=length_exponent,
            divisibility_bonus=divisibility_bonus,
        )
        for b in bases
    ]
    return float(np.mean(vals)) if vals else 0.0


def saturation(
    n: int,
    bases: Iterable[int],
    window: int = 100,
    **kwargs,
) -> float:
    start = max(2, n - window)
    comps: List[int] = []
    for k in range(start, n):
        if k <= 3:
            continue
        is_comp = any(k % d == 0 for d in range(2, int(math.sqrt(k)) + 1))
        if is_comp:
            comps.append(k)
    if not comps:
        return 0.0
    vals = [sigma_avg(k, bases, **kwargs) for k in comps]
    return float(np.mean(vals))


def pbii(
    n: int,
    bases: Optional[Iterable[int]] = None,
    window: int = 100,
    **kwargs,
) -> float:
    """
    Prime Base Instability Index.

    PBII(n) = saturation(composites) - sigma_avg(n)

    Higher values => more prime-like instability.
    """
    if bases is None:
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    bases = list(bases)
    sat = saturation(n, bases, window=window, **kwargs)
    sig = sigma_avg(n, bases, **kwargs)
    return float(sat - sig)


# ==============================
# 2. HALLUCINATION DETECTION (GSM8K-LIKE)
# ==============================

_num_pattern = re.compile(r"\b\d+\b")


def extract_numbers(text: str) -> List[int]:
    return [int(m.group(0)) for m in _num_pattern.finditer(text) if int(m.group(0)) > 1]


def detect_hallucination_pbii(
    chain_text: str,
    threshold: float = 0.10,
) -> Tuple[bool, float]:
    """
    Very simple heuristic:
    - extract all integers in the chain
    - compute PBII for each
    - average PBII
    - if avg PBII > threshold => flag as 'unstable' (potential hallucination)
    """
    nums = extract_numbers(chain_text)
    if not nums:
        return False, 0.0
    scores = [pbii(n) for n in nums]
    avg = float(np.mean(scores))
    return bool(avg > threshold), avg


# Synthetic GSM8K-like chains (same spirit as previous demo, but wrapped)
SYNTH_CORRECT_CHAINS: List[str] = [
    "Sam skipped 16 times per round. Jeff: r1=15, r2=13, r3=20, r4=8. Total=56, avg=14.",
    "Mark bought 50 cans. Jennifer adds 6 for every 5: 10 times, 60 cans. Total=110.",
    "Paityn 20 red + 24 blue = 44. Zola 16 red + 48 blue = 64. Total 108, each 54.",
    "Todd 4, Alisha 8, Bobby 27. Remaining 6. Total 45.",
    "First tank 48/3=16, second 24/2=12. After eating: 15-12=3 more in tank one.",
]

SYNTH_HALLUCINATED_CHAINS: List[str] = [
    "Sam skipped 17 times per round. Jeff: r1=16, r2=14, r3=21, r4=9. Total=60, avg=15.",
    "Mark bought 51 cans. Jennifer adds 7 for every 6: 11 times, 77 cans. Total=118.",
    "Paityn 21 red + 25 blue = 46. Zola 17 red + 50 blue = 67. Total 113, each 56.5.",
    "Todd 5, Alisha 10, Bobby 35. Remaining 7. Total 57.",
    "First tank 49/4=12.25, second 25/3≈8.33. After eating: 11.25-8.33≈3.",
]


@dataclass
class HallucinationMetrics:
    threshold: float
    false_positive_rate: float
    true_positive_rate: float
    num_correct: int
    num_hallucinated: int

    def to_dict(self) -> Dict:
        return asdict(self)


def run_hallucination_benchmark(
    correct_chains: List[str],
    hallucinated_chains: List[str],
    threshold: float = 0.10,
) -> HallucinationMetrics:
    # Correct chains
    fp_flags = []
    for c in correct_chains:
        flag, _ = detect_hallucination_pbii(c, threshold=threshold)
        fp_flags.append(flag)
    false_positive_rate = float(sum(fp_flags) / len(fp_flags)) if fp_flags else 0.0

    # Hallucinated chains
    tp_flags = []
    for c in hallucinated_chains:
        flag, _ = detect_hallucination_pbii(c, threshold=threshold)
        tp_flags.append(flag)
    true_positive_rate = float(sum(tp_flags) / len(tp_flags)) if tp_flags else 0.0

    return HallucinationMetrics(
        threshold=threshold,
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        num_correct=len(correct_chains),
        num_hallucinated=len(hallucinated_chains),
    )


# Hook for REAL GSM8K generations (for labs / xAI)
# The expected JSONL format is:
# {"id": "...", "gold_answer": "...", "model_chain": "...", "is_hallucinated": true/false}
# If such a file is present, the script will run a 'verified' benchmark in addition
# to the synthetic one.

def load_gsm8k_generations(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def run_gsm8k_file_benchmark(
    path: str,
    threshold: float = 0.10,
) -> Optional[HallucinationMetrics]:
    if not os.path.exists(path):
        return None
    rows = load_gsm8k_generations(path)
    correct_chains: List[str] = []
    hallucinated_chains: List[str] = []
    for row in rows:
        chain = row.get("model_chain", "")
        is_h = bool(row.get("is_hallucinated", False))
        if is_h:
            hallucinated_chains.append(chain)
        else:
            correct_chains.append(chain)
    if not correct_chains or not hallucinated_chains:
        return None
    return run_hallucination_benchmark(
        correct_chains=correct_chains,
        hallucinated_chains=hallucinated_chains,
        threshold=threshold,
    )


# ==============================
# 3. PRIME VS COMPOSITE AUC
# ==============================

def is_prime(num: int) -> bool:
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0:
        return False
    r = int(math.sqrt(num))
    for i in range(3, r + 1, 2):
        if num % i == 0:
            return False
    return True


@dataclass
class PrimeAUCResult:
    auc: float
    num_samples: int
    primes: int
    composites: int

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_auc(labels: List[int], scores: List[float]) -> float:
    """
    Simple ROC AUC implementation (no sklearn).
    labels: 1 = positive (prime), 0 = negative (composite)
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    order = np.argsort(scores_arr)[::-1]
    labels_sorted = labels_arr[order]

    pos = int(labels_sorted.sum())
    neg = len(labels_sorted) - pos
    if pos == 0 or neg == 0:
        return 0.0

    tp = 0
    fp = 0
    prev_fp = 0
    auc = 0.0
    for lab in labels_sorted:
        if lab == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return float(auc / (pos * neg))


def run_prime_auc_benchmark(
    num_samples: int = 500,
    low: int = 2,
    high: int = 10_000,
    random_seed: int = 42,
) -> PrimeAUCResult:
    rng = np.random.default_rng(random_seed)
    numbers = rng.integers(low, high + 1, size=num_samples)
    labels = [1 if is_prime(int(n)) else 0 for n in numbers]
    # Invert PBII so that 'higher score' should correspond to 'more prime-like'
    scores = [-pbii(int(n)) for n in numbers]

    auc = compute_auc(labels, scores)

    primes = int(sum(labels))
    composites = num_samples - primes

    # Plot distribution for primes vs composites
    primes_pbii = [pbii(int(n)) for n, l in zip(numbers, labels) if l == 1]
    comps_pbii = [pbii(int(n)) for n, l in zip(numbers, labels) if l == 0]

    ensure_dir("figures")
    plt.figure(figsize=(8, 4))
    plt.hist(primes_pbii, bins=30, alpha=0.5, label="Primes")
    plt.hist(comps_pbii, bins=30, alpha=0.5, label="Composites")
    plt.xlabel("PBII score")
    plt.ylabel("Count")
    plt.title("PBII distribution: primes vs composites")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/pbii_primes_vs_composites.png")
    plt.close()

    return PrimeAUCResult(
        auc=float(auc),
        num_samples=num_samples,
        primes=primes,
        composites=composites,
    )


# ==============================
# 4. OPTIONAL Ω-SCORE CORRELATION DEMO
# ==============================

@dataclass
class OmegaCorrelationResult:
    pearson_r: float
    num_points: int

    def to_dict(self) -> Dict:
        return asdict(self)


def run_omega_correlation_demo(
    random_seed: int = 0,
) -> Optional[OmegaCorrelationResult]:
    """
    Uses OMNIA_TOTALE_v0.6 if available in the same directory.
    We build synthetic 'easy' vs 'hard' sequences and see if Ω-score
    correlates with an error indicator.
    """
    try:
        import OMNIA_TOTALE_v0_6 as omnia  # type: ignore
    except Exception:
        return None

    rng = np.random.default_rng(random_seed)
    t = np.arange(400)

    # Easy: clean sinusoid, stable regime
    series_easy = np.sin(t / 20.0)
    # Hard: sinusoid + regime shift + noise
    series_hard = np.sin(t / 20.0) + 0.8 * (t > 250) + 0.4 * rng.normal(size=t.size)

    # Build dummy series_dict for omniacausa
    base_series_dict = {
        "s1": series_easy,
        "s2": np.roll(series_easy, 2),
        "s3": rng.normal(size=t.size),
    }
    hard_series_dict = {
        "s1": series_hard,
        "s2": np.roll(series_hard, 2),
        "s3": rng.normal(size=t.size),
    }

    # Choose two test integers (prime vs composite) for the base lens
    n_prime = 173
    n_comp = 180

    scores: List[float] = []
    labels: List[int] = []  # 0 = easy, 1 = hard

    # Easy regime
    for n in [n_prime, n_comp] * 20:
        res = omnia.omnia_totale_score(
            n=n,
            series=series_easy,
            series_dict=base_series_dict,
        )
        scores.append(float(res.omega_score))
        labels.append(0)

    # Hard regime
    for n in [n_prime, n_comp] * 20:
        res = omnia.omnia_totale_score(
            n=n,
            series=series_hard,
            series_dict=hard_series_dict,
        )
        scores.append(float(res.omega_score))
        labels.append(1)

    if not scores or not labels:
        return None

    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=float)
    scores_centered = scores_arr - scores_arr.mean()
    labels_centered = labels_arr - labels_arr.mean()
    num = float(np.sum(scores_centered * labels_centered))
    den = math.sqrt(
        float(np.sum(scores_centered ** 2) * np.sum(labels_centered ** 2))
    )
    r = num / den if den > 0 else 0.0

    return OmegaCorrelationResult(
        pearson_r=float(r),
        num_points=len(scores),
    )


# ==============================
# 5. MAIN: RUN ALL BENCHMARKS
# ==============================

def main():
    ensure_dir("results")

    # 1) Synthetic hallucination benchmark
    hallu_metrics_synth = run_hallucination_benchmark(
        SYNTH_CORRECT_CHAINS,
        SYNTH_HALLUCINATED_CHAINS,
        threshold=0.10,
    )

    # 2) Optional REAL GSM8K generations from JSONL (if present)
    gsm8k_path = os.environ.get("OMNIA_GSM8K_FILE", "gsm8k_generations.jsonl")
    hallu_metrics_real = run_gsm8k_file_benchmark(
        gsm8k_path,
        threshold=0.10,
    )

    # 3) Prime vs composite AUC
    prime_auc = run_prime_auc_benchmark(
        num_samples=500,
        low=2,
        high=10_000,
        random_seed=42,
    )

    # 4) Optional Ω-score correlation
    omega_corr = run_omega_correlation_demo()

    # Collect results
    results: Dict[str, Dict] = {
        "hallucination_synthetic": hallu_metrics_synth.to_dict(),
        "prime_auc": prime_auc.to_dict(),
    }
    if hallu_metrics_real is not None:
        results["hallucination_gsm8k_real"] = hallu_metrics_real.to_dict()
    if omega_corr is not None:
        results["omega_correlation_demo"] = omega_corr.to_dict()

    with open("results/OMNIA_TOTALE_BENCHMARK_v1.0.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== OMNIA_TOTALE BENCHMARK v1.0 ===")
    print("Synthetic hallucination detection:")
    print(hallu_metrics_synth)
    if hallu_metrics_real is not None:
        print("Real GSM8K JSONL benchmark:")
        print(hallu_metrics_real)
    else:
        print("Real GSM8K JSONL file not found -> only synthetic benchmark run.")
    print("Prime vs composite AUC:")
    print(prime_auc)
    if omega_corr is not None:
        print("Ω-score correlation demo:")
        print(omega_corr)
    else:
        print("OMNIA_TOTALE_v0.6 not found -> Ω demo skipped.")


if __name__ == "__main__":
    main()