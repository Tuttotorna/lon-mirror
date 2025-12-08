"""
gsm8k_benchmark_demo.py

Benchmark demo for OMNIA_TOTALE / PBII:
- Synthetic hallucination detection on GSM8K-like reasoning chains.
- Prime vs composite separation with PBII-based score + AUC estimate.

NOTE: All metrics here (e.g. “71% hallucination reduction”, “AUC ~0.98”)
are synthetic placeholders for demonstration and review purposes only.
Replace them with real metrics if you run full-scale experiments.

Author: Massimiliano Brighindi (MB-X.01 / OMNIA_TOTALE)
"""

import math
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PBII DEMO IMPLEMENTATION (aligned with OMNIA_TOTALE concept)
# ============================================================

def digits_in_base(n: int, b: int):
    """Return digits of n in base b (MSB first)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if b <= 1:
        raise ValueError("base must be >= 2")
    if n == 0:
        return [0]
    res = []
    while n > 0:
        res.append(n % b)
        n //= b
    return res[::-1]


def sigma_b(n: int, b: int) -> float:
    """
    Base Symmetry Score (semplificata per il benchmark):

    - calcola entropia normalizzata dei digit in base b
    - penalizza rappresentazioni lunghe e rumorose
    - bonus se n è multiplo della base (struttura evidente)
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

    length_term = (1.0 - Hn) / L
    div_bonus = 0.5 if n % b == 0 else 0.0
    return length_term + div_bonus


def sigma_avg(n: int, bases) -> float:
    """Average sigma_b over a set of bases."""
    return sum(sigma_b(n, b) for b in bases) / len(bases)


def saturation(n: int, bases, window: int = 100) -> float:
    """
    Local saturation around n: media di sigma_avg per i compositi
    nell’intervallo [n-window, n).
    """
    start = max(2, n - window)
    comps = []
    for k in range(start, n):
        if k <= 3:
            continue
        # semplice test di composito
        if any(k % d == 0 for d in range(2, int(math.sqrt(k)) + 1)):
            comps.append(k)

    if not comps:
        return 0.0

    vals = [sigma_avg(k, bases) for k in comps]
    return sum(vals) / len(vals)


def pbii(n: int, bases=None, window: int = 100) -> float:
    """
    Prime Base Instability Index (demo):

    PBII(n) = saturation(neighborhood) - sigma_avg(n)

    Valori più alti ~ n più "prime-like" (instabile rispetto ai vicini).
    """
    if bases is None:
        bases = [2, 3, 5, 7, 11, 13, 17, 19]
    sat = saturation(n, bases, window=window)
    sig = sigma_avg(n, bases)
    return sat - sig


# ============================================================
# Utility: estrazione numeri da una chain testuale
# ============================================================

def extract_numbers(chain_text: str):
    """Estrae interi positivi >1 dal testo."""
    return [int(num) for num in re.findall(r"\b\d+\b", chain_text) if int(num) > 1]


# ============================================================
# Sample GSM8K-like chains (corrette vs "allucinate")
# ============================================================

correct_chains = [
    """Sam skipped 16 times per round. Jeff: r1=15, r2=13, r3=20, r4=8. Total Jeff=56, avg=14.""",
    """Mark bought 50 cans. Jennifer adds 6 every 5: 10 times, 60 cans. Total 40+60=100.""",
    """Paityn 20 red +24 blue=44. Zola 16 red +48 blue=64. Total 108, each 54.""",
    """Todd 4, Alisha 8, Bobby 27. Remaining 6. Total 45.""",
    """First tank 48/3=16 fish, second 24/2=12. After eat: 15-12=3 more in first.""",
]

hallucinated_chains = [
    """Sam skipped 17 times per round. Jeff: r1=16, r2=14, r3=21, r4=9. Total Jeff=60, avg=15.""",
    """Mark bought 51 cans. Jennifer adds 7 every 6: 11 times, 77 cans. Total 41+77=118.""",
    """Paityn 21 red +25 blue=46. Zola 17 red +50 blue=67. Total 113, each 56.5.""",
    """Todd 5, Alisha 10, Bobby 35. Remaining 7. Total 57.""",
    """First tank 49/4=12.25 fish, second 25/3≈8.33. After eat: 11.25-8.33≈3.""",
]


# ============================================================
# Benchmark 1 – Hallucination detection (demo)
# ============================================================

def detect_hallucination(chain_text: str, threshold: float = 0.10) -> bool:
    """
    Flagga la chain se la media PBII sui numeri estratti supera la soglia.
    In pratica: alta instabilità strutturale => potenziale allucinazione.
    """
    nums = extract_numbers(chain_text)
    if not nums:
        return False
    scores = [pbii(n) for n in nums]
    avg_pbii = float(np.mean(scores))
    return avg_pbii > threshold


def run_hallucination_demo():
    correct_flags = [detect_hallucination(c) for c in correct_chains]
    halluc_flags = [detect_hallucination(c) for c in hallucinated_chains]

    fp_rate = sum(correct_flags) / len(correct_flags)
    det_rate = sum(halluc_flags) / len(hallucinated_chains)

    print("=== Benchmark 1: Hallucination demo ===")
    print(f"False positives on correct chains: {fp_rate * 100:.1f}%")
    print(f"Detection rate on hallucinated chains: {det_rate * 100:.1f}%")
    print(
        "NOTE: This is a small synthetic demo. "
        "The placeholder claim '71% hallucination reduction on >50-step GSM8K chains' "
        "must be backed by real large-scale runs before being reported as a true metric."
    )
    print()


# ============================================================
# Benchmark 2 – Prime vs composite AUC (demo)
# ============================================================

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


def compute_auc(labels, scores) -> float:
    """
    Simple ROC AUC via rank statistic (no sklearn).
    labels: 1 = positive (prime), 0 = negative (composite)
    scores: higher score => more likely positive
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    order = np.argsort(scores)[::-1]
    labels = labels[order]

    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return 0.0

    tp = 0
    fp = 0
    prev_fp = 0
    auc = 0.0
    for lbl in labels:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return auc / (pos * neg)


def run_prime_auc_demo():
    np.random.seed(42)
    numbers = np.random.randint(2, 2000, 200)
    labels = [1 if is_prime(n) else 0 for n in numbers]
    # PBII: primes tend to have lower PBII; invert sign to get high score for primes
    scores = [-pbii(n) for n in numbers]

    auc = compute_auc(labels, scores)
    print("=== Benchmark 2: Prime vs composite AUC demo ===")
    print(f"Estimated AUC (synthetic sample): {auc:.3f}")
    print(
        "NOTE: This AUC is based on a small synthetic sample. "
        "Use larger ranges and more points for a stable estimate."
    )

    primes_pbii = [pbii(n) for n, lbl in zip(numbers, labels) if lbl == 1]
    comps_pbii = [pbii(n) for n, lbl in zip(numbers, labels) if lbl == 0]

    plt.figure(figsize=(10, 5))
    plt.hist(primes_pbii, bins=20, alpha=0.5, label="Primes")
    plt.hist(comps_pbii, bins=20, alpha=0.5, label="Composites")
    plt.xlabel("PBII score")
    plt.ylabel("Count")
    plt.title("PBII distribution: primes vs composites (demo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pbii_distribution_demo.png")
    # plt.show()  # opzionale in ambiente notebook
    print("Saved histogram to pbii_distribution_demo.png\n")


# ============================================================
# MAIN
# ============================================================

def main():
    run_hallucination_demo()
    run_prime_auc_demo()


if __name__ == "__main__":
    main()