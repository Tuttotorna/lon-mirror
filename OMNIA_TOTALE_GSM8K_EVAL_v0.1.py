"""
OMNIA_TOTALE_GSM8K_EVAL_v0.1.py

Benchmark demo for OMNIA_TOTALE / PBII on GSM8K-style reasoning chains.

- Carica il dataset GSM8K (split "train") da HuggingFace.
- Usa le soluzioni ufficiali come catene "corrette".
- Genera versioni "allucinate" corrompendo numeri nelle catene.
- Applica PBII ai numeri in ogni catena e misura:
    * False Positive Rate (catene corrette marcate come instabili)
    * Detection Rate (catene corrotte marcate come instabili)
- Calcola anche AUC prime vs. composite usando PBII.
- Salva due grafici:
    * pbii_gsm8k_chain_means.png   (distribuzione PBII catene corrette vs corrotte)
    * pbii_primes_vs_composites.png (distribuzione PBII primi vs composti)

NOTE IMPORTANTI:
- Questo script usa una corruzione SINTETICA delle catene GSM8K per simulare allucinazioni.
  Per metriche "reali" va sostituito il generatore di catene allucinate con output
  di un vero modello LLM (es. file JSONL con reasonings del modello).
- Dipendenze:
    pip install datasets numpy matplotlib
"""

import math
import re
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset


# ============================================================
# 1. PBII CORE (versione standalone, coerente con OMNIA_TOTALE)
# ============================================================

def digits_in_base(n: int, b: int) -> List[int]:
    """Rappresenta n in base b (MSB first)."""
    if n == 0:
        return [0]
    res: List[int] = []
    while n > 0:
        res.append(n % b)
        n //= b
    return res[::-1]


def sigma_b(n: int, b: int) -> float:
    """
    Base Symmetry Score minimale, compatibile con PBII.

    - Entropia normalizzata delle cifre in base b.
    - Penalizza rappresentazioni lunghe e rumorose.
    - Bonus se n è multiplo della base.
    """
    if n < 0:
        n = abs(n)

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
    # term principale: più alto = più struttura
    base_term = (1.0 - Hn) / L
    return base_term + bonus


def sigma_avg(n: int, bases: List[int]) -> float:
    """Media di sigma_b su più basi."""
    vals = [sigma_b(n, b) for b in bases]
    return float(sum(vals) / len(vals)) if vals else 0.0


def saturation(n: int, bases: List[int], window: int = 100) -> float:
    """
    Saturazione di struttura dei composti in un intorno prima di n.

    Usa solo numeri composti nell'intervallo [n-window, n).
    """
    start = max(2, n - window)
    comps: List[int] = []
    for k in range(start, n):
        if not is_prime(k):
            comps.append(k)

    if not comps:
        return 0.0

    vals = [sigma_avg(k, bases) for k in comps]
    return float(sum(vals) / len(vals))


def pbii(n: int,
         bases: List[int] = None,
         window: int = 100) -> float:
    """
    Prime Base Instability Index.

    PBII(n) = saturation(composites) - sigma_avg(n)

    - Valori alti: struttura "debole" rispetto ai composti vicini
      (tipicamente associata a numeri primi).
    - Valori bassi o negativi: più struttura, più simile ai composti.
    """
    if bases is None:
        bases = [2, 3, 5, 7, 11, 13, 17, 19]

    sat = saturation(n, bases, window)
    sig_n = sigma_avg(n, bases)
    return float(sat - sig_n)


# ============================================================
# 2. UTILITY: PRIMI / ESTRARRE NUMERI / CORRUZIONE CATENE
# ============================================================

def is_prime(num: int) -> bool:
    """Test semplice di primalità (sufficiente per il benchmark)."""
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    limit = int(math.sqrt(num)) + 1
    for i in range(3, limit, 2):
        if num % i == 0:
            return False
    return True


def extract_numbers(text: str) -> List[int]:
    """
    Estrae interi positivi dal testo.

    GSM8K contiene formati tipo '#### 42' alla fine della risposta:
    qui raccogliamo tutti i numeri > 1 nella chain.
    """
    nums = [int(m) for m in re.findall(r"\b\d+\b", text)]
    return [n for n in nums if n > 1]


def corrupt_number(n: int, rng: np.random.Generator) -> int:
    """
    Crea una versione "allucinata" di un numero:
    - a volte + o - un offset
    - a volte moltiplicato o diviso approssimativamente
    """
    if n <= 2:
        return n + 1
    mode = rng.integers(0, 3)
    if mode == 0:
        # offset
        delta = rng.integers(1, max(2, n // 5 + 2))
        return n + int(delta)
    elif mode == 1:
        # riduzione
        delta = rng.integers(1, max(2, n // 5 + 2))
        return max(2, n - int(delta))
    else:
        # fattore
        factor = rng.uniform(0.5, 1.8)
        return max(2, int(round(n * factor)))


def corrupt_chain(text: str,
                  corruption_prob: float,
                  rng: np.random.Generator) -> str:
    """
    Crea una "catena allucinata" alterando parte dei numeri nel testo.

    Mantiene la struttura linguistica, ma cambia
    alcuni valori numerici per introdurre incoerenza.
    """
    def repl(match):
        original = int(match.group(0))
        if original <= 1:
            return match.group(0)
        if rng.random() < corruption_prob:
            return str(corrupt_number(original, rng))
        return match.group(0)

    return re.sub(r"\b\d+\b", repl, text)


def chain_pbii_mean(text: str) -> Tuple[float, int]:
    """
    Calcola la media PBII sui numeri presenti nella catena.

    Ritorna:
    - pbii_mean: media PBII (0.0 se nessun numero valido)
    - count: quanti numeri sono stati considerati
    """
    nums = extract_numbers(text)
    if not nums:
        return 0.0, 0
    scores = [pbii(n) for n in nums]
    return float(np.mean(scores)), len(nums)


def detect_hallucination(pbii_mean: float,
                         threshold: float) -> bool:
    """
    Regola di decisione: catena "instabile" se PBII medio supera threshold.
    """
    return pbii_mean > threshold


# ============================================================
# 3. BENCHMARK GSM8K (SINTETICO MA SU DATASET REALE)
# ============================================================

def evaluate_gsm8k(num_samples: int = 300,
                   corruption_prob: float = 0.4,
                   threshold: float = 0.10,
                   seed: int = 0) -> dict:
    """
    Esegue benchmark sintetico su subset GSM8K.

    - num_samples: quanti esempi usare dal train set.
    - corruption_prob: probabilità di corrompere ogni numero nella chain.
    - threshold: soglia PBII per marcare allucinazione.
    """
    rng = np.random.default_rng(seed)

    print("Carico dataset GSM8K (split=train, config=main)...")
    ds = load_dataset("gsm8k", "main", split="train")

    if num_samples > len(ds):
        num_samples = len(ds)

    idx = rng.choice(len(ds), size=num_samples, replace=False)
    idx = list(idx)

    pbii_correct: List[float] = []
    pbii_corrupted: List[float] = []
    flags_correct: List[bool] = []
    flags_corrupted: List[bool] = []
    counts_correct: List[int] = []
    counts_corrupted: List[int] = []

    for i in idx:
        ex = ds[int(i)]
        answer_text: str = ex["answer"]

        # catena corretta
        pb_mean_corr, cnt_corr = chain_pbii_mean(answer_text)

        # catena allucinata (numeri corrotti)
        halluc_text = corrupt_chain(answer_text, corruption_prob, rng)
        pb_mean_hall, cnt_hall = chain_pbii_mean(halluc_text)

        pbii_correct.append(pb_mean_corr)
        pbii_corrupted.append(pb_mean_hall)
        counts_correct.append(cnt_corr)
        counts_corrupted.append(cnt_hall)

        flags_correct.append(detect_hallucination(pb_mean_corr, threshold))
        flags_corrupted.append(detect_hallucination(pb_mean_hall, threshold))

    false_positive_rate = float(np.mean(flags_correct))
    detection_rate = float(np.mean(flags_corrupted))

    avg_nums_correct = float(np.mean(counts_correct))
    avg_nums_corrupted = float(np.mean(counts_corrupted))

    print("\n=== GSM8K synthetic hallucination benchmark ===")
    print(f"Samples used:                  {num_samples}")
    print(f"Corruption probability:        {corruption_prob:.2f}")
    print(f"PBII threshold:                {threshold:.3f}")
    print(f"Avg numbers per correct chain: {avg_nums_correct:.2f}")
    print(f"Avg numbers per corrupt chain: {avg_nums_corrupted:.2f}")
    print(f"False positive rate (correct): {false_positive_rate * 100:.1f}%")
    print(f"Detection rate (corrupted):    {detection_rate * 100:.1f}%")

    # Plot distribuzioni PBII (mean per chain)
    plt.figure()
    plt.hist(pbii_correct, bins=30, alpha=0.5, label="Correct chains")
    plt.hist(pbii_corrupted, bins=30, alpha=0.5, label="Corrupted chains")
    plt.xlabel("Mean PBII per chain")
    plt.ylabel("Count")
    plt.title("GSM8K: PBII mean distribution (correct vs corrupted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pbii_gsm8k_chain_means.png")
    print("Saved plot: pbii_gsm8k_chain_means.png")

    return {
        "num_samples": num_samples,
        "corruption_prob": corruption_prob,
        "threshold": threshold,
        "false_positive_rate": false_positive_rate,
        "detection_rate": detection_rate,
        "pbii_correct": pbii_correct,
        "pbii_corrupted": pbii_corrupted,
    }


# ============================================================
# 4. AUC PRIME VS COMPOSITES (COMPLEMENTO)
# ============================================================

def compute_auc(labels: List[int], scores: List[float]) -> float:
    """
    AUC semplice (ROC) senza librerie esterne.

    labels: 1 = prime, 0 = composite
    scores: valore continuo (più alto = più "prime-like")
    """
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    order = np.argsort(scores_arr)[::-1]
    labels_sorted = labels_arr[order]

    pos = int(np.sum(labels_sorted))
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


def benchmark_primes_vs_composites(seed: int = 0) -> float:
    """
    Usa PBII per separare numeri primi vs composti e calcola AUC.
    """
    rng = np.random.default_rng(seed)
    nums = rng.integers(2, 2000, size=200)

    labels = [1 if is_prime(int(n)) else 0 for n in nums]
    # score alto = più "prime-like": usiamo -PBII (primi hanno PBII basso)
    scores = [-pbii(int(n)) for n in nums]

    auc = compute_auc(labels, scores)
    print("\n=== PBII prime vs composite benchmark ===")
    print(f"AUC (PBII-based separation): {auc:.3f}")

    # distribuzione PBII
    primes_pbii = [pbii(int(n)) for n, lab in zip(nums, labels) if lab == 1]
    comps_pbii = [pbii(int(n)) for n, lab in zip(nums, labels) if lab == 0]

    plt.figure()
    plt.hist(primes_pbii, bins=30, alpha=0.5, label="Primes")
    plt.hist(comps_pbii, bins=30, alpha=0.5, label="Composites")
    plt.xlabel("PBII score")
    plt.ylabel("Count")
    plt.title("PBII distribution: primes vs composites")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pbii_primes_vs_composites.png")
    print("Saved plot: pbii_primes_vs_composites.png")

    return auc


# ============================================================
# 5. MAIN
# ============================================================

def main():
    # 1) Benchmark GSM8K sintetico
    gsm8k_results = evaluate_gsm8k(
        num_samples=300,
        corruption_prob=0.4,
        threshold=0.10,
        seed=0,
    )

    # 2) Benchmark primi vs composti
    auc = benchmark_primes_vs_composites(seed=0)

    print("\nSummary:")
    print(f"- GSM8K synthetic detection rate: {gsm8k_results['detection_rate'] * 100:.1f}%")
    print(f"- GSM8K synthetic false positives: {gsm8k_results['false_positive_rate'] * 100:.1f}%")
    print(f"- Prime/composite AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
```0