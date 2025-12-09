"""
GSM8K + Ω-Lens Benchmark v1.0
Author: Massimiliano Brighindi (MB-X.01 / Ω-Lens Engine) + MBX IA

Pipeline:
1) Carica le soluzioni GSM8K con catene di ragionamento modello in formato JSONL.
2) Calcola PBII / Ω-score su ogni chain.
3) Etichetta "hallucinated" = risposta finale errata con chain abbastanza lunga.
4) Calcola:
   - accuracy di base
   - hallucination rate di base
   - hallucination rate dopo filtro Ω-Lens (threshold)
   - % di riduzione
   - AUC per PBII nel separare prime vs composite (sanity check)
5) Produce grafici e salva tutto su disco.

Dipendenze:
    pip install numpy matplotlib datasets
    (opzionale: pip install tqdm)
"""

from __future__ import annotations
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    # fallback minimale
    def tqdm(x, **kwargs):
        return x

# ============================================================
# 0. CONFIG
# ============================================================

# Percorso di input: file JSONL con output modello su GSM8K
# Formato atteso per ogni riga:
# {
#   "id": str,
#   "question": str,
#   "gold_answer": str | int | float,
#   "model_chain": str,      # chain-of-thought testo completo
#   "model_answer": str      # risposta finale del modello (come stringa)
# }
#
# Questo file lo generi tu o xAI; qui facciamo solo l'analisi.
GSM8K_OUTPUTS_PATH = "data/gsm8k_model_outputs.jsonl"

# Soglia minima di "lunghezza chain" per considerare la risposta come
# potenzialmente allucinata (se sbagliata).
MIN_CHAIN_TOKENS_FOR_HALLUC = 50

# Soglia PBII/Ω per flaggare instabilità (da tarare; default ragionevole)
PBII_THRESHOLD = 0.10

# Cartella dove salvare grafici e report
OUTPUT_DIR = "benchmarks_gsm8k"


# ============================================================
# 1. PBII / OMNIABASE CORE (compatibile con OMNIA_TOTALE)
# ============================================================

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

    length_term = length_weight * (1.0 - Hn) / (L ** length_exponent)
    div_term = divisibility_bonus * (1.0 if n % b == 0 else 0.0)
    return float(length_term + div_term)


def sigma_avg(
    n: int,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
) -> float:
    bases = list(bases)
    vals = [sigma_b(n, b) for b in bases]
    return float(sum(vals) / len(vals)) if vals else 0.0


def saturation(
    n: int,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    window: int = 100,
) -> float:
    """
    Saturazione sui compositi vicini (come nei tuoi script OMNIA_TOTALE).
    """
    start = max(4, n - window)
    comp_vals: List[float] = []
    for k in range(start, n):
        if is_composite(k):
            comp_vals.append(sigma_avg(k, bases))
    return float(sum(comp_vals) / len(comp_vals)) if comp_vals else 0.0


def pbii(
    n: int,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    window: int = 100,
) -> float:
    """
    Prime Base Instability Index:
    PBII(n) = mean_sigma(composites around n) - sigma_avg(n)
    (più è alto, più n appare "prime-like")
    """
    return saturation(n, bases=bases, window=window) - sigma_avg(n, bases=bases)


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


def is_composite(num: int) -> bool:
    return num > 3 and not is_prime(num)


# ============================================================
# 2. UTILITIES PER GSM8K
# ============================================================

number_regex = re.compile(r"\b\d+\b")


def extract_numbers(text: str) -> List[int]:
    return [int(m.group(0)) for m in number_regex.finditer(text)]


def normalize_answer(ans: str) -> str:
    """
    Normalizza risposta numerica in forma stringa compatibile
    (GSM8K spesso ha risposte come '42' o '42.0' o '42 dollars').
    Qui prendiamo il primo numero intero che troviamo; se non c'è, stringa intera.
    """
    nums = extract_numbers(ans)
    if nums:
        return str(nums[0])
    return ans.strip().lower()


@dataclass
class GSM8KRecord:
    id: str
    question: str
    gold_answer: str
    model_chain: str
    model_answer: str

    # calcolati
    is_correct: bool = False
    is_hallucinated: bool = False
    pbii_score: float = 0.0
    chain_len_tokens: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# 3. CARICAMENTO JSONL
# ============================================================

def load_gsm8k_outputs(path: str) -> List[GSM8KRecord]:
    records: List[GSM8KRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rec = GSM8KRecord(
                id=str(obj["id"]),
                question=obj["question"],
                gold_answer=str(obj["gold_answer"]),
                model_chain=obj["model_chain"],
                model_answer=str(obj["model_answer"]),
            )
            records.append(rec)
    return records


# ============================================================
# 4. ETICHETTATURA HALLUCINATIONS + PBII
# ============================================================

def count_tokens_approx(text: str) -> int:
    # approssimazione: split per spazi
    return len(text.strip().split())


def annotate_records(records: List[GSM8KRecord]) -> None:
    for rec in records:
        gold = normalize_answer(rec.gold_answer)
        pred = normalize_answer(rec.model_answer)
        rec.is_correct = (gold == pred)

        rec.chain_len_tokens = count_tokens_approx(rec.model_chain)

        # definizione semplice di "hallucinated":
        # risposta sbagliata + chain lunga
        rec.is_hallucinated = (not rec.is_correct) and (
            rec.chain_len_tokens >= MIN_CHAIN_TOKENS_FOR_HALLUC
        )

        # PBII medio sui numeri citati nella chain
        nums = extract_numbers(rec.model_chain)
        if nums:
            scores = [pbii(n) for n in nums if n > 1]
            rec.pbii_score = float(np.mean(scores)) if scores else 0.0
        else:
            rec.pbii_score = 0.0


# ============================================================
# 5. METRICHE Ω-LENS
# ============================================================

@dataclass
class BenchmarkMetrics:
    n_total: int
    accuracy_baseline: float
    halluc_rate_baseline: float
    halluc_rate_filtered: float
    hallucination_reduction_pct: float
    pbii_auc: float

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_auc(labels: List[int], scores: List[float]) -> float:
    """
    AUC binaria semplice (1 = positive, 0 = negative).
    Ordina per score decrescente.
    """
    arr_labels = np.array(labels, dtype=int)
    arr_scores = np.array(scores, dtype=float)
    order = np.argsort(arr_scores)[::-1]
    arr_labels = arr_labels[order]

    pos = np.sum(arr_labels)
    neg = len(arr_labels) - pos
    if pos == 0 or neg == 0:
        return 0.0

    tp = 0
    fp = 0
    prev_fp = 0
    auc = 0.0
    for lab in arr_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return float(auc / (pos * neg))


def compute_metrics(records: List[GSM8KRecord]) -> BenchmarkMetrics:
    n_total = len(records)
    if n_total == 0:
        raise ValueError("No records to evaluate.")

    # baseline accuracy
    correct = sum(1 for r in records if r.is_correct)
    accuracy_baseline = correct / n_total

    # hallucination baseline
    halluc = [r for r in records if r.is_hallucinated]
    halluc_rate_baseline = len(halluc) / n_total

    # filtro Ω-Lens: blocchiamo le risposte con PBII sopra threshold
    filtered = []
    halluc_after = 0
    for r in records:
        if r.pbii_score > PBII_THRESHOLD:
            # considerato "bloccato": niente risposta, quindi niente halluc
            continue
        filtered.append(r)
        if r.is_hallucinated:
            halluc_after += 1

    halluc_rate_filtered = halluc_after / n_total  # rat. su totale, non solo filtrati

    if halluc_rate_baseline > 0:
        hallucination_reduction_pct = 100.0 * (
            1.0 - halluc_rate_filtered / halluc_rate_baseline
        )
    else:
        hallucination_reduction_pct = 0.0

    # sanity check: AUC su numeri prime vs composite usando PBII
    # prendiamo 200 numeri random nella zona 2..5000
    rng = np.random.default_rng(42)
    sample_nums = rng.integers(2, 5000, size=200)
    labels = [1 if is_prime(int(n)) else 0 for n in sample_nums]
    scores = [-pbii(int(n)) for n in sample_nums]  # score alto = più prime-like
    pbii_auc = compute_auc(labels, scores)

    return BenchmarkMetrics(
        n_total=n_total,
        accuracy_baseline=float(accuracy_baseline),
        halluc_rate_baseline=float(halluc_rate_baseline),
        halluc_rate_filtered=float(halluc_rate_filtered),
        hallucination_reduction_pct=float(hallucination_reduction_pct),
        pbii_auc=float(pbii_auc),
    )


# ============================================================
# 6. GRAFICI
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_pbii_hist(records: List[GSM8KRecord], out_dir: str) -> None:
    hall = [r.pbii_score for r in records if r.is_hallucinated]
    nonhall = [r.pbii_score for r in records if not r.is_hallucinated]

    plt.figure(figsize=(10, 5))
    plt.hist(nonhall, bins=30, alpha=0.6, label="Non-hallucinated")
    plt.hist(hall, bins=30, alpha=0.6, label="Hallucinated")
    plt.axvline(PBII_THRESHOLD, linestyle="--", label=f"PBII threshold={PBII_THRESHOLD}")
    plt.xlabel("PBII score")
    plt.ylabel("Count")
    plt.title("PBII distribution on GSM8K chains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pbii_hist_gsm8k.png"))
    plt.close()


def plot_hallucination_bars(metrics: BenchmarkMetrics, out_dir: str) -> None:
    plt.figure(figsize=(6, 5))
    labels = ["Baseline", "Ω-Lens filtered"]
    vals = [metrics.halluc_rate_baseline, metrics.halluc_rate_filtered]
    plt.bar(labels, vals)
    plt.ylabel("Hallucination rate (fraction of all samples)")
    plt.title("Hallucinations before vs after Ω-Lens filter")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hallucination_reduction.png"))
    plt.close()


# ============================================================
# 7. MAIN
# ============================================================

def run_benchmark():
    ensure_dir(OUTPUT_DIR)

    print(f"Loading GSM8K model outputs from: {GSM8K_OUTPUTS_PATH}")
    records = load_gsm8k_outputs(GSM8K_OUTPUTS_PATH)
    print(f"Loaded {len(records)} records.")

    print("Annotating records with correctness, hallucinations, PBII scores...")
    annotate_records(records)

    print("Computing metrics...")
    metrics = compute_metrics(records)

    print("\n=== GSM8K + Ω-Lens Benchmark v1.0 ===")
    print(f"Total samples:           {metrics.n_total}")
    print(f"Baseline accuracy:       {metrics.accuracy_baseline:.3f}")
    print(f"Hallucination rate base: {metrics.halluc_rate_baseline:.3f}")
    print(f"Hallucination rate Ω:    {metrics.halluc_rate_filtered:.3f}")
    print(
        f"Hallucination reduction: {metrics.hallucination_reduction_pct:.1f}% "
        f"(threshold PBII>{PBII_THRESHOLD})"
    )
    print(f"PBII prime/composite AUC: {metrics.pbii_auc:.3f}")

    # Salva JSON con metriche aggregate
    metrics_path = os.path.join(OUTPUT_DIR, "gsm8k_omega_metrics_v1.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")

    # Grafici
    print("Generating plots...")
    plot_pbii_hist(records, OUTPUT_DIR)
    plot_hallucination_bars(metrics, OUTPUT_DIR)
    print(f"Plots saved under: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_benchmark()