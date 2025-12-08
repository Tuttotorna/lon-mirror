"""
OMNIA_TOTALE_LLM_INTEGRATION_v0.1
PBII + OMNIA_TOTALE pipeline for real LLM chains on GSM8K.

Requires:
    pip install datasets numpy
Optional (example integration):
    pip install openai
"""

import math
import os
import re
from dataclasses import dataclass
from typing import List, Iterable, Tuple, Dict, Any

import numpy as np
from datasets import load_dataset


# =========================
# 1. PBII CORE (stessa logica del file full)
# =========================

def digits_in_base(n: int, b: int) -> List[int]:
    if n == 0:
        return [0]
    res = []
    while n > 0:
        res.append(n % b)
        n //= b
    return res[::-1]


def sigma_b(n: int, b: int) -> float:
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
    bases = list(bases)
    start = max(2, n - W)
    comps = []
    for k in range(start, n):
        if any(k % d == 0 for d in range(2, int(math.sqrt(k)) + 1)):
            comps.append(k)
    if not comps:
        return 0.0
    vals = [sigma_avg(k, bases) for k in comps]
    return sum(vals) / len(vals)


def pbii(n: int,
         bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
         W: int = 100) -> float:
    bases = list(bases)
    sat = saturation(n, bases, W=W)
    sig = sigma_avg(n, bases)
    return sat - sig


# =========================
# 2. UTILS PBII + GSM8K
# =========================

NUMBER_RE = re.compile(r"\b\d+\b")


def extract_numbers(text: str) -> List[int]:
    return [int(m.group()) for m in NUMBER_RE.finditer(text) if int(m.group()) > 1]


def avg_pbii_in_text(text: str) -> float:
    nums = extract_numbers(text)
    if not nums:
        return 0.0
    scores = [pbii(n) for n in nums]
    return float(np.mean(scores))


def parse_gsm8k_answer(answer_text: str) -> int | None:
    """
    GSM8K answer format: chain + final '#### 42'.
    Estrae l'ultimo intero dopo '####'.
    """
    if "####" not in answer_text:
        nums = extract_numbers(answer_text)
        return nums[-1] if nums else None
    tail = answer_text.split("####")[-1]
    nums = extract_numbers(tail)
    return nums[-1] if nums else None


# =========================
# 3. LLM WRAPPER (STUB + ESEMPIO)
# =========================

def generate_chain_stub(question: str) -> str:
    """
    STUB di default: restituisce solo la domanda.
    Sostituisci questa funzione con una chiamata al tuo LLM interno.

    Esempio (commentato) per OpenAI-style client:

    from openai import OpenAI
    client = OpenAI()

    def generate_chain_stub(question: str) -> str:
        prompt = (
            "You are a step-by-step math solver. "
            "Solve the following GSM8K problem with explicit reasoning, "
            "then end with '#### <final_answer>'.\\n\\n"
            f"Question: {question}"
        )
        resp = client.chat.completions.create(
            model=\"gpt-4.1-mini\",
            messages=[{\"role\": \"user\", \"content\": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    """
    return f"[STUB ONLY – replace with LLM call]\nQuestion: {question}"


@dataclass
class LLMExampleResult:
    idx: int
    question: str
    gold_answer: int | None
    llm_chain: str
    llm_answer: int | None
    correct: bool
    avg_pbii: float


@dataclass
class LLMEvalStats:
    n: int
    n_correct: int
    acc: float
    mean_pbii_correct: float
    mean_pbii_incorrect: float
    corr_pbii_vs_error: float | None  # correlazione (più alto = più instabile)


# =========================
# 4. EVAL PIPELINE
# =========================

def run_llm_gsm8k_eval(
    max_samples: int = 50,
    start_index: int = 0,
) -> Tuple[LLMEvalStats, List[LLMExampleResult]]:
    """
    Esegue:
      - load GSM8K 'main'
      - per i primi max_samples a partire da start_index:
          * genera chain con LLM
          * calcola PBII medio
          * confronta con risposta gold
    Ritorna stats globali + lista di esempi.
    """
    ds = load_dataset("openai/gsm8k", "main")["train"]
    end_index = min(start_index + max_samples, len(ds))

    results: List[LLMExampleResult] = []

    for i in range(start_index, end_index):
        row = ds[i]
        q = row["question"]
        gold = parse_gsm8k_answer(row["answer"])

        chain = generate_chain_stub(q)
        llm_ans = parse_gsm8k_answer(chain)
        correct = (llm_ans is not None and gold is not None
                   and int(llm_ans) == int(gold))

        score = avg_pbii_in_text(chain)

        results.append(
            LLMExampleResult(
                idx=i,
                question=q,
                gold_answer=gold,
                llm_chain=chain,
                llm_answer=llm_ans,
                correct=bool(correct),
                avg_pbii=score,
            )
        )

    if not results:
        return LLMEvalStats(
            n=0,
            n_correct=0,
            acc=0.0,
            mean_pbii_correct=0.0,
            mean_pbii_incorrect=0.0,
            corr_pbii_vs_error=None,
        ), results

    n = len(results)
    n_correct = sum(1 for r in results if r.correct)
    acc = n_correct / n

    pbii_correct = [r.avg_pbii for r in results if r.correct]
    pbii_incorrect = [r.avg_pbii for r in results if not r.correct]

    mean_pc = float(np.mean(pbii_correct)) if pbii_correct else 0.0
    mean_pi = float(np.mean(pbii_incorrect)) if pbii_incorrect else 0.0

    # correlazione PBII vs errore (error=1 se sbagliato, 0 se corretto)
    errors = np.array([0 if r.correct else 1 for r in results], dtype=float)
    scores = np.array([r.avg_pbii for r in results], dtype=float)
    if np.std(errors) == 0 or np.std(scores) == 0:
        corr = None
    else:
        corr = float(np.corrcoef(errors, scores)[0, 1])

    stats = LLMEvalStats(
        n=n,
        n_correct=n_correct,
        acc=float(acc),
        mean_pbii_correct=mean_pc,
        mean_pbii_incorrect=mean_pi,
        corr_pbii_vs_error=corr,
    )
    return stats, results


# =========================
# 5. MAIN
# =========================

def main():
    print("=== OMNIA_TOTALE_LLM_INTEGRATION_v0.1 ===")
    print("Nota: generate_chain_stub è solo un placeholder.")
    print("Sostituiscilo con una chiamata al tuo LLM (Grok/xAI, OpenAI, ecc.).\n")

    stats, _ = run_llm_gsm8k_eval(
        max_samples=20,
        start_index=0,
    )

    print(f"Samples evaluated: {stats.n}")
    print(f"Correct answers:   {stats.n_correct}")
    print(f"Accuracy:          {stats.acc * 100:.2f}%")
    print(f"Mean PBII (correct):   {stats.mean_pbii_correct:.4f}")
    print(f"Mean PBII (incorrect): {stats.mean_pbii_incorrect:.4f}")
    if stats.corr_pbii_vs_error is not None:
        print(f"Corr(PBII, error):     {stats.corr_pbii_vs_error:.3f}")
    else:
        print("Corr(PBII, error):     n/a (varianza nulla)")

    print("\nPer usare un vero LLM:")
    print("  - modifica generate_chain_stub() con la tua chiamata a Grok/xAI o altro modello;")
    print("  - eventualmente aumenta max_samples e aggiungi logging su file per analisi interne.")


if __name__ == "__main__":
    main()