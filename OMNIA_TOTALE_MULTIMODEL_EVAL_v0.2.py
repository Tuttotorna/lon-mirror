"""
OMNIA_TOTALE — Multi-Model Ω Evaluation v0.2
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Scopo:
- Valutare più modelli LLM in parallelo usando PBII/Ω come metrica di stabilità.
- Calcolare Ω per ogni catena di ragionamento (estratta da GSM8K o simili).
- Fornire statistiche per modello (media, deviazione, drift) e ranking.

Dipendenze:
    pip install numpy
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Tuple
import math
import re

import numpy as np

# =========================
# 1. CORE PBII (semplificato)
# =========================

def digits_in_base(n: int, b: int) -> List[int]:
    """Return digits of n in base b (MSB first)."""
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


def sigma_b(n: int, b: int) -> float:
    """
    Base Symmetry Score (versione compatta).

    - Entropia normalizzata delle cifre in base b.
    - Penalizza rappresentazioni lunghe / rumorose.
    - Bonus se n è multiplo della base (struttura evidente).
    """
    digits = digits_in_base(n, b)
    L = len(digits)
    if L == 0:
        return 0.0

    freq = np.bincount(np.array(digits, dtype=int), minlength=b).astype(float)
    probs = freq[freq > 0] / L
    if probs.size == 0:
        Hn = 0.0
    else:
        H = -np.sum(probs * np.log2(probs))
        Hmax = math.log2(b)
        Hn = float(H / Hmax) if Hmax > 0 else 0.0

    length_term = (1.0 - Hn) / L
    bonus = 0.5 if n % b == 0 else 0.0
    return float(length_term + bonus)


def sigma_avg(n: int, bases: Iterable[int]) -> float:
    bases = list(bases)
    vals = [sigma_b(n, b) for b in bases]
    return float(np.mean(vals)) if vals else 0.0


def saturation(n: int, bases: Iterable[int], window: int = 100) -> float:
    """
    Saturazione strutturale media sui composti vicini a n.
    """
    bases = list(bases)
    start = max(4, n - window)
    comps: List[int] = []
    for k in range(start, n):
        if k <= 3:
            continue
        # composito se ha almeno un divisore non banale
        if any(k % d == 0 for d in range(2, int(math.sqrt(k)) + 1)):
            comps.append(k)
    if not comps:
        return 0.0
    vals = [sigma_avg(k, bases) for k in comps]
    return float(np.mean(vals))


def pbii(n: int,
         bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29),
         window: int = 100) -> float:
    """
    Prime Base Instability Index.

    PBII(n) = mean_sigma(composites_near_n) - mean_sigma(n)

    - Valori alti -> n più "prime-like" (instabile rispetto ai composti).
    - Valori bassi / negativi -> n più strutturato / composito.
    """
    bases = list(bases)
    sig_n = sigma_avg(n, bases)
    sat = saturation(n, bases, window=window)
    return float(sat - sig_n)


# =========================
# 2. Ω PER CATENE DI RAGIONAMENTO
# =========================

number_pattern = re.compile(r"\b\d+\b")


def extract_numbers(chain_text: str) -> List[int]:
    """Estrae interi positivi > 1 dal testo."""
    nums: List[int] = []
    for m in number_pattern.findall(chain_text):
        v = int(m)
        if v > 1:
            nums.append(v)
    return nums


@dataclass
class ChainOmega:
    model_name: str
    sample_id: str
    tokens: List[str]
    token_scores: List[float]
    omega_raw: float
    omega_revised: float
    delta_omega: float

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_chain_omega(
    model_name: str,
    sample_id: str,
    chain_text: str,
    pbii_bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29),
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> ChainOmega:
    """
    Calcola Ω per una singola catena di ragionamento di un modello.

    - Estrae i numeri dal testo.
    - Calcola PBII per ogni numero.
    - Definisce Ω_raw come media dei punteggi invertiti (stabilità).
    - Applica clipping e normalizzazione leggera per Ω_revised.
    - ΔΩ = Ω_revised - Ω_raw (indice di correzione).
    """
    nums = extract_numbers(chain_text)
    if not nums:
        # Nessuna struttura numerica: catena non valutabile -> Ω neutro.
        return ChainOmega(
            model_name=model_name,
            sample_id=sample_id,
            tokens=[],
            token_scores=[],
            omega_raw=0.0,
            omega_revised=0.0,
            delta_omega=0.0,
        )

    scores_pbii = np.array([pbii(n, bases=pbii_bases) for n in nums], dtype=float)
    # Stabilità = -PBII (primi instabili -> punteggio più basso)
    stab = -scores_pbii

    omega_raw = float(stab.mean())

    # Clipping e riscalamento morbido in [clip_min, clip_max]
    omega_clipped = float(np.clip(omega_raw, clip_min, clip_max))
    omega_revised = omega_clipped
    delta = float(omega_revised - omega_raw)

    tokens = [str(n) for n in nums]
    token_scores = stab.tolist()

    return ChainOmega(
        model_name=model_name,
        sample_id=sample_id,
        tokens=tokens,
        token_scores=token_scores,
        omega_raw=omega_raw,
        omega_revised=omega_revised,
        delta_omega=delta,
    )


# =========================
# 3. STATISTICHE PER MODELLO
# =========================

@dataclass
class ModelStats:
    model_name: str
    mean_omega: float
    std_omega: float
    drift_index: float
    n_samples: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MultiModelEvalResult:
    chains: List[ChainOmega]
    per_model: Dict[str, ModelStats]
    ranking: List[Tuple[str, float]]  # (model_name, mean_omega)

    def to_dict(self) -> Dict:
        return {
            "chains": [c.to_dict() for c in self.chains],
            "per_model": {k: v.to_dict() for k, v in self.per_model.items()},
            "ranking": self.ranking,
        }


def _compute_drift_index(omega_series: np.ndarray) -> float:
    """
    Drift index semplice: media |ΔΩ_t| su serie temporale.
    Più alto = modello più instabile nel tempo.
    """
    if omega_series.size < 2:
        return 0.0
    diffs = np.diff(omega_series)
    return float(np.mean(np.abs(diffs)))


def evaluate_models_omega(
    data: Dict[str, List[Tuple[str, str]]],
    # data[model] = [(sample_id, chain_text), ...]
) -> MultiModelEvalResult:
    """
    Valuta più modelli in parallelo.

    Ritorna:
    - lista di ChainOmega (una per catena, per modello),
    - statistiche aggregate per modello,
    - ranking per mean_omega decrescente.
    """
    all_chains: List[ChainOmega] = []

    for model_name, samples in data.items():
        for sample_id, chain_text in samples:
            co = compute_chain_omega(
                model_name=model_name,
                sample_id=sample_id,
                chain_text=chain_text,
            )
            all_chains.append(co)

    # aggrega per modello
    per_model: Dict[str, ModelStats] = {}
    by_model: Dict[str, List[float]] = {}

    for co in all_chains:
        by_model.setdefault(co.model_name, []).append(co.omega_revised)

    for model_name, omegas in by_model.items():
        arr = np.asarray(omegas, dtype=float)
        mean_omega = float(arr.mean()) if arr.size else 0.0
        std_omega = float(arr.std(ddof=0)) if arr.size else 0.0
        drift = _compute_drift_index(arr)
        per_model[model_name] = ModelStats(
            model_name=model_name,
            mean_omega=mean_omega,
            std_omega=std_omega,
            drift_index=drift,
            n_samples=len(omegas),
        )

    ranking = sorted(
        [(m, ms.mean_omega) for m, ms in per_model.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    return MultiModelEvalResult(
        chains=all_chains,
        per_model=per_model,
        ranking=ranking,
    )


# =========================
# 4. DEMO MULTI-MODELLO
# =========================

def demo():
    """
    Demo sintetica con tre modelli fittizi su poche catene GSM8K-like.

    In pratica:
    - model_A: catene più stabili (vicine al ground truth)
    - model_B: catene miste
    - model_C: catene più corrotte / instabili
    """
    model_A = [
        ("q1", "Sam skipped 16 times per round. Jeff: 15, 13, 20, 8 -> 56 total, avg 14."),
        ("q2", "Mark bought 50 cans. Jennifer adds 6 for every 5: 10 times -> 60. Total 110."),
        ("q3", "Paityn 20+24=44, Zola 16+48=64, total 108, each 54."),
    ]

    model_B = [
        ("q1", "Sam skipped 17 times per round. Jeff 16, 14, 21, 9 -> 60, avg 15."),
        ("q2", "Mark bought 51 cans. Jennifer adds 6 for every 5, but miscounts: 11 times -> 66."),
        ("q3", "Paityn 21+24=45, Zola 17+48=65, total 110, each 55."),
    ]

    model_C = [
        ("q1", "Sam skipped 19 times. Jeff: 21, 22, 19, 25 -> 87, avg 22."),
        ("q2", "Mark 60 cans. Jennifer random 7,9,11 -> 97 total."),
        ("q3", "Paityn 30+40=90, Zola 50+70=130, total 250, each 80."),
    ]

    data = {
        "model_A": model_A,
        "model_B": model_B,
        "model_C": model_C,
    }

    res = evaluate_models_omega(data)

    print("=== OMNIA_TOTALE — Multi-Model Ω Evaluation v0.2 ===\n")
    print("Per-model stats:")
    for name, stats in res.per_model.items():
        print(
            f"  {name}: "
            f"mean_Ω={stats.mean_omega:.4f}, "
            f"std_Ω={stats.std_omega:.4f}, "
            f"drift={stats.drift_index:.4f}, "
            f"n={stats.n_samples}"
        )
    print("\nRanking (best first):")
    for rank, (name, score) in enumerate(res.ranking, start=1):
        print(f"  {rank}. {name}  (mean_Ω={score:.4f})")

    print("\nSample chain Ω values:")
    for co in res.chains:
        print(
            f"  {co.model_name} | {co.sample_id} | "
            f"Ω_raw={co.omega_raw:.4f} Ω_rev={co.omega_revised:.4f} "
            f"ΔΩ={co.delta_omega:.4f} | tokens={co.tokens}"
        )


if __name__ == "__main__":
    demo()