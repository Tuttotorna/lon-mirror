"""
OMNIA_TOTALE v2.0 — fused Ω score using omnia.core lenses

Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Requires:
    pip install numpy
    (and the local package structure:
        omnia/
          __init__.py
          core/
            __init__.py
            omniabase.py
            omniatempo.py
            omniacausa.py
    )
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List

import math
import numpy as np

# Importa le tre lenti strutturali dal package omnia.core
from omnia.core import (
    OmniabaseSignature,
    omniabase_signature,
    pbii_index,
    OmniatempoResult,
    omniatempo_analyze,
    OmniaEdge,
    OmniacausaResult,
    omniacausa_analyze,
)


# =========================
# 1. OMNIA_TOTALE RESULT
# =========================

@dataclass
class OmniaTotaleResult:
    """
    Fused result container for OMNIA_TOTALE.

    Fields:
    - n: integer under omniabase analysis
    - omniabase: multi-base numeric signature (PBII, sigmas, entropy)
    - omniatempo: temporal stability metrics (means, std, regime_change_score)
    - omniacausa: lagged causal graph (edges)
    - omega_score: fused scalar Ω
    - components: per-lens contributions to Ω
    """
    n: int
    omniabase: OmniabaseSignature
    omniatempo: OmniatempoResult
    omniacausa: OmniacausaResult
    omega_score: float
    components: Dict[str, float]

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Dataclass nested objects are already dict-serializable via asdict
        return d


# =========================
# 2. FUSED Ω SCORE
# =========================

def omnia_totale_score(
    n: int,
    series: Iterable[float],
    series_dict: Dict[str, Iterable[float]],
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
    short_window: int = 20,
    long_window: int = 100,
    hist_bins: int = 20,
    max_lag: int = 5,
    strength_threshold: float = 0.3,
    # fusion weights
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
    epsilon: float = 1e-9,
) -> OmniaTotaleResult:
    """
    Compute fused Ω score combining:
      - omniabase (PBII-style instability)
      - omniatempo (regime-change divergence)
      - omniacausa (mean lagged-correlation strength)

    Components:
      base_instability  ~ PBII(n)
      tempo_log_regime  ~ log(1 + regime_change_score)
      causa_mean_strength ~ mean |corr| over accepted edges
    """

    # ---- omniabase lens ----
    base_sig = omniabase_signature(
        n,
        bases=bases,
        length_weight=length_weight,
        length_exponent=length_exponent,
        divisibility_bonus=divisibility_bonus,
    )
    base_instability = pbii_index(n, bases=bases)

    # ---- omniatempo lens ----
    tempo_res = omniatempo_analyze(
        series,
        short_window=short_window,
        long_window=long_window,
        hist_bins=hist_bins,
        epsilon=epsilon,
    )
    tempo_val = math.log(1.0 + tempo_res.regime_change_score)

    # ---- omniacausa lens ----
    causa_res = omniacausa_analyze(
        series_dict,
        max_lag=max_lag,
        strength_threshold=strength_threshold,
    )
    if causa_res.edges:
        strengths = np.array([abs(e.strength) for e in causa_res.edges], dtype=float)
        causa_val = float(strengths.mean())
    else:
        causa_val = 0.0

    # ---- fusion ----
    omega = w_base * base_instability + w_tempo * tempo_val + w_causa * causa_val

    components = {
        "base_instability": base_instability,
        "tempo_log_regime": tempo_val,
        "causa_mean_strength": causa_val,
    }

    return OmniaTotaleResult(
        n=n,
        omniabase=base_sig,
        omniatempo=tempo_res,
        omniacausa=causa_res,
        omega_score=float(omega),
        components=components,
    )


# =========================
# 3. DEMO
# =========================

def demo() -> None:
    """
    Minimal demo for OMNIA_TOTALE v2.0.

    Run:
        python OMNIA_TOTALE_v2.0.py

    to see:
      - fused Ω for a prime vs a composite
      - causal edges discovered by omniacausa
    """
    import random

    n_prime = 173
    n_comp = 180

    random.seed(0)
    np.random.seed(0)

    # Synthetic time series with regime shift
    t = np.arange(300)
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    # Three related series for causal lens
    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)

    series_dict = {"s1": s1, "s2": s2, "s3": s3}

    res_prime = omnia_totale_score(n_prime, series, series_dict)
    res_comp = omnia_totale_score(n_comp, series, series_dict)

    print("=== OMNIA_TOTALE v2.0 demo ===")
    print(
        f"n={n_prime} (prime)  Ω ≈ {res_prime.omega_score:.4f}  "
        f"components={res_prime.components}"
    )
    print(
        f"n={n_comp} (comp.)  Ω ≈ {res_comp.omega_score:.4f}  "
        f"components={res_comp.components}"
    )
    print("Causal edges (from omniacausa):")
    for e in res_prime.omniacausa.edges:
        print(f"  {e.source} -> {e.target}  lag={e.lag}  strength={e.strength:.3f}")


if __name__ == "__main__":
    demo()