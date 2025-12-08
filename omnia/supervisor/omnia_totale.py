"""
OMNIA_TOTALE — fused supervisor lens
Author: Massimiliano Brighindi (concepts) + MBX IA (structure)

Fusion of:
- Omniabase  → numeric multi-base instability (PBII)
- Omniatempo → temporal regime-change
- Omniacausa → lagged causal structure

This module provides a single Ω-score for a number n
+ associated time series + multivariate signals.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Mapping, Sequence
import math

import numpy as np

from omnia.base.omniabase import (
    OmniabaseSignature,
    omniabase_signature,
    pbii_index,
)
from omnia.tempo.omniatempo import (
    OmniatempoResult,
    analyze_temporal,
)
from omnia.causa.omniacausa import (
    OmniacausaResult,
    analyze_causal,
)


@dataclass
class OmniaTotaleResult:
    """
    Container for the fused OMNIA_TOTALE output.
    """
    n: int
    omniabase: OmniabaseSignature
    omniatempo: OmniatempoResult
    omniacausa: OmniacausaResult
    omega_score: float
    components: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


def omnia_totale_score(
    n: int,
    series: Sequence[float],
    series_dict: Mapping[str, Sequence[float]],
    # Omniabase params
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
    # Omniatempo params
    short_window: int = 20,
    long_window: int = 100,
    hist_bins: int = 20,
    epsilon: float = 1e-9,
    # Omniacausa params
    max_lag: int = 5,
    strength_threshold: float = 0.3,
    # fusion weights
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
) -> OmniaTotaleResult:
    """
    Compute fused Ω-score.

    Components:
      - base_instability  : PBII-style instability (higher ~ prime-like)
      - tempo_log_regime  : log(1 + regime_change_score)
      - causa_mean_strength: mean |lagged-correlation| across accepted edges
    """

    # 1) Omniabase (numeric structure)
    base_sig = omniabase_signature(
        n,
        bases=bases,
        length_weight=length_weight,
        length_exponent=length_exponent,
        divisibility_bonus=divisibility_bonus,
    )
    base_instability = pbii_index(
        n,
        bases=bases,
    )

    # 2) Omniatempo (temporal stability)
    tempo_res = analyze_temporal(
        series,
        short_window=short_window,
        long_window=long_window,
        hist_bins=hist_bins,
        epsilon=epsilon,
    )
    tempo_val = math.log(1.0 + tempo_res.regime_change_score)

    # 3) Omniacausa (causal edges)
    causa_res = analyze_causal(
        series_dict,
        max_lag=max_lag,
        threshold=strength_threshold,
    )

    if causa_res.edges:
        strengths = np.array([abs(e.strength) for e in causa_res.edges], dtype=float)
        causa_val = float(strengths.mean())
    else:
        causa_val = 0.0

    # 4) Fusion
    omega = (
        w_base * base_instability
        + w_tempo * tempo_val
        + w_causa * causa_val
    )

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


def demo() -> None:
    """
    Minimal demo: compare a prime vs a composite
    under the same temporal/causal context.
    """
    import random

    n_prime = 173
    n_comp = 180

    random.seed(0)
    np.random.seed(0)

    # Time series with regime shift
    t = np.arange(300)
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    # Causal toy system: s1 → s2 (lag 2), s3 as noise
    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)

    series_dict = {"s1": s1, "s2": s2, "s3": s3}

    res_prime = omnia_totale_score(n_prime, series, series_dict)
    res_comp = omnia_totale_score(n_comp, series, series_dict)

    print("=== OMNIA_TOTALE demo ===")
    print(
        f"n={n_prime} (prime)  Ω ≈ {res_prime.omega_score:.4f}  "
        f"components={res_prime.components}"
    )
    print(
        f"n={n_comp} (comp.)  Ω ≈ {res_comp.omega_score:.4f}  "
        f"components={res_comp.components}"
    )
    print("Causal edges (from same context):")
    for