"""
omnia.core.omnia_totale — fused Ω-score
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable
import math

import numpy as np

from .omniabase import OmniabaseSignature, omniabase_signature, pbii_index
from .omniatempo import OmniatempoResult, omniatempo_analyze
from .omniacausa import OmniaEdge, OmniacausaResult, omniacausa_analyze


# =========================
# 4. OMNIA_TOTALE FUSED SCORE
# =========================

@dataclass
class OmniaTotaleResult:
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
    Fused Ω score combining Omniabase, Omniatempo, and Omniacausa.

    - base_component: PBII-style instability (higher for primes).
    - tempo_component: log(1 + regime_change_score).
    - causa_component: mean |strength| of accepted edges.
    """
    base_sig = omniabase_signature(
        n,
        bases=bases,
        length_weight=length_weight,
        length_exponent=length_exponent,
        divisibility_bonus=divisibility_bonus,
    )
    base_instability = pbii_index(n, bases=bases)

    tempo_res = omniatempo_analyze(
        series,
        short_window=short_window,
        long_window=long_window,
        hist_bins=hist_bins,
        epsilon=epsilon,
    )
    tempo_val = math.log(1.0 + tempo_res.regime_change_score)

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