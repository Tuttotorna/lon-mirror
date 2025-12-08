"""
OMNIA_TOTALE — Core Engine (Stable)
Author: Massimiliano Brighindi (MBX)
Package version: 1.0

This module exposes the stable API for:
- omniabase_signature
- omniatempo_analyze
- omniacausa_analyze
- omnia_totale_score
"""

from .omniabase import omniabase_signature, pbii_index
from .omniatempo import omniatempo_analyze
from .omniacausa import omniacausa_analyze

import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class OmniaTotaleResult:
    n: int
    omniabase: object
    omniatempo: object
    omniacausa: object
    omega_score: float
    components: Dict[str, float]


def omnia_totale_score(
    n: int,
    series: Iterable[float],
    series_dict: Dict[str, Iterable[float]],
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
) -> OmniaTotaleResult:
    """Unified Ω-score (multi-base + temporal + causal)."""

    base_sig = omniabase_signature(n, bases=bases)
    base_inst = pbii_index(n, bases=bases)

    tempo_res = omniatempo_analyze(series)
    tempo_val = math.log(1 + tempo_res.regime_change_score)

    causa_res = omniacausa_analyze(series_dict)
    causa_val = (
        float(np.mean([abs(e.strength) for e in causa_res.edges]))
        if causa_res.edges
        else 0.0
    )

    omega = w_base * base_inst + w_tempo * tempo_val + w_causa * causa_val

    return OmniaTotaleResult(
        n=n,
        omniabase=base_sig,
        omniatempo=tempo_res,
        omniacausa=causa_res,
        omega_score=float(omega),
        components={
            "base_instability": base_inst,
            "tempo_log_regime": tempo_val,
            "causa_mean_strength": causa_val,
        },
    )