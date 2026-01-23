"""
omnia.core.omnia_totale — fused Ω-score (Unified with Sovereign Kernel)
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization) + ARK ASCENDANCE (Sovereign Integration)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Optional, Any
import math

import numpy as np

from .omniabase import OmniabaseSignature, omniabase_signature, pbii_index
from .omniatempo import OmniatempoResult, omniatempo_analyze
from .omniacausa import OmniaEdge, OmniacausaResult, omniacausa_analyze
from omnia.sovereign import SovereignKernel, GovernanceResult


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
    # Sovereign Integration
    governance: Optional[Dict[str, Any]] = None

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
    # Sovereign Control
    enable_sovereign: bool = True,
) -> OmniaTotaleResult:
    """
    Fused Ω score combining Omniabase, Omniatempo, and Omniacausa.

    Includes ARK ASCENDANCE Sovereign Governance to validate metrics.
    """
    # 1. Base Component (Instability)
    base_sig = omniabase_signature(
        n,
        bases=bases,
        length_weight=length_weight,
        length_exponent=length_exponent,
        divisibility_bonus=divisibility_bonus,
    )
    base_instability = pbii_index(n, bases=bases)

    # 2. Tempo Component (Regime Change)
    tempo_res = omniatempo_analyze(
        series,
        short_window=short_window,
        long_window=long_window,
        hist_bins=hist_bins,
        epsilon=epsilon,
    )
    tempo_val = math.log(1.0 + tempo_res.regime_change_score)

    # 3. Causa Component (Connectivity)
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

    # 4. Sovereign Governance (The Brain)
    gov_result = None
    if enable_sovereign:
        kernel = SovereignKernel()
        # Feed context: The computed metrics
        context = {
            "base": base_instability,
            "tempo": tempo_val,
            "causa": causa_val,
            "edges": len(causa_res.edges)
        }
        # Ask Sovereign to govern the Fusion process
        # If entropy is high (metrics disagree or are volatile), Sovereign might HALT or WARN
        gov = kernel.govern(context, intent="FUSION")

        # Structure the result
        gov_result = {
            "decision": gov.decision,
            "s_lang": gov.s_lang_trace,
            "risk": gov.risk_assessment,
            "note": gov.note
        }

        # Dynamic Weighting (Sovereign Adjustment)
        # If Risk is high (entropy > 1.0), we might dampen the causal component (likely spurious)
        if gov.risk_assessment > 0.5:
             w_causa *= 0.5 # Dampen causal noise
             gov_result["adjustment"] = "Dampened w_causa due to high entropy"

    # 5. Final Calculation
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
        governance=gov_result
    )
