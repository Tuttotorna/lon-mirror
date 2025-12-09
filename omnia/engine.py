"""
omnia.engine — OMNIA_TOTALE engine on top of OMNIA_KERN

Wraps:
- omniabase   (PBII, multi-base signatures)
- omniatempo  (regime-change detection)
- omniacausa  (lagged causal edges)
- tokenlens   (token-level Ω-map for LLM traces)

Exposes:
- build_default_engine(): OmniaKernel preconfigurato
- run_omnia_totale(...): scorciatoia ad alto livello
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, Mapping, Any, Optional, List

import numpy as np

from .omniabase import (
    OmniabaseSignature,
    omniabase_signature,
    pbii_index,
)
from .omniatempo import (
    OmniatempoResult,
    omniatempo_analyze,
)
from .omniacausa import (
    OmniaEdge,
    OmniacausaResult,
    omniacausa_analyze,
)
from .tokenlens import (
    TokenOmegaMap,
    compute_token_omega_map,
)
from .kernel import (
    OmniaContext,
    LensResult,
    KernelResult,
    OmniaKernel,
)


# =========================
# 1. LENS WRAPPERS
# =========================

def _lens_omniabase(ctx: OmniaContext) -> LensResult:
    """
    Lens wrapper for Omniabase / PBII.

    Richiede:
    - ctx.n (int)

    Produce:
    - scores: {
        'omega': base_instability (PBII),
        'sigma_mean': sigma_mean,
        'entropy_mean': entropy_mean,
      }
    - metadata: full OmniabaseSignature as dict
    """
    if ctx.n is None:
        # nessun intero → lens “spenta”
        return LensResult(
            name="omniabase",
            scores={"omega": 0.0},
            metadata={"warning": "ctx.n is None"},
        )

    sig: OmniabaseSignature = omniabase_signature(ctx.n)
    base_instability = pbii_index(ctx.n)

    scores = {
        "omega": float(base_instability),
        "sigma_mean": float(sig.sigma_mean),
        "entropy_mean": float(sig.entropy_mean),
    }

    meta: Dict[str, Any] = sig.to_dict()
    return LensResult(
        name="omniabase",
        scores=scores,
        metadata=meta,
    )


def _lens_omniatempo(ctx: OmniaContext) -> LensResult:
    """
    Lens wrapper per Omniatempo.

    Richiede:
    - ctx.series (Iterable[float])

    Produce:
    - scores: {
        'omega': log(1 + regime_change_score),
        'regime_change_score': raw,
      }
    - metadata: OmniatempoResult as dict
    """
    import math

    if ctx.series is None:
        return LensResult(
            name="omniatempo",
            scores={"omega": 0.0},
            metadata={"warning": "ctx.series is None"},
        )

    res: OmniatempoResult = omniatempo_analyze(ctx.series)
    regime = float(res.regime_change_score)
    omega = math.log(1.0 + regime)

    scores = {
        "omega": omega,
        "regime_change_score": regime,
    }

    meta: Dict[str, Any] = asdict(res)
    return LensResult(
        name="omniatempo",
        scores=scores,
        metadata=meta,
    )


def _lens_omniacausa(ctx: OmniaContext) -> LensResult:
    """
    Lens wrapper per Omniacausa.

    Richiede:
    - ctx.series_dict (Mapping[str, Iterable[float]])

    Produce:
    - scores: {
        'omega': mean |strength|,
        'edge_count': number of edges,
      }
    - metadata: list of edges as dict
    """
    if ctx.series_dict is None or not ctx.series_dict:
        return LensResult(
            name="omniacausa",
            scores={"omega": 0.0},
            metadata={"warning": "ctx.series_dict is None or empty"},
        )

    res: OmniacausaResult = omniacausa_analyze(ctx.series_dict)
    if not res.edges:
        return LensResult(
            name="omniacausa",
            scores={"omega": 0.0, "edge_count": 0.0},
            metadata={"edges": []},
        )

    strengths = np.array([abs(e.strength) for e in res.edges], dtype=float)
    mean_strength = float(strengths.mean())

    scores = {
        "omega": mean_strength,
        "edge_count": float(len(res.edges)),
    }

    meta = {
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "lag": e.lag,
                "strength": e.strength,
            }
            for e in res.edges
        ]
    }
    return LensResult(
        name="omniacausa",
        scores=scores,
        metadata=meta,
    )


def _lens_token(ctx: OmniaContext) -> LensResult:
    """
    Lens wrapper per la mappa token-level Ω.

    Richiede in ctx.extra:
    - 'tokens': list[str]
    - 'token_numbers': list[int] (proxy numerico per token)

    Produce:
    - scores: {
        'omega': mean |z|,
        'mean_pbii': media PBII sui token,
      }
    - metadata: TokenOmegaMap as dict
    """
    extra = ctx.extra or {}
    tokens: Optional[List[str]] = extra.get("tokens")
    token_numbers: Optional[List[int]] = extra.get("token_numbers")

    if not tokens or not token_numbers or len(tokens) != len(token_numbers):
        return LensResult(
            name="tokenlens",
            scores={"omega": 0.0},
            metadata={
                "warning": "missing or inconsistent tokens/token_numbers in ctx.extra",
            },
        )

    tok_map: TokenOmegaMap = compute_token_omega_map(tokens, token_numbers)

    scores = {
        "omega": float(tok_map.mean_abs_z),
        "mean_pbii": float(tok_map.mean_pbii),
    }

    meta: Dict[str, Any] = {
        "tokens": tok_map.tokens,
        "token_numbers": tok_map.token_numbers,
        "pbii_scores": tok_map.pbii_scores,
        "z_scores": tok_map.z_scores,
        "mean_abs_z": tok_map.mean_abs_z,
        "mean_pbii": tok_map.mean_pbii,
    }

    return LensResult(
        name="tokenlens",
        scores=scores,
        metadata=meta,
    )


# =========================
# 2. ENGINE BUILDERS
# =========================

def build_default_engine(
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
    w_token: float = 1.0,
) -> OmniaKernel:
    """
    Crea un OmniaKernel con quattro lenti registrate:

    - 'omniabase'  (peso w_base)
    - 'omniatempo' (peso w_tempo)
    - 'omniacausa' (peso w_causa)
    - 'tokenlens'  (peso w_token)
    """
    kern = OmniaKernel()
    kern.register_lens("omniabase", _lens_omniabase, weight=w_base)
    kern.register_lens("omniatempo", _lens_omniatempo, weight=w_tempo)
    kern.register_lens("omniacausa", _lens_omniacausa, weight=w_causa)
    kern.register_lens("tokenlens", _lens_token, weight=w_token)
    return kern


def run_omnia_totale(
    n: Optional[int],
    series: Optional[Iterable[float]],
    series_dict: Optional[Mapping[str, Iterable[float]]],
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
    w_token: float = 1.0,
    extra: Optional[Dict[str, Any]] = None,
) -> KernelResult:
    """
    Scorciatoia ad alto livello:

    - costruisce un OmniaContext
    - crea un engine di default
    - esegue tutte le lenti
    - restituisce KernelResult

    Questo sostituisce l'uso diretto di OMNIA_TOTALE_v2.0.py monolitico.
    Per attivare la lente TOKEN, passa in extra:
      extra = {
          "tokens": [...],
          "token_numbers": [...],
      }
    """
    if extra is None:
        extra = {}

    ctx = OmniaContext(
        n=n,
        series=series,
        series_dict=series_dict,
        extra=extra,
    )
    engine = build_default_engine(
        w_base=w_base,
        w_tempo=w_tempo,
        w_causa=w_causa,
        w_token=w_token,
    )
    return engine.run(ctx)