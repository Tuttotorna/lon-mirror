"""
omnia.engine — OMNIA_TOTALE v2.0 engine on top of OMNIA_KERN

Wraps three structural lenses:

- omniabase   → multi-base numeric lens (PBII, signatures)
- omniatempo  → temporal stability / regime-change lens
- omniacausa  → lagged causal-structure lens

Exposes:

- build_default_engine(...)
    → returns a preconfigured OmniaKernel with the three lenses.

- run_omnia_totale(...)
    → high-level entrypoint: builds context, runs all lenses,
      returns KernelResult with fused Ω-score and per-lens details.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, Mapping, Any, Optional

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

    Requires:
    - ctx.n (int) as target integer.

    Produces:
    - scores:
        {
          "omega":        PBII(n),
          "sigma_mean":   mean σ over bases,
          "entropy_mean": mean normalized entropy over bases,
        }
    - metadata:
        full OmniabaseSignature as dict.
    """
    if ctx.n is None:
        # No integer provided → lens is effectively off
        return LensResult(
            name="omniabase",
            scores={"omega": 0.0},
            metadata={"warning": "ctx.n is None"},
        )

    sig: OmniabaseSignature = omniabase_signature(ctx.n)
    base_instability = pbii_index(ctx.n)

    scores: Dict[str, float] = {
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
    Lens wrapper for Omniatempo (temporal regime changes).

    Requires:
    - ctx.series (Iterable[float])

    Produces:
    - scores:
        {
          "omega":               log(1 + regime_change_score),
          "regime_change_score": raw symmetric KL-like divergence,
        }
    - metadata:
        OmniatempoResult as dict.
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

    scores: Dict[str, float] = {
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
    Lens wrapper for Omniacausa (lagged causal edges).

    Requires:
    - ctx.series_dict (Mapping[str, Iterable[float]])

    Produces:
    - scores:
        {
          "omega":      mean |strength| over accepted edges,
          "edge_count": number of edges,
        }
    - metadata:
        {
          "edges": [
             { "source": ..., "target": ..., "lag": ..., "strength": ... },
             ...
          ]
        }
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

    scores: Dict[str, float] = {
        "omega": mean_strength,
        "edge_count": float(len(res.edges)),
    }

    meta_edges = [
        {
            "source": e.source,
            "target": e.target,
            "lag": e.lag,
            "strength": e.strength,
        }
        for e in res.edges
    ]
    metadata: Dict[str, Any] = {"edges": meta_edges}

    return LensResult(
        name="omniacausa",
        scores=scores,
        metadata=metadata,
    )


# =========================
# 2. ENGINE BUILDERS
# =========================

def build_default_engine(
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
) -> OmniaKernel:
    """
    Build a default OmniaKernel with three lenses registered:

    - 'omniabase'   (weight = w_base)
    - 'omniatempo'  (weight = w_tempo)
    - 'omniacausa'  (weight = w_causa)
    """
    kern = OmniaKernel()
    kern.register_lens("omniabase", _lens_omniabase, weight=w_base)
    kern.register_lens("omniatempo", _lens_omniatempo, weight=w_tempo)
    kern.register_lens("omniacausa", _lens_omniacausa, weight=w_causa)
    return kern


def run_omnia_totale(
    n: Optional[int],
    series: Optional[Iterable[float]],
    series_dict: Optional[Mapping[str, Iterable[float]]],
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
    extra: Optional[Dict[str, Any]] = None,
) -> KernelResult:
    """
    High-level entrypoint for OMNIA_TOTALE:

    - Builds an OmniaContext from inputs.
    - Constructs a default OmniaKernel with the three lenses.
    - Runs all lenses and returns a KernelResult, with:

        - fused_omega: weighted sum of lens 'omega' scores,
        - per-lens scores and metadata.

    This replaces the monolithic OMNIA_TOTALE_v2.0.py script-style usage.
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
    )
    return engine.run(ctx)