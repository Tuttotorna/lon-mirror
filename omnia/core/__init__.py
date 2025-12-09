"""
omnia.core — structural lenses (base, time, causal, token)

Exposes:
- omniabase   → multi-base numeric lens + PBII
- omniatempo  → temporal regime-change lens
- omniacausa  → lagged causal-structure lens
- tokenlens   → token-level Ω-map for LLM traces
"""

from __future__ import annotations

from .omniabase import (
    digits_in_base_np,
    normalized_entropy_base,
    sigma_b,
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

__all__ = [
    # omniabase
    "digits_in_base_np",
    "normalized_entropy_base",
    "sigma_b",
    "OmniabaseSignature",
    "omniabase_signature",
    "pbii_index",
    # omniatempo
    "OmniatempoResult",
    "omniatempo_analyze",
    # omniacausa
    "OmniaEdge",
    "OmniacausaResult",
    "omniacausa_analyze",
    # tokenlens
    "TokenOmegaMap",
    "compute_token_omega_map",
]