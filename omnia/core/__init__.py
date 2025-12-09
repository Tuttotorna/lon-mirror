"""
OMNIA package

Unified import surface for:
- omniabase   → multi-base lenses + PBII
- omniatempo  → temporal regime lenses
- omniacausa  → causal lenses
- omnia_totale → fused Ω-score
"""

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

from .omnia_totale import (
    OmniaTotaleResult,
    omnia_totale_score,
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
    # omnia_totale
    "OmniaTotaleResult",
    "omnia_totale_score",
]