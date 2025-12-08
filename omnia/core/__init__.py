"""
omnia.core — structural lenses package

Exposes:
- omniabase: multi-base numeric lens (PBII, signatures)
- omniatempo: temporal stability lens
- omniacausa: lagged causal-structure lens
"""

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

__all__ = [
    # omniabase
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
]