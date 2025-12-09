"""
omnia.core — structural lenses + Ω-engine

Unified import surface for:

- omniabase      → multi-base numeric lens (PBII, signatures)
- omniatempo     → temporal regime-change lens (KL drift)
- omniacausa     → lagged causal-structure lens
- tokenlens      → token-level Ω-map for LLM traces
- omnia_totale   → fused Ω-score (monolithic helper, optional)
- kernel / engine → generic lens kernel + OMNIA_TOTALE engine
"""

# -------------------------
# omniabase (BASE / PBII)
# -------------------------
from .omniabase import (
    digits_in_base_np,
    normalized_entropy_base,
    sigma_b,
    OmniabaseSignature,
    omniabase_signature,
    pbii_index,
)

# -------------------------
# omniatempo (TIME)
# -------------------------
from .omniatempo import (
    OmniatempoResult,
    omniatempo_analyze,
)

# -------------------------
# omniacausa (CAUSA)
# -------------------------
from .omniacausa import (
    OmniaEdge,
    OmniacausaResult,
    omniacausa_analyze,
)

# -------------------------
# tokenlens (TOKEN Ω-map)
# -------------------------
from .tokenlens import (
    TokenOmegaMap,
    compute_token_omega_map,
)

# -------------------------
# fused score helper (v2.0)
# -------------------------
from .omnia_totale import (
    OmniaTotaleResult,
    omnia_totale_score,
)

# -------------------------
# kernel + high-level engine
# -------------------------
from .kernel import (
    OmniaContext,
    LensResult,
    KernelResult,
    OmniaKernel,
)

from .engine import (
    build_default_engine,
    run_omnia_totale,
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
    # fused helper
    "OmniaTotaleResult",
    "omnia_totale_score",
    # kernel / engine
    "OmniaContext",
    "LensResult",
    "KernelResult",
    "OmniaKernel",
    "build_default_engine",
    "run_omnia_totale",
]