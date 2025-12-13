"""
OMNIA / MB-X.01

Public API surface (stable).

OMNIA is a structural measurement layer.
It does not decide. It does not generate. It measures.

This package exposes:
- Structural metrics (TruthΩ, Δ, κ, ε)
- ICE envelope (confidence / impossibility)
- Legacy ICE gate (backward compatible)
"""

# -----------------------------
# New structural core (Omniabase)
# -----------------------------

from .metrics import (
    OmegaMetrics,
    compute_metrics,
)

from .envelope import (
    ICEEnvelope,
    build_ice,
)

# -----------------------------
# Legacy / compatibility layer
# -----------------------------
# (do NOT remove: used by existing OMNIA_TOTALE code)

from .ice import (
    ICEStatus,
    ICEInput,
    ICEResult,
    ice_gate,
)

# -----------------------------
# Public exports
# -----------------------------

__all__ = [
    # Omniabase / structural metrics
    "OmegaMetrics",
    "compute_metrics",
    "ICEEnvelope",
    "build_ice",

    # Legacy ICE gate
    "ICEStatus",
    "ICEInput",
    "ICEResult",
    "ice_gate",
]