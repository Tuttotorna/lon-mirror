"""
OMNIA — Unified Structural Lenses
MB-X.01 / OMNIA_TOTALE

Package initialization.

OMNIA is a structural measurement engine.
It does not decide, optimize, or generate.
It measures coherence, invariance, and drift.

Core outputs:
- TruthΩ (structural invariance)
- Δ-coherence
- κ-alignment
- ε-drift
- ICE status (Impossibility & Confidence Envelope)
"""

# ---- Metrics -------------------------------------------------

from .metrics import (
    truth_omega,     # TruthΩ scalar
    delta_coherence, # Δ
    kappa_alignment, # κ
    epsilon_drift,   # ε
)

# ---- ICE Envelope --------------------------------------------

from .envelope import (
    ICEStatus,
    ICEInput,
    ICEResult,
    ice_gate,
)

# ---- Omniabase -----------------------------------------------

from .omniabase import (
    omni_signature,
    omni_transform,
)

__all__ = [
    # Metrics
    "truth_omega",
    "delta_coherence",
    "kappa_alignment",
    "epsilon_drift",

    # ICE
    "ICEStatus",
    "ICEInput",
    "ICEResult",
    "ice_gate",

    # Omniabase
    "omni_signature",
    "omni_transform",
]