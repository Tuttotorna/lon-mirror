from __future__ import annotations

from .state import InferenceState
from .signature import StructuralSignature


def classify_state(sig: StructuralSignature) -> InferenceState:
    """
    Deterministic pre-limit inference state classifier (S1–S5).

    This function:
    - does not interpret meaning
    - does not learn
    - does not optimize
    - applies fixed structural thresholds only
    """

    # S1 — Rigid invariance
    if (sig.sei < 0.20) and (sig.omega_variance < 0.01) and (sig.drift < 0.10):
        return InferenceState.RIGID_INVARIANCE

    # S2 — Elastic invariance
    if (sig.sei < 0.40) and (sig.omega_variance < 0.05) and (sig.drift < 0.20):
        return InferenceState.ELASTIC_INVARIANCE

    # S3 — Meta-stable (order sensitive)
    if (sig.order_sensitivity > 0.50) and (sig.sei < 0.60):
        return InferenceState.META_STABLE

    # S4 — Coherent drift (directional)
    if (sig.drift > 0.30) and (sig.drift_vector > 0.70):
        return InferenceState.COHERENT_DRIFT

    # S5 — Pre-limit fragmentation
    return InferenceState.PRE_LIMIT_FRAGMENTATION