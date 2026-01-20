from .state import InferenceState
from .signature import StructuralSignature

def classify_state(sig: StructuralSignature) -> InferenceState:
    if sig.sei < 0.2 and sig.omega_variance < 0.01:
        return InferenceState.RIGID_INVARIANCE

    if sig.sei < 0.4 and sig.omega_variance < 0.05:
        return InferenceState.ELASTIC_INVARIANCE

    if sig.order_sensitivity > 0.5 and sig.sei < 0.6:
        return InferenceState.META_STABLE

    if sig.drift > 0.3 and sig.drift_vector > 0.7:
        return InferenceState.COHERENT_DRIFT

    return InferenceState.PRE_LIMIT_FRAGMENTATION