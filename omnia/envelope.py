# OMNIA / MB-X.01
# ICE — Impossibility & Confidence Envelope
# Converts structural metrics into a stable decision-support envelope.
# OMNIA does not decide: it reports confidence + impossibility + flags.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import math

from .metrics import OmegaMetrics


@dataclass(frozen=True)
class ICEEnvelope:
    confidence: float          # [0,1] reliability of structure (not semantic truth)
    impossibility: float       # [0,1] structural contradiction / collapse likelihood
    flags: List[str]           # machine-readable warnings
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # ensure JSON-stable ordering by returning plain dict; callers can json.dumps(sort_keys=True)
        return d


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def build_ice(
    m: OmegaMetrics,
    *,
    # Default thresholds (tunable per deployment)
    min_truth_omega: float = 0.72,
    max_delta: float = 0.35,
    min_kappa: float = 0.70,
    max_epsilon: float = 0.55,
) -> ICEEnvelope:
    """
    Build the ICE envelope from OmegaMetrics.

    Interpretation:
    - confidence increases with TruthΩ and κ-alignment, decreases with Δ and ε.
    - impossibility increases when multiple thresholds are violated at once (structural collapse).
    - flags are deterministic and stable (no narrative).

    This is NOT a classifier; it is a calibrated structural envelope.
    """

    flags: List[str] = []

    # Normalize “badness” channels
    # Δ and ε are unbounded in theory, so map via saturating transform.
    delta_bad = _clamp01(m.delta_coherence / (max_delta if max_delta > 0 else 1.0))
    eps_bad = _clamp01(m.epsilon_drift / (max_epsilon if max_epsilon > 0 else 1.0))

    # TruthΩ, κ are already [0,1]
    omega_good = _clamp01(m.truth_omega)
    kappa_good = _clamp01(m.kappa_alignment)

    # Violations -> flags
    if omega_good < min_truth_omega:
        flags.append("LOW_TRUTH_OMEGA")
    if m.delta_coherence > max_delta:
        flags.append("HIGH_DELTA_COHERENCE")
    if kappa_good < min_kappa:
        flags.append("LOW_KAPPA_ALIGNMENT")
    if m.epsilon_drift > max_epsilon:
        flags.append("HIGH_EPSILON_DRIFT")

    # Confidence: weighted geometric mean of “good” and inverse “bad”
    # Use logs to avoid underflow.
    inv_delta = _clamp01(1.0 - delta_bad)
    inv_eps = _clamp01(1.0 - eps_bad)

    weights = {
        "omega": 0.40,
        "kappa": 0.25,
        "inv_delta": 0.20,
        "inv_eps": 0.15,
    }

    def wlog(x: float, w: float) -> float:
        x = max(x, 1e-12)
        return w * math.log(x)

    log_c = (
        wlog(omega_good, weights["omega"]) +
        wlog(kappa_good, weights["kappa"]) +
        wlog(inv_delta, weights["inv_delta"]) +
        wlog(inv_eps, weights["inv_eps"])
    )
    confidence = math.exp(log_c)
    confidence = _clamp01(confidence)

    # Impossibility: not “1-confidence”.
    # It spikes when contradictions cluster (multiple flags).
    # Combine two components:
    # (A) threshold-violation density
    # (B) worst-case fragility (ε + Δ)
    violation_density = len(flags) / 4.0  # 0..1
    fragility = _clamp01(0.5 * delta_bad + 0.5 * eps_bad)

    impossibility = _clamp01(0.65 * violation_density + 0.35 * fragility)

    return ICEEnvelope(
        confidence=confidence,
        impossibility=impossibility,
        flags=flags,
        thresholds={
            "min_truth_omega": float(min_truth_omega),
            "max_delta": float(max_delta),
            "min_kappa": float(min_kappa),
            "max_epsilon": float(max_epsilon),
        },
    )