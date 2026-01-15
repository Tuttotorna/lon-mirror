from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence, Optional, Dict, Any

from .metrics import truth_omega, delta_coherence, kappa_alignment


class ICEStatus(str, Enum):
    OK = "OK"                    # structurally coherent, safe to pass through
    WARN = "WARN"                # mildly unstable, pass with caution
    FAIL = "FAIL"                # structurally unstable, should be blocked/escalated
    ESCALATE = "ESCALATE"        # ambiguous edge-case, requires external judge


@dataclass(frozen=True)
class ICEInput:
    """
    signatures: base -> signature vector (same dimension for all bases)
    Optional metadata can be used by decision layers, but ICE itself measures only structure.
    """
    signatures: Mapping[int, Sequence[float]]
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ICEResult:
    status: ICEStatus
    truth_omega: float
    delta: float
    kappa: float
    confidence: float
    reasons: Sequence[str]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def ice_gate(
    ice_in: ICEInput,
    *,
    # Default thresholds are intentionally conservative and deterministic.
    thr_fail_truth: float = 0.35,
    thr_warn_truth: float = 0.55,
    thr_low_kappa: float = 0.25,
    thr_high_delta: float = 1.25,
    thr_escalate_band: float = 0.05,
) -> ICEResult:
    """
    ICE gate: produces a status + confidence envelope from structural metrics.

    Rules (aligned with the boundary you already fixed publicly):
    - Flags do NOT overwrite thresholds: status is derived from metrics only.
    - Edge cases -> ESCALATE (narrow bands around thresholds).
    - Confidence = reliability of the structural signal, not certainty of the content.

    confidence heuristic:
      conf = clamp01(0.60*TruthΩ + 0.40*((kappa+1)/2)) penalized by high Δ
      penalty = clamp01(delta / (delta + 1))
      conf = conf * (1 - 0.35*penalty)
    """
    sig = ice_in.signatures

    d = delta_coherence(sig)
    t = truth_omega(sig)
    k = kappa_alignment(sig)

    reasons = []

    # Edge band detector around truth thresholds
    def in_band(x: float, thr: float) -> bool:
        return abs(x - thr) <= thr_escalate_band

    # Primary fail conditions
    if t < thr_fail_truth or d > thr_high_delta:
        reasons.append("low_truth_or_high_delta")
        status = ICEStatus.FAIL

        # Escalate if near boundary (ambiguous)
        if in_band(t, thr_fail_truth) or in_band(d, thr_high_delta):
            reasons.append("edge_case_near_threshold")
            status = ICEStatus.ESCALATE

    # Warning conditions
    elif t < thr_warn_truth or k < thr_low_kappa:
        reasons.append("moderate_instability")
        status = ICEStatus.WARN

        if in_band(t, thr_warn_truth) or in_band(k, thr_low_kappa):
            reasons.append("edge_case_near_threshold")
            status = ICEStatus.ESCALATE

    else:
        reasons.append("structurally_coherent")
        status = ICEStatus.OK

    # Confidence envelope (signal reliability)
    base_conf = 0.60 * _clamp01(t) + 0.40 * _clamp01((k + 1.0) / 2.0)
    penalty = _clamp01(d / (d + 1.0))
    conf = _clamp01(base_conf * (1.0 - 0.35 * penalty))

    return ICEResult(
        status=status,
        truth_omega=float(t),
        delta=float(d),
        kappa=float(k),
        confidence=float(conf),
        reasons=tuple(reasons),
    )