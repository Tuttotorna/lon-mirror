from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

from .types import (ActionBundle, OmniaMetrics, WorldProxy,
                    Decision, Boundary, GovDecision, Certificate)
from .metrics import compute_cvi, compute_iri_act, compute_hci

@dataclass(frozen=True)
class GovThresholds:
    theta_act: float = 0.55
    theta_hci: float = 0.70
    theta_cvi: float = 0.60
    sei_near: float = 0.10
    sei_sat: float = 0.03
    iri_hard: float = 0.80

def boundary_state(sei: float, th: GovThresholds) -> Boundary:
    if sei <= th.sei_sat:
        return Boundary.SATURATED
    if sei <= th.sei_near:
        return Boundary.NEAR
    return Boundary.OPEN

def decide(action: ActionBundle,
           omnia: OmniaMetrics,
           world: Optional[WorldProxy] = None,
           th: GovThresholds = GovThresholds()) -> GovDecision:

    cvi = compute_cvi(action)
    iri_act = compute_iri_act(world)
    hci = compute_hci(omnia)
    bnd = boundary_state(omnia.sei, th)

    reasons: List[str] = []
    cert: Optional[Certificate] = None

    # Hard safety stops
    if iri_act >= th.theta_act:
        reasons.append("IRI_act above threshold (irreversible action surface).")
        return GovDecision(
            decision=Decision.REFUSE,
            boundary=bnd,
            reasons=reasons,
            scores={"omega": omnia.omega, "delta_omega": omnia.delta_omega, "sei": omnia.sei, "iri": omnia.iri,
                    "cvi": cvi, "iri_act": iri_act, "hci": hci},
            certificate=None
        )

    if cvi >= th.theta_cvi:
        reasons.append("CVI above threshold (external constraints mismatch).")
        return GovDecision(
            decision=Decision.REFUSE,
            boundary=bnd,
            reasons=reasons,
            scores={"omega": omnia.omega, "delta_omega": omnia.delta_omega, "sei": omnia.sei, "iri": omnia.iri,
                    "cvi": cvi, "iri_act": iri_act, "hci": hci},
            certificate=None
        )

    if hci >= th.theta_hci:
        reasons.append("HCI above threshold (hypercoherence / sterile saturation risk).")
        decision = Decision.BOUNDARY_ONLY
        if omnia.iri >= th.iri_hard:
            reasons.append("IRI hard threshold also exceeded.")
            decision = Decision.REFUSE
        return GovDecision(
            decision=decision,
            boundary=bnd,
            reasons=reasons,
            scores={"omega": omnia.omega, "delta_omega": omnia.delta_omega, "sei": omnia.sei, "iri": omnia.iri,
                    "cvi": cvi, "iri_act": iri_act, "hci": hci},
            certificate=None
        )

    # Boundary handling
    if bnd == Boundary.SATURATED:
        reasons.append("SEI indicates saturated regime (stop recommended).")
        cert = Certificate(type="SNRC-ACT", issued=True, notes="Action-space boundary reached (SEI saturated).")
        return GovDecision(
            decision=Decision.BOUNDARY_ONLY,
            boundary=bnd,
            reasons=reasons,
            scores={"omega": omnia.omega, "delta_omega": omnia.delta_omega, "sei": omnia.sei, "iri": omnia.iri,
                    "cvi": cvi, "iri_act": iri_act, "hci": hci},
            certificate=cert
        )

    if bnd == Boundary.NEAR:
        reasons.append("SEI indicates near-boundary regime (restricted output).")
        return GovDecision(
            decision=Decision.BOUNDARY_ONLY,
            boundary=bnd,
            reasons=reasons,
            scores={"omega": omnia.omega, "delta_omega": omnia.delta_omega, "sei": omnia.sei, "iri": omnia.iri,
                    "cvi": cvi, "iri_act": iri_act, "hci": hci},
            certificate=None
        )

    reasons.append("Open regime with compatible constraints.")
    return GovDecision(
        decision=Decision.ALLOW,
        boundary=bnd,
        reasons=reasons,
        scores={"omega": omnia.omega, "delta_omega": omnia.delta_omega, "sei": omnia.sei, "iri": omnia.iri,
                "cvi": cvi, "iri_act": iri_act, "hci": hci},
        certificate=None
    )