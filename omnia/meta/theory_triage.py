from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TheoryTriageInput:
    """
    Input bundle for theory triage.

    All fields are meaning-blind, post-hoc structural measurements.

    omega_hat:  Omega-set invariance estimate (higher = more invariant residue)
    sei:        Saturation / Exhaustion trend (approx: marginal yield)
    iri:        Irreversibility / hysteresis index (higher = more irreversible loss)
    opi:        Observer Perturbation Index (higher = more damage from observation/interpretation)
    sci:        Structural compatibility score in [0,1] (optional, higher = more coexistent)
    zone:       Structural zone label if available (STABLE/TENSE/FRAGILE/IMPOSSIBLE), optional
    """
    omega_hat: float
    sei: float
    iri: float
    opi: float
    sci: Optional[float] = None
    zone: Optional[str] = None


@dataclass(frozen=True)
class TheoryTriageResult:
    """
    Output is classification only.
    No semantics, no judgment of truth, no policy.
    """
    label: str  # STRUCTURAL | SATURATED | NARRATIVE
    reason_codes: List[str]
    scores: Dict[str, float]
    zone: Optional[str] = None


class TheoryTriage:
    """
    THEORY-TRIAGE-1.0

    Classifies a line of work (or a "theory artifact") by structural behavior when
    observer-like operations are introduced.

    Labels:
    - STRUCTURAL: invariants survive, low observer cost, admissible continuation possible
    - SATURATED: stable but no marginal yield (continuation likely non-productive in same domain)
    - NARRATIVE: collapses without observer; observer cost high and/or irreversibility dominates

    This is NOT a truth test.
    This is a structural admissibility classifier.
    """

    def __init__(
        self,
        *,
        # Thresholds are intentionally simple and conservative.
        omega_hat_min_structural: float = 0.30,
        opi_max_structural: float = 0.10,
        iri_max_structural: float = 0.10,

        sei_eps: float = 1e-3,  # "near zero" saturation band

        opi_min_narrative: float = 0.25,
        iri_min_narrative: float = 0.20,
        omega_hat_max_narrative: float = 0.15,

        sci_min_ok: float = 0.50,
    ):
        self.omega_hat_min_structural = float(omega_hat_min_structural)
        self.opi_max_structural = float(opi_max_structural)
        self.iri_max_structural = float(iri_max_structural)

        self.sei_eps = float(sei_eps)

        self.opi_min_narrative = float(opi_min_narrative)
        self.iri_min_narrative = float(iri_min_narrative)
        self.omega_hat_max_narrative = float(omega_hat_max_narrative)

        self.sci_min_ok = float(sci_min_ok)

    def classify(self, x: TheoryTriageInput) -> TheoryTriageResult:
        rc: List[str] = []

        # Basic score pack
        scores: Dict[str, float] = {
            "omega_hat": float(x.omega_hat),
            "sei": float(x.sei),
            "iri": float(x.iri),
            "opi": float(x.opi),
        }
        if x.sci is not None:
            scores["sci"] = float(x.sci)

        # Optional zone gates
        zone = (x.zone or "").upper().strip() or None
        if zone is not None:
            if zone in {"FRAGILE", "IMPOSSIBLE"}:
                rc.append(f"ZONE_{zone}")
            elif zone in {"TENSE", "STABLE"}:
                rc.append(f"ZONE_{zone}")
            else:
                rc.append("ZONE_UNKNOWN")

        # SCI hint (never decisive alone)
        sci_ok = True
        if x.sci is not None:
            sci_ok = float(x.sci) >= self.sci_min_ok
            rc.append("SCI_OK" if sci_ok else "SCI_LOW")

        # ---- Rule set (ordered) ----
        # 1) Narrative / collapse detection:
        # High observer cost OR high irreversibility combined with weak invariant residue.
        narrative = False
        if (x.opi >= self.opi_min_narrative and x.omega_hat <= self.omega_hat_max_narrative):
            narrative = True
            rc.append("OPI_HIGH_AND_OH_LOW")
        if (x.iri >= self.iri_min_narrative and x.omega_hat <= self.omega_hat_max_narrative):
            narrative = True
            rc.append("IRI_HIGH_AND_OH_LOW")
        if zone in {"FRAGILE", "IMPOSSIBLE"} and (x.opi >= self.opi_min_narrative or x.iri >= self.iri_min_narrative):
            narrative = True
            rc.append("ZONE_STOP_WITH_PERTURB")

        if narrative:
            return TheoryTriageResult(
                label="NARRATIVE",
                reason_codes=sorted(set(rc)),
                scores=scores,
                zone=zone,
            )

        # 2) Structural / robust detection:
        structural = (
            x.omega_hat >= self.omega_hat_min_structural
            and x.opi <= self.opi_max_structural
            and x.iri <= self.iri_max_structural
        )
        if structural:
            rc.append("OH_HIGH")
            rc.append("OPI_LOW")
            rc.append("IRI_LOW")
            if x.sei > self.sei_eps:
                rc.append("SEI_POSITIVE")
            else:
                rc.append("SEI_NEAR_ZERO")
            if not sci_ok:
                rc.append("SCI_LOW_WARNING")
            return TheoryTriageResult(
                label="STRUCTURAL",
                reason_codes=sorted(set(rc)),
                scores=scores,
                zone=zone,
            )

        # 3) Saturated detection:
        # Not collapsing, but marginal yield ~ 0 (no growth in same domain).
        if abs(x.sei) <= self.sei_eps:
            rc.append("SEI_NEAR_ZERO")
            if x.omega_hat >= self.omega_hat_max_narrative:
                rc.append("OH_NONTRIVIAL")
            if x.opi > self.opi_max_structural:
                rc.append("OPI_NONZERO")
            if x.iri > self.iri_max_structural:
                rc.append("IRI_NONZERO")
            if not sci_ok:
                rc.append("SCI_LOW_WARNING")
            return TheoryTriageResult(
                label="SATURATED",
                reason_codes=sorted(set(rc)),
                scores=scores,
                zone=zone,
            )

        # 4) Default: if none matched, treat as SATURATED (conservative)
        rc.append("DEFAULT_SATURATED")
        if x.sei > self.sei_eps:
            rc.append("SEI_POSITIVE")
        elif x.sei < -self.sei_eps:
            rc.append("SEI_NEGATIVE")
        return TheoryTriageResult(
            label="SATURATED",
            reason_codes=sorted(set(rc)),
            scores=scores,
            zone=zone,
        )