# omnia/runtime/compatibility_guard.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from omnia.meta.structural_compatibility import CompatibilityResult, StructuralCompatibility


@dataclass(frozen=True)
class GuardDecision:
    """
    GuardDecision is a *runtime STOP certificate*, not a recommendation.

    action:
      - "CONTINUE"  -> structurally admissible
      - "STOP"      -> structurally inadmissible (FRAGILE/IMPOSSIBLE or below floor)

    reason:
      - short structural label, no narrative

    snapshot:
      - minimal measured fields for audit (no raw text)
    """
    action: str
    reason: str
    result: CompatibilityResult
    snapshot: Dict[str, Any]


class CompatibilityGuard:
    """
    CompatibilityGuard (CG-1.0)

    This is the *runtime envelope* that turns SCI into a STOP/CONTINUE gate.

    - No semantics
    - No policy
    - No optimization
    - No retries

    It only:
      1) computes SCI from lens outputs
      2) emits STOP if the ensemble enters structurally unsafe zones
    """

    def __init__(
        self,
        *,
        sci: Optional[StructuralCompatibility] = None,
        stop_zones: Tuple[str, ...] = ("FRAGILE", "IMPOSSIBLE"),
        stop_below_floor: bool = True,
    ):
        self.sci = sci or StructuralCompatibility()
        self.stop_zones = tuple(stop_zones)
        self.stop_below_floor = bool(stop_below_floor)

    def evaluate(
        self,
        *,
        aperspective: Optional[Mapping[str, Any]] = None,
        saturation: Optional[Mapping[str, Any]] = None,
        irreversibility: Optional[Mapping[str, Any]] = None,
        redundancy: Optional[Mapping[str, Any]] = None,
        distribution: Optional[Mapping[str, Any]] = None,
        nondecision: Optional[Mapping[str, Any]] = None,
        tag: str = "",
    ) -> GuardDecision:
        r = self.sci.measure(
            aperspective=aperspective,
            saturation=saturation,
            irreversibility=irreversibility,
            redundancy=redundancy,
            distribution=distribution,
            nondecision=nondecision,
        )

        # Hard STOP by zone
        if r.zone in self.stop_zones:
            return GuardDecision(
                action="STOP",
                reason=f"ZONE_{r.zone}",
                result=r,
                snapshot=self._snapshot(r, tag=tag),
            )

        # Optional STOP by compatibility floor (even if zone is TENSE)
        if self.stop_below_floor and (r.sci < self.sci.compatible_floor):
            return GuardDecision(
                action="STOP",
                reason="SCI_BELOW_FLOOR",
                result=r,
                snapshot=self._snapshot(r, tag=tag),
            )

        return GuardDecision(
            action="CONTINUE",
            reason="ADMISSIBLE",
            result=r,
            snapshot=self._snapshot(r, tag=tag),
        )

    @staticmethod
    def _snapshot(r: CompatibilityResult, *, tag: str = "") -> Dict[str, Any]:
        d = dict(r.diagnostics)
        # keep only compact floats + core fields (no narrative)
        keep = {
            "omega_ap",
            "di_min",
            "nd_mean",
            "nd_disp",
            "iri",
            "is_saturated",
            "c_star",
            "is_redundant",
            "collapse_cost",
            "buffer",
            "penalty",
            "compatible_floor",
            "stable_floor",
            "tense_floor",
            "iri_hard",
            "dispersion_hard",
        }
        out: Dict[str, Any] = {"tag": tag, "zone": r.zone, "sci": r.sci, "compatible": r.is_compatible}
        for k in keep:
            if k in d:
                out[k] = d[k]
        # include tensions compactly
        out["tensions"] = {k: float(v) for k, v in r.tensions.items()}
        return out


# -------------------------
# Minimal demo (standalone)
# -------------------------
if __name__ == "__main__":
    guard = CompatibilityGuard()

    decision_ok = guard.evaluate(
        aperspective={"omega_score": 0.82},
        saturation={"is_saturated": False, "c_star": 5.0},
        irreversibility={"iri": 0.0},
        redundancy={"is_redundant": True, "collapse_cost": 5.0},
        distribution={"min_score": 0.86},
        nondecision={"mean_score": 0.80, "dispersion": 0.03},
        tag="case_ok",
    )
    print("case_ok:", decision_ok.action, decision_ok.reason, decision_ok.snapshot)

    decision_stop = guard.evaluate(
        aperspective={"omega_score": 0.22},
        saturation={"is_saturated": True, "c_star": 2.0},
        irreversibility={"iri": 0.22},
        redundancy={"is_redundant": False},
        distribution={"min_score": 0.30},
        nondecision={"mean_score": 0.55, "dispersion": 0.12},
        tag="case_stop",
    )
    print("case_stop:", decision_stop.action, decision_stop.reason, decision_stop.snapshot)