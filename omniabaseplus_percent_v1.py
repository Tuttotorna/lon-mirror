from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# ============================
# OmniabasePlus% v1.0 (patched)
# Deterministic diagnostic layer
# ============================

# Canonical thresholds (structural)
THETA = {"B2": 30, "B4": 40, "B8": 50}

@dataclass
class PenaltyHit:
    code: str
    points: int
    reason: str

@dataclass
class OmniabasePlusResult:
    statement: str
    omega2: int
    omega4: int
    omega8: int
    pass_b2: int
    pass_b4: int
    pass_b8: int
    bsp_structural: float
    state: str
    penalties: Dict[str, List[PenaltyHit]]
    notes: List[str]

def _clamp(v: int) -> int:
    return max(0, min(100, v))

def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())

def _has_any(s: str, kws: List[str]) -> bool:
    return any(k in s for k in kws)

def evaluate(statement: str, context: Optional[Dict[str, Any]] = None) -> OmniabasePlusResult:
    """
    context (optional) example:
      {
        "distance_km": 500,
        "requires_physical_presence": True,
        "agent_present": False,
        "time_available_hours": 0,
        "blame_or_merit_frame": True,

        # NEW (for causal hallucinations / "magic causation"):
        "claims_physical_effect": True
      }

    NEW RULE (P2.6):
      If depends_on_unobservable AND claims_physical_effect => treat as structural impossibility (Ω2 cap to 20).
    """
    ctx = context or {}
    s = _norm(statement)

    penalties: Dict[str, List[PenaltyHit]] = {"B2": [], "B4": [], "B8": []}
    notes: List[str] = []

    # -----------------------
    # B2: Possibility (Ω2)
    # -----------------------
    omega2 = 100

    # P2.1 explicit logical/physical impossibility triggers
    imposs_triggers = [
        "impossibile", "non può essere", "non puo essere", "contraddizione",
        "violando la fisica", "violazione fisica"
    ]
    if _has_any(s, imposs_triggers) or ctx.get("explicit_impossibility") is True:
        penalties["B2"].append(PenaltyHit("P2.1", 80, "Explicit logical/physical impossibility"))
        omega2 -= 80
        omega2 = min(omega2, 20)  # cap rule

    # P2.2 spatial constraint (distance/presence)
    distance_km = ctx.get("distance_km")
    requires_presence = bool(ctx.get("requires_physical_presence", False))
    agent_present = ctx.get("agent_present")
    if requires_presence:
        if agent_present is False:
            penalties["B2"].append(PenaltyHit("P2.2", 60, "Physical presence required but agent not present"))
            omega2 -= 60
            omega2 = min(omega2, 20)
        elif isinstance(distance_km, (int, float)) and distance_km >= 100:
            penalties["B2"].append(PenaltyHit("P2.2", 60, f"Presence required; distance_km={distance_km}"))
            omega2 -= 60
            omega2 = min(omega2, 20)

    # P2.3 temporal constraint
    if ctx.get("temporal_incompatibility") is True:
        penalties["B2"].append(PenaltyHit("P2.3", 40, "Temporal constraint incompatible"))
        omega2 -= 40

    # P2.4 missing agent/means
    action_verbs = ["fare", "aiut", "complet", "risolver", "costru", "mand", "scriver", "riparar", "implement"]
    if _has_any(s, action_verbs) and ctx.get("agent_defined") is False:
        penalties["B2"].append(PenaltyHit("P2.4", 30, "Action asserted but agent/means not defined in context"))
        omega2 -= 30

    # P2.5 non-observable dependency (weak penalty)
    if ctx.get("depends_on_unobservable") is True:
        penalties["B2"].append(PenaltyHit("P2.5", 20, "Depends on unobservable / inaccessible condition"))
        omega2 -= 20

    # P2.6 NEW: unobservable agent claims direct physical causation (strong)
    # This converts "unverifiable" into "structurally impossible" in the B2 sense.
    if ctx.get("depends_on_unobservable") is True and ctx.get("claims_physical_effect") is True:
        penalties["B2"].append(PenaltyHit("P2.6", 80, "Unobservable agent claims direct physical causation"))
        omega2 -= 80
        omega2 = min(omega2, 20)  # cap rule

    omega2 = _clamp(omega2)

    # -----------------------
    # B4: Causality (Ω4)
    # -----------------------
    omega4 = 100

    # P4.1 correlation-as-causation
    causation_words = ["quindi", "porta a", "causa", "provoca", "perché", "perche", "dunque"]
    if _has_any(s, causation_words) and ctx.get("mechanism_provided") is False:
        penalties["B4"].append(PenaltyHit("P4.1", 40, "Causation implied without mechanism"))
        omega4 -= 40

    # P4.2 post hoc trigger
    if _has_any(s, ["dopo", "da quando", "da allora"]) and ctx.get("post_hoc_risk") is True:
        penalties["B4"].append(PenaltyHit("P4.2", 30, "Post hoc / temporal jump risk"))
        omega4 -= 30

    # P4.3 alternatives not excluded
    if ctx.get("alternative_causes_not_excluded") is True:
        penalties["B4"].append(PenaltyHit("P4.3", 20, "Alternative causes not excluded"))
        omega4 -= 20

    # P4.4 blame/merit without possibility conditions
    blame_words = ["colpa", "vergogn", "non fai", "non mi aiuti", "sei sempre", "mai", "inconcludente"]
    blame_frame = bool(ctx.get("blame_or_merit_frame", False)) or _has_any(s, blame_words)
    if blame_frame:
        penalties["B4"].append(PenaltyHit("P4.4", 40, "Attribution (blame/merit) without explicit possibility conditions"))
        omega4 -= 40

        # hard rule: if Ω2 < 30 then force Ω4 ≤ 20
        if omega2 < THETA["B2"]:
            omega4 = min(omega4, 20)
            notes.append("Rule: Ω2 < 30 ⇒ force Ω4 ≤ 20 for blame/merit frames")

    # P4.5 declared cause but not operationally linked
    if ctx.get("cause_not_operationally_linked") is True:
        penalties["B4"].append(PenaltyHit("P4.5", 20, "Cause declared but not operationally linked to effect"))
        omega4 -= 20

    omega4 = _clamp(omega4)

    # -----------------------
    # B8: Context (Ω8)
    # -----------------------
    omega8 = 100

    # P8.1 critical premise missing
    if ctx.get("missing_critical_premise") is True:
        penalties["B8"].append(PenaltyHit("P8.1", 60, "Critical premise missing"))
        omega8 -= 60
        omega8 = min(omega8, THETA["B8"] - 1)  # hard fail threshold

    # P8.2 domain unspecified
    if ctx.get("domain_unspecified") is True or _has_any(s, ["in generale", "spesso", "molti casi"]):
        penalties["B8"].append(PenaltyHit("P8.2", 30, "Domain unspecified / generic claim"))
        omega8 -= 30

    # P8.3 missing quantities/limits
    if ctx.get("missing_quantities") is True or _has_any(s, ["significativamente", "molto", "poco", "tanto"]):
        penalties["B8"].append(PenaltyHit("P8.3", 20, "Quantities/limits missing where required"))
        omega8 -= 20

    # P8.4 ambiguous referents
    if ctx.get("ambiguous_referents") is True:
        penalties["B8"].append(PenaltyHit("P8.4", 20, "Ambiguous referents (who/what/when)"))
        omega8 -= 20

    # P8.5 generalization without perimeter
    if ctx.get("unbounded_generalization") is True:
        penalties["B8"].append(PenaltyHit("P8.5", 20, "Generalization without perimeter"))
        omega8 -= 20

    omega8 = _clamp(omega8)

    # -----------------------
    # Pass/fail + BSP
    # -----------------------
    pass_b2 = 1 if omega2 >= THETA["B2"] else 0
    pass_b4 = 1 if omega4 >= THETA["B4"] else 0
    pass_b8 = 1 if omega8 >= THETA["B8"] else 0

    bsp = 100.0 * (pass_b2 + pass_b4 + pass_b8) / 3.0

    # State
    # Hard rule: B2 fail => REJECT
    if omega2 < THETA["B2"]:
        state = "REJECT"
    else:
        if bsp == 100.0:
            state = "STRUCTURALLY_ADMISSIBLE"
        elif bsp == 0.0:
            state = "REJECT"
        else:
            state = "PARTIAL"

    return OmniabasePlusResult(
        statement=statement,
        omega2=omega2, omega4=omega4, omega8=omega8,
        pass_b2=pass_b2, pass_b4=pass_b4, pass_b8=pass_b8,
        bsp_structural=round(bsp, 1),
        state=state,
        penalties=penalties,
        notes=notes,
    )

def render(result: OmniabasePlusResult) -> str:
    lines = []
    lines.append("OmniabasePlus% v1.0 (patched)\n")
    lines.append("Truth Vector (structural):")
    lines.append(f"Ω2={result.omega2}%  Ω4={result.omega4}%  Ω8={result.omega8}%\n")
    lines.append("Pass:")
    lines.append(f"B2={result.pass_b2}  B4={result.pass_b4}  B8={result.pass_b8}\n")
    lines.append(f"BSP-Structural: {result.bsp_structural}%")
    lines.append(f"State: {result.state}\n")
    lines.append("Penalties:")
    for b in ["B2", "B4", "B8"]:
        hits = result.penalties.get(b, [])
        if not hits:
            lines.append(f"- {b}: none")
            continue
        lines.append(f"- {b}:")
        for h in hits:
            lines.append(f"  - {h.code}: -{h.points} ({h.reason})")
    if result.notes:
        lines.append("\nNotes:")
        for n in result.notes:
            lines.append(f"- {n}")
    return "\n".join(lines)

if __name__ == "__main__":
    # Demo: wife/husband case (requires context to expose the truncation)
    stmt = "Tu non mi aiuti in casa."
    ctx = {
        "requires_physical_presence": True,
        "distance_km": 500,
        "blame_or_merit_frame": True,
        "missing_critical_premise": True,
        "agent_defined": True,
        "mechanism_provided": False,
    }
    res = evaluate(stmt, ctx)
    print(render(res))