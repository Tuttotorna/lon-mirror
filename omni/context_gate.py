import math
from dataclasses import dataclass
from typing import Dict, Any, Optional
from omniabaseplus_percent_v1 import evaluate, OmniabasePlusResult

DOMAIN_VR = {
    "medical": 1.0,
    "legal": 1.0,
    "science": 0.9,
    "engineering": 0.85,
    "general": 0.7,
    "narrative": 0.3,
    "gossip": 0.1,
}

@dataclass
class GateDecision:
    domain: str
    vr: float
    threshold: int
    passed: int
    allow: bool
    reason: str
    result: OmniabasePlusResult

def decide(statement: str, context: Optional[Dict[str, Any]] = None, domain: str = "general") -> GateDecision:
    ctx = context or {}
    vr = DOMAIN_VR.get(domain, DOMAIN_VR["general"])
    r = evaluate(statement, ctx)

    # Hard invariant: B2 fail blocks always
    if r.pass_b2 == 0:
        return GateDecision(domain, vr, 3, 0, False, "B2 failed (impossibility / contradiction)", r)

    passed = r.pass_b2 + r.pass_b4 + r.pass_b8
    threshold = int(math.ceil(3 * vr))
    allow = passed >= threshold

    if allow:
        reason = f"ALLOW: passed {passed}/3 >= threshold {threshold} (VR={vr})"
    else:
        failed = []
        if r.pass_b4 == 0: failed.append("B4")
        if r.pass_b8 == 0: failed.append("B8")
        reason = f"BLOCK: failed {failed}; passed {passed}/3 < threshold {threshold} (VR={vr})"

    return GateDecision(domain, vr, threshold, passed, allow, reason, r)