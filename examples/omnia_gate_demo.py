"""
OMNIA — Gate Demo (neutral)
OMNIA measures coherence. An external system decides.

Run:
  python examples/omnia_gate_demo.py
"""

from dataclasses import dataclass
from typing import Any, Dict, List


# -----------------------------
#  Expected OMNIA output contract
# -----------------------------
@dataclass
class OmniaResult:
    score: float               # [0,1]
    flags: List[str]
    metrics: Dict[str, float]  # delta_coherence, kappa_alignment, epsilon_drift


# -----------------------------
#  Adapter: import OMNIA if available, else stub
# -----------------------------
def truth_omega(x: Any) -> OmniaResult:
    """
    Adapter boundary.
    Replace this with the real OMNIA call in your project, e.g.:
        from omnia.engine.omnia_totale.kernel import truth_omega
        return truth_omega(x)
    """
    # Safe stub for demo-only, deterministic, not meaningful:
    s = str(x)
    score = 0.85 if len(s) < 120 and "not" not in s.lower() else 0.35
    flags = []
    if score < 0.5:
        flags.append("LOW_STABILITY")
    return OmniaResult(
        score=score,
        flags=flags,
        metrics={
            "delta_coherence": score,
            "kappa_alignment": max(0.0, score - 0.1),
            "epsilon_drift": 1.0 - score,
        },
    )


# -----------------------------
#  External decision layer (example)
# -----------------------------
def external_decision(result: OmniaResult) -> str:
    """
    External policy. OMNIA must never contain this logic.
    """
    if result.score >= 0.80:
        return "ACCEPT"
    if result.score >= 0.55:
        return "ESCALATE"
    return "REJECT"


def main() -> None:
    samples = [
        "All humans are mortal. Socrates is a human. Socrates is mortal.",
        "All humans are mortal. Socrates is a human. Socrates is not mortal.",
        "Generate a step-by-step proof, but keep changing assumptions mid-way.",
    ]

    for i, x in enumerate(samples, 1):
        r = truth_omega(x)
        decision = external_decision(r)
        print(f"\n--- SAMPLE {i} ---")
        print("X:", x)
        print("OMNIA.score:", round(r.score, 4))
        print("OMNIA.flags:", r.flags)
        print("OMNIA.metrics:", {k: round(v, 4) for k, v in r.metrics.items()})
        print("EXTERNAL.decision:", decision)


if __name__ == "__main__":
    main()