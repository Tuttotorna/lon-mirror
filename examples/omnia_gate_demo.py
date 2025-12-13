"""
OMNIA — Gate Demo (neutral)
OMNIA measures coherence. An external system decides.

Run:
  python examples/omnia_gate_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


# -----------------------------
# Expected OMNIA output contract
# -----------------------------
@dataclass(frozen=True)
class OmniaResult:
    score: float               # [0,1]
    flags: List[str]
    metrics: Dict[str, float]  # delta_coherence, kappa_alignment, epsilon_drift


# -----------------------------
# Adapter: use real OMNIA if available, else deterministic stub
# -----------------------------
def truth_omega(x: Any) -> OmniaResult:
    """
    Adapter boundary.

    If OMNIA is installed in this repo, this demo uses the real kernel:
        from omnia.engine.omnia_totale.kernel import truth_omega

    Otherwise it falls back to a deterministic stub (demo-only).
    """
    try:
        from omnia.engine.omnia_totale.kernel import truth_omega as _truth_omega  # type: ignore

        r = _truth_omega(x)

        # Normalize to this demo contract (in case the real kernel returns a dict/object)
        if isinstance(r, OmniaResult):
            return r

        if isinstance(r, dict):
            return OmniaResult(
                score=float(r.get("score", r.get("omega_score", 0.0))),
                flags=list(r.get("flags", [])),
                metrics=dict(r.get("metrics", r.get("components", {}))),
            )

        # Generic object with attributes
        score = float(getattr(r, "score", getattr(r, "omega_score", 0.0)))
        flags = list(getattr(r, "flags", []))
        metrics = dict(getattr(r, "metrics", getattr(r, "components", {})))
        return OmniaResult(score=score, flags=flags, metrics=metrics)

    except Exception:
        # Deterministic stub for demo-only (not meaningful)
        s = str(x)
        score = 0.85 if len(s) < 120 and "not" not in s.lower() else 0.35
        flags: List[str] = ["LOW_STABILITY"] if score < 0.5 else []
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
# External decision layer (example)
# -----------------------------
def external_decision(result: OmniaResult) -> str:
    """
    External policy.
    OMNIA must never contain this logic.
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
        print("OMNIA.score:", round(float(r.score), 4))
        print("OMNIA.flags:", list(r.flags))
        print("OMNIA.metrics:", {k: round(float(v), 4) for k, v in dict(r.metrics).items()})
        print("EXTERNAL.decision:", decision)


if __name__ == "__main__":
    main()