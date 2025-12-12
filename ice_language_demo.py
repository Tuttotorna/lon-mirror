"""
ice_language_demo.py

OMNIA ICE — Language/Idiom ambiguity demo (v0.1)

Purpose:
  Show why "0% is false" must be enforced at the language layer too:
    - idioms ("I'm dying laughing")
    - ellipsis ("The system deleted the user data" vs "deleted the user")
    - polysemy ("kill the pain" vs "kill the patient")

This demo DOES NOT call any model.
It only feeds ICE with:
  - a synthetic omega_total (structural instability proxy)
  - an ambiguity_score (multi-interpretation proxy)
  - optional omega_ext (LCR-like coherence, 0..1)

Author: Massimiliano Brighindi (MB-X.01 / OMNIA)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, List

# Adjust the import path if your ICE file has a different name.
from ICE.OMNIA_ICE_v0_1 import ICEInput, ice_gate


def _mk_case(
    case_id: str,
    text: str,
    omega_total: float,
    ambiguity_score: float,
    omega_ext: float | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    x = ICEInput(
        omega_total=omega_total,
        lens_scores={"BASE": 0.0, "TIME": 0.0, "CAUSA": 0.0, "TOKEN": 0.0, "LCR": 0.0},
        lens_metadata={"LANG": {"text": text, "case_id": case_id}},
        omega_ext=omega_ext,
        gold_match=None,
        ambiguity_score=ambiguity_score,
        notes=notes,
    )
    y = ice_gate(x)
    return {
        "case_id": case_id,
        "text": text,
        "input": {
            "omega_total": omega_total,
            "omega_ext": omega_ext,
            "ambiguity_score": ambiguity_score,
            "notes": notes,
        },
        "result": y.to_dict(),
    }


def main() -> None:
    cases: List[Dict[str, Any]] = []

    # 1) Idiom: literally impossible, but human-acceptable as idiom.
    # ICE should ESCALATE (ambiguity high) rather than PASS.
    cases.append(
        _mk_case(
            "IDIOM_1",
            "I'm dying laughing.",
            omega_total=0.10,
            ambiguity_score=0.80,
            omega_ext=0.65,
            notes="Idiom: literal reading is impossible; figurative reading is valid.",
        )
    )

    # 2) Polysemy: "kill" can mean eliminate (pain), not murder.
    cases.append(
        _mk_case(
            "POLYSEMY_1",
            "The drug killed the pain in minutes.",
            omega_total=0.08,
            ambiguity_score=0.60,
            omega_ext=0.75,
            notes="Polysemy: 'kill' applies to pain, not a person.",
        )
    )

    # 3) Ellipsis (safe): explicit object makes the statement well-formed.
    cases.append(
        _mk_case(
            "ELLIPSIS_SAFE",
            "The system deleted the user's data.",
            omega_total=0.06,
            ambiguity_score=0.20,
            omega_ext=0.80,
            notes="Explicit object: likely PASS if risk remains low.",
        )
    )

    # 4) Ellipsis (danger): missing object can flip meaning into a 0%-class claim.
    cases.append(
        _mk_case(
            "ELLIPSIS_DANGER",
            "The system deleted the user.",
            omega_total=0.25,
            ambiguity_score=0.85,
            omega_ext=0.40,
            notes="Underdetermined: could be account removal (valid) or physical harm (impossible in most contexts).",
        )
    )

    # 5) Contradiction example: literal impossible claim.
    cases.append(
        _mk_case(
            "HARD_CONTRA_1",
            "Socrates is immortal.",
            omega_total=0.60,
            ambiguity_score=0.55,
            omega_ext=0.20,
            notes="Likely BLOCK/ESCALATE depending on thresholds; literal claim is impossible.",
        )
    )

    print("=== OMNIA ICE language demo (v0.1) ===\n")
    for c in cases:
        r = c["result"]
        print(f"[{c['case_id']}] {c['text']}")
        print(
            f"  input: omega_total={c['input']['omega_total']:.3f}  "
            f"omega_ext={c['input']['omega_ext']}  "
            f"ambiguity={c['input']['ambiguity_score']:.2f}"
        )
        print(
            f"  ICE: status={r['status']}  confidence={r['confidence']:.3f}  "
            f"impossibility={r['impossibility']:.3f}  ambiguity={r['ambiguity']:.3f}"
        )
        print(
            f"  reasons: risk_struct={r['reasons']['risk_struct']:.3f}  "
            f"risk_ext={r['reasons']['risk_ext']:.3f}  "
            f"risk_gold={r['reasons']['risk_gold']:.3f}"
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()