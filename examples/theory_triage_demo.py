from __future__ import annotations

from omnia.meta.theory_triage import TheoryTriage, TheoryTriageInput

def main() -> None:
    triage = TheoryTriage()

    # Example A: robust structural line (low observer cost, low irreversibility, decent residue)
    a = TheoryTriageInput(
        omega_hat=0.45,
        sei=0.02,
        iri=0.03,
        opi=0.05,
        sci=0.78,
        zone="STABLE",
    )

    # Example B: saturated line (non-trivial residue but no marginal yield)
    b = TheoryTriageInput(
        omega_hat=0.28,
        sei=0.0002,
        iri=0.08,
        opi=0.12,
        sci=0.62,
        zone="TENSE",
    )

    # Example C: narrative/collapse (high perturbation, weak residue)
    c = TheoryTriageInput(
        omega_hat=0.08,
        sei=0.0001,
        iri=0.25,
        opi=0.40,
        sci=0.30,
        zone="FRAGILE",
    )

    for name, x in [("A", a), ("B", b), ("C", c)]:
        r = triage.classify(x)
        print("\nCase", name)
        print(" label:", r.label)
        print(" zone:", r.zone)
        print(" reason_codes:", ", ".join(r.reason_codes))
        print(" scores:", {k: round(v, 6) for k, v in r.scores.items()})

if __name__ == "__main__":
    main()