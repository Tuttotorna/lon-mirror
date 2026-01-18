from __future__ import annotations

from omnia.lenses.observer_perturbation import (
    ObserverPerturbation,
    o_add_explanation,
    o_reformat_bullets,
    o_optimize_for_clarity,
    o_identity,
)

def main() -> None:
    x = """
    Double-slit interference: 2 slits yield a stable fringe pattern.
    OMNIA measures structure only: length, repetition, compressibility, n-grams.
    2026 2025 2024 12345
    """

    opi = ObserverPerturbation()

    observers = [
        ("identity", o_identity),
        ("explanation", o_add_explanation("This means that: ")),
        ("bullets", o_reformat_bullets()),
        ("clarity", o_optimize_for_clarity()),
    ]

    print("Observer Perturbation Index (OPI-1.0)")
    for name, obs in observers:
        r = opi.measure(x=x, observer=obs)
        print(f"\n[{name}]")
        print("Ω_ap   =", round(r.omega_aperspective, 4))
        print("Ω_obs  =", round(r.omega_observed, 4))
        print("OPI    =", round(r.opi, 6))
        print("ratio  =", round(r.ratio, 6))

if __name__ == "__main__":
    main()