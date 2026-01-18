from __future__ import annotations

from omnia.lenses.observer_perturbation import (
    ObserverPerturbation,
    o_identity,
    o_add_explanation,
    o_reformat_bullets,
    o_optimize_for_clarity,
)

def main() -> None:
    x = """
    Double-slit interference produces a stable fringe pattern.
    OMNIA measures structure only.
    2026 2025 2024 12345
    """

    lens = ObserverPerturbation()

    observers = [
        ("identity", o_identity),
        ("explanation", o_add_explanation("This means that: ")),
        ("bullets", o_reformat_bullets()),
        ("clarity", o_optimize_for_clarity()),
    ]

    print("Observer Perturbation Cost (OPI-1.0)")
    print("----------------------------------")

    for name, obs in observers:
        r = lens.measure(x=x, observer=obs)
        print(f"\n{name}")
        print("  Ω_ap  =", round(r.omega_aperspective, 5))
        print("  Ω_obs =", round(r.omega_observed, 5))
        print("  OPI   =", round(r.opi, 6))
        print("  ratio =", round(r.ratio, 6))

if __name__ == "__main__":
    main()