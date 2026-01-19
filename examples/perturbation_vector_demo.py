from __future__ import annotations

from omnia.meta.perturbation_vector import default_pv


def main() -> None:
    pv = default_pv()

    x = """
    Double-slit interference: 2 slits yield a stable fringe pattern.
    OMNIA measures structure only: length, repetition, compressibility, n-grams.
    2026 2025 2024 12345
    """

    r = pv.measure(x)

    print("Perturbation Vector (PV-1.0)")
    print("Ω_ap:", round(r.omega_ap, 6))
    print("Ω by type:")
    for k, v in sorted(r.omega_by_type.items()):
        print(" ", k, "->", round(v, 6))

    print("PI by type (Ω_ap - Ω_k):")
    for k, v in sorted(r.pi_by_type.items()):
        print(" ", k, "->", round(v, 6))

    print("dominant:", r.dominant_type)
    print("notes:", {k: round(v, 6) for k, v in r.notes.items()})


if __name__ == "__main__":
    main(