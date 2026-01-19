from __future__ import annotations

from omnia.meta.total_collapse import default_tco


def main() -> None:
    tco = default_tco()

    x = """
    Double-slit interference: 2 slits yield a stable fringe pattern.
    OMNIA measures structure only: length, repetition, compressibility, n-grams.
    2026 2025 2024 12345
    """

    r = tco.measure(x)

    print("Total Collapse Operator (TCO-1.0)")
    print("Ω_ap (baseline):", round(r.omega0, 6))
    print("Ω curve:", [round(v, 6) for v in r.omega_curve])
    print("cost curve:", r.cost_curve)
    print("c* (collapse point):", r.c_star)
    print("collapse index:", round(r.collapse_index, 6))
    print("notes:", {k: round(v, 6) for k, v in r.notes.items()})


if __name__ == "__main__":
    main()