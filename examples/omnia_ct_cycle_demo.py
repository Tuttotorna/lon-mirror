from __future__ import annotations

from omnia.meta.omnia_ct_cycle import default_cycle


def main() -> None:
    cycle = default_cycle()

    x = """
    Double-slit interference: 2 slits yield a stable fringe pattern.
    OMNIA measures structure only: length, repetition, compressibility, n-grams.
    2026 2025 2024 12345
    """

    r = cycle.run(x)

    print("OMNIA↔CT Cycle (OC-1.0)")
    print("omega_curve:", [round(v, 6) for v in r.omega_curve])
    print("ctc_curve:", [round(v, 6) for v in r.ctc_curve])
    print("converged:", r.converged)
    print("collapsed:", r.collapsed)
    print("steps:")
    for s in r.steps:
        print(" ", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()