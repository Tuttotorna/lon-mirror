from __future__ import annotations

from omnia.meta.total_collapse import default_tco, t_identity, t_whitespace_collapse, t_reverse, t_drop_vowels, t_shuffle_words
from omnia.meta.convergence_certificate import default_convergence_harness


def main() -> None:
    # Baseline transforms must match the aperspective "no privileged view" set.
    baseline = [
        ("id", t_identity),
        ("ws", t_whitespace_collapse),
        ("rev", t_reverse),
        ("vow-", t_drop_vowels),
        ("shuf", t_shuffle_words(seed=3)),
    ]

    tco = default_tco()
    h = default_convergence_harness(baseline_transforms=baseline, tco=tco)

    x = """
    Double-slit interference: 2 slits yield a stable fringe pattern.
    OMNIA measures structure only: length, repetition, compressibility, n-grams.
    2026 2025 2024 12345
    """

    cert = h.certify(x)

    print("OMNIA ↔ TCO Convergence Certificate (CTC-1.0)")
    print("label:", cert.label)
    print("CTC:", round(cert.ctc, 6))
    print("CI:", round(cert.ci, 6))
    print("c*:", cert.c_star)
    print("notes:", {k: round(v, 6) for k, v in cert.notes.items()})


if __name__ == "__main__":
    main()