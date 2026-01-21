from __future__ import annotations

import re
import zlib

from omnia.meta.measurement_projection_loss import MeasurementProjectionLoss


def omega_compressibility(x: str) -> float:
    s = x.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s).strip()
    if not s:
        return 0.0
    comp = zlib.compress(s.encode("utf-8", errors="ignore"), level=9)
    ratio = len(comp) / max(1, len(s))
    return max(0.0, min(1.0, 1.0 - ratio))


def omega_digit_skeleton(x: str) -> float:
    digits = re.findall(r"\d+", x)
    if not digits:
        return 0.1
    total = sum(len(d) for d in digits)
    return max(0.0, min(1.0, 0.2 + (total / 200.0)))


def _project_keep_only_numbers(x: str) -> str:
    return re.sub(r"[^\d ]+", "", x)


def _project_keep_only_words(x: str) -> str:
    return re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ ]+", "", x)


def omega_projected_numbers(x: str) -> float:
    return omega_compressibility(_project_keep_only_numbers(x))


def omega_projected_words(x: str) -> float:
    return omega_compressibility(_project_keep_only_words(x))


def main() -> None:
    aperspective = [
        ("compressibility", omega_compressibility),
        ("digit_skeleton", omega_digit_skeleton),
    ]

    projected = [
        ("proj_numbers", omega_projected_numbers),
        ("proj_words", omega_projected_words),
    ]

    spl = MeasurementProjectionLoss(
        aperspective_measurers=aperspective,
        projected_measurers=projected,
        aggregator="trimmed_mean",
        trim_q=0.2,
    )

    x = """
    Double-slit interference: 2 slits yield a stable fringe pattern.
    OMNIA measures structure only: repetition, compressibility, n-grams.
    2026 2025 2024 12345
    """

    r = spl.measure(x)

    print("OMNIA Case Study 02 — Observer / Projection as Measurable Loss (SPL)")
    print()
    print("Ω_ap   :", round(r.omega_aperspective, 6))
    print("Ω_proj :", round(r.omega_projected, 6))
    print("SPL_abs:", round(r.spl_abs, 6))
    print("SPL_rel:", round(r.spl_rel, 6))
    print()
    print("Meaning-blind conclusion:")
    print("If SPL > 0, the projection (observer/basis forcing) is collapsing accessible structure.")
    print("This quantifies collapse as a measurement artifact, not as semantics.")


if __name__ == "__main__":
    main()