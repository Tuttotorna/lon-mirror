from __future__ import annotations

import re
import zlib
from typing import Dict

from omnia.meta.measurement_projection_loss import MeasurementProjectionLoss


# -----------------------------
# Minimal Ω-like measurers
# (semantics-free, deterministic)
# -----------------------------

def omega_compressibility(x: str) -> float:
    s = x.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s).strip()
    if not s:
        return 0.0
    comp = zlib.compress(s.encode("utf-8", errors="ignore"), level=9)
    # lower ratio => "more structure" (more compressible)
    ratio = len(comp) / max(1, len(s))
    # map to [0,1] (simple monotone)
    return max(0.0, min(1.0, 1.0 - ratio))


def omega_digit_skeleton(x: str) -> float:
    # measures how stable the numeric skeleton is
    digits = re.findall(r"\d+", x)
    if not digits:
        return 0.1
    total = sum(len(d) for d in digits)
    return max(0.0, min(1.0, 0.2 + (total / 200.0)))


# -----------------------------
# "Projection" measurers
# These emulate basis forcing / observer preference:
# they first apply an asymmetric, selective transform
# then measure with same Ω-like primitive.
# -----------------------------

def _project_keep_only_numbers(x: str) -> str:
    # hard projection: keeps only digits and spaces
    return re.sub(r"[^\d ]+", "", x)

def _project_keep_only_words(x: str) -> str:
    # another hard projection: removes digits/punctuation
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

    op = MeasurementProjectionLoss(
        aperspective_measurers=aperspective,
        projected_measurers=projected,
        aggregator="trimmed_mean",
        trim_q=0.2,
    )

    x = """
    Double-slit interference: 2 slits yield a stable fringe pattern.
    OMNIA measures structure only: length, repetition, compressibility, n-grams.
    2026 2025 2024 12345
    """

    r = op.measure(x)

    print("Measurement Projection Loss (SPL-1.0)")
    print("Ω_ap  :", round(r.omega_aperspective, 6))
    print("Ω_proj:", round(r.omega_projected, 6))
    print("SPL_abs:", round(r.spl_abs, 6))
    print("SPL_rel:", round(r.spl_rel, 6))
    print("details keys:", sorted(list(r.details.keys()))[:10], "...")

if __name__ == "__main__":
    main()