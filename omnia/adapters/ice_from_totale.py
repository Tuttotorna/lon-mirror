from __future__ import annotations
from typing import Any, Dict, Sequence

from omnia.envelope import ICEInput
from omnia.metrics import truth_omega, delta_coherence, kappa_alignment


def ice_input_from_omnia_totale(
    omnia_result: Any,
) -> ICEInput:
    """
    Adapter:
    OMNIA_TOTALE result -> ICEInput (structural only)

    Rules:
    - ICE sees ONLY structure
    - fused omega is NOT trusted directly
    - lens scores are converted into a pseudo multi-base signature
    """

    lens_scores: Dict[str, float] = getattr(omnia_result, "lens_scores", {}) or {}

    # Build a fake-but-deterministic multi-base signature
    # Each lens becomes a "base"
    signatures: Dict[int, Sequence[float]] = {}

    for i, (name, score) in enumerate(sorted(lens_scores.items())):
        # signature vector must be fixed-dimension
        # [score, score^2, 1]
        signatures[i + 2] = [
            float(score),
            float(score * score),
            1.0,
        ]

    return ICEInput(
        signatures=signatures,
        meta={
            "source": "OMNIA_TOTALE",
            "lens_names": list(lens_scores.keys()),
        },
    )