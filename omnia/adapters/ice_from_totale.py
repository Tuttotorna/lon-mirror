from __future__ import annotations

from typing import Any, Dict, Sequence, List

from omnia.envelope import ICEInput


def ice_input_from_omnia_totale(omnia_result: Any) -> ICEInput:
    """
    Adapter:
      OMNIA_TOTALE result -> structural ICEInput

    ICE must see ONLY structure.
    We convert per-lens omega scores into a deterministic multi-base signature map:
      base_id -> fixed-dim vector (dim=5)

    This keeps:
    - ICE independent from fused omega_total
    - no decision logic
    - deterministic mapping
    """

    lens_scores: Dict[str, float] = getattr(omnia_result, "lens_scores", {}) or {}

    # Stable order (deterministic)
    items = sorted((str(k), float(v)) for k, v in lens_scores.items())

    signatures: Dict[int, Sequence[float]] = {}

    # Map each lens to a pseudo-base id starting at 2 (avoid base<2)
    # Fixed dimension = 5:
    # [score, score^2, 1-score, abs(score-0.5), 1]
    for idx, (name, score) in enumerate(items):
        b = idx + 2
        s = float(score)
        signatures[b] = [
            s,
            s * s,
            1.0 - s,
            abs(s - 0.5),
            1.0,
        ]

    meta = {
        "source": "OMNIA_TOTALE",
        "lens_names": [name for name, _ in items],
        "mapping": {name: (i + 2) for i, (name, _) in enumerate(items)},
    }

    return ICEInput(signatures=signatures, meta=meta)