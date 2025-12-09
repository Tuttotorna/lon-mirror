"""
omnia.core.tokenlens — token-level Ω-map for LLM traces

This module maps a sequence of tokens (and associated integer proxies)
to PBII-based instability scores and z-scores.

It is model-agnostic: you can feed any tokenization as long as you
provide an integer per token (e.g. vocab ID, digit-sum, hash, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Iterable

import numpy as np

from .omniabase import pbii_index


@dataclass
class TokenOmegaMap:
    """
    Container for token-level Ω information.

    Attributes:
        tokens: original tokens (strings).
        token_numbers: integer proxies used for PBII.
        pbii_scores: PBII per token.
        z_scores: z-score of PBII over the sequence.
        mean_abs_z: mean absolute z-score (global instability indicator).
    """
    tokens: List[str]
    token_numbers: List[int]
    pbii_scores: List[float]
    z_scores: List[float]
    mean_abs_z: float

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_token_omega_map(
    tokens: Iterable[str],
    token_numbers: Iterable[int],
) -> TokenOmegaMap:
    """
    Compute token-level PBII + z-scores for a sequence.

    Args:
        tokens: sequence of tokens (strings).
        token_numbers: integers associated to each token; must have
                       the same length as `tokens`.

    Returns:
        TokenOmegaMap dataclass with per-token PBII, z-scores and mean |z|.
    """
    tokens = list(tokens)
    nums = list(token_numbers)

    if len(tokens) != len(nums):
        raise ValueError("tokens and token_numbers must have the same length")

    if not tokens:
        return TokenOmegaMap(
            tokens=[],
            token_numbers=[],
            pbii_scores=[],
            z_scores=[],
            mean_abs_z=0.0,
        )

    # PBII per token
    pbii_vals = np.array([pbii_index(int(n)) for n in nums], dtype=float)

    # z-score normalisation
    mean = float(pbii_vals.mean())
    std = float(pbii_vals.std(ddof=0))
    if std == 0.0:
        z = np.zeros_like(pbii_vals, dtype=float)
    else:
        z = (pbii_vals - mean) / std

    mean_abs_z = float(np.mean(np.abs(z)))

    return TokenOmegaMap(
        tokens=tokens,
        token_numbers=nums,
        pbii_scores=pbii_vals.tolist(),
        z_scores=z.tolist(),
        mean_abs_z=mean_abs_z,
    )