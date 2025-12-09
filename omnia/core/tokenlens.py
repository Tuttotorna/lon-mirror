"""
omnia.core.tokenlens — token-level Ω-map for LLM traces

This module turns a sequence of tokens + integer proxies into:

- PBII scores per token (via omniabase.pbii_index)
- z-scores over the PBII sequence
- a single instability summary: mean |z|

Intended use:
- pass tokens + token_numbers in OmniaContext.extra
- the engine wraps this as an additional Ω-lens.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Dict

import numpy as np

from .omniabase import pbii_index


@dataclass
class TokenOmegaMap:
    tokens: List[str]
    numbers: List[int]
    pbii_scores: List[float]
    z_scores: List[float]
    mean_abs_z: float

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_token_omega_map(
    tokens: Iterable[str],
    token_numbers: Iterable[int],
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
) -> TokenOmegaMap:
    """
    Compute PBII + z-scores for a token sequence.

    - tokens: textual tokens (as produced by a tokenizer or simple split)
    - token_numbers: integer proxies aligned 1:1 with tokens
                     (e.g., token IDs, digit-sums, etc.)
    - bases: bases used by PBII

    Returns:
        TokenOmegaMap with per-token PBII, z-scores and mean |z|.
    """
    tok_list = list(tokens)
    num_list = list(token_numbers)

    if len(tok_list) != len(num_list):
        raise ValueError("tokens and token_numbers must have the same length")

    if not tok_list:
        return TokenOmegaMap(
            tokens=[],
            numbers=[],
            pbii_scores=[],
            z_scores=[],
            mean_abs_z=0.0,
        )

    # PBII per token-number
    pbii_vals = np.array(
        [pbii_index(int(n), bases=bases) for n in num_list],
        dtype=float,
    )

    # z-scores over PBII
    mean = float(pbii_vals.mean())
    std = float(pbii_vals.std(ddof=0))
    if std > 0:
        z_vals = (pbii_vals - mean) / std
    else:
        z_vals = np.zeros_like(pbii_vals)

    mean_abs_z = float(np.mean(np.abs(z_vals)))

    return TokenOmegaMap(
        tokens=tok_list,
        numbers=[int(n) for n in num_list],
        pbii_scores=pbii_vals.tolist(),
        z_scores=z_vals.tolist(),
        mean_abs_z=mean_abs_z,
    )


__all__ = [
    "TokenOmegaMap",
    "compute_token_omega_map",
]