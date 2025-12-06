"""
OMNIA_TOTALE_TOKENMAP v0.1
Token-level Ω-maps built on top of OMNIA_TOTALE v0.5.

Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

This module computes a per-token Ω-score using OMNIA_TOTALE for
sliding windows over a text. It is a prototype for:
- visualizing coherence / instability along a sequence,
- feeding Ω-maps into LLM eval or training pipelines.

NOTE:
    Adjust the import below so that omnia_totale_score is imported
    from your OMNIA_TOTALE v0.5 core file.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List

import math
import numpy as np

# Import the fused Ω-score from your core module.
# For example, if you rename OMNIA_TOTALE_v0.5.py to omnia_totale_core.py:
# from omnia_totale_core import omnia_totale_score
from OMNIA_TOTALE_v0_5 import omnia_totale_score  # adjust to your setup


# =========================
# 1. TOKENIZATION UTILITIES
# =========================

def simple_tokenize(text: str) -> List[str]:
    """
    Ultra-simple tokenizer: whitespace split + strip.
    For real use, replace with model tokenizer (e.g., tiktoken, sentencepiece, etc.).
    """
    raw = text.strip().split()
    return [t for t in raw if t]


def token_to_int(token: str, modulus: int = 10**9 + 7) -> int:
    """
    Map a token to a non-negative integer via hash.
    This integer is fed into the BASE lens (Omniabase).
    """
    return abs(hash(token)) % modulus


# =========================
# 2. WINDOW FEATURES
# =========================

def _window_indices(center: int, radius: int, n_tokens: int) -> range:
    start = max(0, center - radius)
    end = min(n_tokens, center + radius + 1)
    return range(start, end)


def build_series_for_token_window(tokens: List[str],
                                  idx: int,
                                  radius: int = 5) -> Dict[str, np.ndarray]:
    """
    Build simple numeric series for a token-centered window.

    These are generic structural features; in a real LLM integration
    they would be replaced by logprobs, activations, etc.
    """
    n = len(tokens)
    win = list(_window_indices(idx, radius, n))

    lens = np.array([len(tokens[i]) for i in win], dtype=float)
    alpha_ratio = np.array(
        [
            sum(ch.isalpha() for ch in tokens[i]) / max(1, len(tokens[i]))
            for i in win
        ],
        dtype=float,
    )
    digit_ratio = np.array(
        [
            sum(ch.isdigit() for ch in tokens[i]) / max(1, len(tokens[i]))
            for i in win
        ],
        dtype=float,
    )

    # Single "time series" for Omniatempo (lengths),
    # and multivariate for Omniacausa.
    return {
        "series": lens,
        "series_dict": {
            "len": lens,
            "alpha_ratio": alpha_ratio,
            "digit_ratio": digit_ratio,
        },
    }


# =========================
# 3. TOKEN-LEVEL Ω MAP
# =========================

@dataclass
class TokenOmega:
    index: int
    token: str
    omega: float
    base_instability: float
    tempo_log_regime: float
    causa_mean_strength: float

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_token_omega_map(
    text: str,
    radius: int = 5,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
) -> List[TokenOmega]:
    """
    Compute a token-level Ω-map for the given text.

    For each token:
      - n = hash(token)  -> BASE lens
      - local window     -> TIME + CAUSA lenses
      - call omnia_totale_score(...)
    """
    tokens = simple_tokenize(text)
    n_tokens = len(tokens)
    results: List[TokenOmega] = []

    for i, tok in enumerate(tokens):
        n = token_to_int(tok)
        feats = build_series_for_token_window(tokens, i, radius=radius)
        series = feats["series"]
        series_dict = feats["series_dict"]

        res = omnia_totale_score(
            n=n,
            series=series,
            series_dict=series_dict,
            bases=bases,
        )

        comp = res.components
        results.append(
            TokenOmega(
                index=i,
                token=tok,
                omega=res.omega_score,
                base_instability=comp.get("base_instability", 0.0),
                tempo_log_regime=comp.get("tempo_log_regime", 0.0),
                causa_mean_strength=comp.get("causa_mean_strength", 0.0),
            )
        )

    return results


# =========================
# 4. SIMPLE TEXT REPORT
# =========================

def format_omega_map(tokens_omega: List[TokenOmega],
                     max_width: int = 80) -> str:
    """
    Return a compact textual visualization of Ω along the sequence.
    Each token is annotated with a small bar proportional to Ω.
    """
    if not tokens_omega:
        return ""

    omegas = np.array([t.omega for t in tokens_omega], dtype=float)
    if omegas.size == 0:
        return ""

    om_min = float(omegas.min())
    om_max = float(omegas.max())
    span = max(1e-9, om_max - om_min)

    lines = []
    for t in tokens_omega:
        norm = (t.omega - om_min) / span
        bar_len = max(1, int(norm * (max_width // 4)))
        bar = "█" * bar_len
        lines.append(
            f"[{t.index:03d}] {t.token:<15} Ω={t.omega: .4f}  {bar}"
        )
    return "\n".join(lines)


# =========================
# 5. DEMO
# =========================

def demo():
    """
    Minimal demo: run
        python OMNIA_TOTALE_TOKENMAP_v0.1.py
    after making sure OMNIA_TOTALE v0.5 is importable.
    """
    sample_text = (
        "OMNIA_TOTALE builds a fused Ω-score across base, time, "
        "and causal lenses to detect structural instability and coherence."
    )

    omegas = compute_token_omega_map(sample_text, radius=4)
    print("=== OMNIA_TOTALE_TOKENMAP v0.1 demo ===")
    print(format_omega_map(omegas))


if __name__ == "__main__":
    demo()