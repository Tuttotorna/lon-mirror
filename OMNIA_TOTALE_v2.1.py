"""
OMNIA_TOTALE v2.1 — Unified Ω-score engine
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Fuses 4 lenses into a single Ω-score:

- BASE   → Omniabase (numeric multi-base + PBII)
- TIME   → Omniatempo (temporal drift)
- CAUSA  → Omniacausa (lagged causal structure)
- TOKEN  → PBII over token proxy ints + z-score instability
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Sequence, Union, Optional
import math
import hashlib
import re

import numpy as np

# Import core lenses from the omnia package
from omnia.omniabase import omniabase_signature, pbii_index
from omnia.omniatempo import omniatempo_analyze
from omnia.omniacausa import omniacausa_analyze, OmniacausaResult


# =========================
# TOKEN LENS (PBII on text)
# =========================

TOKEN_SPLIT_REGEX = re.compile(r"\S+")


def tokenize(text_or_tokens: Union[str, Sequence[str]]) -> List[str]:
    """
    Very simple tokenizer.

    If input is:
      - str: split on whitespace
      - Sequence[str]: assume already tokenized
    """
    if isinstance(text_or_tokens, str):
        return TOKEN_SPLIT_REGEX.findall(text_or_tokens)
    else:
        return list(text_or_tokens)


def token_to_int(token: str) -> int:
    """
    Deterministic token → integer mapping using SHA256.
    Avoids Python's built-in hash (not stable across runs).

    Output range: [0, 2^32 - 1]
    """
    h = hashlib.sha256(token.encode("utf-8")).digest()
    # Take first 4 bytes as unsigned integer
    return int.from_bytes(h[:4], byteorder="big", signed=False)


@dataclass
class OmniaTokenResult:
    tokens: List[str]
    token_ints: List[int]
    pbii_scores: List[float]
    mean_pbii: float
    mean_abs_z: float  # TOKEN Ω-component

    def to_dict(self) -> Dict:
        return asdict(self)


def token_lens_pbii(
    text_or_tokens: Union[str, Sequence[str]],
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
) -> OmniaTokenResult:
    """
    Apply PBII to each token via a stable integer proxy, then compute
    z-score instability.

    Steps:
      1) tokenize
      2) map each token → integer via SHA256
      3) compute PBII(int) for each
      4) compute z-scores of PBII and mean |z| as TOKEN Ω-component
    """
    tokens = tokenize(text_or_tokens)
    if not tokens:
        return OmniaTokenResult(
            tokens=[],
            token_ints=[],
            pbii_scores=[],
            mean_pbii=0.0,
            mean_abs_z=0.0,
        )

    ints = [token_to_int(tok) for tok in tokens]
    scores = [pbii_index(v, bases=bases) for v in ints]

    arr = np.asarray(scores, dtype=float)
    mean_pb = float(arr.mean())
    std_pb = float(arr.std(ddof=0))

    if std_pb == 0.0:
        mean_abs_z = 0.0
    else:
        z = (arr - mean_pb) / (std_pb + 1e-12)
        mean_abs_z = float(np.mean(np.abs(z)))

    return OmniaTokenResult(
        tokens=tokens,
        token_ints=ints,
        pbii_scores=scores,
        mean_pbii=mean_pb,
        mean_abs_z=mean_abs_z,
    )


# =========================
# FUSED Ω ENGINE
# =========================

@dataclass
class OmniaTotaleResult:
    n: int
    omniabase: Dict[str, float]
    omniatempo: Dict[str, float]
    omniacausa: OmniacausaResult
    token: OmniaTokenResult
    omega_score: float
    components: Dict[str, float]

    def to_dict(self) -> Dict:
        # Convert nested dataclasses to JSON-safe dicts
        return {
            "n": self.n,
            "omniabase": self.omniabase,
            "omniatempo": self.omniatempo,
            "omniacausa": {
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "lag": e.lag,
                        "strength": e.strength,
                    }
                    for e in self.omniacausa.edges
                ]
            },
            "token": self.token.to_dict(),
            "omega_score": self.omega_score,
            "components": dict(self.components),
        }


def omnia_totale_score(
    n: int,
    series: Iterable[float],
    series_dict: Dict[str, Iterable[float]],
    text_or_tokens: Optional[Union[str, Sequence[str]]] = None,
    # numeric lens
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
    # temporal lens
    short_window: int = 20,
    long_window: int = 100,
    hist_bins: int = 20,
    # causal lens
    max_lag: int = 5,
    strength_threshold: float = 0.3,
    # fusion weights
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
    w_token: float = 1.0,
) -> OmniaTotaleResult:
    """
    Compute unified Ω-score using 4 lenses:

      - BASE   → PBII-style numeric instability
      - TIME   → log(1 + regime_change_score)
      - CAUSA  → mean |lagged_corr|
      - TOKEN  → mean |z(PBII_token)|

    Returns an OmniaTotaleResult with scalar Ω and detailed metadata.
    """

    # --- BASE lens (numeric) ---
    base_sig = omniabase_signature(
        n,
        bases=bases,
        length_weight=length_weight,
        length_exponent=length_exponent,
        divisibility_bonus=divisibility_bonus,
    )
    base_instability = pbii_index(n, bases=bases)

    base_meta = {
        "sigma_mean": base_sig.sigma_mean,
        "entropy_mean": base_sig.entropy_mean,
        "pbii": base_instability,
    }

    # --- TIME lens (temporal drift) ---
    tempo_res = omniatempo_analyze(
        series,
        short_window=short_window,
        long_window=long_window,
        hist_bins=hist_bins,
    )
    tempo_val = math.log(1.0 + tempo_res.regime_change_score)

    tempo_meta = {
        "global_mean": tempo_res.global_mean,
        "global_std": tempo_res.global_std,
        "short_mean": tempo_res.short_mean,
        "short_std": tempo_res.short_std,
        "long_mean": tempo_res.long_mean,
        "long_std": tempo_res.long_std,
        "regime_change_score": tempo_res.regime_change_score,
    }

    # --- CAUSA lens (causal structure) ---
    causa_res = omniacausa_analyze(
        series_dict,
        max_lag=max_lag,
        strength_threshold=strength_threshold,
    )
    if causa_res.edges:
        strengths = np.array([abs(e.strength) for e in causa_res.edges], dtype=float)
        causa_val = float(strengths.mean())
    else:
        causa_val = 0.0

    # --- TOKEN lens (PBII on tokens) ---
    if text_or_tokens is None:
        token_res = OmniaTokenResult(
            tokens=[],
            token_ints=[],
            pbii_scores=[],
            mean_pbii=0.0,
            mean_abs_z=0.0,
        )
        token_val = 0.0
    else:
        token_res = token_lens_pbii(text_or_tokens, bases=bases)
        token_val = token_res.mean_abs_z

    # --- FUSION ---
    omega = (
        w_base * base_instability
        + w_tempo * tempo_val
        + w_causa * causa_val
        + w_token * token_val
    )

    components = {
        "base_instability": base_instability,
        "tempo_log_regime": tempo_val,
        "causa_mean_strength": causa_val,
        "token_mean_abs_z": token_val,
    }

    return OmniaTotaleResult(
        n=n,
        omniabase=base_meta,
        omniatempo=tempo_meta,
        omniacausa=causa_res,
        token=token_res,
        omega_score=float(omega),
        components=components,
    )


# =========================
# DEMO
# =========================

def demo():
    """
    Minimal demo for OMNIA_TOTALE v2.1.
    Run this file directly to see example output.

    This does NOT represent any official benchmark, just a smoke test.
    """
    np.random.seed(0)

    # Numeric target
    n_prime = 173
    n_comp = 180

    # Time series with a regime shift
    t = np.arange(300)
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    # Simple causal structure
    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)
    series_dict = {"s1": s1, "s2": s2, "s3": s3}

    # Token chain example
    chain_text = (
        "Model attempts a 4-step reasoning chain, introduces subtle numeric shifts, "
        "then lands on an incorrect but coherent-looking answer."
    )

    print("=== OMNIA_TOTALE v2.1 demo ===")

    res_prime = omnia_totale_score(
        n_prime, series, series_dict, text_or_tokens=chain_text
    )
    res_comp = omnia_totale_score(
        n_comp, series, series_dict, text_or_tokens=chain_text
    )

    print(f"n={n_prime} (prime)  Ω ≈ {res_prime.omega_score:.4f}")
    print(f"  components={res_prime.components}")
    print(f"n={n_comp} (comp.)  Ω ≈ {res_comp.omega_score:.4f}")
    print(f"  components={res_comp.components}")

    print("\nCausal edges (from CAUSA lens):")
    for e in res_prime.omniacausa.edges:
        print(f"  {e.source} -> {e.target}  lag={e.lag}  strength={e.strength:.3f}")

    print("\nTOKEN lens:")
    print(f"  tokens={len(res_prime.token.tokens)}")
    print(f"  mean_pbii={res_prime.token.mean_pbii:.4f}")
    print(f"  mean_abs_z={res_prime.token.mean_abs_z:.4f}")


if __name__ == "__main__":
    demo()