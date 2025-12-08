"""
OMNIA_TOTALE v2.0 — Unified Ω-fusion engine
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Dependencies:
    pip install numpy
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Optional, Any
import math
import json

import numpy as np

# =========================
# 1. OMNIABASE (multi-base)
# =========================

def digits_in_base_np(n: int, b: int) -> np.ndarray:
    """Return digits of n in base b as numpy array (MSB first)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if b <= 1:
        raise ValueError("base must be >= 2")
    if n == 0:
        return np.array([0], dtype=int)
    digits: List[int] = []
    while n > 0:
        digits.append(n % b)
        n //= b
    return np.array(digits[::-1], dtype=int)


def normalized_entropy_base(n: int, b: int) -> float:
    """Normalized Shannon entropy of digits of n in base b."""
    digits = digits_in_base_np(n, b)
    L = len(digits)
    if L == 0:
        return 0.0
    counts = np.bincount(digits, minlength=b).astype(float)
    probs = counts[counts > 0] / L
    if probs.size == 0:
        return 0.0
    H = -np.sum(probs * np.log2(probs))
    Hmax = math.log2(b)
    return float(H / Hmax) if Hmax > 0 else 0.0


def sigma_b(
    n: int,
    b: int,
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> float:
    """
    Base Symmetry Score (NumPy version).

    sigma_b(n) = length_weight * (1 - H_norm) / L^length_exponent
                 + divisibility_bonus * I[n % b == 0]
    """
    digits = digits_in_base_np(n, b)
    L = len(digits)
    if L == 0:
        return 0.0

    counts = np.bincount(digits, minlength=b).astype(float)
    probs = counts[counts > 0] / L
    if probs.size == 0:
        Hn = 0.0
    else:
        H = -np.sum(probs * np.log2(probs))
        Hmax = math.log2(b)
        Hn = float(H / Hmax) if Hmax > 0 else 0.0

    length_term = length_weight * (1.0 - Hn) / (L ** length_exponent)
    div_term = divisibility_bonus * (1.0 if n % b == 0 else 0.0)
    return float(length_term + div_term)


@dataclass
class OmniabaseSignature:
    n: int
    bases: List[int]
    sigmas: Dict[int, float]
    entropy: Dict[int, float]
    sigma_mean: float
    entropy_mean: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def omniabase_signature(
    n: int,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> OmniabaseSignature:
    """Compute multi-base signature for integer n."""
    bases_list = list(bases)
    sigmas: Dict[int, float] = {}
    entropy: Dict[int, float] = {}
    for b in bases_list:
        sig = sigma_b(
            n,
            b,
            length_weight=length_weight,
            length_exponent=length_exponent,
            divisibility_bonus=divisibility_bonus,
        )
        Hn = normalized_entropy_base(n, b)
        sigmas[b] = sig
        entropy[b] = Hn
    sigma_vals = np.array(list(sigmas.values()), dtype=float)
    ent_vals = np.array(list(entropy.values()), dtype=float)
    return OmniabaseSignature(
        n=n,
        bases=bases_list,
        sigmas=sigmas,
        entropy=entropy,
        sigma_mean=float(sigma_vals.mean()) if sigma_vals.size else 0.0,
        entropy_mean=float(ent_vals.mean()) if ent_vals.size else 0.0,
    )


def pbii_index(
    n: int,
    composite_window: Iterable[int] = (4, 6, 8, 9, 10, 12, 14, 15),
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
) -> float:
    """
    Prime Base Instability Index (NumPy variant).

    PBII(n) = mean_sigma(composites) - mean_sigma(n)
    (Higher values ~ more prime-like instability)
    """
    bases_list = list(bases)
    comp = list(composite_window)
    comp_sigmas: List[float] = []
    for c in comp:
        sig_c = omniabase_signature(c, bases=bases_list).sigma_mean
        comp_sigmas.append(sig_c)
    sat = float(np.mean(comp_sigmas)) if comp_sigmas else 0.0
    sig_n = omniabase_signature(n, bases=bases_list).sigma_mean
    return float(sat - sig_n)


# =========================
# 2. OMNIATEMPO (time lens)
# =========================

@dataclass
class OmniatempoResult:
    global_mean: float
    global_std: float
    short_mean: float
    short_std: float
    long_mean: float
    long_std: float
    regime_change_score: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _histogram_probs(x: np.ndarray, bins: int = 20) -> np.ndarray:
    """Return normalized histogram probabilities for x."""
    if x.size == 0:
        return np.zeros(bins, dtype=float)
    hist, _ = np.histogram(x, bins=bins, density=False)
    total = hist.sum()
    if total == 0:
        return np.zeros_like(hist, dtype=float)
    return hist.astype(float) / total


def omniatempo_analyze(
    series: Iterable[float],
    short_window: int = 20,
    long_window: int = 100,
    hist_bins: int = 20,
    epsilon: float = 1e-9,
) -> OmniatempoResult:
    """
    Analyze 1D time series stability using NumPy.

    Returns global stats and symmetric KL-like divergence
    between recent-short vs. recent-long distributions.
    """
    x = np.asarray(list(series), dtype=float)
    if x.size == 0:
        return OmniatempoResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    g_mean = float(x.mean())
    g_std = float(x.std(ddof=0))

    sw = min(short_window, x.size)
    lw = min(long_window, x.size)

    short_seg = x[-sw:]
    long_seg = x[-lw:]

    s_mean = float(short_seg.mean())
    s_std = float(short_seg.std(ddof=0))
    l_mean = float(long_seg.mean())
    l_std = float(long_seg.std(ddof=0))

    p = _histogram_probs(short_seg, bins=hist_bins) + epsilon
    q = _histogram_probs(long_seg, bins=hist_bins) + epsilon
    p /= p.sum()
    q /= q.sum()

    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    regime = 0.5 * (kl_pq + kl_qp)

    return OmniatempoResult(
        global_mean=g_mean,
        global_std=g_std,
        short_mean=s_mean,
        short_std=s_std,
        long_mean=l_mean,
        long_std=l_std,
        regime_change_score=regime,
    )


# =========================
# 3. OMNIACAUSA (causal lens)
# =========================

@dataclass
class OmniaEdge:
    source: str
    target: str
    lag: int
    strength: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OmniacausaResult:
    edges: List[OmniaEdge]

    def to_dict(self) -> Dict[str, Any]:
        return {"edges": [e.to_dict() for e in self.edges]}


def _lagged_corr_np(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    Pearson correlation between x and y with given lag.
    Positive lag means x leads y (x at t-lag -> y at t).
    """
    if lag > 0:
        x_l = x[:-lag]
        y_l = y[lag:]
    elif lag < 0:
        lag_abs = -lag
        x_l = x[lag_abs:]
        y_l = y[:-lag_abs]
    else:
        x_l = x
        y_l = y

    if x_l.size < 2 or y_l.size < 2:
        return 0.0

    x_mean = float(x_l.mean())
    y_mean = float(y_l.mean())
    num = float(np.sum((x_l - x_mean) * (y_l - y_mean)))
    den = math.sqrt(
        float(np.sum((x_l - x_mean) ** 2) * np.sum((y_l - y_mean) ** 2))
    )
    if den == 0.0:
        return 0.0
    return float(num / den)


def omniacausa_analyze(
    series_dict: Dict[str, Iterable[float]],
    max_lag: int = 5,
    strength_threshold: float = 0.3,
) -> OmniacausaResult:
    """
    Heuristic causal structure: finds strongest lagged correlation between pairs.

    Returns edges for |corr| >= strength_threshold with lag in [-max_lag, max_lag].
    """
    keys = list(series_dict.keys())
    arrays: Dict[str, np.ndarray] = {
        k: np.asarray(list(series_dict[k]), dtype=float) for k in keys
    }

    edges: List[OmniaEdge] = []
    lags = list(range(-max_lag, max_lag + 1))
    for src in keys:
        for tgt in keys:
            if src == tgt:
                continue
            x = arrays[src]
            y = arrays[tgt]
            best_lag = 0
            best_corr = 0.0
            for lag in lags:
                c = _lagged_corr_np(x, y, lag)
                if abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag = lag
            if abs(best_corr) >= strength_threshold:
                edges.append(
                    OmniaEdge(
                        source=src,
                        target=tgt,
                        lag=best_lag,
                        strength=best_corr,
                    )
                )
    return OmniacausaResult(edges=edges)


# =========================
# 4. TOKEN-LEVEL Ω MAP
# =========================

@dataclass
class TokenOmega:
    token: str
    score: float
    pbii: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenMapResult:
    tokens: List[TokenOmega]
    z_scores: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": [t.to_dict() for t in self.tokens],
            "z_scores": self.z_scores,
        }


def token_level_omega_map(
    tokens: List[str],
    token_numbers: List[int],
) -> TokenMapResult:
    """
    Simple token-level Ω map:
    - maps each token to an integer proxy (token_numbers)
    - computes PBII
    - converts PBII to z-score across the sequence
    """
    if len(tokens) != len(token_numbers):
        raise ValueError("tokens and token_numbers must have same length")

    pbii_vals: List[float] = [pbii_index(int(n)) for n in token_numbers]
    arr = np.array(pbii_vals, dtype=float)
    mu = float(arr.mean()) if arr.size else 0.0
    sigma = float(arr.std(ddof=0)) if arr.size else 0.0
    if sigma == 0.0:
        z = [0.0 for _ in pbii_vals]
    else:
        z = [float((v - mu) / sigma) for v in pbii_vals]

    token_objs = [
        TokenOmega(token=tokens[i], score=z[i], pbii=pbii_vals[i])
        for i in range(len(tokens))
    ]
    return TokenMapResult(tokens=token_objs, z_scores=z)


# =========================
# 5. CORE FUSION ENGINE v2.0
# =========================

@dataclass
class OmniaInput:
    n: int
    series: np.ndarray
    series_dict: Dict[str, np.ndarray]
    tokens: Optional[List[str]] = None
    token_numbers: Optional[List[int]] = None


@dataclass
class OmniaFusionResult:
    omega_raw: float
    omega_revised: float
    components: Dict[str, float]
    thresholds: Dict[str, float]
    omniabase: OmniabaseSignature
    omniatempo: OmniatempoResult
    omniacausa: OmniacausaResult
    tokenmap: Optional[TokenMapResult]
    meta: Dict[str, Any]

    def to_json(self) -> str:
        payload = {
            "omega_raw": self.omega_raw,
            "omega_revised": self.omega_revised,
            "components": self.components,
            "thresholds": self.thresholds,
            "omniabase": self.omniabase.to_dict(),
            "omniatempo": self.omniatempo.to_dict(),
            "omniacausa": self.omniacausa.to_dict(),
            "tokenmap": self.tokenmap.to_dict() if self.tokenmap else None,
            "meta": self.meta,
        }
        return json.dumps(payload, indent=2)


class OmniaTotaleEngine:
    """
    OMNIA_TOTALE v2.0

    Modular fusion of:
      - Omniabase / PBII (BASE)
      - Omniatempo (TIME)
      - Omniacausa (CAUSA)
      - Token-level Ω maps (TOKEN)

    Exposes:
      - compute(input) -> OmniaFusionResult
      - step_log_json(...) helper for LLM logs
    """

    def __init__(
        self,
        w_base: float = 1.0,
        w_tempo: float = 1.0,
        w_causa: float = 1.0,
        w_token: float = 1.0,
        pbii_prime_like_threshold: float = 0.1,
        z_abs_threshold: float = 2.0,
    ) -> None:
        self.w_base = w_base
        self.w_tempo = w_tempo
        self.w_causa = w_causa
        self.w_token = w_token
        self.pbii_prime_like_threshold = pbii_prime_like_threshold
        self.z_abs_threshold = z_abs_threshold

    # ---- core compute ----

    def compute(self, inp: OmniaInput) -> OmniaFusionResult:
        # BASE
        base_sig = omniabase_signature(inp.n)
        base_instability = pbii_index(inp.n)

        # TIME
        tempo_res = omniatempo_analyze(inp.series)
        tempo_val = math.log(1.0 + tempo_res.regime_change_score)

        # CAUSA
        causa_res = omniacausa_analyze(inp.series_dict)
        if causa_res.edges:
            strengths = np.array(
                [abs(e.strength) for e in causa_res.edges], dtype=float
            )
            causa_val = float(strengths.mean())
        else:
            causa_val = 0.0

        # TOKEN
        if inp.tokens is not None and inp.token_numbers is not None:
            token_res = token_level_omega_map(inp.tokens, inp.token_numbers)
            # token component = mean absolute z-score
            if token_res.z_scores:
                token_val = float(
                    np.mean([abs(z) for z in token_res.z_scores])
                )
            else:
                token_val = 0.0
        else:
            token_res = None
            token_val = 0.0

        omega_raw = (
            self.w_base * base_instability
            + self.w_tempo * tempo_val
            + self.w_causa * causa_val
            + self.w_token * token_val
        )

        # adaptive thresholds
        base_thr = self.pbii_prime_like_threshold
        token_thr = self.z_abs_threshold

        base_flag = base_instability > base_thr
        token_flag = token_val > token_thr

        # revised Ω: down-weight unstable regions
        penalty = 0.0
        if base_flag:
            penalty += 0.5 * abs(base_instability)
        if token_flag:
            penalty += 0.5 * abs(token_val)

        omega_revised = omega_raw - penalty

        components = {
            "base_instability": base_instability,
            "tempo_log_regime": tempo_val,
            "causa_mean_strength": causa_val,
            "token_mean_abs_z": token_val,
        }
        thresholds = {
            "pbii_prime_like": base_thr,
            "token_abs_z": token_thr,
        }
        meta = {
            "base_flag": base_flag,
            "token_flag": token_flag,
        }

        return OmniaFusionResult(
            omega_raw=float(omega_raw),
            omega_revised=float(omega_revised),
            components=components,
            thresholds=thresholds,
            omniabase=base_sig,
            omniatempo=tempo_res,
            omniacausa=causa_res,
            tokenmap=token_res,
            meta=meta,
        )

    # ---- helper for LLM logs ----

    def step_log_json(
        self,
        step_index: int,
        prompt: str,
        completion: str,
        omega_before: float,
        omega_after: float,
        fusion: OmniaFusionResult,
    ) -> str:
        """
        Compact JSON record for a single LLM step.
        Intended for supervisor / guardrail pipelines.
        """
        rec = {
            "step": step_index,
            "prompt": prompt,
            "completion": completion,
            "omega_before": omega_before,
            "omega_after": omega_after,
            "delta_omega": omega_after - omega_before,
            "components": fusion.components,
            "thresholds": fusion.thresholds,
            "flags": fusion.meta,
        }
        return json.dumps(rec, ensure_ascii=False)


# =========================
# 6. MINIMAL DEMO
# =========================

def demo() -> None:
    """
    Minimal demo for OMNIA_TOTALE v2.0.
    Shows:
      - BASE/TIME/CAUSA fusion
      - optional token map
      - JSON log stub
    """
    np.random.seed(0)

    # toy numeric target
    n = 173  # prime-like

    # time series with regime change
    t = np.arange(300)
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.7

    # three correlated channels
    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)
    series_dict = {"s1": s1, "s2": s2, "s3": s3}

    # fake token-level mapping: use ord sum as proxy for numbers
    tokens = ["The", "prime", "candidate", "is", "173", "."]
    token_numbers = [sum(ord(ch) for ch in tok) for tok in tokens]

    inp = OmniaInput(
        n=n,
        series=series,
        series_dict=series_dict,
        tokens=tokens,
        token_numbers=token_numbers,
    )

    engine = OmniaTotaleEngine()
    fusion = engine.compute(inp)

    print("Ω_raw   =", fusion.omega_raw)
    print("Ω_rev   =", fusion.omega_revised)
    print("parts   =", fusion.components)
    print("flags   =", fusion.meta)

    log_line = engine.step_log_json(
        step_index=0,
        prompt="Toy prompt",
        completion="Toy completion",
        omega_before=0.0,
        omega_after=fusion.omega_revised,
        fusion=fusion,
    )
    print("JSON log:", log_line)


if __name__ == "__main__":
    demo()