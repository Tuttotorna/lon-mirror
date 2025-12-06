"""
OMNIA-TOTALE v0.1 · Unified BASE–TIME–CAUSE Lens
Author: Massimiliano Brighindi (concepts) + MBX-IA (formalization)
License: MIT

This module implements:
- Omniabase   → structural (BASE) lens on integers
- Omniatempo  → temporal (TIME) lens on 1D time series
- Omniacausa  → directional (CAUSE) lens on multivariate series
- OmniaTotale → fusion Ω(X) = sqrt(S_base * S_time) * S_cause
"""

from __future__ import annotations
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# =========================
#  BASE LENS (OMNIABASE)
# =========================

def digits_in_base(n: int, b: int) -> List[int]:
    """
    Return digits of n in base b (MSB first).
    Assumes n >= 0, b >= 2.
    """
    if n == 0:
        return [0]
    digits: List[int] = []
    x = n
    while x > 0:
        x, r = divmod(x, b)
        digits.append(r)
    return digits[::-1]


def normalized_entropy(digits: List[int], base: int) -> float:
    """
    Shannon entropy normalized in [0,1] for digits in given base.
    """
    if not digits:
        return 0.0
    L = len(digits)
    counts = [0] * base
    for d in digits:
        if 0 <= d < base:
            counts[d] += 1
    probs = [c / L for c in counts if c > 0]
    if not probs:
        return 0.0
    H = -sum(p * math.log2(p) for p in probs)
    Hmax = math.log2(base) if base > 1 else 1.0
    return H / Hmax if Hmax > 0 else 0.0


def sigma_b(
    n: int,
    b: int,
    length_weight: float = 1.0,
    divisibility_bonus: float = 0.5,
    length_exponent: float = 1.0,
) -> float:
    """
    Base Symmetry Score σ_b(n).

    σ_b(n) = length_weight * (1 - H_norm) / (L ** length_exponent)
             + divisibility_bonus * I[n % b == 0]

    - H_norm: normalized entropy of digits in base b
    - L: number of digits in base b
    """
    if b < 2:
        raise ValueError("Base b must be >= 2")
    if n < 0:
        raise ValueError("n must be >= 0")

    digits = digits_in_base(n, b)
    L = len(digits)
    if L == 0:
        return 0.0

    Hn = normalized_entropy(digits, b)
    base_term = 0.0
    if L > 0:
        base_term = length_weight * (1.0 - Hn) / (L ** max(length_exponent, 0.0))

    div_term = divisibility_bonus if (n % b == 0) else 0.0
    return base_term + div_term


@dataclass
class OmniabaseSignature:
    n: int
    bases: List[int]
    sigmas: Dict[int, float]
    entropies: Dict[int, float]
    sigma_mean: float
    entropy_mean: float
    S_base: float  # structural stability score in [0,1]

    def to_dict(self) -> Dict:
        return {
            "n": self.n,
            "bases": self.bases,
            "sigmas": self.sigmas,
            "entropies": self.entropies,
            "sigma_mean": self.sigma_mean,
            "entropy_mean": self.entropy_mean,
            "S_base": self.S_base,
        }


def omniabase_signature(
    n: int,
    bases: Optional[List[int]] = None,
    length_weight: float = 1.0,
    divisibility_bonus: float = 0.5,
    length_exponent: float = 1.0,
) -> OmniabaseSignature:
    """
    Compute multi-base structural signature for integer n.
    """
    if bases is None:
        # Default: small prime bases for structural sensitivity
        bases = [2, 3, 5, 7, 11, 13, 17, 19]

    sigmas: Dict[int, float] = {}
    entropies: Dict[int, float] = {}

    for b in bases:
        digits = digits_in_base(n, b)
        Hn = normalized_entropy(digits, b)
        entropies[b] = Hn
        sigmas[b] = sigma_b(
            n,
            b,
            length_weight=length_weight,
            divisibility_bonus=divisibility_bonus,
            length_exponent=length_exponent,
        )

    sigma_vals = list(sigmas.values())
    entropy_vals = list(entropies.values())
    sigma_mean = sum(sigma_vals) / len(sigma_vals) if sigma_vals else 0.0
    entropy_mean = sum(entropy_vals) / len(entropy_vals) if entropy_vals else 0.0

    # Structural stability score S_base ∈ [0,1]: low entropy → high stability
    S_base = max(0.0, min(1.0, 1.0 - entropy_mean))

    return OmniabaseSignature(
        n=n,
        bases=bases,
        sigmas=sigmas,
        entropies=entropies,
        sigma_mean=sigma_mean,
        entropy_mean=entropy_mean,
        S_base=S_base,
    )


# Simple PBII-style instability index (optional helper)
def pbii_index(
    n: int,
    composite_window: List[int],
    bases: Optional[List[int]] = None,
) -> float:
    """
    PBII(n) = Sat - Σ_avg(n), where:
    - Σ_avg(n): mean σ_b(n) over bases
    - Sat: mean σ_b(k) over composite_window
    """
    if bases is None:
        bases = [2, 3, 5, 7, 11, 13, 17, 19]

    sig_n = [
        sigma_b(n, b, length_weight=1.0, divisibility_bonus=0.5, length_exponent=1.0)
        for b in bases
    ]
    sigma_avg_n = sum(sig_n) / len(sig_n) if sig_n else 0.0

    sig_comp: List[float] = []
    for k in composite_window:
        for b in bases:
            sig_comp.append(
                sigma_b(k, b, length_weight=1.0, divisibility_bonus=0.5, length_exponent=1.0)
            )
    sat = sum(sig_comp) / len(sig_comp) if sig_comp else 0.0
    return sat - sigma_avg_n


# =========================
#  TIME LENS (OMNIATEMPO)
# =========================

@dataclass
class OmniatempoResult:
    series_len: int
    global_mean: float
    global_std: float
    regime_change_score: float  # higher = stronger regime shift
    S_time: float               # stability score in [0,1]


def _histogram(data: List[float], bins: int, vmin: float, vmax: float) -> List[float]:
    if not data:
        return [0.0] * bins
    if vmin == vmax:
        # All values equal → single bin
        hist = [0.0] * bins
        hist[0] = float(len(data))
        return hist
    width = (vmax - vmin) / bins
    hist = [0.0] * bins
    for x in data:
        idx = int((x - vmin) / width)
        if idx < 0:
            idx = 0
        elif idx >= bins:
            idx = bins - 1
        hist[idx] += 1.0
    return hist


def _sym_kl(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    """
    Symmetrized KL divergence between two discrete distributions.
    """
    if len(p) != len(q):
        raise ValueError("p and q must have same length")
    Zp = sum(p)
    Zq = sum(q)
    if Zp == 0 or Zq == 0:
        return 0.0
    p_norm = [(x / Zp) for x in p]
    q_norm = [(x / Zq) for x in q]
    kl_pq = 0.0
    kl_qp = 0.0
    for pi, qi in zip(p_norm, q_norm):
        pi = max(pi, eps)
        qi = max(qi, eps)
        kl_pq += pi * math.log(pi / qi)
        kl_qp += qi * math.log(qi / pi)
    return kl_pq + kl_qp


def omniatempo_analyze(
    series: List[float],
    short_window: int = 20,
    long_window: int = 100,
    bins: int = 20,
) -> OmniatempoResult:
    """
    Analyze temporal stability of a 1D series.

    - Uses global mean/std.
    - Compares short recent window vs older long window via symmetrized KL.
    - S_time = 1 / (1 + regime_change_score) ∈ (0,1].
    """
    n = len(series)
    if n == 0:
        return OmniatempoResult(0, 0.0, 0.0, 0.0, 1.0)
    if n < long_window + short_window:
        # If series is short, fall back to simple global stats
        gm = statistics.mean(series)
        gs = statistics.pstdev(series) if n > 1 else 0.0
        return OmniatempoResult(n, gm, gs, 0.0, 1.0)

    gm = statistics.mean(series)
    gs = statistics.pstdev(series) if n > 1 else 0.0

    recent = series[-short_window:]
    past = series[-(long_window + short_window):-short_window]

    vmin = min(min(recent), min(past))
    vmax = max(max(recent), max(past))
    hist_recent = _histogram(recent, bins, vmin, vmax)
    hist_past = _histogram(past, bins, vmin, vmax)

    reg_score = _sym_kl(hist_recent, hist_past)
    # Map to stability in (0,1]: high divergence → low stability
    S_time = 1.0 / (1.0 + reg_score)
    return OmniatempoResult(n, gm, gs, reg_score, S_time)


# =========================
#  CAUSE LENS (OMNIACAUSA)
# =========================

@dataclass
class OmniacausaEdge:
    source: str
    target: str
    lag: int
    strength: float  # Pearson-like correlation


@dataclass
class OmniacausaResult:
    edges: List[OmniacausaEdge]
    S_cause: float  # average |strength| over edges in [0,1]


def _pearson_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def omniacausa_analyze(
    signals: Dict[str, List[float]],
    max_lag: int = 5,
    strength_threshold: float = 0.3,
) -> OmniacausaResult:
    """
    Heuristic directional dependency lens.

    For each pair (A,B):
    - scans lag in [-max_lag, +max_lag]
    - finds lag with max |corr|
    - if |corr| >= threshold and lag != 0, creates an edge
      lag > 0  → A leads B
      lag < 0  → B leads A
    """
    names = list(signals.keys())
    if not names:
        return OmniacausaResult(edges=[], S_cause=0.0)

    length_set = {len(signals[name]) for name in names}
    if len(length_set) != 1:
        raise ValueError("All signals must have same length")
    L = length_set.pop()
    if L < 2:
        return OmniacausaResult(edges=[], S_cause=0.0)

    edges: List[OmniacausaEdge] = []

    for i, src in enumerate(names):
        for j, tgt in enumerate(names):
            if i == j:
                continue
            s_src = signals[src]
            s_tgt = signals[tgt]
            best_corr = 0.0
            best_lag = 0
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    continue  # skip instantaneous, too ambiguous
                if lag > 0:
                    x = s_src[:-lag]
                    y = s_tgt[lag:]
                else:
                    # lag < 0: tgt leads src in raw index
                    x = s_src[-lag:]
                    y = s_tgt[:L + lag]
                if len(x) < 2 or len(y) < 2:
                    continue
                corr = _pearson_corr(x, y)
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            if abs(best_corr) >= strength_threshold and best_lag != 0:
                if best_lag > 0:
                    # src leads tgt
                    edges.append(OmniacausaEdge(source=src, target=tgt, lag=best_lag, strength=best_corr))
                else:
                    # tgt leads src
                    edges.append(OmniacausaEdge(source=tgt, target=src, lag=-best_lag, strength=best_corr))

    if edges:
        S_cause = sum(abs(e.strength) for e in edges) / len(edges)
        # clamp for safety
        S_cause = max(0.0, min(1.0, S_cause))
    else:
        S_cause = 0.0

    return OmniacausaResult(edges=edges, S_cause=S_cause)


# =========================
#  FUSION LENS (OMNIA-TOTALE)
# =========================

@dataclass
class OmniaTotalSignature:
    S_base: float
    S_time: float
    S_cause: float
    omega: float

    # Optional: underlying components for inspection
    base_signature: Optional[OmniabaseSignature] = None
    tempo_result: Optional[OmniatempoResult] = None
    causa_result: Optional[OmniacausaResult] = None


def fuse_omnia_total(
    S_base: float,
    S_time: float,
    S_cause: float,
) -> float:
    """
    Ω(X) = sqrt(S_base * S_time) * S_cause
    All inputs are clamped to [0,1] before fusion.
    """
    Sb = max(0.0, min(1.0, S_base))
    St = max(0.0, min(1.0, S_time))
    Sc = max(0.0, min(1.0, S_cause))
    return math.sqrt(Sb * St) * Sc


def omnia_total_for_integer_and_series(
    n: int,
    time_series: List[float],
    multi_signals: Dict[str, List[float]],
    *,
    bases: Optional[List[int]] = None,
    short_window: int = 20,
    long_window: int = 100,
    bins: int = 20,
    max_lag: int = 5,
    strength_threshold: float = 0.3,
) -> OmniaTotalSignature:
    """
    Example wrapper:
    - uses n for Omniabase (BASE)
    - uses time_series for Omniatempo (TIME)
    - uses multi_signals for Omniacausa (CAUSE)
    and fuses them into a single Ω score.
    """
    base_sig = omniabase_signature(n, bases=bases)
    tempo_res = omniatempo_analyze(time_series, short_window=short_window, long_window=long_window, bins=bins)
    causa_res = omniacausa_analyze(multi_signals, max_lag=max_lag, strength_threshold=strength_threshold)

    omega = fuse_omnia_total(base_sig.S_base, tempo_res.S_time, causa_res.S_cause)

    return OmniaTotalSignature(
        S_base=base_sig.S_base,
        S_time=tempo_res.S_time,
        S_cause=causa_res.S_cause,
        omega=omega,
        base_signature=base_sig,
        tempo_result=tempo_res,
        causa_result=causa_res,
    )


# =========================
#  SIMPLE DEMO (optional)
# =========================

if __name__ == "__main__":
    # Minimal demo with synthetic data
    import random

    # BASE: integer
    n = 173

    # TIME: smooth signal with noise
    random.seed(0)
    series = [math.sin(t / 10.0) + 0.1 * random.gauss(0, 1) for t in range(200)]

    # CAUSE: s1 leads s2, s3 is noise
    t = list(range(200))
    s1 = [math.sin(tt / 10.0) for tt in t]
    s2 = [0.5 * s1[i - 2] + 0.1 * random.gauss(0, 1) if i >= 2 else 0.0 for i in range(len(t))]
    s3 = [random.gauss(0, 1) for _ in t]
    signals = {"s1": s1, "s2": s2, "s3": s3}

    omnia_sig = omnia_total_for_integer_and_series(
        n=n,
        time_series=series,
        multi_signals=signals,
    )

    print("S_base :", omnia_sig.S_base)
    print("S_time :", omnia_sig.S_time)
    print("S_cause:", omnia_sig.S_cause)
    print("Ω      :", omnia_sig.omega)