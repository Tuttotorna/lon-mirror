"""
OMNIA LENSES · MBX-Omnia v0.1

This module implements three structural lenses:

1) Omniabase   – multi-base structural analysis on integers
2) Omniatempo  – temporal structure analysis on time series
3) Omniacausa  – causal-structure heuristic on multivariate time series

The goal is NOT prediction magic but consistent, computable structure:
- Omniabase: "how this number behaves across many bases?"
- Omniatempo: "how this signal behaves across time (stability / regime shifts)?"
- Omniacausa: "who influences whom, and how strongly, over time?"

Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable, Optional
from collections import Counter, defaultdict


# ===========================
#  COMMON UTILITIES
# ===========================

def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _chunk(seq: List[float], size: int) -> List[List[float]]:
    return [seq[i : i + size] for i in range(0, len(seq), size) if len(seq[i : i + size]) == size]


# ===========================
#  1. OMNIABASE LENS
# ===========================

def digits_in_base(n: int, b: int) -> List[int]:
    """Return digits of n in base b, most significant first."""
    if n < 0:
        raise ValueError("Only non-negative integers are supported.")
    if b < 2:
        raise ValueError("Base must be >= 2.")
    if n == 0:
        return [0]
    res: List[int] = []
    while n > 0:
        n, r = divmod(n, b)
        res.append(r)
    return res[::-1]


def normalized_entropy_base(n: int, b: int) -> float:
    """
    Normalized Shannon entropy of the digits of n in base b.

    0 → fully structured (all digits equal)
    1 → maximally random (all digits equally likely)
    """
    digits = digits_in_base(n, b)
    L = len(digits)
    if L == 0:
        return 0.0
    freq = Counter(digits)
    probs = [c / L for c in freq.values()]
    H = -sum(p * math.log2(p) for p in probs if p > 0)
    H_max = math.log2(b)
    return H / H_max if H_max > 0 else 0.0


def sigma_b(
    n: int,
    b: int,
    length_weight: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> float:
    """
    Base Symmetry Score σ_b(n).

    σ_b(n) = length_weight * (1 - H_norm) / L + divisibility_bonus * I[n % b == 0]

    - H_norm: normalized entropy in base b
    - L: number of digits in base b
    - divisibility_bonus: extra structure if n is divisible by b
    """
    digits = digits_in_base(n, b)
    L = len(digits)
    if L == 0:
        return 0.0
    Hn = normalized_entropy_base(n, b)
    base_term = length_weight * (1.0 - Hn) / L
    div_term = divisibility_bonus if n % b == 0 else 0.0
    return base_term + div_term


@dataclass
class OmniabaseSignature:
    n: int
    bases: List[int]
    sigmas: Dict[int, float]
    entropy: Dict[int, float]

    @property
    def sigma_mean(self) -> float:
        return _safe_mean(self.sigmas.values())

    @property
    def entropy_mean(self) -> float:
        return _safe_mean(self.entropy.values())

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["sigma_mean"] = self.sigma_mean
        d["entropy_mean"] = self.entropy_mean
        return d


def omniabase_signature(
    n: int,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    length_weight: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> OmniabaseSignature:
    """Compute the multi-base signature of n."""
    bases = list(bases)
    sigmas: Dict[int, float] = {}
    ent: Dict[int, float] = {}
    for b in bases:
        h = normalized_entropy_base(n, b)
        s = sigma_b(n, b, length_weight=length_weight, divisibility_bonus=divisibility_bonus)
        sigmas[b] = s
        ent[b] = h
    return OmniabaseSignature(n=n, bases=bases, sigmas=sigmas, entropy=ent)


def pbii_index(
    n: int,
    composite_window: Iterable[int],
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
) -> float:
    """
    Simple Prime Base Instability Index (PBII) for a single n, relative to a set of composites.

    PBII(n) = Sat - Sigma_avg(n)
    where:
      Sat       = average sigma over composite_window
      Sigma_avg = average sigma over bases for n
    """
    bases = list(bases)
    sig_n = [sigma_b(n, b) for b in bases]
    sigma_avg = _safe_mean(sig_n)
    composite = list(composite_window)
    if not composite:
        return -sigma_avg
    sat_vals = []
    for c in composite:
        for b in bases:
            sat_vals.append(sigma_b(c, b))
    sat = _safe_mean(sat_vals)
    return sat - sigma_avg


# ===========================
#  2. OMNIATEMPO LENS
# ===========================

@dataclass
class OmniatempoResult:
    series_length: int
    global_mean: float
    global_std: float
    local_trend: List[float]
    local_volatility: List[float]
    regime_change_score: float


def omniatempo_analyze(
    series: List[float],
    short_window: int = 10,
    long_window: int = 50,
) -> OmniatempoResult:
    """
    Analyze the temporal structure of a 1D time series.

    Measures:
    - local_trend: rolling mean (short_window)
    - local_volatility: rolling std (short_window)
    - regime_change_score: how different short vs long windows behave structurally

    regime_change_score is high if the recent short window distribution
    is very different from the long-term window.
    """
    if len(series) < max(short_window, long_window):
        raise ValueError("Series too short for the chosen windows.")

    global_mean = statistics.fmean(series)
    global_std = statistics.pstdev(series) if len(series) > 1 else 0.0

    # Rolling mean & std for short window
    local_trend: List[float] = []
    local_vol: List[float] = []
    for i in range(len(series) - short_window + 1):
        window = series[i : i + short_window]
        local_trend.append(statistics.fmean(window))
        local_vol.append(statistics.pstdev(window) if len(window) > 1 else 0.0)

    # Long vs short distribution difference (simple KL-like measure)
    recent_short = series[-short_window:]
    recent_long = series[-long_window:]

    def _hist(values: List[float], bins: int = 10) -> List[float]:
        if not values:
            return [0.0] * bins
        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            return [1.0] + [0.0] * (bins - 1)
        step = (vmax - vmin) / bins
        hist = [0] * bins
        for v in values:
            idx = min(bins - 1, int((v - vmin) / step))
            hist[idx] += 1
        total = sum(hist)
        return [h / total for h in hist]

    p = _hist(recent_long)
    q = _hist(recent_short)

    # Symmetrized divergence
    eps = 1e-12
    div = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            div += pi * math.log((pi + eps) / (qi + eps))
        if qi > 0:
            div += qi * math.log((qi + eps) / (pi + eps))
    regime_change_score = div

    return OmniatempoResult(
        series_length=len(series),
        global_mean=global_mean,
        global_std=global_std,
        local_trend=local_trend,
        local_volatility=local_vol,
        regime_change_score=regime_change_score,
    )


# ===========================
#  3. OMNIACAUSA LENS
# ===========================

@dataclass
class CausalEdge:
    source: str
    target: str
    lag: int
    strength: float  # [-1, 1] correlation-like


@dataclass
class OmniacausaResult:
    variables: List[str]
    edges: List[CausalEdge]


def _lagged_correlation(x: List[float], y: List[float], lag: int) -> float:
    """
    Pearson-like correlation with y shifted by 'lag' relative to x.
    If lag > 0: x[t] vs y[t+lag] (x leads, y follows).
    If lag < 0: x[t] vs y[t+lag] (y leads, x follows).
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("Time series must have same length for lagged correlation.")
    if abs(lag) >= n:
        return 0.0

    if lag > 0:
        xs = x[0 : n - lag]
        ys = y[lag : n]
    elif lag < 0:
        xs = x[-lag : n]
        ys = y[0 : n + lag]
    else:
        xs = x
        ys = y

    if len(xs) < 2:
        return 0.0

    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in xs))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def omniacausa_analyze(
    series: Dict[str, List[float]],
    max_lag: int = 5,
    strength_threshold: float = 0.3,
) -> OmniacausaResult:
    """
    Heuristic causal structure lens.

    Input:
        series: dict {name -> time series}
        max_lag: test lags in [-max_lag, ..., +max_lag]
        strength_threshold: minimal |correlation| to keep an edge

    Output:
        OmniacausaResult with edges:
        - source, target: variable names
        - lag: positive if source leads target, negative if target leads source
        - strength: correlation-like score

    NOTE:
        This is not a full causal discovery algorithm.
        It is a structural "lens" to highlight directional dependencies.
    """
    names = sorted(series.keys())
    length_set = {len(v) for v in series.values()}
    if len(length_set) != 1:
        raise ValueError("All series must have same length.")
    edges: List[CausalEdge] = []

    for i, src in enumerate(names):
        for j, tgt in enumerate(names):
            if i == j:
                continue
            x = series[src]
            y = series[tgt]
            best_lag = 0
            best_strength = 0.0
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    continue
                corr = _lagged_correlation(x, y, lag)
                if abs(corr) > abs(best_strength):
                    best_strength = corr
                    best_lag = lag
            if abs(best_strength) >= strength_threshold:
                edges.append(
                    CausalEdge(
                        source=src,
                        target=tgt,
                        lag=best_lag,
                        strength=best_strength,
                    )
                )

    return OmniacausaResult(variables=names, edges=edges)


# ===========================
#  DEMO / USAGE
# ===========================

if __name__ == "__main__":
    # --- Omniabase demo ---
    n = 173
    sig = omniabase_signature(n)
    print("OMNIABASE DEMO")
    print(sig.to_dict())

    # --- PBII demo ---
    pb = pbii_index(n, composite_window=[4, 6, 8, 9, 10, 12, 14, 15])
    print("\nPBII index for n =", n, "->", pb)

    # --- Omniatempo demo ---
    import random

    random.seed(0)
    series = [math.sin(t / 10.0) + 0.1 * random.gauss(0, 1) for t in range(200)]
    ot = omniatempo_analyze(series)
    print("\nOMNIATEMPO DEMO")
    print("length:", ot.series_length)
    print("global_mean:", ot.global_mean)
    print("global_std:", ot.global_std)
    print("regime_change_score:", ot.regime_change_score)

    # --- Omniacausa demo ---
    t = list(range(200))
    s1 = [math.sin(tt / 10.0) for tt in t]
    s2 = [0.5 * s1[i - 2] + 0.1 * random.gauss(0, 1) if i >= 2 else 0.0 for i in range(len(t))]
    s3 = [random.gauss(0, 1) for _ in t]
    oc = omniacausa_analyze({"s1": s1, "s2": s2, "s3": s3}, max_lag=5, strength_threshold=0.3)
    print("\nOMNIACAUSA DEMO")
    for e in oc.edges:
        print(f"{e.source} -> {e.target} (lag={e.lag}, strength={e.strength:.3f})")