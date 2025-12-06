"""
OMNIA_TOTALE v0.6 — Ω-FLOW ENGINE (NumPy-accelerated)
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Dependencies:
    pip install numpy
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable
import math

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

    def to_dict(self) -> Dict:
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


@dataclass
class OmniacausaResult:
    edges: List[OmniaEdge]


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

    x_mean = x_l.mean()
    y_mean = y_l.mean()
    num = np.sum((x_l - x_mean) * (y_l - y_mean))
    den = math.sqrt(float(np.sum((x_l - x_mean) ** 2) * np.sum((y_l - y_mean) ** 2)))
    if den == 0:
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
# 4. TOKEN MAP (local instability)
# =========================

@dataclass
class TokenMapResult:
    instabilities: List[float]
    mean_instability: float
    max_instability: float


def token_instability_map(
    series: Iterable[float],
    window: int = 10,
    epsilon: float = 1e-9,
) -> TokenMapResult:
    """
    Per-step instability based on rolling z-score magnitude.

    For each t:
        z_t = |x_t - mean_{window}| / (std_{window} + epsilon)
    """
    x = np.asarray(list(series), dtype=float)
    n = x.size
    if n == 0:
        return TokenMapResult(instabilities=[], mean_instability=0.0, max_instability=0.0)

    w = max(2, min(window, n))
    inst: List[float] = []
    for t in range(n):
        start = max(0, t - w + 1)
        seg = x[start:t + 1]
        m = seg.mean()
        s = seg.std(ddof=0)
        z = abs(x[t] - m) / (s + epsilon)
        inst.append(float(z))

    inst_arr = np.asarray(inst, dtype=float)
    return TokenMapResult(
        instabilities=inst,
        mean_instability=float(inst_arr.mean()),
        max_instability=float(inst_arr.max()),
    )


# =========================
# 5. OMNIA-FLOW (fused Ω with self-revision)
# =========================

@dataclass
class OmniaFlowResult:
    n: int
    omniabase: OmniabaseSignature
    omniatempo: OmniatempoResult
    omniacausa: OmniacausaResult
    tokenmap: TokenMapResult
    omega_raw: float
    omega_revised: float
    delta_omega: float
    components: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


def omnia_flow(
    n: int,
    series: Iterable[float],
    series_dict: Dict[str, Iterable[float]],
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
    short_window: int = 20,
    long_window: int = 100,
    hist_bins: int = 20,
    max_lag: int = 5,
    strength_threshold: float = 0.3,
    token_window: int = 10,
    # fusion weights
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
    w_token: float = 0.5,
    epsilon: float = 1e-9,
) -> OmniaFlowResult:
    """
    Full Ω-FLOW:

    1) Omniabase → PBII-like instability.
    2) Omniatempo → regime-change score.
    3) Omniacausa → mean |lagged-correlation|.
    4) Token-map → mean local anomaly.
    5) Ω_raw  = w_base*base + w_tempo*tempo + w_causa*causa.
    6) Ω_rev  = Ω_raw - w_token*token_mean (self-revision penalty).
    """
    # 1) Omniabase
    base_sig = omniabase_signature(
        n,
        bases=bases,
        length_weight=length_weight,
        length_exponent=length_exponent,
        divisibility_bonus=divisibility_bonus,
    )
    base_instability = pbii_index(n, bases=bases)

    # 2) Omniatempo
    tempo_res = omniatempo_analyze(
        series,
        short_window=short_window,
        long_window=long_window,
        hist_bins=hist_bins,
        epsilon=epsilon,
    )
    tempo_val = math.log(1.0 + tempo_res.regime_change_score)

    # 3) Omniacausa
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

    # 4) Token-map
    token_res = token_instability_map(series, window=token_window, epsilon=epsilon)
    token_mean = token_res.mean_instability

    # 5) Ω raw
    omega_raw = w_base * base_instability + w_tempo * tempo_val + w_causa * causa_val

    # 6) self-revision: penalize high local instability
    omega_revised = omega_raw - w_token * token_mean
    delta_omega = omega_revised - omega_raw

    components = {
        "base_instability": base_instability,
        "tempo_log_regime": tempo_val,
        "causa_mean_strength": causa_val,
        "token_mean_instability": token_mean,
    }

    return OmniaFlowResult(
        n=n,
        omniabase=base_sig,
        omniatempo=tempo_res,
        omniacausa=causa_res,
        tokenmap=token_res,
        omega_raw=float(omega_raw),
        omega_revised=float(omega_revised),
        delta_omega=float(delta_omega),
        components=components,
    )


# =========================
# 6. DEMO + REPORT
# =========================

def run_demo() -> List[OmniaFlowResult]:
    """
    Minimal demo for OMNIA_TOTALE v0.6.
    """
    import random

    n_prime = 173
    n_comp = 180

    random.seed(0)
    np.random.seed(0)

    t = np.arange(300)
    # base series with regime shift
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    # causal toy system
    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)

    series_dict = {"s1": s1, "s2": s2, "s3": s3}

    res_prime = omnia_flow(n_prime, series, series_dict)
    res_comp = omnia_flow(n_comp, series, series_dict)

    print("=== OMNIA_TOTALE v0.6 demo ===")
    for label, res in [("prime", res_prime), ("composite", res_comp)]:
        print(
            f"n={res.n} ({label})  "
            f"Ω_raw ≈ {res.omega_raw:.4f}  "
            f"Ω_rev ≈ {res.omega_revised:.4f}  "
            f"ΔΩ ≈ {res.delta_omega:.4f}"
        )
        print(f"  components={res.components}")
    print("Causal edges (from prime run):")
    for e in res_prime.omniacausa.edges:
        print(f"  {e.source} -> {e.target}  lag={e.lag}  strength={e.strength:.3f}")

    return [res_prime, res_comp]


def generate_report(results: List[OmniaFlowResult]) -> str:
    """
    Generate a compact text report from a list of OmniaFlowResult.
    """
    lines: List[str] = []
    lines.append("# OMNIA_TOTALE v0.6 — Ω-FLOW demo report\n")
    for res in results:
        lines.append(f"## n = {res.n}\n")
        lines.append(f"- Ω_raw: {res.omega_raw:.6f}")
        lines.append(f"- Ω_revised: {res.omega_revised:.6f}")
        lines.append(f"- ΔΩ (self-revision): {res.delta_omega:.6f}")
        lines.append(f"- base_instability (PBII-like): {res.components['base_instability']:.6f}")
        lines.append(f"- tempo_log_regime: {res.components['tempo_log_regime']:.6f}")
        lines.append(f"- causa_mean_strength: {res.components['causa_mean_strength']:.6f}")
        lines.append(f"- token_mean_instability: {res.components['token_mean_instability']:.6f}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    demo_results = run_demo()
    report = generate_report(demo_results)
    print("\n=== AUTO-GENERATED REPORT ===")
    print(report)