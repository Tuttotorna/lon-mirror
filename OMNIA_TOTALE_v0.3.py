"""
OMNIA_TOTALE v0.3 — Parametric Ω + NumPy-accelerated lenses
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Core:
- Omniabase (multi-base entropy + sigma_b)
- Omniatempo (temporal stability via symmetric KL)
- Omniacausa (lagged Pearson edges)
- OmniaTotale Ω-score with configurable weights + scaling

Extras:
- Generic benchmarking utilities
- GSM8K-like evaluation stub (you provide omega_scores + correctness)
Dependencies:
    pip install numpy
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Sequence, Tuple
import math

import numpy as np

# =========================
# 0. CONFIG
# =========================

@dataclass
class OmniaConfig:
    # fusion weights
    w_base: float = 1.0
    w_tempo: float = 1.0
    w_causa: float = 1.0

    # per-component scaling (for normalization)
    base_scale: float = 1.0
    tempo_scale: float = 1.0
    causa_scale: float = 1.0

    # normalization mode: "none", "z", "tanh", "scaled"
    fusion_norm: str = "tanh"

    # epsilon for numeric stability
    epsilon: float = 1e-9

    def to_dict(self) -> Dict:
        return asdict(self)


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
    digits = []
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
    bases = list(bases)
    sigmas: Dict[int, float] = {}
    entropy: Dict[int, float] = {}
    for b in bases:
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
        bases=bases,
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
    bases = list(bases)
    comp = list(composite_window)
    comp_sigmas = []
    for c in comp:
        sig_c = omniabase_signature(c, bases=bases).sigma_mean
        comp_sigmas.append(sig_c)
    sat = float(np.mean(comp_sigmas)) if comp_sigmas else 0.0
    sig_n = omniabase_signature(n, bases=bases).sigma_mean
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
# 4. OMNIA_TOTALE FUSED SCORE
# =========================

@dataclass
class OmniaTotaleResult:
    n: int
    omniabase: OmniabaseSignature
    omniatempo: OmniatempoResult
    omniacausa: OmniacausaResult
    omega_score: float
    components_raw: Dict[str, float]
    components_norm: Dict[str, float]
    config: OmniaConfig


def _normalize_component(value: float, scale: float, mode: str, eps: float) -> float:
    if mode == "none":
        return value
    if mode == "scaled":
        return value / (scale + eps)
    if mode == "tanh":
        return math.tanh(value / (scale + eps))
    if mode == "z":
        # here scale is treated as std; mean assumed 0
        return value / (scale + eps)
    # fallback
    return value


def omnia_totale_score(
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
    config: OmniaConfig | None = None,
) -> OmniaTotaleResult:
    """
    Fused Ω score combining Omniabase, Omniatempo, and Omniacausa.

    Raw components:
      - base_instability: PBII-style instability (higher for primes).
      - tempo_log_regime: log(1 + regime_change_score).
      - causa_mean_strength: mean |strength| of accepted edges.

    Normalized via OmniaConfig.fusion_norm and *_scale,
    then fused linearly with weights w_*.
    """
    if config is None:
        config = OmniaConfig()

    # base
    base_sig = omniabase_signature(
        n,
        bases=bases,
        length_weight=length_weight,
        length_exponent=length_exponent,
        divisibility_bonus=divisibility_bonus,
    )
    base_instability = pbii_index(n, bases=bases)

    # tempo
    tempo_res = omniatempo_analyze(
        series,
        short_window=short_window,
        long_window=long_window,
        hist_bins=hist_bins,
        epsilon=config.epsilon,
    )
    tempo_val = math.log(1.0 + tempo_res.regime_change_score)

    # causa
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

    components_raw = {
        "base_instability": base_instability,
        "tempo_log_regime": tempo_val,
        "causa_mean_strength": causa_val,
    }

    # normalization
    base_n = _normalize_component(
        base_instability, config.base_scale, config.fusion_norm, config.epsilon
    )
    tempo_n = _normalize_component(
        tempo_val, config.tempo_scale, config.fusion_norm, config.epsilon
    )
    causa_n = _normalize_component(
        causa_val, config.causa_scale, config.fusion_norm, config.epsilon
    )

    components_norm = {
        "base_instability": base_n,
        "tempo_log_regime": tempo_n,
        "causa_mean_strength": causa_n,
    }

    omega = (
        config.w_base * base_n
        + config.w_tempo * tempo_n
        + config.w_causa * causa_n
    )

    return OmniaTotaleResult(
        n=n,
        omniabase=base_sig,
        omniatempo=tempo_res,
        omniacausa=causa_res,
        omega_score=float(omega),
        components_raw=components_raw,
        components_norm=components_norm,
        config=config,
    )


# =========================
# 5. BENCHMARK UTILITIES
# =========================

@dataclass
class BenchmarkResult:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_like: float  # simple threshold-sweep AUC surrogate


def benchmark_omega_binary(
    omega_scores: Sequence[float],
    labels: Sequence[bool],
    num_thresholds: int = 200,
) -> BenchmarkResult:
    """
    Generic binary benchmark:
    - omega_scores: list of Ω (higher = "more prime / more unstable" by default)
    - labels: True/False (e.g., True = prime, or True = correct answer)
    We scan many thresholds and pick the best F1, returning its metrics.

    Also computes an AUC-like metric via threshold sweep (not ROC exact, but monotone).
    """
    omega = np.asarray(omega_scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if omega.size != y.size or omega.size == 0:
        raise ValueError("omega_scores and labels must be same non-zero length")

    # thresholds between min and max
    lo, hi = omega.min(), omega.max()
    if hi == lo:
        thr = float(lo)
        acc = float((y == (omega >= thr)).mean())
        return BenchmarkResult(thr, acc, acc, acc, acc, auc_like=0.0)

    thresholds = np.linspace(lo, hi, num_thresholds)
    best_f1 = -1.0
    best_metrics: Tuple[float, float, float, float, float] = (0, 0, 0, 0, 0)

    auc_sum = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0

    for thr in thresholds:
        pred = omega >= thr
        tp = np.logical_and(pred, y).sum()
        fp = np.logical_and(pred, ~y).sum()
        fn = np.logical_and(~pred, y).sum()
        tn = np.logical_and(~pred, ~y).sum()

        acc = (tp + tn) / (tp + fp + fn + tn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        # crude ROC point
        tpr = rec
        fpr = fp / (fp + tn + 1e-9)
        # trapezoid in ROC space
        auc_sum += (tpr + prev_tpr) * (fpr - prev_fpr) / 2.0
        prev_tpr, prev_fpr = tpr, fpr

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = (thr, acc, prec, rec, f1)

    thr, acc, prec, rec, f1 = best_metrics
    return BenchmarkResult(
        threshold=float(thr),
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        auc_like=float(abs(auc_sum)),
    )


def gsm8k_like_eval(
    omega_scores: Sequence[float],
    is_correct: Sequence[bool],
) -> BenchmarkResult:
    """
    GSM8K-like wrapper:
    - omega_scores: Ω for each solution chain
    - is_correct: True if chain solves the problem correctly

    Interpretation up to you:
    - If Ω is "coherence", higher Ω should correlate with correctness.
    - If Ω is "instability", invert sign before call.
    """
    return benchmark_omega_binary(omega_scores, is_correct)


# =========================
# 6. DEMO
# =========================

def demo():
    """
    Minimal demo for OMNIA_TOTALE v0.3.
    Run this file directly to see example output.
    """
    import random

    n_prime = 173
    n_comp = 180

    random.seed(0)
    np.random.seed(0)

    t = np.arange(300)
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)

    series_dict = {"s1": s1, "s2": s2, "s3": s3}

    # default config (tanh fusion)
    cfg = OmniaConfig(
        w_base=1.0,
        w_tempo=0.5,
        w_causa=0.5,
        base_scale=1.0,
        tempo_scale=1.0,
        causa_scale=1.0,
        fusion_norm="tanh",
    )

    res_prime = omnia_totale_score(n_prime, series, series_dict, config=cfg)
    res_comp = omnia_totale_score(n_comp, series, series_dict, config=cfg)

    print("=== OMNIA_TOTALE v0.3 demo ===")
    print(
        f"n={n_prime} (prime)  Ω ≈ {res_prime.omega_score:.4f}  "
        f"raw={res_prime.components_raw}  norm={res_prime.components_norm}"
    )
    print(
        f"n={n_comp} (comp.)  Ω ≈ {res_comp.omega_score:.4f}  "
        f"raw={res_comp.components_raw}  norm={res_comp.components_norm}"
    )

    print("\nCausal edges (sample):")
    for e in res_prime.omniacausa.edges[:5]:
        print(f"  {e.source} -> {e.target}  lag={e.lag}  strength={e.strength:.3f}")

    # fake GSM8K-like eval: primes=True, comps=False
    omegas = [res_prime.omega_score, res_comp.omega_score]
    labels = [True, False]
    bench = gsm8k_like_eval(omegas, labels)
    print("\nBinary benchmark (prime vs composite):")
    print(bench)


if __name__ == "__main__":
    demo()