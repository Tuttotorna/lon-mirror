"""
OMNIA_TOTALE v0.4 — Multimodal Ω-fusion (NumPy-accelerated)
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Dependencies:
    pip install numpy
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Optional
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
    comp_sigmas: List[float] = []
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


# =====================================
# 4. OMNIABASE ON IMAGES (multi-modal)
# =====================================

@dataclass
class OmniaImageResult:
    shape: tuple
    bases: List[int]
    sigmas: Dict[int, float]
    entropy: Dict[int, float]
    sigma_mean: float
    entropy_mean: float

    def to_dict(self) -> Dict:
        return asdict(self)


def omniabase_image(
    image: np.ndarray,
    bases: Iterable[int] = (2, 4, 8, 16),
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
) -> OmniaImageResult:
    """
    Apply an Omniabase-like lens to an image.

    - Convert to grayscale if needed.
    - For each base b, map pixel intensities to digits via (value % b),
      compute entropy and a sigma-style compactness term.
    """
    arr = np.asarray(image)
    if arr.ndim == 3:
        # simple grayscale
        arr = arr.mean(axis=2)
    arr = arr.astype(int)
    flat = arr.ravel()

    bases = list(bases)
    sigmas: Dict[int, float] = {}
    entropy: Dict[int, float] = {}

    for b in bases:
        if b <= 1:
            raise ValueError("base must be >= 2")

        digits = flat % b
        L = digits.size
        if L == 0:
            sig = 0.0
            Hn = 0.0
        else:
            counts = np.bincount(digits, minlength=b).astype(float)
            probs = counts[counts > 0] / L
            if probs.size == 0:
                Hn = 0.0
            else:
                H = -np.sum(probs * np.log2(probs))
                Hmax = math.log2(b)
                Hn = float(H / Hmax) if Hmax > 0 else 0.0
            length_term = length_weight * (1.0 - Hn) / (L ** length_exponent)
            sig = length_term

        sigmas[b] = float(sig)
        entropy[b] = float(Hn)

    sigma_vals = np.array(list(sigmas.values()), dtype=float)
    ent_vals = np.array(list(entropy.values()), dtype=float)

    return OmniaImageResult(
        shape=tuple(arr.shape),
        bases=bases,
        sigmas=sigmas,
        entropy=entropy,
        sigma_mean=float(sigma_vals.mean()) if sigma_vals.size else 0.0,
        entropy_mean=float(ent_vals.mean()) if ent_vals.size else 0.0,
    )


# ====================================
# 5. MULTIMODAL OMNIATEMPO (sequences)
# ====================================

@dataclass
class OmniatempoMMResult:
    per_key: Dict[str, OmniatempoResult]
    mean_regime_change: float

    def to_dict(self) -> Dict:
        return {
            "per_key": {k: asdict(v) for k, v in self.per_key.items()},
            "mean_regime_change": self.mean_regime_change,
        }


def omniatempo_multimodal(
    series_dict: Dict[str, Iterable[float]],
    short_window: int = 20,
    long_window: int = 100,
    hist_bins: int = 20,
    epsilon: float = 1e-9,
) -> OmniatempoMMResult:
    """
    Apply Omniatempo to multiple sequences (e.g., loss, reward, sensors).

    Returns per-key OmniatempoResult plus the mean regime-change score.
    """
    per_key: Dict[str, OmniatempoResult] = {}
    regimes: List[float] = []

    for k, seq in series_dict.items():
        res = omniatempo_analyze(
            seq,
            short_window=short_window,
            long_window=long_window,
            hist_bins=hist_bins,
            epsilon=epsilon,
        )
        per_key[k] = res
        regimes.append(res.regime_change_score)

    mean_reg = float(np.mean(regimes)) if regimes else 0.0
    return OmniatempoMMResult(per_key=per_key, mean_regime_change=mean_reg)


# ==============================
# 6. OMNIA_TOTALE FUSED Ω SCORE
# ==============================

@dataclass
class OmniaTotaleResult:
    n: int
    omniabase: OmniabaseSignature
    omniatempo: OmniatempoResult
    omniacausa: OmniacausaResult
    image: Optional[OmniaImageResult]
    tempo_mm: Optional[OmniatempoMMResult]
    omega_score: float
    components: Dict[str, float]

    def to_dict(self) -> Dict:
        d = {
            "n": self.n,
            "omniabase": asdict(self.omniabase),
            "omniatempo": asdict(self.omniatempo),
            "omniacausa": {"edges": [asdict(e) for e in self.omniacausa.edges]},
            "omega_score": self.omega_score,
            "components": self.components,
        }
        if self.image is not None:
            d["image"] = asdict(self.image)
        if self.tempo_mm is not None:
            d["tempo_mm"] = self.tempo_mm.to_dict()
        return d


def _normalize_component(x: float) -> float:
    """
    Simple bounded normalization: maps R -> (-1, 1).

    x_norm = x / (1 + |x|)
    """
    return float(x / (1.0 + abs(x)))


def omnia_totale_multimodal(
    n: int,
    series: Iterable[float],
    series_dict: Dict[str, Iterable[float]],
    image: Optional[np.ndarray] = None,
    seq_dict: Optional[Dict[str, Iterable[float]]] = None,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
    short_window: int = 20,
    long_window: int = 100,
    hist_bins: int = 20,
    max_lag: int = 5,
    strength_threshold: float = 0.3,
    # fusion weights
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
    w_image: float = 1.0,
    w_seq: float = 1.0,
    epsilon: float = 1e-9,
) -> OmniaTotaleResult:
    """
    Multimodal Ω score combining:

    - Omniabase / PBII on integer n
    - Omniatempo on a reference time series
    - Omniacausa on multivariate time series
    - Omniabase-like entropy on images (optional)
    - Omniatempo on multiple sequences (optional)
    """
    # BASE
    base_sig = omniabase_signature(
        n,
        bases=bases,
        length_weight=length_weight,
        length_exponent=length_exponent,
        divisibility_bonus=divisibility_bonus,
    )
    base_instability = pbii_index(n, bases=bases)

    # TEMPO (single)
    tempo_res = omniatempo_analyze(
        series,
        short_window=short_window,
        long_window=long_window,
        hist_bins=hist_bins,
        epsilon=epsilon,
    )
    tempo_val = math.log(1.0 + tempo_res.regime_change_score)

    # CAUSA
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

    # IMAGE (optional)
    img_res: Optional[OmniaImageResult] = None
    img_val = 0.0
    if image is not None:
        img_res = omniabase_image(image)
        # High structure = low entropy
        img_val = 1.0 - img_res.entropy_mean

    # MULTI-SEQUENCE TEMPO (optional)
    tempo_mm_res: Optional[OmniatempoMMResult] = None
    seq_val = 0.0
    if seq_dict is not None:
        tempo_mm_res = omniatempo_multimodal(
            seq_dict,
            short_window=short_window,
            long_window=long_window,
            hist_bins=hist_bins,
            epsilon=epsilon,
        )
        seq_val = tempo_mm_res.mean_regime_change

    # NORMALIZATION
    c_base = _normalize_component(base_instability)
    c_tempo = _normalize_component(tempo_val)
    c_causa = _normalize_component(causa_val)
    c_img = _normalize_component(img_val)
    c_seq = _normalize_component(seq_val)

    # FUSED Ω
    omega = (
        w_base * c_base
        + w_tempo * c_tempo
        + w_causa * c_causa
        + w_image * c_img
        + w_seq * c_seq
    )

    components = {
        "base_instability": base_instability,
        "tempo_log_regime": tempo_val,
        "causa_mean_strength": causa_val,
        "image_1_minus_entropy": img_val,
        "seq_mean_regime": seq_val,
        "norm_base": c_base,
        "norm_tempo": c_tempo,
        "norm_causa": c_causa,
        "norm_image": c_img,
        "norm_seq": c_seq,
    }

    return OmniaTotaleResult(
        n=n,
        omniabase=base_sig,
        omniatempo=tempo_res,
        omniacausa=causa_res,
        image=img_res,
        tempo_mm=tempo_mm_res,
        omega_score=float(omega),
        components=components,
    )


# ==================================
# 7. BINARY EVAL (AUC / GSM8K STUB)
# ==================================

@dataclass
class BinaryEvalResult:
    thresholds: np.ndarray
    tpr: np.ndarray
    fpr: np.ndarray
    auc: float


def binary_eval_omega(
    y_true: Iterable[int],
    omega_scores: Iterable[float],
    num_thresholds: int = 100,
) -> BinaryEvalResult:
    """
    Simple binary-eval stub:

    - y_true: 1 = "good" (e.g., prime / correct chain), 0 = "bad".
    - omega_scores: fused Ω scores.
    - returns ROC curve (TPR/FPR) and AUC, for quick scans.
    """
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(omega_scores), dtype=float)

    if y.size == 0 or s.size == 0 or y.size != s.size:
        return BinaryEvalResult(
            thresholds=np.array([]),
            tpr=np.array([]),
            fpr=np.array([]),
            auc=0.0,
        )

    thr = np.linspace(s.min(), s.max(), num=num_thresholds)
    tpr_list: List[float] = []
    fpr_list: List[float] = []

    for t in thr:
        pred = (s >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))

        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_list.append(tpr_val)
        fpr_list.append(fpr_val)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[idx]
    tpr_sorted = tpr_arr[idx]

    auc = float(np.trapz(tpr_sorted, fpr_sorted))

    return BinaryEvalResult(
        thresholds=thr,
        tpr=tpr_sorted,
        fpr=fpr_sorted,
        auc=auc,
    )


# Minimal GSM8K-style skeleton:
# records = [{"id": ..., "is_correct": bool, "omega": float}, ...]
def gsm8k_eval_stub(records: List[Dict]) -> BinaryEvalResult:
    """
    GSM8K-style skeleton: treat omega as confidence for correctness.

    - records: list of {"is_correct": bool, "omega": float}
    - Uses binary_eval_omega under the hood.
    """
    y_true = [1 if r.get("is_correct", False) else 0 for r in records]
    omega_scores = [float(r.get("omega", 0.0)) for r in records]
    return binary_eval_omega(y_true, omega_scores)


# ===========
# 8. DEMO
# ==========

def demo() -> None:
    """
    Minimal multimodal demo for OMNIA_TOTALE v0.4.

    - Prime vs composite Ω scores.
    - Image lens.
    - Multi-sequence lens.
    - Binary eval on a tiny toy set.
    """
    import random

    random.seed(0)
    np.random.seed(0)

    n_prime = 173
    n_comp = 180

    # Base time series with regime shift
    t = np.arange(300)
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    # Causal toy system
    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)

    series_dict = {"s1": s1, "s2": s2, "s3": s3}

    # Image: simple gradient + noise
    H, W = 32, 32
    img = (
        np.linspace(0, 255, num=H * W).reshape(H, W)
        + 10 * np.random.randn(H, W)
    ).clip(0, 255)

    # Multi-sequence (e.g., RL logs)
    seq_dict = {
        "loss": list(series + 0.05 * np.random.normal(size=t.size)),
        "reward": list(
            0.5 + 0.2 * np.sin(t / 30.0) + 0.05 * np.random.normal