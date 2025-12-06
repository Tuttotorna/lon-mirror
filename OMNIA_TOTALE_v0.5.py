"""
OMNIA_TOTALE v0.5 — Multimodal Ω-fusion + Eval Module (NumPy)
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Dependencies:
    pip install numpy
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Optional, Tuple
import math

import numpy as np

# ============================================================
# 1. OMNIABASE (multi-base numeric lens, NumPy-accelerated)
# ============================================================

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


@dataclass
class OmniabaseConfig:
    bases: Tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19)
    length_weight: float = 1.0
    length_exponent: float = 1.0
    divisibility_bonus: float = 0.5


def omniabase_signature(
    n: int,
    cfg: OmniabaseConfig = OmniabaseConfig(),
) -> OmniabaseSignature:
    """Compute multi-base signature for integer n."""
    bases = list(cfg.bases)
    sigmas: Dict[int, float] = {}
    entropy: Dict[int, float] = {}
    for b in bases:
        sig = sigma_b(
            n,
            b,
            length_weight=cfg.length_weight,
            length_exponent=cfg.length_exponent,
            divisibility_bonus=cfg.divisibility_bonus,
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
    cfg: OmniabaseConfig = OmniabaseConfig(),
) -> float:
    """
    Prime Base Instability Index (NumPy variant).

    PBII(n) = mean_sigma(composites) - mean_sigma(n)
    (Higher values ~ more prime-like instability)
    """
    comp = list(composite_window)
    comp_sigmas = []
    for c in comp:
        sig_c = omniabase_signature(c, cfg).sigma_mean
        comp_sigmas.append(sig_c)
    sat = float(np.mean(comp_sigmas)) if comp_sigmas else 0.0
    sig_n = omniabase_signature(n, cfg).sigma_mean
    return float(sat - sig_n)


# ============================================================
# 2. OMNIATEMPO (time lens, NumPy)
# ============================================================

@dataclass
class OmniatempoConfig:
    short_window: int = 20
    long_window: int = 100
    hist_bins: int = 20
    epsilon: float = 1e-9


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
    cfg: OmniatempoConfig = OmniatempoConfig(),
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

    sw = min(cfg.short_window, x.size)
    lw = min(cfg.long_window, x.size)

    short_seg = x[-sw:]
    long_seg = x[-lw:]

    s_mean = float(short_seg.mean())
    s_std = float(short_seg.std(ddof=0))
    l_mean = float(long_seg.mean())
    l_std = float(long_seg.std(ddof=0))

    p = _histogram_probs(short_seg, bins=cfg.hist_bins) + cfg.epsilon
    q = _histogram_probs(long_seg, bins=cfg.hist_bins) + cfg.epsilon
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


# ============================================================
# 3. OMNIACAUSA (causal lens, NumPy)
# ============================================================

@dataclass
class OmniacausaConfig:
    max_lag: int = 5
    strength_threshold: float = 0.3


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
    cfg: OmniacausaConfig = OmniacausaConfig(),
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
    lags = list(range(-cfg.max_lag, cfg.max_lag + 1))
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
            if abs(best_corr) >= cfg.strength_threshold:
                edges.append(
                    OmniaEdge(
                        source=src,
                        target=tgt,
                        lag=best_lag,
                        strength=best_corr,
                    )
                )
    return OmniacausaResult(edges=edges)


# ============================================================
# 4. OMNIAIMAGE (image entropy lens)
# ============================================================

@dataclass
class OmniaImageConfig:
    bins: int = 64
    epsilon: float = 1e-9


@dataclass
class OmniaImageResult:
    entropy_per_channel: Dict[int, float]
    entropy_mean: float


def _channel_entropy(arr: np.ndarray, bins: int, epsilon: float) -> float:
    """Compute normalized entropy of a single channel image array in [0,1] or [0,255]."""
    if arr.size == 0:
        return 0.0
    arr = arr.astype(float)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        return 0.0
    # Normalize to [0,1] and bin
    norm = (arr - arr_min) / (arr_max - arr_min + epsilon)
    hist, _ = np.histogram(norm, bins=bins, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist.astype(float) / total
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log2(probs))
    Hmax = math.log2(bins)
    return float(H / Hmax) if Hmax > 0 else 0.0


def omniaimage_entropy(
    image: np.ndarray,
    cfg: OmniaImageConfig = OmniaImageConfig(),
) -> OmniaImageResult:
    """
    Compute normalized entropy per channel for an image.

    image: HxW (grayscale) or HxWxC (RGB / multi-channel).
    """
    img = np.asarray(image)
    if img.ndim == 2:  # HxW
        ent = _channel_entropy(img, cfg.bins, cfg.epsilon)
        return OmniaImageResult(entropy_per_channel={0: ent}, entropy_mean=ent)
    elif img.ndim == 3:  # HxWxC
        C = img.shape[2]
        ent_ch: Dict[int, float] = {}
        for c in range(C):
            ent_ch[c] = _channel_entropy(img[:, :, c], cfg.bins, cfg.epsilon)
        vals = np.array(list(ent_ch.values()), dtype=float)
        return OmniaImageResult(
            entropy_per_channel=ent_ch,
            entropy_mean=float(vals.mean()) if vals.size else 0.0,
        )
    else:
        raise ValueError("image must be 2D (H,W) or 3D (H,W,C)")


# ============================================================
# 5. MULTI-SEQUENCE OMNIATEMPO (for larger scale)
# ============================================================

@dataclass
class OmniatempoMultiResult:
    per_key: Dict[str, OmniatempoResult]
    mean_regime_score: float


def omniatempo_multi(
    series_dict: Dict[str, Iterable[float]],
    cfg: OmniatempoConfig = OmniatempoConfig(),
) -> OmniatempoMultiResult:
    """
    Apply Omniatempo to multiple sequences and aggregate regime-change scores.
    """
    per_key: Dict[str, OmniatempoResult] = {}
    scores: List[float] = []
    for key, seq in series_dict.items():
        res = omniatempo_analyze(seq, cfg)
        per_key[key] = res
        scores.append(res.regime_change_score)
    scores_arr = np.asarray(scores, dtype=float)
    mean_regime = float(scores_arr.mean()) if scores_arr.size else 0.0
    return OmniatempoMultiResult(per_key=per_key, mean_regime_score=mean_regime)


# ============================================================
# 6. Ω-FUSION CORE (numeric + time + causal + image)
# ============================================================

@dataclass
class OmegaFusionConfig:
    w_base: float = 1.0
    w_time: float = 1.0
    w_causa: float = 1.0
    w_image: float = 1.0
    normalize_log_time: bool = True
    epsilon: float = 1e-9


@dataclass
class OmniaSampleResult:
    n: int
    omniabase: OmniabaseSignature
    pbii: float
    omniatempo: Optional[OmniatempoResult]
    omniatempo_multi: Optional[OmniatempoMultiResult]
    omniacausa: Optional[OmniacausaResult]
    omniaimage: Optional[OmniaImageResult]
    omega_score: float
    components: Dict[str, float]


def omnia_fused_score(
    n: int,
    base_cfg: OmniabaseConfig,
    fusion_cfg: OmegaFusionConfig,
    tempo_series: Optional[Iterable[float]] = None,
    tempo_cfg: OmniatempoConfig = OmniatempoConfig(),
    multi_series: Optional[Dict[str, Iterable[float]]] = None,
    causa_series: Optional[Dict[str, Iterable[float]]] = None,
    causa_cfg: OmniacausaConfig = OmniacausaConfig(),
    image: Optional[np.ndarray] = None,
    image_cfg: OmniaImageConfig = OmniaImageConfig(),
) -> OmniaSampleResult:
    """
    Compute fused Ω score for a single sample with optional modalities.
    """
    # BASE
    base_sig = omniabase_signature(n, base_cfg)
    base_instability = pbii_index(n, cfg=base_cfg)

    # TIME (single series)
    tempo_res: Optional[OmniatempoResult] = None
    tempo_val = 0.0
    if tempo_series is not None:
        tempo_res = omniatempo_analyze(tempo_series, tempo_cfg)
        if fusion_cfg.normalize_log_time:
            tempo_val = math.log(1.0 + tempo_res.regime_change_score)
        else:
            tempo_val = tempo_res.regime_change_score

    # TIME MULTI
    tempo_multi_res: Optional[OmniatempoMultiResult] = None
    tempo_multi_val = 0.0
    if multi_series is not None:
        tempo_multi_res = omniatempo_multi(multi_series, tempo_cfg)
        if fusion_cfg.normalize_log_time:
            tempo_multi_val = math.log(1.0 + tempo_multi_res.mean_regime_score)
        else:
            tempo_multi_val = tempo_multi_res.mean_regime_score

    # CAUSA
    causa_res: Optional[OmniacausaResult] = None
    causa_val = 0.0
    if causa_series is not None:
        causa_res = omniacausa_analyze(causa_series, causa_cfg)
        if causa_res.edges:
            strengths = np.array([abs(e.strength) for e in causa_res.edges], dtype=float)
            causa_val = float(strengths.mean())
        else:
            causa_val = 0.0

    # IMAGE
    image_res: Optional[OmniaImageResult] = None
    image_val = 0.0
    if image is not None:
        image_res = omniaimage_entropy(image, image_cfg)
        image_val = image_res.entropy_mean

    omega = (
        fusion_cfg.w_base * base_instability
        + fusion_cfg.w_time * (tempo_val + tempo_multi_val)
        + fusion_cfg.w_causa * causa_val
        + fusion_cfg.w_image * image_val
    )

    components = {
        "base_instability": base_instability,
        "tempo_single": tempo_val,
        "tempo_multi": tempo_multi_val,
        "causa_mean_strength": causa_val,
        "image_entropy_mean": image_val,
    }

    return OmniaSampleResult(
        n=n,
        omniabase=base_sig,
        pbii=base_instability,
        omniatempo=tempo_res,
        omniatempo_multi=tempo_multi_res,
        omniacausa=causa_res,
        omniaimage=image_res,
        omega_score=float(omega),
        components=components,
    )


# ============================================================
# 7. SIMPLE AUC / BENCHMARKING UTILITIES (no external deps)
# ============================================================

def roc_curve_from_scores(
    scores: Iterable[float],
    labels: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ROC curve (FPR, TPR) from anomaly/coherence scores and binary labels.
    labels: 1 = positive (e.g., correct), 0 = negative.
    """
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    if s.size == 0 or y.size == 0 or s.size != y.size:
        raise ValueError("scores and labels must be non-empty and same length")

    # Sort by descending score
    order = np.argsort(-s)
    s = s[order]
    y = y[order]

    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        raise ValueError("both positive and negative labels required for ROC")

    tpr_list = []
    fpr_list = []
    tp = 0
    fp = 0
    prev_score = None

    for score, label in zip(s, y):
        if prev_score is None or score != prev_score:
            tpr_list.append(tp / P)
            fpr_list.append(fp / N)
            prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1

    tpr_list.append(tp / P)
    fpr_list.append(fp / N)

    return np.array(fpr_list, dtype=float), np.array(tpr_list, dtype=float)


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute AUC using trapezoidal rule."""
    if fpr.size != tpr.size or fpr.size == 0:
        raise ValueError("fpr and tpr must be same non-zero length")
    # Sort by fpr ascending
    order = np.argsort(fpr)
    f = fpr[order]
    t = tpr[order]
    return float(np.trapz(t, f))


# ============================================================
# 8. OMNIA EVAL MODULE (for LLM outputs / GSM8K-style tasks)
# ============================================================

@dataclass
class OmniaEvalConfig:
    base_cfg: OmniabaseConfig = OmniabaseConfig()
    tempo_cfg: OmniatempoConfig = OmniatempoConfig()
    causa_cfg: OmniacausaConfig = OmniacausaConfig()
    image_cfg: OmniaImageConfig = OmniaImageConfig()
    fusion_cfg: OmegaFusionConfig = OmegaFusionConfig()
    # For benchmarking
    roc_min_points: int = 10


@dataclass
class LLMTrace:
    """
    Minimal container for a single LLM reasoning trace.

    - problem_id: identifier (e.g., GSM8K id)
    - numeric_anchor: integer anchor (e.g., final number predicted, or hash)
    - token_logprobs: sequence of log-probabilities for tokens (if available)
    - step_scores: optional per-step scalar scores (e.g., model confidences)
    - is_correct: ground truth label (1/0) if available
    """
    problem_id: str
    numeric_anchor: int
    token_logprobs: Optional[List[float]] = None
    step_scores: Optional[List[float]] = None
    is_correct: Optional[int] = None


@dataclass
class OmniaEvalResult:
    traces: List[LLMTrace]
    omegas: List[float]
    components_list: List[Dict[str, float]]
    auc: Optional[float]


class OmniaEvalModule:
    """
    Eval module: maps LLM traces -> Ω-scores and optional AUC on correctness.
    """

    def __init__(self, cfg: OmniaEvalConfig = OmniaEvalConfig()):
        self.cfg = cfg

    def score_trace(self, trace: LLMTrace) -> Tuple[float, Dict[str, float]]:
        """
        Compute Ω-score for a single trace using:
        - numeric_anchor via Omniabase/PBII
        - token_logprobs via Omniatempo
        - step_scores via Omniatempo (multi)
        """
        # TIME: token_logprobs as a 1D series
        tempo_series = trace.token_logprobs if trace.token_logprobs is not None else None

        # MULTI-TIME: combine token_logprobs + step_scores if both present
        multi_series: Optional[Dict[str, Iterable[float]]] = None
        if trace.token_logprobs is not None or trace.step_scores is not None:
            multi_series = {}
            if trace.token_logprobs is not None:
                multi_series["logprobs"] = trace.token_logprobs
            if trace.step_scores is not None:
                multi_series["steps"] = trace.step_scores

        # No causal or image in this minimal eval; they can be added later
        sample_res = omnia_fused_score(
            n=trace.numeric_anchor,
            base_cfg=self.cfg.base_cfg,
            fusion_cfg=self.cfg.fusion_cfg,
            tempo_series=tempo_series,
            tempo_cfg=self.cfg.tempo_cfg,
            multi_series=multi_series,
            causa_series=None,
            causa_cfg=self.cfg.causa_cfg,
            image=None,
            image_cfg=self.cfg.image_cfg,
        )
        return sample_res.omega_score, sample_res.components

    def evaluate_traces(self, traces: List[LLMTrace]) -> OmniaEvalResult:
        """
        Compute Ω-scores for a batch of traces and AUC if labels available.
        """
        omegas: List[float] = []
        components_list: List[Dict[str, float]] = []
        labels: List[int] = []

        for tr in traces:
            omega, comps = self.score_trace(tr)
            omegas.append(omega)
            components_list.append(comps)
            if tr.is_correct is not None:
                labels.append(int(tr.is_correct))

        auc_val: Optional[float] = None
        if labels and len(labels) == len(omegas):
            try:
                fpr, tpr = roc_curve_from_scores(omegas, labels)
                auc_val = auc_from_roc(fpr, tpr)
            except ValueError:
                auc_val = None

        return OmniaEvalResult(
            traces=traces,
            omegas=omegas,
            components_list=components_list,
            auc=auc_val,
        )


# ============================================================
# 9. DEMOS
# ============================================================

def demo_numeric_prime_vs_composite():
    """Demo: Ω-score difference between prime and composite integers."""
    base_cfg = OmniabaseConfig()
    fusion_cfg = OmegaFusionConfig()
    tempo_cfg = OmniatempoConfig()
    # Simple synthetic time series shared
    t = np.arange(300)
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    multi_series = {
        "sensor1": series,
        "sensor2": series + 0.05 * np.random.normal(size=t.size),
    }

    n_prime = 173
    n_comp = 180

    res_prime = omnia_fused_score(
        n_prime,
        base_cfg=base_cfg,
        fusion_cfg=fusion_cfg,
        tempo_series=series,
        tempo_cfg=tempo_cfg,
        multi_series=multi_series,
    )
    res_comp = omnia_fused_score(
        n_comp,
        base_cfg=base_cfg,
        fusion_cfg=fusion_cfg,
        tempo_series=series,
        tempo_cfg=tempo_cfg,
        multi_series=multi_series,
    )

    print("=== Demo: prime vs composite ===")
    print(f"Prime   n={n_prime}, Ω={res_prime.omega_score:.4f}, components={res_prime.components}")
    print(f"Composite n={n_comp}, Ω={res_comp.omega_score:.4f}, components={res_comp.components}")


def demo_image_entropy():
    """Demo: image entropy lens on synthetic images."""
    img_low = np.zeros((64, 64), dtype=float)
    img_high = np.random.rand(64, 64)
    cfg = OmniaImageConfig()

    res_low = omniaimage_entropy(img_low, cfg)
    res_high = omniaimage_entropy(img_high, cfg)

    print("=== Demo: image entropy ===")
    print(f"Low-entropy image:  entropy_mean={res_low.entropy_mean:.4f}")
    print(f"High-entropy image: entropy_mean={res_high.entropy_mean:.4f}")


def demo_gsm8k_stub():
    """
    GSM8K-style stub:
    - Simulate some correct/incorrect reasoning traces
    - Compute Ω for each
    - Compute AUC on correctness
    """
    np.random.seed(0)
    eval_cfg = OmniaEvalConfig()
    module = OmniaEvalModule(eval_cfg)

    traces: List[LLMTrace] = []
    for i in range(50):
        # Simulate a "problem id"
        pid = f"gsm8k_{i:03d}"
        # numeric anchor: pretend this is the final integer answer
        numeric_anchor = np.random.randint(10, 500)
        # token logprobs: stable for correct, noisier for incorrect
        is_correct = 1 if np.random.rand() < 0.5 else 0
        length = np.random.randint(30, 80)
        base_logprob = -1.0 if is_correct else -1.5
        noise_scale = 0.1 if is_correct else 0.4
        token_logprobs = (base_logprob + noise_scale * np.random.randn(length)).tolist()
        # step_scores: cumulative moving average of logprobs
        arr = np.asarray(token_logprobs)
        step_scores = np.cumsum(arr) / (np.arange(arr.size) + 1)

        traces.append(
            LLMTrace(
                problem_id=pid,
                numeric_anchor=int(numeric_anchor),
                token_logprobs=token_logprobs,
                step_scores=step_scores.tolist(),
                is_correct=is_correct,
            )
        )

    result = module.evaluate_traces(traces)
    print("=== Demo: GSM8K stub ===")
    print(f"Computed Ω for {len(result.traces)} traces.")
    if result.auc is not None:
        print(f"Approx. AUC on correctness vs Ω-score: {result.auc:.3f}")
    else:
        print("AUC not available (labels missing or degenerate).")


def demo_all():
    """Run all demos."""
    demo_numeric_prime_vs_composite()
    print()
    demo_image_entropy()
    print()
    demo_gsm8k_stub()


if __name__ == "__main__":
    demo_all()
```0