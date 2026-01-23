"""
omnia.core.omniatempo — temporal stability lens (Vectorized)

Provides:
- OmniatempoResult: dataclass with global/local stats and regime-change score
- omniatempo_analyze: analyze 1D time series stability via symmetric KL-like divergence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

import numpy as np


@dataclass
class OmniatempoResult:
    global_mean: float
    global_std: float
    short_mean: float
    short_std: float
    long_mean: float
    long_std: float
    regime_change_score: float


def _histogram_probs_vec(x: np.ndarray, bins: int = 20) -> np.ndarray:
    """Return normalized histogram probabilities for x (Vectorized)."""
    if x.size == 0:
        return np.zeros(bins, dtype=float)

    # Fast numpy histogram
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
    Analyze 1D time series stability using Vectorized NumPy ops.

    Returns global stats and symmetric KL-like divergence
    between recent-short vs. recent-long distributions.
    """
    # 1. Load Data (Memory mapping)
    x = np.asarray(list(series), dtype=float)
    if x.size == 0:
        return OmniatempoResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # 2. Global Stats (Vectorized)
    g_mean = float(np.mean(x))
    g_std = float(np.std(x, ddof=0))

    sw = min(short_window, x.size)
    lw = min(long_window, x.size)

    # 3. Slicing
    short_seg = x[-sw:]
    long_seg = x[-lw:]

    s_mean = float(np.mean(short_seg))
    s_std = float(np.std(short_seg, ddof=0))
    l_mean = float(np.mean(long_seg))
    l_std = float(np.std(long_seg, ddof=0))

    # 4. Probabilities (Vectorized Histogram)
    # Important: To compare KL divergence, bins must be aligned.
    # We must compute a common range for the histograms.
    data_min = min(short_seg.min(), long_seg.min())
    data_max = max(short_seg.max(), long_seg.max())

    if data_min == data_max:
         # Trivial flatline
         return OmniatempoResult(g_mean, g_std, s_mean, s_std, l_mean, l_std, 0.0)

    # Shared bins
    bins_edges = np.linspace(data_min, data_max, hist_bins + 1)

    p, _ = np.histogram(short_seg, bins=bins_edges, density=False)
    q, _ = np.histogram(long_seg, bins=bins_edges, density=False)

    # Normalize + Epsilon smoothing
    p = p.astype(float) / p.sum() + epsilon
    q = q.astype(float) / q.sum() + epsilon

    # Re-normalize after epsilon
    p /= p.sum()
    q /= q.sum()

    # 5. KL Divergence (Vectorized)
    # KL(P||Q) = sum(p * log(p/q))
    # regime = 0.5 * (KL(P||Q) + KL(Q||P))

    # np.log is vectorized
    log_pq = np.log(p / q)
    log_qp = np.log(q / p)

    kl_pq = np.sum(p * log_pq)
    kl_qp = np.sum(q * log_qp)

    regime = float(0.5 * (kl_pq + kl_qp))

    return OmniatempoResult(
        global_mean=g_mean,
        global_std=g_std,
        short_mean=s_mean,
        short_std=s_std,
        long_mean=l_mean,
        long_std=l_std,
        regime_change_score=regime,
    )


__all__ = [
    "OmniatempoResult",
    "omniatempo_analyze",
]
