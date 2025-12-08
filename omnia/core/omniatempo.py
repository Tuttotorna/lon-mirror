"""
OMNIATEMPO — temporal stability & regime-change lens
Author: Massimiliano Brighindi (MBX)

Exposes:
- OmniatempoResult
- omniatempo_analyze
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import math


# =========================
# Result structure
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


# =========================
# Histogram helper
# =========================

def _histogram_probs(x: np.ndarray, bins: int = 20) -> np.ndarray:
    """Return normalized histogram probabilities for x."""
    if x.size == 0:
        return np.zeros(bins, dtype=float)
    hist, _ = np.histogram(x, bins=bins, density=False)
    total = hist.sum()
    if total == 0:
        return np.zeros_like(hist, dtype=float)
    return hist.astype(float) / total


# =========================
# Main temporal lens
# =========================

def omniatempo_analyze(
    series: Iterable[float],
    short_window: int = 20,
    long_window: int = 100,
    hist_bins: int = 20,
    epsilon: float = 1e-9,
) -> OmniatempoResult:
    """
    Temporal stability analysis.

    Returns:
    - global stats
    - short vs long window stats
    - symmetric KL divergence (regime-change score)
    """

    x = np.asarray(list(series), dtype=float)
    if x.size == 0:
        return OmniatempoResult(0, 0, 0, 0, 0, 0, 0)

    # --- Global statistics ---
    g_mean = float(x.mean())
    g_std  = float(x.std(ddof=0))

    sw = min(short_window, x.size)
    lw = min(long_window, x.size)

    short_seg = x[-sw:]
    long_seg  = x[-lw:]

    # --- Local statistics ---
    s_mean = float(short_seg.mean())
    s_std  = float(short_seg.std(ddof=0))
    l_mean = float(long_seg.mean())
    l_std  = float(long_seg.std(ddof=0))

    # --- Histograms ---
    p = _histogram_probs(short_seg, bins=hist_bins) + epsilon
    q = _histogram_probs(long_seg,  bins=hist_bins) + epsilon
    p /= p.sum()
    q /= q.sum()

    # --- Symmetric KL divergence ---
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    regime_change = 0.5 * (kl_pq + kl_qp)

    return OmniatempoResult(
        global_mean=g_mean,
        global_std=g_std,
        short_mean=s_mean,
        short_std=s_std,
        long_mean=l_mean,
        long_std=l_std,
        regime_change_score=regime_change,
    )