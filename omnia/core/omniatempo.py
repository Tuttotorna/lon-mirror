"""
omnia.core.omniatempo — temporal stability lens

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


__all__ = [
    "OmniatempoResult",
    "omniatempo_analyze",
]