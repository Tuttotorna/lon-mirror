"""
OMNIATEMPO — Temporal stability lens
Core module for OMNIA_TOTALE package
Author: Massimiliano Brighindi (concepts) + MBX IA (structure)
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class OmniatempoResult:
    global_mean: float
    global_std: float
    short_mean: float
    short_std: float
    long_mean: float
    long_std: float
    regime_change: float

def _hist_probs(x: np.ndarray, bins: int = 20) -> np.ndarray:
    if x.size == 0:
        return np.zeros(bins)
    hist, _ = np.histogram(x, bins=bins)
    total = hist.sum()
    return hist / total if total > 0 else np.zeros(bins)

def analyze_temporal(
    series, short_window=20, long_window=100, bins=20, eps=1e-9
) -> OmniatempoResult:

    x = np.asarray(list(series), dtype=float)
    if x.size == 0:
        return OmniatempoResult(0,0,0,0,0,0,0)

    g_mean = float(x.mean())
    g_std  = float(x.std())

    sw = min(short_window, x.size)
    lw = min(long_window, x.size)

    short = x[-sw:]
    long  = x[-lw:]

    s_mean = float(short.mean())
    s_std  = float(short.std())
    l_mean = float(long.mean())
    l_std  = float(long.std())

    p = _hist_probs(short, bins) + eps
    q = _hist_probs(long, bins) + eps
    p /= p.sum()
    q /= q.sum()

    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    regime = 0.5 * (kl_pq + kl_qp)

    return OmniatempoResult(
        global_mean=g_mean,
        global_std=g_std,
        short_mean=s_mean,
        short_std=s_std,
        long_mean=l_mean,
        long_std=l_std,
        regime_change=float(regime)
    )