"""
OMNIACAUSA — lagged causal-structure lens
Author: Massimiliano Brighindi (MBX)

Exposes:
- OmniaEdge
- OmniacausaResult
- omniacausa_analyze
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List
import numpy as np
import math


# =========================
# Data structures
# =========================

@dataclass
class OmniaEdge:
    source: str
    target: str
    lag: int          # positive: source leads target
    strength: float   # correlation in [-1, 1]


@dataclass
class OmniacausaResult:
    edges: List[OmniaEdge]


# =========================
# Internal helpers
# =========================

def _lagged_corr_np(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    Pearson correlation between x and y with given lag.

    Conventions:
    - lag > 0: x leads y (x[t-lag] -> y[t])
    - lag < 0: y leads x (y[t+lag] -> x[t])
    - lag = 0: synchronous correlation
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
    num = float(np.sum((x_l - x_mean) * (y_l - y_mean)))
    den = math.sqrt(
        float(np.sum((x_l - x_mean) ** 2) * np.sum((y_l - y_mean) ** 2))
    )
    if den == 0:
        return 0.0
    return num / den


# =========================
# Main causal lens
# =========================

def omniacausa_analyze(
    series_dict: Dict[str, Iterable[float]],
    max_lag: int = 5,
    strength_threshold: float = 0.3,
) -> OmniacausaResult:
    """
    Heuristic causal structure discovery over multivariate time series.

    Parameters
    ----------
    series_dict : dict
        key -> iterable of floats (same length per key ideally)
    max_lag : int
        maximum |lag| to search (from -max_lag to +max_lag)
    strength_threshold : float
        minimum |corr| to keep an edge

    Returns
    -------
    OmniacausaResult
        edges: list of OmniaEdge(source, target, lag, strength)

    Notes
    -----
    - This is NOT full causal discovery.
    - It is a structural "lens": it highlights strong, lagged correlations,
      which are often useful signals for regime analysis and stability checks.
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

            # Align lengths (min common length)
            min_len = min(x.size, y.size)
            if min_len < 3:
                continue
            x = x[:min_len]
            y = y[:min_len]

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
                        strength=float(best_corr),
                    )
                )

    return OmniacausaResult(edges=edges)