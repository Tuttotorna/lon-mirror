"""
omnia.core.omniacausa — lagged causal-structure lens

Provides:
- OmniaEdge: directed edge with lag and strength
- OmniacausaResult: collection of edges
- omniacausa_analyze: heuristic lagged-correlation graph builder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import math
import numpy as np


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


__all__ = [
    "OmniaEdge",
    "OmniacausaResult",
    "omniacausa_analyze",
]