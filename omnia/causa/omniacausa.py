"""
OMNIACAUSA — Causal heuristic lens
Core module for OMNIA_TOTALE package
Author: Massimiliano Brighindi (concepts) + MBX IA (structure)
"""

import numpy as np
from dataclasses import dataclass
import math

@dataclass
class OmniaEdge:
    source: str
    target: str
    lag: int
    strength: float

@dataclass
class OmniacausaResult:
    edges: list

def _lag_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    Pearson correlation between x(t) and y(t+lag).
    Positive lag → x leads y.
    """
    if lag > 0:
        x_l = x[:-lag]
        y_l = y[lag:]
    elif lag < 0:
        lag = -lag
        x_l = x[lag:]
        y_l = y[:-lag]
    else:
        x_l, y_l = x, y

    if x_l.size < 2 or y_l.size < 2:
        return 0.0

    xm = x_l.mean()
    ym = y_l.mean()

    num = np.sum((x_l - xm) * (y_l - ym))
    den = math.sqrt(np.sum((x_l - xm)**2) * np.sum((y_l - ym)**2))
    return float(num / den) if den != 0 else 0.0


def analyze_causal(
    series_dict: dict,
    max_lag: int = 5,
    threshold: float = 0.3
) -> OmniacausaResult:

    keys = list(series_dict.keys())
    arrays = {k: np.asarray(series_dict[k], dtype=float) for k in keys}
    edges = []
    lag_range = range(-max_lag, max_lag + 1)

    for s in keys:
        for t in keys:
            if s == t:
                continue

            x = arrays[s]
            y = arrays[t]

            best_corr = 0.0
            best_lag = 0

            for lag in lag_range:
                c = _lag_corr(x, y, lag)
                if abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag = lag

            if abs(best_corr) >= threshold:
                edges.append(
                    OmniaEdge(
                        source=s,
                        target=t,
                        lag=best_lag,
                        strength=float(best_corr)
                    )
                )

    return OmniacausaResult(edges=edges)