"""
omnia.core.omniacausa — lagged causal-structure lens (Vectorized)

Provides:
- OmniaEdge: directed edge with lag and strength
- OmniacausaResult: collection of edges
- omniacausa_analyze: heuristic lagged-correlation graph builder (Vectorized)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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


def omniacausa_analyze(
    series_dict: Dict[str, Iterable[float]],
    max_lag: int = 5,
    strength_threshold: float = 0.3,
) -> OmniacausaResult:
    """
    Heuristic causal structure: finds strongest lagged correlation between pairs.
    VECTORIZED IMPLEMENTATION (Fluid Dynamics Isomorphism).

    Instead of looping O(N^2 * Lags), we use matrix operations.
    """
    keys = list(series_dict.keys())
    if not keys:
        return OmniacausaResult(edges=[])

    # 1. Prepare Data Matrix (T x N)
    # T = time steps, N = number of series
    data_list = [np.asarray(list(series_dict[k]), dtype=float) for k in keys]
    # Ensure all same length for vectorization; truncate to min length
    min_len = min(len(d) for d in data_list)
    if min_len < 2:
        return OmniacausaResult(edges=[])

    X = np.stack([d[:min_len] for d in data_list], axis=1) # Shape: (T, N)

    # Normalize: (X - mean) / std
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)

    # Avoid division by zero
    valid_cols = stds > 1e-12
    if not np.any(valid_cols):
        return OmniacausaResult(edges=[])

    X_norm = np.zeros_like(X)
    X_norm[:, valid_cols] = (X[:, valid_cols] - means[valid_cols]) / stds[valid_cols]

    T_dim, N_dim = X.shape
    edges: List[OmniaEdge] = []

    # 2. Compute Correlations for all Lags simultaneously
    # Lags range from -max_lag to +max_lag
    # Positive lag k: src(t-k) -> tgt(t). This is corr(src[:-k], tgt[k:])

    lags = range(-max_lag, max_lag + 1)

    # Store best correlation found so far for each pair (src_idx, tgt_idx)
    # best_corrs[i, j] = value, best_lags[i, j] = lag
    best_corrs = np.zeros((N_dim, N_dim))
    best_lags = np.zeros((N_dim, N_dim), dtype=int)

    # We iterate lags, but vectorize the N*N correlation for each lag
    for lag in lags:
        if lag == 0:
            # Standard correlation matrix
            # Covariance is X_norm.T @ X_norm / T
            C = (X_norm.T @ X_norm) / T_dim
        elif lag > 0:
            # X_src leads (shifts back/slices early), X_tgt follows (shifts fwd/slices late)
            # src[:-lag], tgt[lag:]
            X_src = X_norm[:-lag]
            X_tgt = X_norm[lag:]
            # Adjust normalization for the slice?
            # Strictly speaking, we should re-normalize the slice.
            # For speed/isomorphism, we use the global normalization (approximate but fast).
            C = (X_src.T @ X_tgt) / (T_dim - lag)
        else: # lag < 0
            # src follows, tgt leads
            # src[abs_lag:], tgt[:-abs_lag]
            abs_lag = -lag
            X_src = X_norm[abs_lag:]
            X_tgt = X_norm[:-abs_lag]
            C = (X_src.T @ X_tgt) / (T_dim - abs_lag)

        # Update bests
        # We need abs(C) > abs(current_best)
        mask = np.abs(C) > np.abs(best_corrs)
        best_corrs[mask] = C[mask]
        best_lags[mask] = lag

    # 3. Extract Edges
    # Iterate the matrix once to build objects
    for i in range(N_dim):
        for j in range(N_dim):
            if i == j: continue
            strength = best_corrs[i, j]
            if abs(strength) >= strength_threshold:
                edges.append(OmniaEdge(
                    source=keys[i],
                    target=keys[j],
                    lag=int(best_lags[i, j]),
                    strength=float(strength)
                ))

    return OmniacausaResult(edges=edges)


__all__ = [
    "OmniaEdge",
    "OmniacausaResult",
    "omniacausa_analyze",
]
