"""
omnia.api — public high-level OMNIA_TOTALE API

Single, stable entrypoint for external users:

    from omnia.api import omnia_totale

    result = omnia_totale(
        n=173,
        series=[...],
        series_dict={"s1": [...], "s2": [...]},
    )

Returns a plain Python dict, ready to be JSON-serialized.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from .engine import run_omnia_totale


def _to_numpy_series(
    series: Optional[Iterable[float]],
) -> Optional[np.ndarray]:
    """
    Normalize a generic iterable of numbers into a NumPy array.
    If series is None, returns None.
    """
    if series is None:
        return None
    return np.asarray(list(series), dtype=float)


def _to_numpy_series_dict(
    series_dict: Optional[Mapping[str, Iterable[float]]],
) -> Optional[Dict[str, np.ndarray]]:
    """
    Normalize a mapping name -> iterable into name -> NumPy array.
    If series_dict is None or empty, returns None.
    """
    if series_dict is None:
        return None
    if not series_dict:
        return None
    return {
        name: np.asarray(list(values), dtype=float)
        for name, values in series_dict.items()
    }


def omnia_totale(
    n: Optional[int],
    series: Optional[Iterable[float]] = None,
    series_dict: Optional[Mapping[str, Iterable[float]]] = None,
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    High-level, public OMNIA_TOTALE entrypoint.

    Parameters
    ----------
    n : int or None
        Target integer (e.g. numeric answer, key state). If None,
        the omniabase lens is effectively disabled.
    series : iterable of float or None
        Main time series for omniatempo. If None, omniatempo is disabled.
    series_dict : mapping str -> iterable[float] or None
        Multi-channel time series for omniacausa. If None/empty,
        omniacausa is disabled.
    w_base, w_tempo, w_causa : float
        Fusion weights for the three lenses.
    extra : dict or None
        Optional extra context, forwarded into the OmniaContext.

    Returns
    -------
    result_dict : dict
        Plain dict with at least:
        - "omega": fused Ω-score (float)
        - "lenses": per-lens scores (dict)
        - "metadata": per-lens metadata (dict)
    """
    series_np = _to_numpy_series(series)
    series_dict_np = _to_numpy_series_dict(series_dict)

    kernel_result = run_omnia_totale(
        n=n,
        series=series_np,
        series_dict=series_dict_np,
        w_base=w_base,
        w_tempo=w_tempo,
        w_causa=w_causa,
        extra=extra,
    )

    # KernelResult è un dataclass → asdict lo rende JSON-compatibile
    result_dict: Dict[str, Any] = asdict(kernel_result)
    return result_dict