"""
OMNIA_TOTALE v2.0 — Unified Ω-fusion demo

Author: Massimiliano Brighindi (MB-X.01 / Omniabase±)
Engine: omnia.core + omnia.engine + omnia.api

This file is now a thin demo script that calls the public API:

    from omnia.api import omnia_totale

and prints a sample Ω-evaluation for:
- a prime vs a composite integer
- a synthetic time series with a regime shift
- a small 3-channel causal structure.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

import numpy as np

from omnia.api import omnia_totale


def build_demo_timeseries(n_points: int = 300) -> np.ndarray:
    """
    Build a simple time series:
    - sinusoid with noise
    - plus a regime shift in the last third.
    """
    t = np.arange(n_points)
    base = np.sin(t / 15.0)
    noise = 0.1 * np.random.normal(size=n_points)
    series = base + noise
    # regime shift on last third
    shift_start = int(2 * n_points / 3)
    series[shift_start:] += 0.8
    return series


def build_demo_multichannel(n_points: int = 300) -> Dict[str, np.ndarray]:
    """
    Build 3 channels:
    - s1: smooth sinusoid
    - s2: lagged, noisy version of s1
    - s3: pure noise (weakly coupled)
    """
    t = np.arange(n_points)

    s1 = np.sin(t / 10.0)

    s2 = np.zeros_like(s1)
    if n_points > 2:
        s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=n_points - 2)

    s3 = np.random.normal(size=n_points)

    return {"s1": s1, "s2": s2, "s3": s3}


def pretty_print_result(label: str, result: Dict) -> None:
    """
    Compact pretty-printer for omnia_totale() results.
    """
    omega = result.get("omega", None)
    lenses = result.get("lenses", {})
    print(f"\n=== {label} ===")
    print(f"Ω (fused) ≈ {omega:.6f}" if isinstance(omega, (int, float)) else f"Ω: {omega}")
    print("Per-lens ω:")
    for name, scores in lenses.items():
        w = scores.get("omega", 0.0)
        print(f"  - {name}: {w:.6f}")


def main_demo() -> None:
    """
    Run a minimal OMNIA_TOTALE v2.0 demo:

    - evaluates a prime and a composite integer,
    - uses a synthetic time-series for omniatempo,
    - uses a 3-channel synthetic system for omniacausa.
    """
    random.seed(0)
    np.random.seed(0)

    n_prime = 173
    n_comp = 180

    series = build_demo_timeseries(n_points=300)
    series_dict = build_demo_multichannel(n_points=300)

    # Prime
    res_prime = omnia_totale(
        n=n_prime,
        series=series,
        series_dict=series_dict,
        w_base=1.0,
        w_tempo=1.0,
        w_causa=1.0,
        extra={"label": "prime_demo"},
    )

    # Composite
    res_comp = omnia_totale(
        n=n_comp,
        series=series,
        series_dict=series_dict,
        w_base=1.0,
        w_tempo=1.0,
        w_causa=1.0,
        extra={"label": "composite_demo"},
    )

    print("=== OMNIA_TOTALE v2.0 demo ===")
    pretty_print_result(f"n = {n_prime} (prime)", res_prime)
    pretty_print_result(f"n = {n_comp} (composite)", res_comp)


if __name__ == "__main__":
    main_demo()