"""
OMNIA_TOTALE v2.0 — demo runner on top of omnia.engine

Author: Massimiliano Brighindi (MB-X.01 / Omniabase±)
Engine formalization: MBX IA

This script is a thin demo:
- builds synthetic numeric + time-series data
- calls omnia.engine.run_omnia_totale(...)
- prints fused Ω and per-lens scores
"""

from __future__ import annotations

import numpy as np

from omnia.engine import run_omnia_totale


def build_synthetic_example():
    """
    Build a small synthetic example:

    - n = 173 (prime) as target integer.
    - series: 1D time-series with a late regime shift.
    - series_dict:
        s1: smooth sine signal
        s2: lagged, noisy version of s1
        s3: pure noise
    """
    t = np.arange(300)

    # main series with regime shift
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    # multi-channel series
    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)

    series_dict = {
        "s1": s1,
        "s2": s2,
        "s3": s3,
    }

    n = 173  # prime; change to 180 (composite) to compare

    return n, series, series_dict


def main():
    # deterministic demo
    np.random.seed(0)

    n, series, series_dict = build_synthetic_example()

    result = run_omnia_totale(
        n=n,
        series=series,
        series_dict=series_dict,
        w_base=1.0,
        w_tempo=1.0,
        w_causa=1.0,
        extra={"label": "demo_prime_173"},
    )

    print("=== OMNIA_TOTALE v2.0 demo ===")
    print(f"fused_omega = {result.fused_omega:.4f}")
    print("lens scores:")
    for name, lens_result in result.lenses.items():
        omega = float(lens_result.scores.get("omega", 0.0))
        print(f"  - {name}: omega = {omega:.4f}, scores = {lens_result.scores}")


if __name__ == "__main__":
    main()