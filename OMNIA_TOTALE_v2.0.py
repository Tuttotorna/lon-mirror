"""
OMNIA_TOTALE v2.0 — package-based demo

Thin demo script that uses the omnia package:

- imports run_omnia_totale from omnia.engine
- builds a synthetic example (prime vs composite, time series, causal series)
- prints Ω score and per-lens contributions

Author: Massimiliano Brighindi (MB-X.01) + MBX IA
"""

from __future__ import annotations

import numpy as np

from omnia.engine import run_omnia_totale


def demo() -> None:
    # Example integers: one prime, one composite
    n_prime = 173
    n_comp = 180

    # Synthetic time series with regime shift
    np.random.seed(0)
    t = np.arange(300)
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    # Simple causal structure: s1 → s2 with lag 2, s3 = noise
    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)

    series_dict = {"s1": s1, "s2": s2, "s3": s3}

    print("=== OMNIA_TOTALE v2.0 — package demo ===")

    # Prime
    res_prime = run_omnia_totale(
        n=n_prime,
        series=series,
        series_dict=series_dict,
        w_base=1.0,
        w_tempo=1.0,
        w_causa=1.0,
        extra={"label": "prime_demo"},
    )
    print(f"\nn = {n_prime} (prime)")
    print(f"Ω ≈ {res_prime.omega:.4f}")
    print("components:", res_prime.components)
    for lr in res_prime.lens_results:
        print(f"  [{lr.name}] omega={lr.scores.get('omega', 0.0):.4f}")

    # Composite
    res_comp = run_omnia_totale(
        n=n_comp,
        series=series,
        series_dict=series_dict,
        w_base=1.0,
        w_tempo=1.0,
        w_causa=1.0,
        extra={"label": "composite_demo"},
    )
    print(f"\nn = {n_comp} (composite)")
    print(f"Ω ≈ {res_comp.omega:.4f}")
    print("components:", res_comp.components)
    for lr in res_comp.lens_results:
        print(f"  [{lr.name}] omega={lr.scores.get('omega', 0.0):.4f}")

    # Quick comparison
    print("\nΔΩ (prime - composite) ≈ "
          f"{res_prime.omega - res_comp.omega:.4f}")


if __name__ == "__main__":
    demo()