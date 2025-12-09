"""
OMNIA_TOTALE v2.0 — unified Ω-fusion demo

Thin script on top of the omnia package.

- Uses omnia.engine.run_omnia_totale as high-level entrypoint.
- Demonstrates how BASE (Omniabase), TIME (Omniatempo) and
  CAUSA (Omniacausa) are fused into a single Ω-score.

Author: Massimiliano Brighindi (MB-X.01 / Omniabase±)
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from omnia.engine import run_omnia_totale
from omnia.kernel import KernelResult


def build_demo_inputs() -> Dict[str, object]:
    """
    Build synthetic demo inputs:

    - n: an integer to analyse structurally (prime vs composite)
    - series: 1D time series with a regime shift
    - series_dict: multichannel series with a lagged causal link
    """
    import math

    n_prime = 173
    # you can change to 180 to see a composite behaviour
    n = n_prime

    t = np.arange(300)

    # Time series with a late regime shift
    series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    # Causal toy: s1 → s2 with lag 2, s3 = noise
    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)

    series_dict: Dict[str, Iterable[float]] = {"s1": s1, "s2": s2, "s3": s3}

    return {
        "n": n,
        "series": series,
        "series_dict": series_dict,
    }


def demo() -> KernelResult:
    """
    Run the OMNIA_TOTALE v2.0 demo:

    - builds synthetic inputs
    - calls run_omnia_totale(...)
    - prints fused Ω and per-lens scores
    - returns KernelResult for further inspection
    """
    np.random.seed(0)

    inputs = build_demo_inputs()
    result: KernelResult = run_omnia_totale(
        n=inputs["n"],
        series=inputs["series"],
        series_dict=inputs["series_dict"],
        w_base=1.0,
        w_tempo=1.0,
        w_causa=1.0,
    )

    print("=== OMNIA_TOTALE v2.0 demo (engine-based) ===")
    print(f"n = {inputs['n']}")
    print(f"fused Ω = {result.fused_omega:.4f}")
    print("per-lens scores:")

    for name, lens_res in result.lenses.items():
        omega = lens_res.scores.get("omega", 0.0)
        print(f"  - {name}: omega = {omega:.4f}, scores = {lens_res.scores}")

    return result


if __name__ == "__main__":
    demo()