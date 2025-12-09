"""
quick_omnia_test.py

Minimal sanity check for the OMNIA package:
- imports the engine
- runs one fused Ω evaluation
- prints the KernelResult
"""

import numpy as np

from omnia.engine import run_omnia_totale


def main():
    # Simple numeric target (prime-like)
    n = 173

    # Simple time series with a regime shift
    t = np.arange(300)
    series = np.sin(t / 15.0) + 0.05 * np.random.normal(size=t.size)
    series[200:] += 0.8  # regime shift

    # Two correlated channels for Omniacausa
    s1 = np.sin(t / 10.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    series_dict = {"s1": s1, "s2": s2}

    result = run_omnia_totale(
        n=n,
        series=series,
        series_dict=series_dict,
        w_base=1.0,
        w_tempo=1.0,
        w_causa=1.0,
    )

    print("=== QUICK OMNIA TEST ===")
    print(result)


if __name__ == "__main__":
    main()