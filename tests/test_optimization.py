"""
Benchmarks for Vectorized OMNIA components.
"""
import pytest
import numpy as np
import time
from omnia.core.omniacausa import omniacausa_analyze
from omnia.core.omniatempo import omniatempo_analyze

def test_omniacausa_vectorized_speed():
    # Generate synthetic data: 10 series, 1000 steps
    N = 10
    T = 1000
    data = {f"s_{i}": np.random.randn(T) for i in range(N)}

    start = time.time()
    res = omniacausa_analyze(data, max_lag=5)
    end = time.time()

    duration = end - start
    print(f"Vectorized Causa (10x1000): {duration:.4f}s")

    # Assert war speed: Should be essentially instant
    assert duration < 0.1
    assert isinstance(res.edges, list)

def test_omniacausa_correctness():
    # Create clear lagged relationship
    # s2 = s1 shifted by 2
    t = np.arange(100)
    s1 = np.sin(t)
    s2 = np.roll(s1, 2) # s2[2] == s1[0] -> s1 leads s2 by 2
    # roll rotates, so endpoints are messy. Let's fix.
    s2[:2] = 0

    data = {"s1": s1, "s2": s2}
    res = omniacausa_analyze(data, max_lag=5, strength_threshold=0.9)

    # We expect edge s1 -> s2 with lag +2 or s2 -> s1 with lag -2
    found = False
    for e in res.edges:
        if e.source == "s1" and e.target == "s2":
            # omniacausa: positive lag means src leads tgt
            # corr(s1[:-2], s2[2:]) should be 1.0
            if e.lag == 2 and e.strength > 0.9:
                found = True
    assert found

def test_omniatempo_vectorized_speed():
    data = np.random.randn(10000)
    start = time.time()
    res = omniatempo_analyze(data, short_window=50, long_window=200)
    end = time.time()

    duration = end - start
    print(f"Vectorized Tempo (10k): {duration:.4f}s")
    assert duration < 0.05
    assert res.global_mean != 0.0
