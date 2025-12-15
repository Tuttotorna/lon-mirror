# tests/test_inevitability.py
# Test for Ω-INEV Inevitability Lens
# OMNIA / MB-X.01

import numpy as np
from omnia.inevitability import omega_inevitability


def test_inevitability_stable_structure():
    """
    A structure whose signature survives small perturbations
    must show high inevitability.
    """

    signal = np.array([1, 2, 3, 4, 5], dtype=float)

    def signature_fn(x):
        return np.std(x)

    perturbations = [
        lambda x: x + 0.0001,
        lambda x: x * 1.0001,
        lambda x: x[::-1],
    ]

    result = omega_inevitability(
        base_signal=signal,
        perturbations=perturbations,
        signature_fn=signature_fn,
        tolerance=1e-3
    )

    assert result.omega_inev >= 0.66


def test_inevitability_fragile_structure():
    """
    A structure whose signature breaks under perturbations
    must show low inevitability.
    """

    signal = np.array([1, 100, 1000, 10000], dtype=float)

    def signature_fn(x):
        return np.std(x)

    perturbations = [
        lambda x: x * 2,
        lambda x: x + np.random.normal(0, 50, size=len(x)),
        lambda x: x[::-1],
    ]

    result = omega_inevitability(
        base_signal=signal,
        perturbations=perturbations,
        signature_fn=signature_fn,
        tolerance=1e-6
    )

    assert result.omega_inev <= 0.3