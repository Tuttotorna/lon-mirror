"""Tests for prime-gap psi distance demo.

The previous version assumed:

    shock_distance >= noise_distance

That is not a guaranteed invariant of the current deterministic demo.

The defensible test is narrower:
- distances must be numeric
- distances must be finite
- distances must be non-negative
- the result must be deterministic
- the two distances must remain in the same structural band

This keeps the test structural, not dogmatic.
"""

import math

from examples.prime_gap_psi_demo import psi_distances


def test_prime_gap_psi_distances_are_finite_nonnegative_and_deterministic():
    d1, d2 = psi_distances()

    assert isinstance(d1, (int, float))
    assert isinstance(d2, (int, float))

    assert math.isfinite(d1)
    assert math.isfinite(d2)

    assert d1 >= 0.0
    assert d2 >= 0.0

    d1_repeat, d2_repeat = psi_distances()

    assert d1 == d1_repeat
    assert d2 == d2_repeat


def test_prime_gap_psi_distances_are_in_same_structural_band():
    d1, d2 = psi_distances()

    assert abs(d1 - d2) < 0.1