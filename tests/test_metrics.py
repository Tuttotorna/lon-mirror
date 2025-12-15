import math

from omnia import metrics as m


def test_truth_omega_zero_at_one():
    v = m.truth_omega(1.0)
    assert math.isfinite(v)
    assert abs(v) < 1e-12  # accetta anche -0.0


def test_truth_omega_monotonic_basic():
    # se l'input peggiora, TruthΩ non deve migliorare
    a = m.truth_omega(1.0)
    b = m.truth_omega(0.5)
    assert b >= a


def test_co_plus_range():
    c1 = m.co_plus(1.0)
    c2 = m.co_plus(0.5)
    assert 0.0 <= c1 <= 1.0
    assert 0.0 <= c2 <= 1.0