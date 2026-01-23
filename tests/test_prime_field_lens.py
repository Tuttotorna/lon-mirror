# tests/test_prime_field_lens.py
from __future__ import annotations

from omnia.lenses.prime_field_lens import PrimeFieldLens


def test_prime_field_lens_determinism():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lens = PrimeFieldLens(k=2)

    r1 = lens.measure(primes).omega_score
    r2 = lens.measure(primes).omega_score

    assert r1 == r2


def test_prime_field_lens_order_invariant():
    primes_a = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    primes_b = list(reversed(primes_a))

    lens = PrimeFieldLens(k=2)

    r1 = lens.measure(primes_a).omega_score
    r2 = lens.measure(primes_b).omega_score

    assert r1 == r2


def test_prime_field_lens_bounds():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lens = PrimeFieldLens(k=2)

    omega = lens.measure(primes).omega_score
    assert 0.0 <= omega <= 1.0


def test_prime_field_lens_handles_small_inputs():
    lens = PrimeFieldLens(k=2)

    # empty
    r0 = lens.measure([]).omega_score
    assert 0.0 <= r0 <= 1.0

    # single prime (degenerate neighborhood)
    r1 = lens.measure([2]).omega_score
    assert 0.0 <= r1 <= 1.0