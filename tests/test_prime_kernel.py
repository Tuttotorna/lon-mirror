# tests/test_prime_kernel.py
from __future__ import annotations

from omnia.engine.prime_kernel import PrimeKernel


def test_prime_kernel_determinism():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    k = PrimeKernel()
    r1 = k.measure(primes).omega_prime
    r2 = k.measure(primes).omega_prime
    assert r1 == r2


def test_prime_kernel_order_invariant():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    k = PrimeKernel()
    r1 = k.measure(primes).omega_prime
    r2 = k.measure(list(reversed(primes))).omega_prime
    assert r1 == r2


def test_prime_kernel_bounds():
    primes = [2, 3, 5, 7, 11, 13, 17]
    k = PrimeKernel()
    omega = k.measure(primes).omega_prime
    assert 0.0 <= omega <= 1.0