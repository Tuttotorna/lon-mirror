from __future__ import annotations

import math
import pytest

from omnia.lenses.prime_ck import PrimeCKLens


def _small_primes():
    # deterministic small basis
    return (2, 3, 5, 7, 11, 13, 17)


class TestPrimeCKLens:
    def test_determinism(self):
        lens = PrimeCKLens(_small_primes())
        r1 = lens.measure(12345)
        r2 = lens.measure(12345)
        assert r1.Ck == r2.Ck
        assert r1.dCk == r2.dCk
        assert r1.d2Ck == r2.d2Ck
        assert r1.residues == r2.residues

    def test_bounds_finite(self):
        lens = PrimeCKLens(_small_primes())
        r = lens.measure(2026)
        for v in r.Ck:
            assert math.isfinite(v)
            assert v >= 0.0
        for v in r.dCk:
            assert math.isfinite(v)
        for v in r.d2Ck:
            assert math.isfinite(v)

    def test_periodicity_mod_primorial(self):
        """
        Invariant: residues depend only on n mod M where M = ∏ primes.
        Therefore the entire Ck curve must be identical for n and n+M.
        """
        primes = _small_primes()
        M = 1
        for p in primes:
            M *= p

        lens = PrimeCKLens(primes)
        n = 1234
        r1 = lens.measure(n)
        r2 = lens.measure(n + M)

        assert r1.residues == r2.residues
        assert r1.Ck == r2.Ck
        assert r1.dCk == r2.dCk
        assert r1.d2Ck == r2.d2Ck

    def test_k_curve_lengths(self):
        primes = _small_primes()
        lens = PrimeCKLens(primes)
        r = lens.measure(999)

        # k from 3..K
        assert len(r.k_values) == len(primes) - 2
        assert len(r.Ck) == len(r.k_values)
        assert len(r.dCk) == len(r.k_values)
        assert len(r.d2Ck) == len(r.k_values)

    def test_input_validation(self):
        lens = PrimeCKLens(_small_primes())
        with pytest.raises(ValueError):
            lens.measure(-1)
        with pytest.raises(TypeError):
            lens.measure(1.5)  # type: ignore[arg-type]