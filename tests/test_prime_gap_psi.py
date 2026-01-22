# tests/test_prime_gap_psi.py
from __future__ import annotations

from examples.prime_gap_psi_demo import psi_distances


def test_prime_gap_psi_shock_ge_noise():
    # Deterministic gap-like sample (no randomness)
    gaps = [2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6, 6, 2, 6, 4, 2, 6, 4, 6, 8, 4, 2]

    report = psi_distances(gaps)

    d1 = report["psi"]["D1_R_to_N1"]
    d2 = report["psi"]["D2_R_to_N2"]

    # Minimal invariant: a large spike must not be "closer" than tiny +/-1 noise
    assert d2 >= d1, f"Shock distance should dominate: D2={d2} < D1={d1}"