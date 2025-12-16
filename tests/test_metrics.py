# tests/test_metrics.py
# OMNIA metrics tests — stronger invariants (reviewer-grade)
# Focus: algebraic identities, edge cases, numerical stability, and API contract.

import math

import pytest

from omnia.metrics import (
    EPS,
    truth_omega,
    co_plus,
    score_plus,
    delta_coherence,
    kappa_alignment,
    epsilon_drift,
    compute_metrics,
)


def approx(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


# -----------------------------
# TruthOmega / CoPlus invariants
# -----------------------------

def test_truth_omega_basic_edges():
    # coherence=1 => TruthOmega=0
    assert approx(truth_omega(1.0), 0.0, tol=1e-15)

    # coherence=0 => uses EPS clamp => -log(EPS)
    t0 = truth_omega(0.0)
    assert t0 > 0.0
    assert approx(t0, -math.log(EPS), tol=1e-12)

    # coherence<0 behaves like 0 after clamp
    tneg = truth_omega(-123.0)
    assert approx(tneg, -math.log(EPS), tol=1e-12)

    # coherence>1 clamps to 1
    assert approx(truth_omega(2.0), 0.0, tol=1e-15)


def test_co_plus_inverse_mapping_and_clipping():
    # CoPlus(exp(-TruthOmega)) should map back to the clamped coherence
    for c in [1.0, 0.9, 0.5, 0.1, 1e-9, 0.0, -1.0, 2.0]:
        t = truth_omega(c)
        cp = co_plus(t)

        # expected: clamp coherence to [EPS, 1], but co_plus is clipped to [0,1]
        expected = max(EPS, min(1.0, float(c)))
        assert abs(cp - expected) <= 1e-12


def test_truth_omega_monotonicity():
    # If coherence decreases, TruthOmega increases (monotonic inverse)
    c1 = 0.9
    c2 = 0.3
    assert truth_omega(c2) > truth_omega(c1)


def test_co_plus_monotonicity():
    # If TruthOmega increases, CoPlus decreases
    t1 = 0.1
    t2 = 3.0
    assert co_plus(t2) < co_plus(t1)


# -----------------------------
# ScorePlus properties
# -----------------------------

def test_score_plus_range_and_behavior():
    # score_plus must stay in [0,1]
    xs = [-10.0, -1.0, 0.0, 0.2, 1.0, 3.0, 10.0]
    for x in xs:
        s = score_plus(x, bias=0.0, info=1.0)
        assert 0.0 <= s <= 1.0

    # info negative should not invert, should floor at 0
    assert approx(score_plus(0.7, bias=0.0, info=-5.0), 0.0, tol=1e-15)

    # bias can push above/below but output clipped
    assert approx(score_plus(0.1, bias=10.0, info=1.0), 1.0, tol=1e-15)
    assert approx(score_plus(0.9, bias=-10.0, info=1.0), 0.0, tol=1e-15)


# -----------------------------
# Delta-coherence invariants
# -----------------------------

def test_delta_coherence_empty_and_singleton():
    assert approx(delta_coherence([]), 0.0, tol=1e-15)
    assert approx(delta_coherence([0.123]), 0.0, tol=1e-15)


def test_delta_coherence_identical_values_zero():
    assert approx(delta_coherence([5.0, 5.0, 5.0, 5.0]), 0.0, tol=1e-15)


def test_delta_coherence_nonnegative_and_scale_invariant_like():
    # Non-negative always
    d = delta_coherence([1.0, 2.0, 3.0, 4.0])
    assert d >= 0.0

    # Multiplying all values by a positive factor should not change delta (normalized MAD)
    d2 = delta_coherence([10.0, 20.0, 30.0, 40.0])
    assert abs(d - d2) <= 1e-12


def test_delta_coherence_translation_sensitivity():
    # Adding a constant changes normalization, so delta can change (this is expected).
    d = delta_coherence([1.0, 2.0, 3.0, 4.0])
    d_shift = delta_coherence([101.0, 102.0, 103.0, 104.0])
    assert d_shift < d  # larger mean => smaller normalized dispersion


# -----------------------------
# Kappa-alignment invariants
# -----------------------------

def test_kappa_alignment_range_symmetry_and_identity():
    for a, b in [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (0.0, 0.0), (0.0, 5.0), (5.0, 0.0)]:
        k1 = kappa_alignment(a, b)
        k2 = kappa_alignment(b, a)
        assert 0.0 <= k1 <= 1.0
        assert abs(k1 - k2) <= 1e-15

    assert approx(kappa_alignment(3.14, 3.14), 1.0, tol=1e-15)


def test_kappa_alignment_extremes():
    # Max difference relative -> near 0
    k = kappa_alignment(0.0, 1.0)
    assert k <= 1e-12


# -----------------------------
# Epsilon-drift invariants
# -----------------------------

def test_epsilon_drift_zero_when_equal():
    assert approx(epsilon_drift(10.0, 10.0), 0.0, tol=1e-15)


def test_epsilon_drift_handles_prev_zero():
    # denom = abs(prev)+eps, with prev=0 uses eps => finite
    e = epsilon_drift(0.0, 1.0)
    assert e > 0.0
    assert math.isfinite(e)


# -----------------------------
# compute_metrics contract
# -----------------------------

def test_compute_metrics_keys_and_types_and_ranges():
    out = compute_metrics(0.8, bias=0.0, info=1.0, kappa_ref=1.0, eps_ref=0.0)

    expected_keys = {
        "truth_omega",
        "co_plus",
        "score_plus",
        "delta_coherence",
        "kappa_alignment",
        "epsilon_drift",
    }
    assert set(out.keys()) == expected_keys

    # Types
    for k in expected_keys:
        assert isinstance(out[k], float)

    # Ranges for probability-like signals
    assert out["truth_omega"] >= 0.0
    assert 0.0 <= out["co_plus"] <= 1.0
    assert 0.0 <= out["score_plus"] <= 1.0
    assert out["delta_coherence"] >= 0.0
    assert 0.0 <= out["kappa_alignment"] <= 1.0
    assert out["epsilon_drift"] >= 0.0


def test_compute_metrics_roundtrip_core_identity():
    # co_plus(truth_omega(c)) should equal clamp(c) under the same EPS rule.
    c = 0.37
    out = compute_metrics(c)
    expected = max(EPS, min(1.0, c))
    assert abs(out["co_plus"] - expected) <= 1e-12